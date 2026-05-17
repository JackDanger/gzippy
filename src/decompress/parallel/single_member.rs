//! Parallel single-member gzip decompression — rapidgzip-shaped port.
//!
//! Production path on x86_64 + ISA-L when num_threads > 1 and the
//! compressed stream exceeds 10 MiB. Routing lives in
//! [`crate::decompress::decompress_single_member`].
//!
//! This module is a thin driver. It parses the gzip header and trailer,
//! delegates to [`crate::decompress::parallel::chunk_fetcher::drive`]
//! for parallel decode, and verifies the CRC32 + ISIZE.
//!
//! The architecture (worker pool, prefetch loop, shared WindowMap, fast
//! and slow decode paths, async re-dispatch on speculative mismatch)
//! lives in [`crate::decompress::parallel::chunk_fetcher`]. See
//! `docs/rapidgzip-port-reference.md` for ground-truth alignment with
//! rapidgzip's C++.
//!
//! Streaming-write trade-off: bytes flow to the writer as each chunk
//! resolves, so a CRC/ISIZE mismatch at the end leaves partial output
//! behind. The routing layer treats CRC failures as terminal (corruption,
//! not dispatch failure); legitimate dispatch failures are mapped to
//! [`ParallelError::TooSmall`] before any bytes are written.

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::AtomicU64;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::atomic::Ordering;
use std::sync::Mutex;

const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;
const MIN_THREADS_FOR_PARALLEL: usize = 2;
const TARGET_COMPRESSED_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Successful runs of the parallel pipeline. Snapshot before/after a
/// decode to confirm production routing actually called us — see the
/// deletion-trap killer test in `src/tests/routing.rs`.
///
/// `pub(crate)` rather than `pub`: internal diagnostic surface, not a
/// library API.
pub(crate) static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

/// Mutex serializing routing tests that snapshot `MARKER_PIPELINE_RUNS`
/// against each other. Without this, `cargo test`'s default parallel
/// execution can mask a real silent-fallback regression with a false
/// positive.
pub(crate) static MARKER_PIPELINE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    let header_size = crate::decompress::parallel::marker_decode::skip_gzip_header(gzip_data)
        .map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::TooSmall);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    if deflate_data.len() < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::TooSmall);
    }

    // Trailer: gzip stores CRC32 then ISIZE (little-endian) in the last
    // 8 bytes.
    let crc_offset = gzip_data.len() - 8;
    let expected_crc = u32::from_le_bytes([
        gzip_data[crc_offset],
        gzip_data[crc_offset + 1],
        gzip_data[crc_offset + 2],
        gzip_data[crc_offset + 3],
    ]);
    let isize_offset = gzip_data.len() - 4;
    let expected_size = u32::from_le_bytes([
        gzip_data[isize_offset],
        gzip_data[isize_offset + 1],
        gzip_data[isize_offset + 2],
        gzip_data[isize_offset + 3],
    ]) as usize;

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    {
        use crate::decompress::parallel::chunk_data::ChunkConfiguration;
        use crate::decompress::parallel::chunk_fetcher;

        let configuration = ChunkConfiguration {
            split_chunk_size: TARGET_COMPRESSED_CHUNK_BYTES,
            max_decoded_chunk_size: 20 * TARGET_COMPRESSED_CHUNK_BYTES,
            crc32_enabled: true,
        };

        let (total_crc, total_size) =
            chunk_fetcher::drive(deflate_data, writer, num_threads, configuration).map_err(
                |e| {
                    if debug_enabled() {
                        eprintln!("[parallel_sm] fetcher error: {e:?}");
                    }
                    ParallelError::DecodeFailed
                },
            )?;

        if total_size != expected_size {
            return Err(ParallelError::SizeMismatch);
        }
        if total_crc != expected_crc {
            return Err(ParallelError::CrcMismatch);
        }

        MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
        if debug_enabled() {
            let total = t0.elapsed();
            let mbps = total_size as f64 / total.as_secs_f64() / 1e6;
            eprintln!(
                "[parallel_sm] total={:.1}ms isize={} ({:.0} MB/s)",
                total.as_secs_f64() * 1000.0,
                expected_size,
                mbps,
            );
        }
        return Ok(total_size as u64);
    }
    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    {
        let _ = (deflate_data, writer, expected_crc, expected_size, t0);
        Err(ParallelError::TooSmall)
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ParallelError {
    InvalidHeader,
    TooSmall,
    DecodeFailed,
    SizeMismatch,
    CrcMismatch,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl ParallelError {
    pub fn is_routing(&self) -> bool {
        matches!(self, ParallelError::TooSmall)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::TooSmall => write!(f, "file too small for parallel decode"),
            ParallelError::DecodeFailed => write!(f, "chunk decode failed"),
            ParallelError::SizeMismatch => write!(f, "output size mismatch"),
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch"),
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_input_returns_too_small() {
        let small = [0u8; 100];
        let mut out = Vec::new();
        let err = decompress_parallel(&small, &mut out, 4).unwrap_err();
        assert!(matches!(
            err,
            ParallelError::TooSmall | ParallelError::InvalidHeader
        ));
    }

    #[test]
    fn single_thread_returns_too_small() {
        // Construct a minimal-but-valid gzip stream of "hello world" so the
        // header check succeeds.
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&vec![0u8; 5_000_000]).unwrap();
        let gz = enc.finish().unwrap();
        let mut out = Vec::new();
        let err = decompress_parallel(&gz, &mut out, 1).unwrap_err();
        assert!(matches!(err, ParallelError::TooSmall));
    }
}
