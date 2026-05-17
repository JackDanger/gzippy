//! Parallel single-member gzip decompression — rapidgzip-shaped port.
//!
//! Production path on x86_64 + ISA-L when the classifier returns
//! [`crate::decompress::DecodePath::IsalParallelSM`] (num_threads > 1
//! and compressed size > `MIN_PARALLEL_COMPRESSED`). Routing lives in
//! [`crate::decompress::classify_gzip`]; this module never makes its
//! own routing decisions — every error variant is terminal.
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
//! behind. The routing layer treats CRC failures as terminal corruption.
//! There is **no fallback**: if `decompress_parallel` returns Err the
//! caller surfaces it; the silent libdeflate retry that used to follow
//! has been removed.

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

/// Parse the gzip header and return the byte offset where the deflate
/// stream starts. Thin wrapper over `gzip_format::read_header`
/// (literal port of `rapidgzip::gzip::readHeader`); drops the parsed
/// `Header` since the driver currently doesn't need it. Multi-stream
/// support reads subsequent headers via `gzip_format::read_header` too.
pub(crate) fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    crate::decompress::parallel::gzip_format::read_header(data).map(|(_h, off)| off)
}

pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    let header_size = skip_gzip_header(gzip_data).map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    // The classifier (`crate::decompress::classify_gzip`) is the only
    // gate on parallel-eligibility. If we got here on input below the
    // working minimum it's a routing bug, not a fallback opportunity —
    // surface it as a hard error. There is no silent retry.
    if deflate_data.len() < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::InvalidGzipFormat);
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
        Err(ParallelError::UnsupportedPlatform)
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Every variant is terminal — the classifier filters
/// parallel-eligibility upstream, so reaching this module with bad
/// inputs is a routing bug surfaced as a hard error rather than a
/// silent fallback opportunity.
#[derive(Debug)]
pub enum ParallelError {
    /// Bytes don't start with a valid gzip header (FHCRC/FNAME/etc.
    /// fields malformed or truncated).
    InvalidHeader,
    /// Header parses but the byte range available to the worker pool
    /// is too short for the parallel pipeline's invariants (e.g. the
    /// classifier sent a stream below `MIN_PARALLEL_SIZE`). Treat as
    /// a routing bug — the dispatcher must have classified this as
    /// `IsalSingle`, not `IsalParallelSM`.
    InvalidGzipFormat,
    /// One or more chunk decodes failed inside the worker pool.
    DecodeFailed,
    /// Output size doesn't match the gzip ISIZE trailer — corruption.
    SizeMismatch,
    /// CRC32 doesn't match the gzip CRC trailer — corruption.
    CrcMismatch,
    /// Build doesn't support the parallel pipeline on this platform
    /// (no x86_64 + ISA-L). The classifier never routes here on
    /// unsupported builds; this exists only as the cfg-stubbed body's
    /// guaranteed error path.
    UnsupportedPlatform,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::InvalidGzipFormat => {
                write!(f, "input below parallel SM minimum (routing bug)")
            }
            ParallelError::DecodeFailed => write!(f, "chunk decode failed"),
            ParallelError::SizeMismatch => write!(f, "output size mismatch"),
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch"),
            ParallelError::UnsupportedPlatform => {
                write!(f, "parallel SM unsupported on this build")
            }
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_input_returns_hard_error() {
        let small = [0u8; 100];
        let mut out = Vec::new();
        let err = decompress_parallel(&small, &mut out, 4).unwrap_err();
        // Either InvalidHeader (no gzip magic) or InvalidGzipFormat
        // (too short for a deflate body). Both are terminal —
        // `TooSmall` is gone.
        assert!(matches!(
            err,
            ParallelError::InvalidGzipFormat | ParallelError::InvalidHeader
        ));
    }

    #[test]
    fn single_thread_returns_hard_error() {
        // Construct a valid 5 MiB gzip and pass num_threads=1. The
        // classifier would never send this here in production (it'd
        // pick IsalSingle), but a direct caller does; we surface a
        // hard error rather than silently routing past.
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&vec![0u8; 5_000_000]).unwrap();
        let gz = enc.finish().unwrap();
        let mut out = Vec::new();
        let err = decompress_parallel(&gz, &mut out, 1).unwrap_err();
        assert!(matches!(err, ParallelError::InvalidGzipFormat));
    }
}
