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
//! lives in [`crate::decompress::parallel::chunk_fetcher`].
//!
//! Streaming-write trade-off: bytes flow to the writer as each chunk
//! resolves, so a CRC/ISIZE mismatch at the end leaves partial output
//! behind. The routing layer treats CRC failures as terminal corruption.
//! There is **no fallback**: if `decompress_parallel` returns Err the
//! caller surfaces it; the silent libdeflate retry that used to follow
//! has been removed.

use std::io::{self, Write};
use std::sync::atomic::AtomicU64;
#[cfg(parallel_sm)]
use std::sync::atomic::Ordering;
use std::sync::Mutex;

const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;
// 1 (was 2, 2026-05-31): the parallel-SM engine is the production path at EVERY
// thread count (MIN_PARALLEL_SM_THREADS=0, user directive). At num_threads=1 the
// pool has one worker and the consumer runs on the calling thread (2 OS threads,
// no worker==consumer deadlock), so the engine runs single-threaded rather than
// erroring "input below parallel SM minimum (routing bug)". This is what lets us
// measure the engine we are optimizing at T=1 instead of a libdeflate confound.
const MIN_THREADS_FOR_PARALLEL: usize = 1;
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const TARGET_COMPRESSED_CHUNK_BYTES: usize = 4 * 1024 * 1024;
/// Floor on the adjusted chunk size when the file is small.
/// Mirror of `512_Ki` literal at vendor's ParallelGzipReader.hpp:305.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const MIN_ADJUSTED_CHUNK_BYTES: usize = 512 * 1024;

/// Literal port of vendor's small-file chunk-size adjustment at
/// `ParallelGzipReader.hpp:294-306`:
///
/// ```cpp
/// if (fileSize && (m_chunkSizeInBytes * 2U * parallelization > *fileSize)) {
///     m_chunkSizeInBytes =
///         std::max(512_Ki,
///                  ceilDiv(ceilDiv(*fileSize, 3U * parallelization), 512_Ki) * 512_Ki);
/// }
/// ```
///
/// Without this, gzippy decompresses small-to-medium files with the
/// static 4 MiB chunk size, capping effective parallelism at
/// `fileSize / 4 MiB` chunks. On the 221 MB / 10.7 MB-compressed
/// fixture: 4 MiB chunks → 4 chunks → ~4-way parallelism on a 16-core
/// machine. Vendor uses 21 chunks (`--verbose` stats: "Total Fetched:
/// 21") and runs at 4.79 CPUs utilized. After this adjustment, gzippy
/// has the same effective chunk count.
///
/// Formula: when default `chunk_size * 2 * num_threads > file_size`,
/// shrink the chunk size to spread the work across `~3 *
/// num_threads` chunks (vendor's "give the thread pool more time to
/// be filled out" — chosen empirically per the comment), with a
/// 512 KiB floor (block-finder overhead would dominate below that).
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
pub(crate) fn adjusted_chunk_size_bytes(
    file_size: usize,
    num_threads: usize,
    default_chunk_size: usize,
) -> usize {
    let threads = num_threads.max(1);
    if default_chunk_size.saturating_mul(2).saturating_mul(threads) <= file_size {
        return default_chunk_size;
    }
    let denom = 3 * threads;
    let inner = file_size.div_ceil(denom);
    let aligned = inner.div_ceil(MIN_ADJUSTED_CHUNK_BYTES) * MIN_ADJUSTED_CHUNK_BYTES;
    aligned.max(MIN_ADJUSTED_CHUNK_BYTES)
}

/// Counter incremented every time `adjusted_chunk_size_bytes` returns a
/// value strictly less than the default. Mirror of the
/// `PREFETCH_NEXT_FILESIZE_ACCEPT` / `UNSPLIT_BLOCKS_EMPLACED`
/// deletion-trap pattern — proves the adjustment branch is reached on
/// real production decodes.
#[cfg_attr(not(parallel_sm), allow(dead_code))] // incremented on the x86 SM path; routing traps read it under the same cfg
pub static ADJUSTED_CHUNK_SIZE_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Successful runs of the parallel pipeline. Snapshot before/after a
/// decode to confirm production routing actually called us — see the
/// deletion-trap killer test in `src/tests/routing.rs`.
///
/// `pub(crate)` rather than `pub`: internal diagnostic surface, not a
/// library API.
#[allow(dead_code)] // incremented by the x86_64+isal-compression decompress_parallel path; read by tests
pub(crate) static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

/// Mutex serializing routing tests that snapshot `MARKER_PIPELINE_RUNS`
/// against each other. Without this, `cargo test`'s default parallel
/// execution can mask a real silent-fallback regression with a false
/// positive.
#[allow(dead_code)] // wired by #[cfg(test)] consumers in src/tests/routing.rs + src/decompress/mod.rs
pub(crate) static MARKER_PIPELINE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
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
    out_fd: Option<i32>,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    // Routing-eligibility gate — classifier upstream guarantees these
    // bounds; reaching this point with bad inputs is a routing bug
    // surfaced as a hard error. There is no silent retry.
    let header_size = skip_gzip_header(gzip_data).map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let deflate_data_len = gzip_data.len().saturating_sub(header_size + trailer_size);
    if deflate_data_len < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::InvalidGzipFormat);
    }

    #[cfg(parallel_sm)]
    {
        use crate::decompress::parallel::sm_driver::{read_parallel_sm, ReadParallelSmError};

        // Production driver: `sm_driver::read_parallel_sm` → `chunk_fetcher::drive`.
        // `single_member::decompress_parallel` is now a thin classifier-
        // routed wrapper: it owns the routing-eligibility gate and the
        // `MARKER_PIPELINE_RUNS` counter; the trailer parsing + CRC /
        // ISIZE verification + chunk_fetcher::drive orchestration all
        // live in the new driver (mirror of vendor's
        // `ParallelGzipReader::read` at ParallelGzipReader.hpp:553-646).
        // Granularity probe (2026-05-29): GZIPPY_CHUNK_KIB overrides the
        // 4 MiB default chunk target so a T=16 chunk-count sweep can
        // discriminate "T16 regression is straggler/granularity" from
        // "T16 regression is HT microarchitecture" without a rebuild per
        // size. Falls back to TARGET_COMPRESSED_CHUNK_BYTES when unset.
        let default_chunk = std::env::var("GZIPPY_CHUNK_KIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(TARGET_COMPRESSED_CHUNK_BYTES);
        let chunk_size = adjusted_chunk_size_bytes(gzip_data.len(), num_threads, default_chunk);
        if chunk_size < TARGET_COMPRESSED_CHUNK_BYTES {
            ADJUSTED_CHUNK_SIZE_APPLIED.fetch_add(1, Ordering::Relaxed);
        }

        // No pool pre-warm here. A prior experiment touched pool pages
        // on the consumer thread before workers spawn; 20-trial bench
        // on neurotic measured a -50% SM regression because every fresh
        // CLI process paid the pre-touch cost without amortization. The
        // page-fault gap vs vendor (40% gzippy vs 17% rapidgzip) needs a
        // real per-Vec allocator (allocator-api2 + rpmalloc-rs) or
        // daemon-mode CLI to close; not a pre-touch loop. See module
        // docs at `chunk_buffer_pool.rs:57-77`.
        let result =
            read_parallel_sm(gzip_data, writer, out_fd, num_threads, chunk_size).map_err(|e| {
                if debug_enabled() {
                    eprintln!("[parallel_sm] driver error: {e}");
                }
                match e {
                    ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
                    ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
                    ReadParallelSmError::DecodeFailed(detail) => {
                        ParallelError::DecodeFailed(detail)
                    }
                    ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
                    ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
                }
            })?;

        MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
        // MECHANISM instrumentation dump (GZIPPY_MARKER_STATS=1). No-op when
        // the env var is unset. Counts are process-global atomics summed across
        // all worker threads.
        crate::decompress::parallel::deflate_block::marker_instr::dump();
        if debug_enabled() {
            let total = t0.elapsed();
            let mbps = result.total_size as f64 / total.as_secs_f64() / 1e6;
            eprintln!(
                "[parallel_sm:v0.6] total={:.1}ms isize={} ({:.0} MB/s)",
                total.as_secs_f64() * 1000.0,
                result.total_size,
                mbps,
            );
        }
        Ok(result.total_size as u64)
    }
    #[cfg(not(parallel_sm))]
    {
        let _ = (writer, out_fd, t0, deflate_data_len);
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
    /// Carries `sm_driver` / `chunk_fetcher` `Debug` detail (e.g.
    /// `Decode(InflateFailed(InvalidBlock))`).
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    DecodeFailed(String),
    /// Output size doesn't match the gzip ISIZE trailer — corruption.
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    SizeMismatch,
    /// CRC32 doesn't match the gzip CRC trailer — corruption.
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    CrcMismatch,
    /// Build doesn't support the parallel pipeline on this platform
    /// (no x86_64 + ISA-L). The classifier never routes here on
    /// unsupported builds; this exists only as the cfg-stubbed body's
    /// guaranteed error path.
    #[allow(dead_code)] // non-SM-build cfg stub; constructed only off the x86+isal path
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
            ParallelError::DecodeFailed(detail) => write!(f, "chunk decode failed: {detail}"),
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
    fn adjusted_chunk_size_keeps_default_on_large_files() {
        // File big enough that chunkSize * 2 * threads <= fileSize.
        // 4 MiB * 2 * 16 = 128 MiB; pick fileSize = 256 MiB.
        let file_size = 256 * 1024 * 1024;
        let got = adjusted_chunk_size_bytes(file_size, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(got, TARGET_COMPRESSED_CHUNK_BYTES);
    }

    #[test]
    fn adjusted_chunk_size_shrinks_for_221mb_fixture() {
        // The bench fixture: 10.7 MB compressed, 16 threads.
        // Vendor formula: max(512 KiB, ceilDiv(ceilDiv(10726414, 48), 512 KiB) * 512 KiB)
        // ceilDiv(10726414, 48) = 223468
        // ceilDiv(223468, 524288) = 1
        // 1 * 524288 = 524288 = 512 KiB
        let got = adjusted_chunk_size_bytes(10_726_414, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(got, 512 * 1024, "expected 512 KiB floor for 10.7 MB / T=16");
        // 10.7 MB / 512 KiB = ~20.5 chunks (matches vendor's "Total Fetched: 21").
    }

    #[test]
    fn adjusted_chunk_size_scales_with_threads() {
        // For a fixed file size, more threads should produce smaller chunks
        // (more parallelism), but never below the 512 KiB floor.
        let file_size = 80 * 1024 * 1024;
        let t4 = adjusted_chunk_size_bytes(file_size, 4, TARGET_COMPRESSED_CHUNK_BYTES);
        let t16 = adjusted_chunk_size_bytes(file_size, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        let t64 = adjusted_chunk_size_bytes(file_size, 64, TARGET_COMPRESSED_CHUNK_BYTES);
        assert!(t4 >= t16, "more threads → smaller (or equal) chunks");
        assert!(t16 >= t64);
        assert!(t64 >= 512 * 1024, "never below 512 KiB floor");
    }

    #[test]
    fn small_input_returns_hard_error() {
        let small = [0u8; 100];
        let mut out = Vec::new();
        let err = decompress_parallel(&small, &mut out, None, 4).unwrap_err();
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
        let err = decompress_parallel(&gz, &mut out, None, 1).unwrap_err();
        assert!(matches!(err, ParallelError::InvalidGzipFormat));
    }
}
