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
/// Floor on the adjusted chunk size when the file is small.
/// Mirror of `512_Ki` literal at vendor's ParallelGzipReader.hpp:305.
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
pub static ADJUSTED_CHUNK_SIZE_APPLIED: AtomicU64 = AtomicU64::new(0);

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

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    {
        use crate::decompress::parallel::parallel_gzip_reader::{
            read_parallel_sm, ReadParallelSmError,
        };

        // Audit step 7 — driver moves into `parallel_gzip_reader::read_parallel_sm`.
        // `single_member::decompress_parallel` is now a thin classifier-
        // routed wrapper: it owns the routing-eligibility gate and the
        // `MARKER_PIPELINE_RUNS` counter; the trailer parsing + CRC /
        // ISIZE verification + chunk_fetcher::drive orchestration all
        // live in the new driver (mirror of vendor's
        // `ParallelGzipReader::read` at ParallelGzipReader.hpp:553-646).
        let chunk_size =
            adjusted_chunk_size_bytes(gzip_data.len(), num_threads, TARGET_COMPRESSED_CHUNK_BYTES);
        if chunk_size < TARGET_COMPRESSED_CHUNK_BYTES {
            ADJUSTED_CHUNK_SIZE_APPLIED.fetch_add(1, Ordering::Relaxed);
        }

        // **Pre-warm intentionally NOT called for CLI invocations.**
        //
        // First prototype called `chunk_buffer_pool::prewarm(N+2,
        // chunk_size*4)` to serialize page-fault cost onto the consumer
        // thread before workers spawn. Measured on neurotic x86_64
        // silesia-large at T=16, 20 trials:
        //
        //   without prewarm: 666 MB/s SM throughput (0.44× vendor)
        //   with prewarm:    329 MB/s            (0.26× vendor)
        //
        // The bench creates a fresh process per trial; pre-warming
        // ~864 MiB on the consumer thread (u8 + u16 pools × chunk_size*4
        // × N+2 buffers × 1+2 bytes) adds ~170 ms of pure overhead to a
        // ~750 ms decode. Total work increases.
        //
        // Vendor's per-process advantage isn't pre-warming; it's
        // rpmalloc handling `mmap` differently — large pre-mapped
        // arenas parcel out warm pages without page-faulting per
        // allocation. Stable Rust can't easily port rpmalloc's
        // allocator semantics. Future options if this band needs to
        // close:
        //   - Daemon-mode CLI (single process, many decodes): prewarm
        //     fires once and amortizes across files.
        //   - Custom `Allocator` parameter on `Vec` when allocator_api
        //     stabilizes — wrap rpmalloc-rs as a per-Vec allocator.
        //   - `mmap(MAP_POPULATE)` for the largest buffers via memmap2.
        //
        // The `prewarm` function in chunk_buffer_pool stays available
        // for daemon-mode callers; only the CLI dispatch leaves it
        // unfired. The 50% regression was caught by 20-trial bench;
        // the comment + this empty branch document why future ports
        // should not re-introduce the call without daemon-mode wiring.
        let result = read_parallel_sm(gzip_data, writer, num_threads, chunk_size).map_err(|e| {
            if debug_enabled() {
                eprintln!("[parallel_sm] driver error: {e}");
            }
            match e {
                ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
                ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
                ReadParallelSmError::DecodeFailed(_) => ParallelError::DecodeFailed,
                ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
                ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
            }
        })?;

        MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
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
    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    {
        let _ = (writer, t0, deflate_data_len);
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

    // Note: `test_window_map_uses_compressed_vector` (spec §I new
    // tests #5) lives in `decompress::tests` — the type-level fence
    // there asserts the same `WindowMap` default = `CompressionType::Zlib`.

    /// Spec §I "Tests required (new)" #6 — multi-stream gzip fed
    /// directly to the parallel SM path. The classifier routes
    /// multi-member to bgzf, so this test bypasses routing and calls
    /// `decompress_parallel` directly. After the cutover, the
    /// multi-stream Footer loop in `gzip_chunk::decode_chunk_with_window`
    /// reads each per-stream gzip footer via
    /// `IsalInflateWrapper::read_footer_at_current`, calls
    /// `reset_for_next_stream`, and parses the next gzip header via
    /// `gzip_format::read_header` — producing byte-correct output even
    /// though our trailer-validation in `decompress_parallel` is
    /// keyed off the last 8 bytes (the FINAL stream's footer). The
    /// per-stream CRCs are combined through `chunk.crc32s` +
    /// `ChunkData::append_footer`; the cumulative CRC matches.
    #[test]
    fn test_multi_stream_in_parallel_sm() {
        #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
        {
            use std::io::Write as _;
            // Build a multi-member gzip: enough total compressed size
            // to exceed MIN_PARALLEL_SIZE (4 MiB). Three 2 MiB
            // sub-streams concatenated, all from the same plaintext
            // generator so byte-perfect verification is possible.
            let mut original: Vec<u8> = Vec::new();
            let mut rng: u64 = 0xfacefacefaceface;
            for _ in 0..(8 * 1024 * 1024) {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                original.push((rng >> 24) as u8);
            }
            let stream_size = original.len() / 3;
            let mut multi: Vec<u8> = Vec::new();
            for chunk in original.chunks(stream_size) {
                let mut enc =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
                enc.write_all(chunk).unwrap();
                multi.extend_from_slice(&enc.finish().unwrap());
            }
            // For the trailer to validate against the FULL output we'd
            // need a single-member gzip. Multi-stream's final footer's
            // CRC + ISIZE cover only its own stream. So in this test
            // we accept that `decompress_parallel` may return
            // `CrcMismatch` or `SizeMismatch` — what we're locking in
            // is the BYTES that did make it to the writer up until
            // that mismatch. The decoder itself must drive through all
            // streams via the Footer loop; bytes accumulate to the
            // full original.
            //
            // If `decompress_parallel` returned Ok, even better — that
            // means the trailer happened to match (vanishingly
            // unlikely on random data). Either way, the decoded bytes
            // must equal `original`.
            let mut out: Vec<u8> = Vec::new();
            let res = decompress_parallel(&multi, &mut out, 4);
            // Accept Ok OR Err(CrcMismatch|SizeMismatch) — both are
            // consistent with the parallel SM path having driven the
            // multi-stream loop end-to-end via Footer reads. The
            // critical assertion is that the BYTES match.
            match res {
                Ok(_) => {}
                Err(ParallelError::CrcMismatch) | Err(ParallelError::SizeMismatch) => {}
                Err(other) => {
                    panic!("multi-stream parallel SM unexpectedly errored: {other:?}")
                }
            }
            assert_eq!(
                out,
                original,
                "multi-stream parallel SM bytes mismatch (got {} bytes, want {})",
                out.len(),
                original.len(),
            );
        }
        #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
        let _ = (); // x86_64 + ISA-L only — no-op elsewhere
    }
}
