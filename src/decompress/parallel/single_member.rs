//! Parallel single-member gzip decompression — rapidgzip-shaped port.
//!
//! Production path on x86_64/arm64 with `pure-rust-inflate` when the classifier
//! returns [`crate::decompress::DecodePath::ParallelSM`] (parallel SM
//! enabled and compressed size > `MIN_PARALLEL_COMPRESSED`). Routing lives in
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

// (Removed 2026-06-04, task #8) `MIN_PARALLEL_SIZE`: was a 4 MiB floor below
// which a C-FFI one-shot decoded small inputs. The ParallelSM pipeline is now
// the SOLE single-member path at any size (verified byte-exact for tiny /
// incompressible / stored at T1+T4), so there is no floor and no one-shot FFI
// fallback. (That pipeline is pure-Rust on gzippy-native; on gzippy-isal its
// clean tail decodes via ISA-L FFI — see gzip_chunk.rs `finish_decode_chunk_impl`.)
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

/// Step-13 decoded-size-adaptive chunk policy: compile-time DECODED-bytes target.
///
/// rapidgzip (and gzippy's base) split the COMPRESSED stream at a fixed spacing
/// (`TARGET_COMPRESSED_CHUNK_BYTES`, 4 MiB). Step-12 Gate-2-confirmed that the
/// parallelism-relevant quantity is the DECODED chunk size: smaller decoded
/// chunks raise P (avg busy CPUs) and drop the multi-thread wall *where it
/// helps* — but the sweet spot tracks decoded bytes, which is corpus×T-dependent
/// because the compression ratio differs per corpus. This policy derives the
/// compressed spacing from a fixed DECODED target:
///   `compressed_chunk = target_decoded / ratio`,  `ratio = ISIZE / comp_len`.
///
/// `0` = adaptive DISABLED → ship the fixed 4 MiB compressed spacing (the base).
/// A nonzero value (bytes) turns the policy on by default. Always overridable
/// per-run via `GZIPPY_TARGET_DECODED_KIB` (the step-13 sweep knob that selects
/// the most ROBUST target before it is baked here).
///
/// SELECTED = 8 MiB (step-13 frozen-Intel matrix sweep, 2026-06-16, N=9
/// interleaved, sha-verified, A/A rig PASS every cell). 8 MiB decoded is the
/// most ROBUST of {4,5,6,8} MiB — the ONLY target that WINS silesia-T7 (−9.0%
/// sig, P 4.84→5.54, gz/rg 1.18→1.076) AND nasa-T4 (−11% sig, P 3.14→3.33)
/// while NOT regressing any cell (silesia-T4 +2.1% is within the 6.2% inter-run
/// spread ⇒ TIE; nasa-T7 / monorepo-T4 / monorepo-T7 TIE). Smaller targets
/// sig-regress silesia-T4 (d4096 +9.8%, d5120 +9.9%), and d5120 hits a
/// reproducible +80–120% chunk-boundary pathology at ~515 KiB on nasa. This is
/// the adaptive advantage no fixed compressed constant could achieve (it picks
/// silesia→2.6 MiB and nasa→0.8 MiB simultaneously). NOT-YET-LAW: single-arch
/// (Intel i7-13700T); AMD (solvency) replication owed before this is universal.
#[allow(dead_code)] // read by the parallel_sm decompress_parallel path
const TARGET_DECODED_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[allow(dead_code)]
fn env_kib(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Resolved decoded-size target for the adaptive policy: the per-run env knob
/// `GZIPPY_TARGET_DECODED_KIB` wins, else the compile-time default (when > 0),
/// else `None` (adaptive off → fixed compressed spacing).
#[allow(dead_code)]
fn adaptive_target_bytes() -> Option<usize> {
    if let Some(kib) = env_kib("GZIPPY_TARGET_DECODED_KIB") {
        return Some(kib * 1024);
    }
    if TARGET_DECODED_CHUNK_BYTES != 0 {
        return Some(TARGET_DECODED_CHUNK_BYTES);
    }
    None
}

/// Adaptive compressed chunk spacing derived from a decoded-size target.
///
/// `compressed_chunk = target_decoded / ratio = target_decoded * comp_len / isize`
/// (ratio = ISIZE / comp_len, guarded ≥ 1).
///
/// Edge cases (the policy NEVER produces a value that risks correctness — only
/// optimality — and falls back to the fixed default when the estimate is
/// untrustworthy):
/// * `isize == 0` or `comp_len == 0` → fixed default.
/// * `comp_len > u32::MAX` (member > 4 GiB compressed) → its true decoded size
///   is certainly > 4 GiB so the trailer ISIZE (mod 2^32) is a wrap → default.
/// * `isize < comp_len` (implied ratio < 1: incompressible/stored, OR a > 4 GiB
///   decoded member whose wrapped ISIZE landed below comp_len) → default.
/// * otherwise clamp the result to `[MIN_ADJUSTED_CHUNK_BYTES, fallback_default]`
///   — adaptive only ever SHRINKS the compressed spacing below the 4 MiB base
///   (high ratio ⇒ more, smaller chunks); it never grows it past the base, so a
///   low-ratio corpus can never be made *coarser* (and thus less parallel) than
///   today.
#[allow(dead_code)]
pub(crate) fn adaptive_compressed_chunk_bytes(
    isize_estimate: u64,
    comp_len: usize,
    target_decoded: usize,
    fallback_default: usize,
) -> usize {
    if isize_estimate == 0 || comp_len == 0 {
        return fallback_default;
    }
    if comp_len as u64 > u32::MAX as u64 {
        return fallback_default;
    }
    if isize_estimate < comp_len as u64 {
        return fallback_default;
    }
    let cc = (target_decoded as u128 * comp_len as u128 / isize_estimate as u128) as usize;
    cc.clamp(MIN_ADJUSTED_CHUNK_BYTES, fallback_default)
}

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

/// Cheap, sound guard for the multi-member resume path: does a SECOND gzip
/// member actually begin right after the first member's deflate body + trailer?
///
/// Walks the first member's deflate stream to its byte-aligned `BFINAL` end
/// (pure-Rust, bounded memory), then checks whether `[member1_end + 8..]` starts
/// with a gzip magic. Returns `false` for a genuinely-corrupt single member (the
/// walk errors) so the caller surfaces the original decode error instead of
/// attempting a pointless resume. Only invoked on the error path (rare).
#[cfg(parallel_sm)]
fn trailing_member_after_first(gzip_data: &[u8]) -> bool {
    let header_size = match skip_gzip_header(gzip_data) {
        Ok(h) => h,
        Err(_) => return false,
    };
    if gzip_data.len() < header_size + 8 {
        return false;
    }
    let deflate_len =
        match crate::decompress::scan_inflate::deflate_stream_byte_len(&gzip_data[header_size..]) {
            Ok(n) => n,
            Err(_) => return false,
        };
    let next = header_size + deflate_len + 8;
    next + 2 <= gzip_data.len() && gzip_data[next] == 0x1f && gzip_data[next + 1] == 0x8b
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
    let _deflate_data_len = gzip_data.len().saturating_sub(header_size + trailer_size);
    // No size floor (task #8: the ParallelSM pipeline is the sole single-member
    // path at any size — pure-Rust on gzippy-native, ISA-L clean tail on
    // gzippy-isal). Only num_threads is gated — T=0 is a caller bug.
    if num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::InvalidGzipFormat);
    }

    #[cfg(parallel_sm)]
    {
        use crate::decompress::parallel::sm_driver::{
            read_parallel_sm_capturing, read_parallel_sm_resume_multi, ReadParallelSmError,
        };

        // Production driver: `sm_driver::read_parallel_sm` → `chunk_fetcher::drive`.
        // `single_member::decompress_parallel` is now a thin classifier-
        // routed wrapper: it owns the routing-eligibility gate and the
        // `MARKER_PIPELINE_RUNS` counter; the trailer parsing + CRC /
        // ISIZE verification + chunk_fetcher::drive orchestration all
        // live in the new driver (mirror of vendor's
        // `ParallelGzipReader::read` at ParallelGzipReader.hpp:553-646).
        // ISIZE — decoded size mod 2^32 — from the gzip trailer (last 4 bytes
        // of the member, LE). The routing-eligibility gate above already proved
        // `gzip_data.len() >= header_size + 8`, so these indices are in bounds.
        // Feeds the step-13 decoded-size-adaptive chunk policy.
        let isize_estimate = {
            let n = gzip_data.len();
            u32::from_le_bytes([
                gzip_data[n - 4],
                gzip_data[n - 3],
                gzip_data[n - 2],
                gzip_data[n - 1],
            ]) as u64
        };

        // Chunk-spacing policy. Precedence:
        //   1. GZIPPY_CHUNK_KIB — explicit FIXED compressed spacing (KIB*1024).
        //      The granularity probe (2026-05-29) and the step-13 sweep BASE arm
        //      (=4096 ⇒ 4 MiB, identical to the compile-time base). Bypasses the
        //      adaptive computation entirely.
        //   2. GZIPPY_TARGET_DECODED_KIB / TARGET_DECODED_CHUNK_BYTES — step-13
        //      decoded-size-adaptive spacing: compressed = target_decoded / ratio
        //      so the chunk COUNT adapts to the corpus compression ratio.
        //   3. else the fixed 4 MiB base (TARGET_COMPRESSED_CHUNK_BYTES).
        let default_chunk = if let Some(kib) = env_kib("GZIPPY_CHUNK_KIB") {
            kib * 1024
        } else if let Some(target) = adaptive_target_bytes() {
            adaptive_compressed_chunk_bytes(
                isize_estimate,
                gzip_data.len(),
                target,
                TARGET_COMPRESSED_CHUNK_BYTES,
            )
        } else {
            TARGET_COMPRESSED_CHUNK_BYTES
        };
        let chunk_size = adjusted_chunk_size_bytes(gzip_data.len(), num_threads, default_chunk);
        if chunk_size < TARGET_COMPRESSED_CHUNK_BYTES {
            ADJUSTED_CHUNK_SIZE_APPLIED.fetch_add(1, Ordering::Relaxed);
        }
        if debug_enabled() {
            let n_chunks = gzip_data.len().div_ceil(chunk_size.max(1));
            eprintln!(
                "[parallel_sm] isize={} comp_len={} ratio={:.3} chunk_size={} (~{} KiB) est_chunks={}",
                isize_estimate,
                gzip_data.len(),
                isize_estimate as f64 / (gzip_data.len() as f64).max(1.0),
                chunk_size,
                chunk_size / 1024,
                n_chunks,
            );
        }

        // No pool pre-warm here. A prior experiment touched pool pages
        // on the consumer thread before workers spawn; 20-trial bench
        // on neurotic measured a -50% SM regression because every fresh
        // CLI process paid the pre-touch cost without amortization. The
        // page-fault gap vs vendor (40% gzippy vs 17% rapidgzip) needs a
        // real per-Vec allocator (allocator-api2 + rpmalloc-rs) or
        // daemon-mode CLI to close; not a pre-touch loop. See module
        // docs at `chunk_buffer_pool.rs:57-77`.
        // Decode as a single member, capturing the bytes streamed so far so the
        // multi-member resume can pick up past them on a boundary error.
        let mut bytes_written = 0usize;
        let sm_result = read_parallel_sm_capturing(
            gzip_data,
            writer,
            out_fd,
            num_threads,
            chunk_size,
            &mut bytes_written,
        );

        let result = match sm_result {
            Ok(r) => r,
            Err(e) => {
                if debug_enabled() {
                    eprintln!("[parallel_sm] driver error: {e}");
                }
                // A decode/size/CRC failure on a stream the classifier called
                // single-member is the multi-member-misroute signature: the
                // second member begins past the 16 MiB detection window, so the
                // single-stream finder cannot cross member 1's gzip footer.
                // Resume the remaining members (pure-Rust, per-member CRC+ISIZE
                // verified), skipping the validated prefix already streamed.
                // `InvalidHeader`/`InvalidFormat` are genuine malformation — not
                // resumable. (Truly corrupt single-member input also lands here;
                // the resume then finds no further valid member and surfaces the
                // original failure, never silently truncating.)
                let resumable = matches!(
                    e,
                    ReadParallelSmError::DecodeFailed(_)
                        | ReadParallelSmError::SizeMismatch { .. }
                        | ReadParallelSmError::CrcMismatch { .. }
                );
                if resumable && trailing_member_after_first(gzip_data) {
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm] single-member decode failed at multi-member \
                             boundary; resuming members past {bytes_written} streamed bytes"
                        );
                    }
                    read_parallel_sm_resume_multi(
                        gzip_data,
                        writer,
                        bytes_written,
                        num_threads,
                        chunk_size,
                    )
                    .map_err(|me| {
                        ParallelError::DecodeFailed(format!("multi-member resume: {me}"))
                    })?
                } else {
                    return Err(match e {
                        ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
                        ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
                        ReadParallelSmError::DecodeFailed(detail) => {
                            ParallelError::DecodeFailed(detail)
                        }
                        ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
                        ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
                    });
                }
            }
        };

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
    #[cfg(not(parallel_sm))]
    {
        let _ = (writer, out_fd, t0, _deflate_data_len);
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
    /// `SingleMember`, not `ParallelSM`.
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

    // ---- step-13 adaptive decoded-size chunk policy ----
    const D4: usize = 4 * 1024 * 1024; // fixed default / fallback

    #[test]
    fn adaptive_normal_ratio_shrinks_to_target_over_ratio() {
        // silesia-like: 211 MiB decoded, 68 MiB compressed → ratio ≈ 3.1.
        // target 6 MiB decoded → compressed ≈ 6/3.1 ≈ 1.94 MiB.
        let isize = 211 * 1024 * 1024;
        let comp = 68 * 1024 * 1024;
        let cc = adaptive_compressed_chunk_bytes(isize, comp, 6 * 1024 * 1024, D4);
        // 6Mi * 68Mi / 211Mi = 2_027_578 bytes ≈ 1.93 MiB
        assert!(
            (2_000_000..=2_060_000).contains(&cc),
            "silesia target-6 → ~1.93 MiB, got {cc}"
        );
        assert!(cc < D4, "adaptive shrinks below the 4 MiB base");
    }

    #[test]
    fn adaptive_high_ratio_clamps_to_floor() {
        // nasa-like ratio 10, but a huge decoded so the raw value < 512 KiB floor.
        // ratio 100: target 6 MiB → 60 KiB → clamped up to the 512 KiB floor.
        let cc =
            adaptive_compressed_chunk_bytes(100 * 1024 * 1024, 1024 * 1024, 6 * 1024 * 1024, D4);
        assert_eq!(cc, MIN_ADJUSTED_CHUNK_BYTES);
    }

    #[test]
    fn adaptive_nasa_ratio_ten_target_six() {
        // ratio 10 (nasa): target 6 MiB → 614 KiB compressed (above the floor).
        let isize = 200 * 1024 * 1024;
        let comp = 20 * 1024 * 1024;
        let cc = adaptive_compressed_chunk_bytes(isize, comp, 6 * 1024 * 1024, D4);
        assert!(
            (600_000..=640_000).contains(&cc),
            "nasa target-6 → ~614 KiB, got {cc}"
        );
    }

    #[test]
    fn adaptive_low_ratio_clamps_to_default_ceiling() {
        // ratio 1.1 (barely compressible): target 6 MiB → 5.45 MiB → clamp to 4 MiB.
        let cc = adaptive_compressed_chunk_bytes(
            110 * 1024 * 1024,
            100 * 1024 * 1024,
            6 * 1024 * 1024,
            D4,
        );
        assert_eq!(
            cc, D4,
            "low ratio never produces a chunk coarser than the base"
        );
    }

    #[test]
    fn adaptive_ratio_below_one_falls_back() {
        // incompressible/stored: ISIZE < comp_len (ratio < 1) → fixed default.
        let cc = adaptive_compressed_chunk_bytes(
            10 * 1024 * 1024,
            11 * 1024 * 1024,
            6 * 1024 * 1024,
            D4,
        );
        assert_eq!(cc, D4);
    }

    #[test]
    fn adaptive_zero_isize_falls_back() {
        assert_eq!(
            adaptive_compressed_chunk_bytes(0, 10 * 1024 * 1024, 6 * 1024 * 1024, D4),
            D4
        );
        assert_eq!(
            adaptive_compressed_chunk_bytes(5, 0, 6 * 1024 * 1024, D4),
            D4
        );
    }

    #[test]
    fn adaptive_member_over_4gib_compressed_falls_back() {
        // comp_len > u32::MAX ⇒ decoded certainly > 4 GiB ⇒ ISIZE wrapped ⇒ default.
        let comp = (u32::MAX as usize) + 1;
        let cc = adaptive_compressed_chunk_bytes(123, comp, 6 * 1024 * 1024, D4);
        assert_eq!(cc, D4);
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

    // Gated on `parallel_sm`: this asserts a SUCCESSFUL decode, which only
    // the `parallel_sm` body of `decompress_parallel` can produce. In the
    // default (`not(parallel_sm)`) build the function is a cfg-stub that
    // returns `UnsupportedPlatform` by design (no pure-Rust engine compiled
    // in), so an ungated test would fail on every default `cargo test` even
    // though nothing is wrong. The correctness assertions (byte-exact output,
    // ISIZE) are unchanged — they run in full wherever the engine exists.
    #[cfg(parallel_sm)]
    #[test]
    fn single_thread_decodes_small_input() {
        // Pure-Rust-sole (task #8): the engine is the ONLY single-member
        // decode path at every size/T. A small input at num_threads=1
        // DECODES — it used to hard-error below the 4 MiB floor (when a
        // C-FFI one-shot existed to catch it); that floor and that
        // fallback are both gone, so the engine handles it directly.
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&vec![0u8; 5_000_000]).unwrap();
        let gz = enc.finish().unwrap();
        let mut out = Vec::new();
        let n = decompress_parallel(&gz, &mut out, None, 1)
            .expect("pure-Rust SM decodes a small input at T=1");
        assert_eq!(n, 5_000_000);
        assert_eq!(out, vec![0u8; 5_000_000]);
    }
}
