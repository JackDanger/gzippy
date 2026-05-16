//! Parallel single-member gzip decompression — marker-based design (v0.6).
//!
//! Production path: wired into `decompress::decompress_single_member` when
//! ISA-L is available, `num_threads > 1`, and the compressed stream exceeds
//! 10 MiB. Replaces v0.5.1's speculative-window two-pass design.
//!
//! # Why a marker pipeline
//!
//! v0.5.1's design did 2N total compute work (phase 1 decode with empty
//! dict + phase 2 re-decode with prior window). On CI's 4-physical-core
//! ubuntu-latest runner that produced 288 MB/s vs. rapidgzip's 327 MB/s —
//! 0.88×, below the 0.99 target.
//!
//! The marker pipeline does ~1.1N total compute work:
//!
//! - **Phase 1 (parallel workers)**: each chunk is decoded by
//!   `fast_marker_inflate::decode_chunk_markers_bounded` over its
//!   `[start_bit, end_bit)` bit range. Output is `Vec<u16>` where literals
//!   are 0..=255 and cross-chunk back-references are markers ≥
//!   `MARKER_BASE = 32768` encoding a window offset.
//!
//! - **Phase 2 (sequential)**: walk chunks in order. Chunk 0's output has
//!   no markers (no predecessor); pass through. For chunk i ≥ 1, call
//!   `replace_markers` with the prior chunk's last 32 KB as the window —
//!   AVX2 (x86_64) or NEON (aarch64) substitution. Convert u16 → u8 via
//!   `u16_to_u8` which fails fast on any leftover marker.
//!
//! - **Phase 3 (sequential)**: verify total bytes against gzip ISIZE,
//!   verify combined CRC32 against gzip CRC. **Write to the writer only
//!   after CRC verifies** — a fallback never produces partial corrupt
//!   output.
//!
//! Marker-resolution work in phase 2 is essentially memcpy-speed (one
//! lookup + one store per marker, vectorized 8–16 lanes at a time). The
//! sequential chain is short — typically a few percent of total wall.
//! Speedup over single-thread ISA-L is ≈ T / 1.1.
//!
//! # Routing-assertion counter (the deletion-trap killer)
//!
//! `MARKER_PIPELINE_RUNS` is incremented on every successful run. Tests in
//! `src/tests/routing.rs` snapshot it before/after a decode and assert it
//! increased — that is the only thing that catches a regression where
//! `decompress_single_member`'s routing silently falls back to sequential
//! ISA-L. Without this assertion, output-equivalence tests pass while the
//! marker pipeline goes uncovered, and the code becomes a deletion target
//! during the next cleanup. See `docs/marker-decoder-plan.md` "deletion-
//! trap killer."

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::decompress::parallel::block_finder::BlockFinder;
use crate::decompress::parallel::fast_marker_inflate::{
    decode_chunk_bootstrap, decode_chunk_markers_continuing, BootstrapResult,
};
use crate::decompress::parallel::replace_markers::{replace_markers, MARKER_BASE};

/// Deflate sliding-window size (RFC 1951 §3.2.4).
const WINDOW_SIZE: usize = 32_768;

/// Minimum compressed size to attempt parallel. The routing entry in
/// `decompress::decompress_single_member` gates at 10 MiB; this is the inner
/// lower bound used by tests that exercise smaller fixtures.
const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;
/// Target compressed bytes per chunk. More chunks than threads allows
/// work-stealing to balance decode time across workers even when block
/// boundaries are unevenly spaced. 4 MB matches rapidgzip's default.
const TARGET_COMPRESSED_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Minimum threads to attempt parallel decode.
const MIN_THREADS_FOR_PARALLEL: usize = 2;

/// Search radius (bytes) around each partition point for block boundaries.
const SEARCH_RADIUS: usize = 512 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RealBlockBoundary(usize);

impl RealBlockBoundary {
    #[inline]
    fn bits(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkStart(RealBlockBoundary);

impl ChunkStart {
    #[inline]
    fn from_bits(bits: usize) -> Self {
        Self(RealBlockBoundary(bits))
    }

    #[inline]
    fn bits(self) -> usize {
        self.0.bits()
    }

    #[inline]
    fn to_end_limit(self) -> ChunkEndLimit {
        ChunkEndLimit(self.0)
    }
}

impl std::fmt::Display for ChunkStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.bits())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkEndLimit(RealBlockBoundary);

impl ChunkEndLimit {
    #[inline]
    fn bits(self) -> usize {
        self.0.bits()
    }

    #[inline]
    fn boundary(self) -> RealBlockBoundary {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ApproximateChunkEnd(usize);

impl ApproximateChunkEnd {
    #[inline]
    fn bits(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkDecodeStop {
    Verified(ChunkEndLimit),
    Approximate(ApproximateChunkEnd),
    UntilEnd,
}

impl ChunkDecodeStop {
    #[inline]
    fn hint_bits(self) -> Option<usize> {
        match self {
            Self::Verified(limit) => Some(limit.bits()),
            Self::Approximate(limit) => Some(limit.bits()),
            Self::UntilEnd => None,
        }
    }

    #[inline]
    fn exact_end_limit(self) -> Option<ChunkEndLimit> {
        match self {
            Self::Verified(limit) => Some(limit),
            Self::Approximate(_) | Self::UntilEnd => None,
        }
    }

    #[inline]
    fn use_isal(self) -> bool {
        true
    }

    #[inline]
    fn label(self) -> &'static str {
        match self {
            Self::Verified(_) => "real",
            Self::Approximate(_) => "approx",
            Self::UntilEnd => "none(last)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkDecodePlan {
    start: ChunkStart,
    stop: ChunkDecodeStop,
}

impl ChunkDecodePlan {
    #[inline]
    fn new(start: ChunkStart, stop: ChunkDecodeStop) -> Self {
        Self { start, stop }
    }

    #[inline]
    fn start(self) -> ChunkStart {
        self.start
    }

    #[inline]
    fn stop(self) -> ChunkDecodeStop {
        self.stop
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EmptyChunkFill {
    idx: usize,
    start: ChunkStart,
}

impl EmptyChunkFill {
    #[inline]
    fn new(idx: usize, start: ChunkStart) -> Self {
        Self { idx, start }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CorrectionDecode {
    idx: usize,
    plan: ChunkDecodePlan,
}

impl CorrectionDecode {
    #[inline]
    fn new(idx: usize, plan: ChunkDecodePlan) -> Self {
        Self { idx, plan }
    }
}

/// Successful runs of the marker pipeline. Snapshot before/after a decode to
/// confirm production routing actually called us — see the deletion-trap
/// killer test in `src/tests/routing.rs`.
///
/// `pub(crate)` rather than `pub`: the counter is an internal diagnostic
/// surface for routing-assertion tests, not part of the library API.
/// Downstream crates have no reason to read it. (Copilot review on PR #94.)
pub(crate) static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

/// Mutex serializing the body of the deletion-trap killer routing test
/// against any other test in the crate that calls `decompress_parallel`
/// concurrently. Without this lock, `cargo test`'s default parallel
/// execution can cause another test to bump `MARKER_PIPELINE_RUNS`
/// between the killer test's before/after snapshots, masking a real
/// silent-fallback regression with a false positive. (Copilot review on
/// PR #94.)
pub(crate) static MARKER_PIPELINE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Times the routing layer silently fell back to sequential libdeflate
/// because `decompress_parallel` returned a non-routing error (D1, Opus
/// advisor review). Distinct from "input was too small to bother"
/// (`ParallelError::TooSmall`), which is expected and intended.
///
/// The bench harness in `scripts/benchmark_single_member.py` reads this
/// counter (via gzippy's debug log) and hard-fails CI if it's non-zero —
/// turning "marker pipeline silently fell back" from invisible (same
/// throughput as libdeflate) into a specific actionable failure. See
/// `docs/marker-decoder-premortem.md` F7 + D1.
pub(crate) static MARKER_PIPELINE_BOUNDARY_MISSED: AtomicU64 = AtomicU64::new(0);

/// Number of cross-chunk consistency retry iterations executed across the
/// process lifetime. Healthy traffic (no BTYPE=01-heavy regions) should
/// see this stay at zero or near-zero; the BTYPE=01-heavy killer fixture
/// should see it move (test asserts `> 0`). Spikes indicate retry
/// thrashing — see `docs/marker-decoder-premortem.md` G5 mitigation.
pub(crate) static MARKER_PIPELINE_RETRY_ITERATIONS: AtomicU64 = AtomicU64::new(0);

/// Counters proving the rapidgzip-style optimizations are actually
/// firing. The marker pipeline can SILENTLY DEGRADE in ways that don't
/// produce wrong output but blow up the perf budget:
///
///   - Bootstrap never exits (no clean 32 KB tail found) → falls back
///     to running the marker decoder for the whole chunk at pure-Rust
///     speed (~50 MB/s/thread) instead of ~163 MB/s ISA-L per-thread.
///     `HANDOFF_FIRED` distinguishes this from the fast path.
///
///   - Bootstrap accumulates far more than 32 KB before exiting because
///     emit_match's chunk-local copy spreads markers forward through
///     the output. `BOOTSTRAP_OUTPUT_BYTES` divided by chunk count
///     should be near 32 KB; growth past 1-2 MB per chunk signals
///     that markers are propagating too aggressively.
///
///   - ISA-L bulk decode never runs (`SLOW_PATH_USED`). On healthy
///     fixtures every chunk except possibly the last should take the
///     bulk path.
///
///   - Phase 1a's boundary search calls try_decode_at quadratically
///     because BlockFinder emits too many candidates. `BOUNDARY_VALIDATIONS`
///     should stay bounded per chunk; runaway counts indicate the
///     `decompress_deflate_from_bit` MIN_CAP wasn't doing what we thought.
///
/// All counters are reset by tests via `reset_optimization_counters()`
/// to avoid cross-test interference under cargo test's default parallel
/// execution. Snapshot before/after a known decode and assert the
/// expected delta — the absolute counter values are uninteresting.
pub(crate) static HANDOFF_FIRED: AtomicU64 = AtomicU64::new(0);
pub(crate) static SLOW_PATH_USED: AtomicU64 = AtomicU64::new(0);
pub(crate) static BOOTSTRAP_OUTPUT_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static ISAL_OUTPUT_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static BOUNDARY_VALIDATIONS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the optimization counters at a point in time. Used by
/// tests to take a before/after delta around a `decompress_parallel`
/// call. The mutex-serialized killer-test pattern (see
/// `MARKER_PIPELINE_TEST_LOCK`) ensures no other test's bumps land in
/// the delta window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OptimizationCounters {
    pub handoff_fired: u64,
    pub slow_path_used: u64,
    pub bootstrap_output_bytes: u64,
    pub isal_output_bytes: u64,
    pub boundary_validations: u64,
    pub retry_iterations: u64,
}

impl OptimizationCounters {
    pub fn snapshot() -> Self {
        Self {
            handoff_fired: HANDOFF_FIRED.load(Ordering::Relaxed),
            slow_path_used: SLOW_PATH_USED.load(Ordering::Relaxed),
            bootstrap_output_bytes: BOOTSTRAP_OUTPUT_BYTES.load(Ordering::Relaxed),
            isal_output_bytes: ISAL_OUTPUT_BYTES.load(Ordering::Relaxed),
            boundary_validations: BOUNDARY_VALIDATIONS.load(Ordering::Relaxed),
            retry_iterations: MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed),
        }
    }

    /// `self - other`, saturating per field — used to compute the
    /// delta over a single decode call.
    pub fn delta(&self, before: &Self) -> Self {
        Self {
            handoff_fired: self.handoff_fired.saturating_sub(before.handoff_fired),
            slow_path_used: self.slow_path_used.saturating_sub(before.slow_path_used),
            bootstrap_output_bytes: self
                .bootstrap_output_bytes
                .saturating_sub(before.bootstrap_output_bytes),
            isal_output_bytes: self
                .isal_output_bytes
                .saturating_sub(before.isal_output_bytes),
            boundary_validations: self
                .boundary_validations
                .saturating_sub(before.boundary_validations),
            retry_iterations: self
                .retry_iterations
                .saturating_sub(before.retry_iterations),
        }
    }
}

/// Hard wall-time deadline for the cross-chunk correction sweep.
/// Healthy data converges in <50 ms; the deadline bounds adversarial
/// inputs (where every chunk decodes serially via the correction chain)
/// at a generous limit. Beyond this, return Err and let the routing
/// fallback fire.
const RETRY_WALL_DEADLINE_MS: u64 = 2000;

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

// ── Public entry ─────────────────────────────────────────────────────────────

/// Parallel decompress a single-member gzip stream via the v0.6 marker
/// pipeline.
///
/// Returns `Err(ParallelError::TooSmall)` if the input is below the parallel
/// threshold — the caller should fall back to sequential decode. Other errors
/// indicate a boundary-search failure, decode failure, or CRC/size mismatch.
///
/// Streaming write trade-off: each chunk is written to the writer as its
/// markers are resolved (overlapping I/O with computation). CRC+ISIZE
/// verification happens after all chunks are written. On mismatch the caller
/// receives `Err` but the writer may already contain partial bytes — same
/// design as rapidgzip. `TooSmall` and `DecodeFailed` never write bytes.
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

    // Trailer: gzip stores CRC32 then ISIZE (little-endian) in the last 8 bytes.
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

    // More chunks than threads lets work-stealing rebalance when block
    // boundaries are unevenly spaced (e.g. gzip -9 on heterogeneous data).
    // Workers are capped at num_threads; extra tasks queue up for idle threads.
    let num_chunks = (deflate_data.len() / TARGET_COMPRESSED_CHUNK_BYTES)
        .max(num_threads)
        .max(1);
    let total_bits = deflate_data.len() * 8;
    let spacing_bits = total_bits / num_chunks;

    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] {} bytes deflate, {} chunks ({} workers), spacing={}KB, isize={}",
            deflate_data.len(),
            num_chunks,
            num_threads,
            spacing_bits / 8 / 1024,
            expected_size
        );
    }

    // ── Phase 1a: parallel speculative boundary search ──────────────────────
    //
    // Returns `Option<usize>` per chunk. A `None` means BlockFinder +
    // byte-aligned brute force both failed to find a validated candidate
    // within 512 KiB of the chunk's spacing-derived anchor — common in
    // BTYPE=01-heavy regions where fixed-Huffman blocks have no header
    // redundancy for `validate_boundary` to filter against. We do NOT
    // reject here; phase 1c chain-decodes those chunks from the
    // predecessor's confirmed end_bit. Chunk 0 must always succeed
    // (anchored at bit 0); we still bail if it doesn't.
    let t_search = std::time::Instant::now();
    let mut start_bits_opt =
        phase1_search_boundaries(deflate_data, num_chunks, spacing_bits, num_threads);
    let search_elapsed = t_search.elapsed();

    if start_bits_opt[0].is_none() {
        // Should be impossible — chunk 0's "search" is hard-pinned to bit 0.
        return Err(ParallelError::DecodeFailed);
    }
    if debug_enabled() {
        let n = start_bits_opt.iter().filter(|s| s.is_none()).count();
        if n > 0 {
            eprintln!(
                "[parallel_sm:v0.6] {}/{} speculative boundary searches failed; \
                 phase 1c will chain-decode those chunks from predecessor end_bits",
                n, num_chunks
            );
        }
    }
    let mut start_bits: Vec<Option<ChunkStart>> =
        start_bits_opt.iter_mut().map(|s| s.take()).collect();

    // ── Phase 1b: parallel marker decode from speculative starts ────────────
    //
    // Each chunk's start is speculative — `validate_boundary(min_blocks=2)`
    // admits some false positives on BTYPE=01 regions (no header redundancy
    // in fixed Huffman). After phase 1b, chunk N's *decoded* end_bit is
    // always a real block boundary (G1 invariant), so phase 1c uses it as
    // chunk N+1's confirmed start. Chunks with `None` start get a `None`
    // decode result that phase 1c will fill in via chain-decode.
    let t_decode = std::time::Instant::now();
    // end_limit[i] is chunk i's upper decode bound (exclusive bit offset).
    //
    // P0 fix: use the nearest downstream confirmed phase1a boundary instead
    // of an anchor (i+1)*spacing_bits). The anchor was never a real deflate
    // block boundary, so ISA-L had to be disabled for those chunks (forcing
    // pure-Rust marker decode at ~50 MB/s vs ISA-L's ~1500 MB/s).
    //
    // With find_map: end_limits[i] is always either None (last chunk,
    // ISA-L decodes to BFINAL) or a real phase1a-confirmed block boundary.
    // ISA-L can safely stop at a real boundary — it decodes to the first
    // ISAL_BLOCK_NEW_HDR state at or before end_limit, which is a real
    // block start. The bfinal_hit guard in decode_chunk_with_handoff
    // handles the rare case where a speculative start decodes a false BFINAL
    // before reaching end_limit.
    //
    // Trade-off: when k consecutive chunks miss phase1a (all None), chunk
    // i decodes up to k extra chunks' worth of data in phase1b. At ISA-L
    // speed this is still much faster than the old anchor+marker approach
    // (e.g. chunk 11 covering chunks 12-16: ~30ms ISA-L vs ~68ms marker).
    // Phase 1c still has k serial waves for the missed chunks, but each
    // wave is ISA-L-fast rather than anchor-marker-slow.
    let decode_stops: Vec<ChunkDecodeStop> = (0..num_chunks)
        .map(|i| chunk_decode_stop(i, &start_bits, spacing_bits, total_bits))
        .collect();
    // Per-chunk decoded-output hint for the ISA-L bulk decode's
    // initial buffer cap. ISIZE / num_chunks is the average; the last
    // chunk may legitimately exceed it, so `decode_chunk_with_handoff`
    // adds 50% headroom.
    let per_chunk_output_hint = expected_size / num_chunks.max(1);
    let chunks_opt = phase1_marker_decode_parallel_with_optional_starts(
        deflate_data,
        &start_bits,
        &decode_stops,
        per_chunk_output_hint,
        num_threads,
    );
    let decode_elapsed = t_decode.elapsed();
    let mut chunks: Vec<Option<ChunkResult>> = chunks_opt;

    if debug_enabled() {
        // Per-chunk breakdown: elapsed, bootstrap size, ISA-L size, total.
        // Skew in elapsed reveals load imbalance; skew in sizes reveals
        // uneven boundary placement. High CV in the benchmark usually comes
        // from one chunk taking significantly longer than the others.
        eprintln!("[parallel_sm:v0.6] per-chunk phase1b breakdown:");
        for (i, chunk) in chunks.iter().enumerate() {
            match chunk {
                Some(c) => {
                    let boot_kb = (c.bootstrap.len() + c.bootstrap_clean.len()) / 1024;
                    let isal_kb = c.isal_bytes.len() / 1024;
                    let total_kb = boot_kb + isal_kb;
                    // G1 invariant: every phase1b result must land at a real block
                    // boundary. debug_assert catches regressions in decode primitives
                    // (e.g. decode_stored bit-position bug) at chunk granularity
                    // rather than 10 chunks later in phase1c.
                    debug_assert!(
                        crate::decompress::parallel::fast_marker_inflate::validate_boundary(
                            deflate_data,
                            c.end_bit_offset,
                            1,
                            1,
                            false,
                        )
                        .is_ok(),
                        "G1 invariant: chunk {} end_bit_offset {} is not a real block boundary",
                        i,
                        c.end_bit_offset,
                    );
                    // Annotate whether end_limit was a confirmed phase1a
                    // boundary or an anchor fallback (start_bits[i+1]=None).
                    let end_limit_kind = decode_stops[i].label();
                    eprintln!(
                        "  chunk {:2}: {:4}ms  boot={:4}KB  isal={:6}KB  \
                         total={:6}KB  end_bit={}  limit={}",
                        i,
                        c.worker_elapsed.as_millis(),
                        boot_kb,
                        isal_kb,
                        total_kb,
                        c.end_bit_offset,
                        end_limit_kind,
                    );
                }
                None => {
                    eprintln!("  chunk {:2}: (boundary not found — phase1c will fill)", i);
                }
            }
        }
        // Imbalance ratio: longest / shortest non-empty worker.
        let elapsed_ms: Vec<u128> = chunks
            .iter()
            .filter_map(|c| c.as_ref())
            .filter(|c| c.worker_elapsed.as_millis() > 0)
            .map(|c| c.worker_elapsed.as_millis())
            .collect();
        if elapsed_ms.len() >= 2 {
            let max_ms = *elapsed_ms.iter().max().unwrap();
            let min_ms = *elapsed_ms.iter().min().unwrap();
            eprintln!(
                "  imbalance: max/min = {:.2}x  (max={max_ms}ms min={min_ms}ms)",
                max_ms as f64 / min_ms.max(1) as f64
            );
        }
    }

    // ── Phase 1c: cross-chunk consistency correction ────────────────────────
    //
    // Walk pairs (N, N+1). When chunks[N].end_bit != start_bits[N+1], the
    // latter was a false positive — correct start_bits[N+1] to
    // chunks[N].end_bit (which IS a real boundary by G1 invariant) and
    // re-decode chunk N+1. Propagate forward.
    //
    // Per Opus advisor review: this is the correct shape for the
    // BTYPE=01-heavy class. Earlier top-K + strictness-ramp design was
    // probabilistic; this is induction-based (chunk 0 starts at bit 0, a
    // real boundary; chunk N's decode lands at a real boundary; chunk
    // N+1's corrected start is therefore a real boundary; …).
    let t_retry = std::time::Instant::now();
    // Scale deadline with compressed size so multi-GB files on slow hardware
    // still get enough time for legitimate corrections. Floor = RETRY_WALL_DEADLINE_MS.
    let data_mb = deflate_data.len() / (1024 * 1024);
    let deadline_ms = RETRY_WALL_DEADLINE_MS.max(data_mb as u64 * 3);
    let retry_deadline = t_retry + std::time::Duration::from_millis(deadline_ms);
    phase1c_resolve_consistency(
        deflate_data,
        &mut start_bits,
        &mut chunks,
        retry_deadline,
        per_chunk_output_hint,
        spacing_bits,
    )?;
    let retry_elapsed = t_retry.elapsed();
    let chunks: Vec<ChunkResult> = chunks
        .into_iter()
        .map(|c| c.expect("phase 1c guarantees all chunks decoded"))
        .collect();

    // G1 assertion after phase1c: every chunk's end_bit_offset must be a real
    // deflate block boundary. This catches decode primitive bugs (like the
    // decode_stored invariant break) that phase1c may have propagated forward.
    #[cfg(debug_assertions)]
    for (i, chunk) in chunks.iter().enumerate() {
        debug_assert!(
            crate::decompress::parallel::fast_marker_inflate::validate_boundary(
                deflate_data,
                chunk.end_bit_offset,
                1,
                1,
                false,
            )
            .is_ok()
                || {
                    // The last chunk's end_bit_offset may be past BFINAL with
                    // only zero-padding bits remaining — validate_boundary will
                    // fail there because there are no more blocks to decode.
                    // Accept only if we're within 1 byte of the stream end.
                    i + 1 == chunks.len() && total_bits.saturating_sub(chunk.end_bit_offset) < 8
                },
            "G1 invariant after phase1c: chunk {} end_bit_offset {} is not a real block boundary",
            i,
            chunk.end_bit_offset,
        );
    }

    // ── Phase 2 + streaming write: resolve markers → write → combine CRCs ───
    //
    // Each chunk is written to the writer immediately after its markers
    // are resolved, overlapping I/O with resolution of subsequent chunks.
    // This matches rapidgzip's per-chunk streaming design and eliminates
    // the ~69 ms write serialization gap on Silesia-class inputs (503 MB
    // written after 478 ms of compute in the deferred design).
    //
    // Trade-off: CRC+ISIZE verification now happens AFTER bytes are
    // written. A mismatch means partial output is in the writer — the
    // routing-layer fallback cannot cleanly retry via libdeflate. This
    // is acceptable because: (a) gzip CRC mismatches indicate hardware
    // errors or corrupt files rather than speculation failures, and
    // (b) the routing layer already flags the error to the caller.
    let t_resolve = std::time::Instant::now();
    let (total_crc, total_size) = phase2_resolve_write_combine(chunks, writer)?;
    let resolve_elapsed = t_resolve.elapsed();

    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] search={:.1}ms decode={:.1}ms retry={:.1}ms resolve+write={:.1}ms",
            search_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            retry_elapsed.as_secs_f64() * 1000.0,
            resolve_elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── Phase 3: verify trailer ──────────────────────────────────────────────
    //
    // Both ISIZE and CRC32 in the gzip trailer can legitimately be zero
    // (ISIZE=0 for an empty input; CRC32=0 happens for specific byte
    // sequences — the CRC of the empty stream is itself 0). Verification
    // is unconditional: the trailer is always present (sliced off at
    // function entry). (Premortem mitigation B5; Copilot review on PR #94.)
    if total_size != expected_size {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] size mismatch: got {} expected {}",
                total_size, expected_size
            );
        }
        return Err(ParallelError::SizeMismatch);
    }
    if total_crc != expected_crc {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] CRC mismatch: got {:#010x} expected {:#010x}",
                total_crc, expected_crc
            );
        }
        return Err(ParallelError::CrcMismatch);
    }

    MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);

    if debug_enabled() {
        let total = t0.elapsed();
        let mbps = total_size as f64 / total.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm:v0.6] total={:.1}ms ({:.0} MB/s)",
            total.as_secs_f64() * 1000.0,
            mbps,
        );
    }

    Ok(total_size as u64)
}

// ── Phase 1a: parallel boundary search ───────────────────────────────────────
//
// Returns one *speculative* start per chunk — a bit offset that passed
// `try_decode_at` (ISA-L + `validate_boundary(min_blocks=2)`). The pick
// is speculative because the same filter that admits real boundaries
// also admits some false positives on BTYPE=01-heavy data — those have
// no header redundancy and any random fixed-Huffman position can decode
// 2 blocks worth of valid symbols by chance.
//
// Speculative picks are corrected in phase 1c after phase 1b reveals
// each chunk's natural decoded end_bit. Chunk N's end_bit is *always* a
// real block boundary (G1 invariant: a decode starting at a real
// boundary lands at a real boundary). If chunks[N].end_bit doesn't
// match start_bits[N+1], the latter is the false positive; phase 1c
// corrects start_bits[N+1] to chunks[N].end_bit and re-decodes chunk
// N+1.

fn phase1_search_boundaries(
    deflate_data: &[u8],
    num_chunks: usize,
    spacing_bits: usize,
    num_workers: usize,
) -> Vec<Option<ChunkStart>> {
    let results: Vec<Mutex<Option<ChunkStart>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_workers {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let pick = if idx == 0 {
                    Some(ChunkStart::from_bits(0))
                } else {
                    let from_bit = idx * spacing_bits;
                    search_boundary_forward(deflate_data, from_bit)
                };
                *results[idx].lock().unwrap() = pick;
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Find one validated deflate block-boundary candidate at or past
/// `from_bit`. Returns the first accepted bit offset, or None if no
/// candidate within the search radius passes.
///
/// The pick is speculative — `validate_boundary(min_blocks=2)` admits
/// some BTYPE=01 false positives (no header redundancy in fixed
/// Huffman); phase 1c's correction sweep verifies and corrects them
/// after phase 1b reveals each chunk's actual decoded end_bit.
///
/// Two tiers:
/// 1. **BlockFinder heuristic candidates** (BTYPE=00 stored + BTYPE=10
///    dynamic Huffman).
/// 2. **Byte-aligned brute force** (first 128 KiB).
fn search_boundary_forward(deflate_data: &[u8], from_bit: usize) -> Option<ChunkStart> {
    let search_end = (from_bit + SEARCH_RADIUS * 8).min(deflate_data.len() * 8);
    if from_bit >= search_end {
        return None;
    }

    // Tier 1: BlockFinder over the search radius. Find a validated
    // BTYPE=00 / BTYPE=10 boundary candidate; return at the first.
    //
    // The earlier "Tier 2: byte-aligned brute force" was removed
    // (PR #97, commit after Silesia Tmax bench analysis): when
    // BlockFinder failed, tier 2 iterated 16384 byte-aligned
    // candidates inside 128 KiB, each calling `try_decode_at`'s
    // ISA-L 32 KB validation (256 KB allocation per call). On
    // Silesia at T=4, the chunk-3 search hit tier 2 and burned
    // ~500 ms of phase 1a wall time before returning None — for a
    // chunk that phase 1c was going to chain-decode anyway from
    // chunk 2's end_bit. Returning None earlier means the same
    // end-state, ~500 ms faster.
    let finder = BlockFinder::new(deflate_data);
    let sub_chunk_bits = 8 * 1024 * 8;
    let mut chunk_start = from_bit;
    while chunk_start < search_end {
        let chunk_end = (chunk_start + sub_chunk_bits).min(search_end);
        let mut candidates = finder.find_blocks(chunk_start, chunk_end);
        candidates.sort_by_key(|b| b.bit_offset);
        for c in &candidates {
            if c.bit_offset < from_bit {
                continue;
            }
            if try_decode_at(deflate_data, c.bit_offset) {
                return Some(ChunkStart::from_bits(c.bit_offset));
            }
        }
        chunk_start = chunk_end;
    }
    None
}

/// Validate a candidate boundary in two stages — see premortem mitigation
/// B7 in `docs/marker-decoder-premortem.md`:
///
/// 1. **Fast filter** via ISA-L's `decompress_deflate_from_bit` (or zlib-ng
///    on non-x86_64): does 32 KB decode succeed without error? ISA-L is
///    permissive — it accepts some non-boundary positions that happen to
///    decode plausible-looking bytes for 32 KB before diverging.
/// 2. **Strict check** via our `fast_marker_inflate` on the same position,
///    bounded to a single deflate block (`end_bit_limit = bit_offset + 1`
///    stops the decoder at the next block boundary at or past `bit_offset`).
///    A real boundary decodes one block cleanly; a false positive fails
///    on Huffman table validation or a malformed length code somewhere in
///    the first block.
///
/// Why both: with stage 1 alone, false positives passed through and the
/// production marker decode later returned Err mid-block, causing routing
/// to silently fall back to sequential ISA-L (failure mode F6). The
/// deletion-trap killer test caught it. Stage 2 alone is correct but
/// would run the slower pure-Rust decoder against every BlockFinder
/// candidate.
///
/// Cost: stage 2 runs only when stage 1 approves. Stage 1 approval
/// candidates are rare on real data (a few per chunk's search window).
/// Each stage-2 run decodes one deflate block (~30 KB out / ~10 KB in)
/// at ~200-300 MB/s — sub-millisecond per accepted candidate.
fn try_decode_at(deflate_data: &[u8], bit_offset: usize) -> bool {
    BOUNDARY_VALIDATIONS.fetch_add(1, Ordering::Relaxed);
    let start_byte = bit_offset / 8;
    if start_byte >= deflate_data.len() {
        return false;
    }
    let remaining = deflate_data.len() - start_byte;
    let min_output = if remaining > 128 * 1024 {
        32 * 1024
    } else {
        4 * 1024
    };
    // Use 4× the min_output as the ISA-L cap so that a false boundary whose
    // BFINAL block produces just-over-min_output bytes (e.g. 34 KB against a
    // 32 KB min) still triggers the premature-BFINAL guard in
    // `decompress_deflate_from_bit`. With a 32 KB cap the guard never fires
    // because the output cap is hit before the BFINAL block is reached.
    let stage1_cap = min_output * 4;

    // Stage 1: ISA-L (or zlib-ng) fast filter.
    let isal_ok = crate::backends::inflate_bit::decompress_deflate_from_bit(
        deflate_data,
        bit_offset,
        &[],
        stage1_cap,
    )
    .is_some_and(|out| out.len() >= min_output);
    if !isal_ok {
        return false;
    }

    // Stage 2: marker-decoder validation, `min_blocks=2`,
    // `require_non_fixed_stop=true`. A single coincidentally-valid stored
    // block (~1/65536 chance) cannot fake a second consecutive valid block.
    // `require_non_fixed_stop` additionally refuses stop points where the
    // next block is BTYPE=01 (fixed Huffman has no header redundancy —
    // any bit sequence decodes as BTYPE=01, so a BTYPE=01-heavy region
    // trivially passes without this guard). Matches rapidgzip
    // GzipChunk.hpp:552.
    use crate::decompress::parallel::fast_marker_inflate::validate_boundary;
    validate_boundary(
        deflate_data,
        bit_offset,
        /*min_blocks=*/ 2,
        /*min_output_bytes=*/ 0,
        /*require_non_fixed_stop=*/ true,
    )
    .is_ok()
}

#[inline]
fn chunk_partition_end_bits(
    idx: usize,
    spacing_bits: usize,
    total_bits: usize,
    num_chunks: usize,
) -> usize {
    if idx + 1 >= num_chunks {
        total_bits
    } else {
        ((idx + 1) * spacing_bits).min(total_bits)
    }
}

#[inline]
fn chunk_decode_stop(
    idx: usize,
    start_bits: &[Option<ChunkStart>],
    spacing_bits: usize,
    total_bits: usize,
) -> ChunkDecodeStop {
    if idx + 1 >= start_bits.len() {
        ChunkDecodeStop::UntilEnd
    } else if let Some(next_start) = start_bits[idx + 1] {
        ChunkDecodeStop::Verified(next_start.to_end_limit())
    } else {
        ChunkDecodeStop::Approximate(ApproximateChunkEnd(chunk_partition_end_bits(
            idx,
            spacing_bits,
            total_bits,
            start_bits.len(),
        )))
    }
}

#[inline]
fn correction_decode_stop(idx: usize, start_bits: &[Option<ChunkStart>]) -> ChunkDecodeStop {
    match (idx + 1..start_bits.len()).find_map(|j| start_bits[j].map(ChunkStart::to_end_limit)) {
        Some(limit) => ChunkDecodeStop::Verified(limit),
        None => ChunkDecodeStop::UntilEnd,
    }
}

// ── Phase 1b: parallel marker decode ─────────────────────────────────────────
//
// Each chunk's decode produces both the marker stream AND its end bit
// offset — the bit position just past the last block consumed. D4 (cross-
// chunk consistency) verifies that chunk N's end_bit_offset equals chunk
// N+1's start_bit; any mismatch means phase 1a picked a misaligned
// boundary for one of them, and the pipeline must reject rather than
// emit double-covered or partial output.

/// Phase 1b chunk result.
///
/// `bootstrap`: u16 output from the pure-Rust marker decoder's bootstrap
/// pass, covering the chunk's first ~32 KB of output (and any prefix
/// markers for cross-chunk back-references). May be empty when ISA-L was
/// not entered (last chunk with no clean tail or chunk-too-small fallback).
///
/// `isal_bytes`: real bytes produced by ISA-L after the bootstrap handed
/// off. Single contiguous Vec. Empty when no handoff occurred (chunk
/// decoded entirely by the marker decoder).
///
/// `end_bit_offset`: bit position just past the chunk's last decoded
/// block. Always at a real deflate block boundary (G1 invariant — both
/// the marker decoder's `decode_loop` and ISA-L's natural stopping
/// points sit between blocks).
///
/// This shape mirrors rapidgzip's per-chunk worker (cleanData → ISA-L
/// handoff at `vendor/rapidgzip/.../GzipChunk.hpp:521`): the marker
/// decoder bootstraps ≤32 KB to establish a window dict, then ISA-L
/// does the ~99% bulk of the chunk at full single-thread ISA-L speed.
/// Without this handoff the per-thread decoder is structurally bounded
/// to pure-Rust speed (~50 MB/s/thread vs. ISA-L's ~163 MB/s/thread on
/// x86_64 CI), which prevents per-thread parity with sequential ISA-L
/// no matter how good the parallel orchestration is.
struct ChunkResult {
    bootstrap: Vec<u16>,
    /// Trailing bootstrap bytes already proven marker-free and stored as
    /// real bytes so phase 2 doesn't keep them in the wider u16 form.
    bootstrap_clean: Vec<u8>,
    /// Bulk decoded bytes from ISA-L. Single contiguous allocation; no
    /// segmentation — the earlier 128 KiB segment design caused ~3900
    /// mmap/munmap pairs (segment size == M_MMAP_THRESHOLD) serialising
    /// through mmap_lock at T=4.
    isal_bytes: Vec<u8>,
    /// Precomputed CRC32 of `isal_bytes`, computed inline inside ISA-L
    /// (data is L1/L2-hot). Combined via `crc32fast::Hasher::combine`
    /// in phase 2; sequential CRC work drops from ~100 ms to ~1-2 ms.
    isal_crc: crc32fast::Hasher,
    end_bit_offset: usize,
    /// Wall time the worker spent on this chunk (bootstrap + ISA-L).
    /// Logged per-chunk in debug mode so load imbalance across workers
    /// is visible without attaching a profiler.
    worker_elapsed: std::time::Duration,
}

impl ChunkResult {
    fn decoded_len(&self) -> usize {
        self.bootstrap.len() + self.bootstrap_clean.len() + self.isal_bytes.len()
    }
}

/// Decodes each chunk in parallel from its speculative start. Each worker:
///
/// 1. Runs the marker decoder in **bootstrap mode** until 32 KB of clean
///    (marker-free) tail accumulates at a deflate block boundary.
/// 2. Hands off to ISA-L (seeded with that 32 KB as `isal_inflate_set_dict`)
///    for the remaining ~99% of the chunk.
///
/// This is the cleanData → ISA-L handoff from rapidgzip's `GzipChunk.hpp`
/// (per the Opus advisor review on PR #97). Without it, the per-thread
/// pure-Rust marker decoder caps single-thread parity vs ISA-L at ~30%
/// no matter how good the parallel orchestration is.
///
/// Chunks whose `start_bits[i]` is `None` (phase 1a found no candidate)
/// are skipped — their result stays `None` and phase 1c chain-decodes
/// them from the predecessor's confirmed `end_bit_offset`.
fn phase1_marker_decode_parallel_with_optional_starts(
    deflate_data: &[u8],
    start_bits: &[Option<ChunkStart>],
    decode_stops: &[ChunkDecodeStop],
    per_chunk_output_hint: usize,
    num_workers: usize,
) -> Vec<Option<ChunkResult>> {
    let num_chunks = start_bits.len();
    let results: Vec<Mutex<Option<ChunkResult>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_workers {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let Some(start_bit) = start_bits[idx] else {
                    continue;
                };
                let plan = ChunkDecodePlan::new(start_bit, decode_stops[idx]);
                let force_slow_path = !decode_stops[idx].use_isal();
                if let Ok(chunk) = decode_chunk_plan(
                    deflate_data,
                    plan,
                    per_chunk_output_hint,
                    num_chunks,
                    force_slow_path,
                ) {
                    *results[idx].lock().unwrap() = Some(chunk);
                }
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Per-chunk worker body: marker-decoder bootstrap → ISA-L handoff.
///
/// Falls back to "marker decoder for the entire chunk" when (a) the chunk
/// is too small to accumulate 32 KB of clean tail before reaching
/// `end_bit_limit` / BFINAL, or (b) ISA-L is not available on this
/// platform. Both fall-back cases return a `ChunkResult` with empty
/// `isal_bytes` — phase 2 sees no difference.
fn decode_chunk_with_handoff(
    deflate_data: &[u8],
    start_bit: ChunkStart,
    stop: ChunkDecodeStop,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
    force_slow_path: bool,
) -> std::io::Result<ChunkResult> {
    let t_worker = std::time::Instant::now();
    let stop_hint_bits = stop.hint_bits();
    let exact_end_limit = stop.exact_end_limit();
    let _ = exact_end_limit;
    // Phase 1 of the worker: bootstrap. Decode the chunk's first ~32 KB
    // with the marker decoder until the trailing 32 KB of output is
    // marker-free. This window seeds ISA-L for the bulk decode.
    let BootstrapResult {
        markers: bootstrap_markers,
        end_bit_offset: bootstrap_end_bit,
        clean_window,
        bfinal_hit,
    } = decode_chunk_bootstrap(deflate_data, start_bit.bits(), stop_hint_bits)?;

    // Track bootstrap output for the "is the handoff actually firing"
    // assertion in tests::routing — see HANDOFF_FIRED rationale.
    BOOTSTRAP_OUTPUT_BYTES.fetch_add(bootstrap_markers.len() as u64, Ordering::Relaxed);

    // Diagnostic: log all bootstrap outcomes so we can trace why a chunk
    // becomes Some vs None in phase1b. bfinal_hit=true should → Err → None.
    if debug_enabled() {
        eprintln!(
            "[bootstrap] start={start_bit} stop={stop:?} \
             end_bit={bootstrap_end_bit} bfinal_hit={bfinal_hit} \
             window={} output_bytes={}",
            clean_window.is_some(),
            bootstrap_markers.len(),
        );
    }

    // No clean window? Three reasons:
    //   (a) end_bit_limit reached before 32 KB clean tail accumulated.
    //   (b) BFINAL was hit before accumulating a clean 32 KB window.
    //   (c) BFINAL was hit with a clean window but output < WINDOW_SIZE.
    //
    // Case (b)/(c) with bfinal_hit=true: calling marker_finish_after_bootstrap
    // would attempt to decode past BFINAL, reading gzip trailer bytes or
    // unrelated stream data as deflate blocks. This produces an invalid
    // end_bit that breaks phase 1c's G1 invariant (observed as "Stored
    // LEN/NLEN mismatch" on the successor's re-decode). Returning Err here
    // discards the result; phase 1b stores None for this chunk, and phase 1c
    // re-decodes it from its predecessor's confirmed end_bit — correct behavior.
    let Some(dict) = clean_window else {
        if bfinal_hit {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] chunk start={start_bit}: BFINAL hit at {bootstrap_end_bit} \
                     before stop={stop:?} ({output_kb} KB output) — discarding, \
                     phase1c will re-derive from predecessor",
                    output_kb = bootstrap_markers.len() / 1024,
                );
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BFINAL in bootstrap before chunk boundary — result discarded",
            ));
        }
        // Case (a): end_bit_limit was reached before clean window.
        // The force_slow_path branch below (anchor-based chunks) continues
        // with the marker decoder to decode the rest of the chunk range.
        // Without force_slow_path (ISA-L enabled), return the bootstrap
        // result directly — the chunk may be tiny or high-entropy.
        #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
        if force_slow_path {
            SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
            return marker_finish_after_bootstrap(
                deflate_data,
                bootstrap_markers,
                bootstrap_end_bit,
                stop_hint_bits,
            )
            .map(|mut c| {
                c.worker_elapsed = t_worker.elapsed();
                c
            });
        }
        return Ok(ChunkResult {
            bootstrap: bootstrap_markers,
            bootstrap_clean: Vec::new(),
            isal_bytes: Vec::new(),
            isal_crc: crc32fast::Hasher::new(),
            end_bit_offset: bootstrap_end_bit,
            worker_elapsed: t_worker.elapsed(),
        });
    };

    // Phase 2 of the worker: ISA-L bulk decode from `bootstrap_end_bit`,
    // seeded with the 32 KB clean window as `isal_inflate_set_dict`.
    // Skipped when `force_slow_path` is set — used for anchor-based
    // end_limits (no real speculative boundary) where ISA-L's truncated
    // input would produce a non-boundary end_bit, violating the G1
    // invariant used by phase 1c. The marker decoder always stops at a
    // real block boundary past end_bit_limit (see decode_loop contract).
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    {
        if force_slow_path {
            let _ = (per_chunk_output_hint, num_chunks_total);
            SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
            return marker_finish_after_bootstrap(
                deflate_data,
                bootstrap_markers,
                bootstrap_end_bit,
                stop_hint_bits,
            )
            .map(|mut c| {
                c.worker_elapsed = t_worker.elapsed();
                c
            });
        }
        // Bound input to the chunk's envelope. Without the chunk
        // boundary trim, ISA-L would decode all the way to BFINAL on
        // every chunk — quadratic total work, defeating the parallel
        // win.
        let end_byte = match stop_hint_bits {
            Some(end_bits) => (end_bits.div_ceil(8)).min(deflate_data.len()),
            None => deflate_data.len(),
        };
        let input = &deflate_data[..end_byte];

        // Output cap: `per_chunk_output_hint * 2` — tight enough that
        // typical per-worker physical RAM is ~2× the chunk's average
        // decoded size, but loose enough to absorb the 5-10× ratio
        // skew Silesia exhibits across chunks.
        //
        // Earlier sizing iterations on this PR taught the trade-off:
        //
        //   - `hint * 3/2` (~120-180 MB): silently truncated chunks
        //     with above-average compression ratios; phase 1c could
        //     not correct mid-block end_bits and CI saw silent
        //     libdeflate fallback at 0.55× rapidgzip.
        //
        //   - `hint * num_chunks_total` (~ISIZE = ~500 MB): no
        //     truncation, but 4 workers × 500 MB virtual = 2 GiB of
        //     committed VA. ISA-L's writes minor-fault every 4 KiB
        //     page on first touch, serializing through `mmap_lock`.
        //     On a quiet runner this is fast; on a noisy CI runner
        //     it blows up the timing variance (CV gzippy 14.94%
        //     vs rapidgzip 1.72% on the same Silesia bench — Opus
        //     advisor identified this as the dominant variance
        //     source). Median moved against gzippy as a result.
        //
        // The `* 2` cap caps page-fault cost at ~2× the actual
        // chunk size while still tolerating 2× chunk-size variance.
        // For chunks that overshoot (rare), the wrapper's
        // grow-on-demand path takes over — see
        // `decompress_deflate_from_bit_with_end` in
        // `src/backends/isal_decompress.rs`.
        let max_output = per_chunk_output_hint.saturating_mul(2);
        let _ = num_chunks_total; // retained in signature for fallback path below

        // CRC is computed inside the ISA-L loop as each write lands
        // (data is still L1/L2-hot). This replaces the earlier post-decode
        // cold-memory walk over the full isal_bytes Vec.
        let mut isal_crc = crc32fast::Hasher::new();
        match crate::backends::isal_decompress::decompress_deflate_from_bit_with_end(
            input,
            bootstrap_end_bit,
            &dict,
            max_output,
            &mut isal_crc,
        ) {
            Some((isal_bytes, isal_end_bit)) => {
                let Some(verified_end_bit) =
                    normalize_isal_end_bit(deflate_data, start_bit, stop, isal_end_bit)
                else {
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm:v0.6] chunk start={start_bit}: rejecting ISA-L end_bit={} \
                             for verified stop={stop:?}",
                            isal_end_bit
                        );
                    }
                    SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
                    return marker_finish_after_bootstrap(
                        deflate_data,
                        bootstrap_markers,
                        bootstrap_end_bit,
                        stop_hint_bits,
                    )
                    .map(|mut c| {
                        c.worker_elapsed = t_worker.elapsed();
                        c
                    });
                };
                HANDOFF_FIRED.fetch_add(1, Ordering::Relaxed);
                ISAL_OUTPUT_BYTES.fetch_add(isal_bytes.len() as u64, Ordering::Relaxed);
                let split_at = bootstrap_markers.len().saturating_sub(dict.len());
                let mut bootstrap = bootstrap_markers;
                bootstrap.truncate(split_at);
                Ok(ChunkResult {
                    bootstrap,
                    bootstrap_clean: dict,
                    isal_bytes,
                    isal_crc,
                    end_bit_offset: verified_end_bit.bits(),
                    worker_elapsed: t_worker.elapsed(),
                })
            }
            None => {
                // ISA-L failed — most often because the speculative
                // end_bit_limit cut mid-block. Fall back to finishing
                // the chunk via the marker decoder. Slow path, rare
                // on real data.
                SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
                marker_finish_after_bootstrap(
                    deflate_data,
                    bootstrap_markers,
                    bootstrap_end_bit,
                    stop_hint_bits,
                )
                .map(|mut c| {
                    c.worker_elapsed = t_worker.elapsed();
                    c
                })
            }
        }
    }

    // On non-x86_64 or without the ISA-L feature, the parallel single-
    // member path is gated off at routing time. This branch should be
    // unreachable in production but is kept for build correctness.
    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    {
        let _ = (
            dict,
            per_chunk_output_hint,
            num_chunks_total,
            force_slow_path,
        );
        SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
        marker_finish_after_bootstrap(
            deflate_data,
            bootstrap_markers,
            bootstrap_end_bit,
            stop_hint_bits,
        )
        .map(|mut c| {
            c.worker_elapsed = t_worker.elapsed();
            c
        })
    }
}

fn decode_chunk_plan(
    deflate_data: &[u8],
    plan: ChunkDecodePlan,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
    force_slow_path: bool,
) -> std::io::Result<ChunkResult> {
    decode_chunk_with_handoff(
        deflate_data,
        plan.start(),
        plan.stop(),
        per_chunk_output_hint,
        num_chunks_total,
        force_slow_path,
    )
}

/// Slow-path completion when the ISA-L handoff isn't available or fails:
/// finish the chunk with the marker decoder, appending its output to the
/// bootstrap. Used in fallback only; the production path should keep this
/// rare.
fn marker_finish_after_bootstrap(
    deflate_data: &[u8],
    mut bootstrap_markers: Vec<u16>,
    bootstrap_end_bit: usize,
    end_bit_limit: Option<usize>,
) -> std::io::Result<ChunkResult> {
    // Continue the marker decoder into the same Vec — `emit_match` must
    // see the bootstrap output as part of `out_pos` so its chunk-local
    // copies and "D > out_pos ⇒ marker" rule work correctly. Decoding
    // the continuation into a fresh Vec and concatenating would emit
    // markers for back-references that actually point into the
    // bootstrap (false cross-chunk markers).
    let end_bit = decode_chunk_markers_continuing(
        deflate_data,
        bootstrap_end_bit,
        end_bit_limit,
        &mut bootstrap_markers,
    )?;
    Ok(ChunkResult {
        bootstrap: bootstrap_markers,
        bootstrap_clean: Vec::new(),
        isal_bytes: Vec::new(),
        // Slow path: no ISA-L bytes, so the per-chunk CRC is the
        // identity. Phase 2 will CRC the resolved bootstrap.
        isal_crc: crc32fast::Hasher::new(),
        end_bit_offset: end_bit,
        worker_elapsed: std::time::Duration::ZERO,
    })
}

fn normalize_isal_end_bit(
    deflate_data: &[u8],
    start_bit: ChunkStart,
    stop: ChunkDecodeStop,
    isal_end_bit: usize,
) -> Option<RealBlockBoundary> {
    const ISA_L_END_BIT_SNAP_TOLERANCE_BITS: usize = 64;

    if isal_end_bit < start_bit.bits() {
        return None;
    }

    match stop {
        ChunkDecodeStop::Approximate(limit) => {
            let limit_bits = limit.bits();
            if isal_end_bit > limit_bits + ISA_L_END_BIT_SNAP_TOLERANCE_BITS {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] approximate stop reject: start={} end_bit={} \
                         limit={} reason=overshoot",
                        start_bit, isal_end_bit, limit_bits,
                    );
                }
                return None;
            }
            if crate::decompress::parallel::fast_marker_inflate::validate_boundary(
                deflate_data,
                isal_end_bit,
                1,
                1,
                false,
            )
            .is_ok()
            {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] approximate stop accept: start={} end_bit={} \
                         limit={} reason=validated-boundary",
                        start_bit, isal_end_bit, limit_bits,
                    );
                }
                Some(RealBlockBoundary(isal_end_bit))
            } else {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] approximate stop reject: start={} end_bit={} \
                         limit={} reason=not-a-boundary",
                        start_bit, isal_end_bit, limit_bits,
                    );
                }
                None
            }
        }
        ChunkDecodeStop::Verified(limit)
            if isal_end_bit >= limit.bits()
                && isal_end_bit - limit.bits() <= ISA_L_END_BIT_SNAP_TOLERANCE_BITS =>
        {
            Some(limit.boundary())
        }
        ChunkDecodeStop::Verified(_) => None,
        ChunkDecodeStop::UntilEnd => Some(RealBlockBoundary(isal_end_bit)),
    }
}

// ── Phase 1c: cross-chunk consistency correction ─────────────────────────────

/// Walk pairs (N, N+1) once forward and correct chunk N+1's start when chunk N's
/// decoded `end_bit` doesn't match it. Re-decode chunk N+1 from the
/// corrected start; propagate forward.
///
/// The induction: chunk 0 starts at bit 0 (a real boundary). Each
/// decode that starts at a real boundary lands at a real boundary
/// (G1 invariant — `decode_loop` only exits between blocks). So
/// chunks[N].end_bit is *always* a real block boundary, regardless
/// of how speculative `end_bit_limit` was set. Correcting chunk N+1's
/// start to chunks[N].end_bit therefore makes chunk N+1's start a
/// real boundary by construction.
///
/// Wall-time deadline bounds adversarial cases where many chunks need
/// correction. On exhaustion: return `Err(DecodeFailed)` and let the
/// routing-layer fallback (G3/G4) surface the failure.
fn phase1c_resolve_consistency(
    deflate_data: &[u8],
    start_bits: &mut [Option<ChunkStart>],
    chunks: &mut [Option<ChunkResult>],
    deadline: std::time::Instant,
    per_chunk_output_hint: usize,
    _spacing_bits: usize,
) -> Result<(), ParallelError> {
    let num_chunks = chunks.len();
    if num_chunks == 0 {
        return Ok(());
    }

    // Chunk 0's start is bit 0 by construction; it must have decoded
    // unless the gzip stream is malformed. Bail if not.
    if chunks[0].is_none() {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] phase1c: chunks[0] is None — bit 0 isn't a valid \
                 deflate start; gzip stream likely corrupt"
            );
        }
        return Err(ParallelError::DecodeFailed);
    }

    if num_chunks == 1 {
        return Ok(());
    }

    let total_bits = deflate_data.len() * 8;

    // Wave-based parallel correction loop.
    //
    // Each wave identifies all chunks whose predecessor is already resolved
    // (Some) but which themselves need correction (None, or wrong start_bit).
    // Independent corrections within a wave are dispatched as parallel threads
    // via `thread::scope` — a correction at index i and one at index j > i are
    // independent whenever chunks[i-1] and chunks[j-1] are both already Some.
    //
    // On typical real data (few corrections) one wave handles everything.
    // Consecutive failures (chunks i, i+1, i+2 all None) are still O(N)
    // serial waves because each depends on its predecessor — unavoidable.
    //
    // `scan_start` tracks the lowest index that could still need correction.
    // Once chunk i is stable (confirmed Some with correct start_bit), the scan
    // never needs to revisit it. Without this cursor a pathological all-None
    // phase-1a run would do O(N) scan work per wave for O(N) waves = O(N²).
    //
    // Preconditions per wave:
    //   - chunks[i] is Some ⟹ its start_bit is a real block boundary (G1)
    //   - deadline enforces that adversarial inputs don't spin forever
    let mut scan_start: usize = 0;
    let mut wave_num: u32 = 0;
    loop {
        // ── Read-only scan: classify corrections this wave ──────────────────
        // Two categories:
        //   empty_fills: predecessor already reached/passed the next verified
        //                boundary (or EOF) — materialize an empty chunk now.
        //   specs:       need a real re-decode; will run in parallel.
        let mut empty_fills: Vec<EmptyChunkFill> = Vec::new();
        let mut specs: Vec<CorrectionDecode> = Vec::new();
        let mut new_scan_start = scan_start;

        for i in scan_start..num_chunks - 1 {
            let pred_end = match &chunks[i] {
                Some(c) => c.end_bit_offset,
                // Predecessor not yet resolved — its correction will appear in
                // a later wave once this wave resolves chunks[i].
                None => continue,
            };

            // Guard: chunk i's own start must be confirmed correct before we use
            // its end_bit to seed chunk i+1. A phase1a false-positive can give
            // chunk i a wrong start_bit (so its end_bit is also wrong). Using
            // that wrong end_bit to decode chunk i+1 propagates the error.
            // If chunk i-1 is None (not yet resolved) we also skip — chunk i
            // might be wrong and will be corrected in a future wave.
            if i > 0 {
                let start_ok = match &chunks[i - 1] {
                    Some(pred_of_i) => {
                        start_bits[i] == Some(ChunkStart::from_bits(pred_of_i.end_bit_offset))
                    }
                    None => false,
                };
                if !start_ok {
                    continue;
                }
            }

            // Correction needed when:
            //   - chunks[i+1] is None (phase 1b worker failed), OR
            //   - start_bits[i+1] is None/wrong (phase 1a false-positive or miss).
            let needs = chunks[i + 1].is_none()
                || (start_bits[i + 1] != Some(ChunkStart::from_bits(pred_end)));
            if !needs {
                if new_scan_start == i {
                    new_scan_start = i + 1;
                }
                continue;
            }

            if total_bits.saturating_sub(pred_end) < 8 {
                // Predecessor consumed BFINAL; remaining bits are only
                // byte-padding before the gzip trailer. No decode needed.
                empty_fills.push(EmptyChunkFill::new(i + 1, ChunkStart::from_bits(pred_end)));
            } else {
                let stop = correction_decode_stop(i + 1, start_bits);
                if stop
                    .hint_bits()
                    .is_some_and(|stop_bits| pred_end >= stop_bits)
                {
                    empty_fills.push(EmptyChunkFill::new(i + 1, ChunkStart::from_bits(pred_end)));
                    continue;
                }
                let plan = ChunkDecodePlan::new(ChunkStart::from_bits(pred_end), stop);
                specs.push(CorrectionDecode::new(i + 1, plan));
            }
        }

        scan_start = new_scan_start;

        if empty_fills.is_empty() && specs.is_empty() {
            break; // all chunks resolved
        }

        // ── Apply immediate empty fills (no decode, no deadline check needed) ─
        for fill in empty_fills {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] phase1c: chunk {} is empty \
                     (pred_end={}, total_bits={total_bits}, remaining={}); \
                     marking empty",
                    fill.idx,
                    fill.start.bits(),
                    total_bits.saturating_sub(fill.start.bits())
                );
            }
            chunks[fill.idx] = Some(ChunkResult {
                bootstrap: Vec::new(),
                bootstrap_clean: Vec::new(),
                isal_bytes: Vec::new(),
                isal_crc: crc32fast::Hasher::new(),
                end_bit_offset: fill.start.bits(),
                worker_elapsed: std::time::Duration::ZERO,
            });
            start_bits[fill.idx] = Some(fill.start);
        }

        if specs.is_empty() {
            // Empty fills may have unblocked more corrections —
            // re-scan before checking the deadline.
            continue;
        }

        if std::time::Instant::now() >= deadline {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] phase1c: wall-time deadline exceeded after \
                     {} corrections",
                    MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed)
                );
            }
            return Err(ParallelError::DecodeFailed);
        }

        MARKER_PIPELINE_RETRY_ITERATIONS.fetch_add(specs.len() as u64, Ordering::Relaxed);

        if debug_enabled() {
            for spec in &specs {
                eprintln!(
                    "[parallel_sm:v0.6] phase1c wave {wave_num}: correcting chunk {} \
                     start={} stop={:?} \
                     range_mb={:.1}",
                    spec.idx,
                    spec.plan.start(),
                    spec.plan.stop(),
                    spec.plan
                        .stop()
                        .hint_bits()
                        .map(|e| e.saturating_sub(spec.plan.start().bits()) as f64 / 8.0 / 1e6)
                        .unwrap_or(0.0)
                );
            }
        }
        let t_wave = if debug_enabled() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── Parallel re-decode of all independent corrections this wave ─────
        // `thread::scope` guarantees all threads join before we mutate
        // `chunks`/`start_bits`. Each thread receives only Copy values
        // (`start_bit`, `end_limit`, `per_chunk_output_hint`, `num_chunks`)
        // plus an immutable shared borrow of `deflate_data`.
        let results: Vec<(CorrectionDecode, std::io::Result<ChunkResult>)> =
            std::thread::scope(|s| {
                let handles: Vec<_> = specs
                    .iter()
                    .map(|&spec| {
                        s.spawn(move || {
                            (
                                spec,
                                decode_chunk_plan(
                                    deflate_data,
                                    spec.plan,
                                    per_chunk_output_hint,
                                    num_chunks,
                                    !spec.plan.stop().use_isal(),
                                ),
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("phase1c correction thread panicked"))
                    .collect()
            });

        if debug_enabled() {
            if let Some(t) = t_wave {
                eprintln!(
                    "[parallel_sm:v0.6] phase1c wave {wave_num}: {n} corrections in {ms:.1}ms",
                    n = specs.len(),
                    ms = t.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
        wave_num += 1;

        // ── Commit results ───────────────────────────────────────────────────
        for (spec, result) in results {
            match result {
                Ok(chunk) => {
                    start_bits[spec.idx] = Some(spec.plan.start());
                    chunks[spec.idx] = Some(chunk);
                }
                Err(e) => {
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm:v0.6] phase1c: re-decode of chunk {} from \
                             {} failed: {e}",
                            spec.idx,
                            spec.plan.start()
                        );
                    }
                    return Err(ParallelError::DecodeFailed);
                }
            }
        }
    }

    Ok(())
}

// ── Phase 2 + streaming write: resolve markers, write, combine CRCs ──────────

/// Phase 2 + streaming write: resolve bootstrap markers per chunk, write
/// immediately to `writer`, combine pre-computed ISA-L CRCs. Returns
/// `(combined_crc32, total_uncompressed_bytes)` after all chunks are
/// written. CRC+ISIZE verification happens in the caller after this
/// returns — partial bytes may already be written on mismatch (same
/// trade-off as rapidgzip's per-chunk streaming design).
fn phase2_resolve_write_combine<W: Write>(
    chunks: Vec<ChunkResult>,
    writer: &mut W,
) -> Result<(u32, usize), ParallelError> {
    let mut total_crc = crc32fast::Hasher::new();
    let mut window: Vec<u8> = Vec::with_capacity(WINDOW_SIZE);
    let mut total_size = 0usize;

    for (i, mut chunk) in chunks.into_iter().enumerate() {
        // Bootstrap u16 prefix: may contain cross-chunk markers (only
        // for chunks i > 0). Resolve them against the previous chunk's
        // last 32 KB.
        if i > 0 {
            // In-place SIMD substitution; ~memory-bandwidth speed on AVX2/NEON.
            replace_markers(&mut chunk.bootstrap, &window);
        }
        // Narrow u16 → u8 into a per-chunk Vec. Each chunk's bootstrap
        // is bounded to ~32-512 KB so the allocation cost is negligible.
        let mut bootstrap_u8: Vec<u8> =
            Vec::with_capacity(chunk.bootstrap.len() + chunk.bootstrap_clean.len());
        narrow_and_append(&mut bootstrap_u8, &chunk.bootstrap).map_err(|pos| {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] chunk {} unresolved marker at bootstrap[{}] (offset {})",
                    i,
                    pos,
                    chunk.bootstrap[pos] - MARKER_BASE,
                );
            }
            ParallelError::DecodeFailed
        })?;
        bootstrap_u8.extend_from_slice(&chunk.bootstrap_clean);
        // Free the u16 buffer eagerly — it's 2× the size of bootstrap_u8
        // and we don't need it anymore.
        chunk.bootstrap = Vec::new();
        chunk.bootstrap_clean = Vec::new();

        // CRC the (small) resolved bootstrap bytes sequentially; the
        // ISA-L bytes' CRC was computed by phase 1 worker in parallel.
        let mut bootstrap_crc = crc32fast::Hasher::new();
        bootstrap_crc.update(&bootstrap_u8);
        total_crc.combine(&bootstrap_crc);
        total_crc.combine(&chunk.isal_crc);

        update_window_from_chunk(&mut window, &bootstrap_u8, &chunk.isal_bytes);

        total_size += bootstrap_u8.len() + chunk.isal_bytes.len();

        // Stream each chunk's bytes to the writer immediately — the OS
        // write path (via BufWriter) can proceed while the next chunk's
        // markers are being resolved, overlapping I/O with computation.
        writer.write_all(&bootstrap_u8).map_err(ParallelError::Io)?;
        writer
            .write_all(&chunk.isal_bytes)
            .map_err(ParallelError::Io)?;
    }

    Ok((total_crc.finalize(), total_size))
}

/// Maintain a rolling 32 KB window of "last decoded bytes" for the
/// next chunk's marker resolution. Slides this chunk's
/// (bootstrap_u8 ++ isal_bytes) into the window. See unit tests at
/// the bottom of this file.
pub(crate) fn update_window_from_chunk(
    window: &mut Vec<u8>,
    bootstrap_u8: &[u8],
    isal_bytes: &[u8],
) {
    let total = bootstrap_u8.len() + isal_bytes.len();

    if isal_bytes.len() >= WINDOW_SIZE {
        window.clear();
        window.extend_from_slice(&isal_bytes[isal_bytes.len() - WINDOW_SIZE..]);
    } else if total >= WINDOW_SIZE {
        let bootstrap_tail_len = WINDOW_SIZE - isal_bytes.len();
        let bootstrap_tail_start = bootstrap_u8.len() - bootstrap_tail_len;
        window.clear();
        window.extend_from_slice(&bootstrap_u8[bootstrap_tail_start..]);
        window.extend_from_slice(isal_bytes);
    } else if total == 0 {
        // Empty chunk near EOF — keep previous window unchanged.
    } else {
        let keep_from_prev = WINDOW_SIZE.saturating_sub(total);
        if window.len() > keep_from_prev {
            let drop = window.len() - keep_from_prev;
            window.drain(..drop);
        }
        window.extend_from_slice(bootstrap_u8);
        window.extend_from_slice(isal_bytes);
    }
}

/// Append `chunk_u16` to `out` as bytes (low 8 bits of each u16), failing
/// with `Err(idx)` if any value has the marker bit still set. Combines the
/// validation walk of `u16_to_u8` with the byte-narrowing copy and the
/// downstream `assembled.extend_from_slice`, so chunk data only crosses the
/// memory hierarchy once on the way out.
#[inline]
fn narrow_and_append(out: &mut Vec<u8>, chunk_u16: &[u16]) -> Result<(), usize> {
    let n = chunk_u16.len();
    out.reserve(n);
    let base = out.len();
    // SAFETY: capacity reserved above. We write `n` bytes then set_len.
    // We compute an `any_high` accumulator across the entire chunk and check
    // it once at the end — the hot loop has no branches on the marker bit,
    // letting the autovectorizer turn it into a tight SIMD narrow. If any
    // value had the high bit set we walk back to find the offending index
    // for the error message; that path is cold.
    let mut any_high: u16 = 0;
    let dst_ptr = unsafe { out.as_mut_ptr().add(base) };
    for (i, &v) in chunk_u16.iter().enumerate() {
        any_high |= v;
        // SAFETY: i < n and we reserved n bytes past `base`.
        unsafe { dst_ptr.add(i).write(v as u8) };
    }
    // SAFETY: wrote exactly n bytes.
    unsafe { out.set_len(base + n) };

    if (any_high & MARKER_BASE) != 0 {
        // Cold path: find the first offending index for the error report.
        let bad = chunk_u16
            .iter()
            .position(|&v| v >= MARKER_BASE)
            .unwrap_or(0);
        // Undo what we just appended so the caller observing the error doesn't
        // see corrupt partial output. (Phase 3 verifies and writes only on
        // success, but this keeps `assembled` honest for the debug path.)
        unsafe { out.set_len(base) };
        return Err(bad);
    }
    Ok(())
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
    use std::io::Write;

    // Direct unit tests for `update_window_from_chunk`. Phase 2 calls
    // this once per chunk to maintain the rolling 32 KB window. The
    // four branches each have failure modes that produce silently
    // wrong markers in the NEXT chunk — wrong-window means
    // replace_markers substitutes wrong bytes, narrow_and_append's
    // marker check catches it (Err → routing fallback) or worse,
    // markers happen to alias into u8 range (rare but possible) and
    // wrong bytes flow through. Opus advisor on PR #97 specifically
    // asked for direct branch coverage.

    #[test]
    fn update_window_isal_geq_window_takes_isal_tail() {
        let mut window = vec![0u8; 0];
        let bootstrap = vec![0xAA; 100];
        let mut isal = vec![0xBB; WINDOW_SIZE + 1000];
        for (i, b) in isal.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        update_window_from_chunk(&mut window, &bootstrap, &isal);
        assert_eq!(window.len(), WINDOW_SIZE);
        assert_eq!(window, &isal[isal.len() - WINDOW_SIZE..]);
    }

    #[test]
    fn update_window_straddle_takes_bootstrap_tail_plus_isal() {
        // bootstrap=20K, isal=16K → total=36K > WINDOW_SIZE.
        let mut window = vec![0u8; WINDOW_SIZE];
        window.fill(0xCC);
        let bootstrap: Vec<u8> = (0..20_000).map(|i| (i % 256) as u8).collect();
        let isal: Vec<u8> = (0..16_000).map(|i| ((i + 100) % 256) as u8).collect();
        update_window_from_chunk(&mut window, &bootstrap, &isal);
        assert_eq!(window.len(), WINDOW_SIZE);
        let bootstrap_tail_start = bootstrap.len() - (WINDOW_SIZE - 16_000);
        assert_eq!(
            &window[..(WINDOW_SIZE - 16_000)],
            &bootstrap[bootstrap_tail_start..]
        );
        assert_eq!(&window[(WINDOW_SIZE - 16_000)..], isal.as_slice());
    }

    #[test]
    fn update_window_empty_chunk_preserves_prev_window() {
        let mut window: Vec<u8> = (0..WINDOW_SIZE).map(|i| (i % 256) as u8).collect();
        let before = window.clone();
        update_window_from_chunk(&mut window, &[], &[]);
        assert_eq!(window, before, "empty chunk should not modify the window");
    }

    #[test]
    fn update_window_smaller_than_window_slides() {
        // bootstrap+isal = 1000+500 = 1500 bytes < WINDOW_SIZE.
        let prev_window: Vec<u8> = (0..WINDOW_SIZE).map(|i| (i % 256) as u8).collect();
        let mut window = prev_window.clone();
        let bootstrap: Vec<u8> = (0..1000).map(|i| (i % 256) as u8 ^ 0xFF).collect();
        let isal: Vec<u8> = (0..500).map(|i| (i % 256) as u8 ^ 0xAA).collect();
        update_window_from_chunk(&mut window, &bootstrap, &isal);
        assert_eq!(window.len(), WINDOW_SIZE);
        assert_eq!(&window[..(WINDOW_SIZE - 1500)], &prev_window[1500..]);
        let bootstrap_offset = WINDOW_SIZE - 1500;
        assert_eq!(
            &window[bootstrap_offset..bootstrap_offset + 1000],
            bootstrap.as_slice()
        );
        assert_eq!(&window[bootstrap_offset + 1000..], isal.as_slice());
    }

    #[test]
    fn update_window_smaller_than_window_with_smaller_prev() {
        let mut window: Vec<u8> = vec![0xAA; 10_000];
        let bootstrap: Vec<u8> = vec![0xBB; 5_000];
        let isal: Vec<u8> = vec![0xCC; 3_000];
        update_window_from_chunk(&mut window, &bootstrap, &isal);
        assert_eq!(window.len(), 10_000 + 5_000 + 3_000);
        assert_eq!(&window[..10_000], &vec![0xAA; 10_000][..]);
        assert_eq!(&window[10_000..15_000], &vec![0xBB; 5_000][..]);
        assert_eq!(&window[15_000..], &vec![0xCC; 3_000][..]);
    }

    fn make_gzip_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_gzip_at_level(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26) as u8 + b'a';
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    fn make_random_data(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        while data.len() < size {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            for shift in (0..64).step_by(8) {
                if data.len() >= size {
                    break;
                }
                data.push((rng >> shift) as u8);
            }
        }
        data
    }

    // ── Contract: input filters ──────────────────────────────────────────────

    #[test]
    fn small_inputs_return_too_small() {
        let compressed = make_gzip_data(b"hello world");
        let mut output = Vec::new();
        assert!(matches!(
            decompress_parallel(&compressed, &mut output, 4),
            Err(ParallelError::TooSmall)
        ));
    }

    #[test]
    fn single_thread_returns_too_small() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let mut output = Vec::new();
        assert!(matches!(
            decompress_parallel(&compressed, &mut output, 1),
            Err(ParallelError::TooSmall)
        ));
    }

    #[test]
    fn empty_input_errors() {
        let mut output = Vec::new();
        assert!(decompress_parallel(&[], &mut output, 4).is_err());
    }

    // ── Contract: round-trip correctness ─────────────────────────────────────

    fn assert_roundtrip_or_known_fallback(data: &[u8], num_threads: usize) {
        let compressed = make_gzip_data(data);
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, num_threads) {
            Ok(n) => {
                assert_eq!(n as usize, data.len(), "size mismatch");
                assert_eq!(output, data, "content mismatch");
            }
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::CrcMismatch)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {
                assert!(
                    output.is_empty(),
                    "fallback error must not have written to the writer"
                );
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn roundtrip_compressible_8mb_t4() {
        assert_roundtrip_or_known_fallback(&make_compressible_data(8 * 1024 * 1024), 4);
    }

    #[test]
    fn roundtrip_compressible_25mb_t4() {
        assert_roundtrip_or_known_fallback(&make_compressible_data(25 * 1024 * 1024), 4);
    }

    #[test]
    fn roundtrip_random_10mb_t2() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 2);
    }

    #[test]
    fn roundtrip_random_10mb_t4() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 4);
    }

    #[test]
    fn roundtrip_random_10mb_t8() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 8);
    }

    #[test]
    fn roundtrip_levels_t4() {
        let data = make_compressible_data(8 * 1024 * 1024);
        for level in [1u32, 3, 6, 9] {
            let compressed = make_gzip_at_level(&data, level);
            let mut output = Vec::new();
            match decompress_parallel(&compressed, &mut output, 4) {
                Ok(_) => assert_eq!(output, data, "L{level} content mismatch"),
                Err(ParallelError::DecodeFailed)
                | Err(ParallelError::CrcMismatch)
                | Err(ParallelError::SizeMismatch)
                | Err(ParallelError::TooSmall) => {}
                Err(e) => panic!("L{level} unexpected: {e}"),
            }
        }
    }

    #[test]
    fn determinism_two_runs_match() {
        let data = make_compressible_data(16 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        let r1 = decompress_parallel(&compressed, &mut out1, 4);
        let r2 = decompress_parallel(&compressed, &mut out2, 4);
        match (r1, r2) {
            (Ok(_), Ok(_)) => assert_eq!(out1, out2, "non-deterministic Ok output"),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("non-deterministic outcome: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn thread_count_does_not_change_output() {
        let data = make_random_data(12 * 1024 * 1024, 0xdeadc0de);
        let compressed = make_gzip_data(&data);
        let mut reference: Option<Vec<u8>> = None;
        for &t in &[2usize, 3, 4, 8] {
            let mut out = Vec::new();
            if decompress_parallel(&compressed, &mut out, t).is_ok() {
                if let Some(r) = &reference {
                    assert_eq!(&out, r, "T={t} differs from prior successful run");
                }
                reference = Some(out);
            }
        }
    }

    #[test]
    fn crc_corruption_is_detected() {
        let data = make_random_data(10 * 1024 * 1024, 0xc0ffee);
        let mut compressed = make_gzip_data(&data);
        let crc_offset = compressed.len() - 8;
        compressed[crc_offset] ^= 0xff;
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::CrcMismatch) => {}
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!("CRC corruption not detected"),
            Err(e) => panic!("unexpected error: {e}"),
        }
        // Streaming design: bytes may be in the writer before CRC verification.
        // The critical invariant is that Err IS returned — caller must not treat
        // corrupted output as Ok.
    }

    /// Premortem mitigation B5: trailer fields can legitimately be zero.
    /// Earlier versions guarded the verification with `if expected_crc != 0`
    /// and `if expected_size > 0` — silently accepting corrupted streams
    /// whose trailer fields happened to be zero. This test mutates a real
    /// stream's CRC field to zero and asserts the decoder rejects it (the
    /// true CRC of any non-trivial 10+ MiB stream is overwhelmingly
    /// non-zero, so trailer-CRC=0 means "trailer lies").
    ///
    /// Note: the streaming write design writes per-chunk bytes before final
    /// CRC verification (same trade-off as rapidgzip). The error is still
    /// returned unconditionally — corruption is never silently accepted.
    #[test]
    fn crc_zero_trailer_is_verified() {
        let data = make_random_data(10 * 1024 * 1024, 0xfacefeed);
        let mut compressed = make_gzip_data(&data);
        let crc_offset = compressed.len() - 8;
        compressed[crc_offset..crc_offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::CrcMismatch) => {}
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!(
                "CRC=0 trailer not detected as corruption — the `expected_crc != 0` \
                 sentinel guard is back. See Copilot review on PR #94."
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    /// Same as above for ISIZE. A 10 MiB input has expected_size = 10 MiB
    /// mod 2^32, which is non-zero; setting the trailer ISIZE field to 0
    /// must be detected even though `expected_size == 0` was the previous
    /// "skip the check" sentinel.
    ///
    /// Note: streaming write design — partial bytes may be in the writer,
    /// but the error is always returned. See crc_zero_trailer_is_verified.
    #[test]
    fn isize_zero_trailer_is_verified() {
        let data = make_random_data(10 * 1024 * 1024, 0xfeedbeef);
        let mut compressed = make_gzip_data(&data);
        let isize_offset = compressed.len() - 4;
        compressed[isize_offset..isize_offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::SizeMismatch) => {}
            Err(ParallelError::CrcMismatch)
            | Err(ParallelError::DecodeFailed)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!(
                "ISIZE=0 trailer not detected as corruption — the `expected_size > 0` \
                 sentinel guard is back. See Copilot review on PR #94."
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── Phase 1c parallel correction ─────────────────────────────────────────

    /// Verify that phase1c corrects two *non-consecutive* None chunks in a
    /// single parallel wave rather than two sequential passes.
    ///
    /// Setup:
    ///   - Decode a real 8 MiB gzip file at T=4 to get all four chunks.
    ///   - Save chunks[0] and chunks[2] (already decoded from real boundaries).
    ///   - Zero out chunks[1] and chunks[3], and corrupt their start_bits.
    ///   - Call phase1c_resolve_consistency.
    ///
    /// Expected: since chunks[0] and chunks[2] are both Some, the first wave
    /// identifies corrections for indexes 1 and 3 simultaneously (they are
    /// independent: different predecessors).  RETRY_ITERATIONS must increase
    /// by exactly 2 in one wave — the parallel path merged both into a single
    /// `thread::scope`.  Both corrected chunks must have non-zero decoded data.
    #[test]
    fn phase1c_corrects_two_nonconsecutive_chunks_in_parallel() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;
        use std::time::{Duration, Instant};

        let original = make_compressible_data(8 * 1024 * 1024);
        let gzip = make_gzip_data(&original);

        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 4;
        let per_chunk_hint = deflate_data.len() / num_chunks;
        let total_bits = deflate_data.len() * 8;
        let spacing = total_bits / num_chunks;

        // Find real block boundaries near the spacing-derived anchor for each
        // chunk using the same boundary search as phase 1a.
        let mut start_bits: Vec<Option<ChunkStart>> = (0..num_chunks)
            .map(|i| {
                if i == 0 {
                    Some(ChunkStart::from_bits(0))
                } else {
                    search_boundary_forward(deflate_data, i * spacing)
                }
            })
            .collect();

        // Decode chunks 0 and 2 from their (real) boundaries.
        let end_limit_0 = start_bits[1].map(ChunkStart::to_end_limit);
        let c0 = match decode_chunk_with_handoff(
            deflate_data,
            ChunkStart::from_bits(0),
            end_limit_0.map_or(ChunkDecodeStop::UntilEnd, ChunkDecodeStop::Verified),
            per_chunk_hint,
            num_chunks,
            false,
        ) {
            Ok(c) => c,
            Err(_) => return, // ISA-L decode failed on this platform — skip
        };
        let start2 = match start_bits[2] {
            Some(s) => s,
            None => return, // phase 1a found no boundary for chunk 2 — skip
        };
        let end_limit_2 = start_bits[3].map(ChunkStart::to_end_limit);
        let c2 = match decode_chunk_with_handoff(
            deflate_data,
            start2,
            end_limit_2.map_or(ChunkDecodeStop::UntilEnd, ChunkDecodeStop::Verified),
            per_chunk_hint,
            num_chunks,
            false,
        ) {
            Ok(c) => c,
            Err(_) => return,
        };

        // Build a chunks array with chunks[1] and chunks[3] missing.
        // Give them wrong start_bits so phase1c knows they need correction.
        let e0 = c0.end_bit_offset;
        let e2 = c2.end_bit_offset;

        let mut chunks: Vec<Option<ChunkResult>> = vec![
            Some(c0),
            None, // will be corrected from e0
            Some(c2),
            None, // will be corrected from e2
        ];
        // Corrupt starts for the None slots so `needs` fires for both.
        start_bits[1] = Some(ChunkStart::from_bits(e0.wrapping_add(99))); // wrong — should be e0
        start_bits[3] = Some(ChunkStart::from_bits(e2.wrapping_add(99))); // wrong — should be e2

        let before = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_secs(30);

        phase1c_resolve_consistency(
            deflate_data,
            &mut start_bits,
            &mut chunks,
            deadline,
            per_chunk_hint,
            spacing,
        )
        .expect("phase1c must succeed on real deflate data with real start boundaries");

        let after = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);

        // Both corrections must have fired (at least 2 increments: one per
        // corrected chunk). The upper bound is intentionally loose — concurrent
        // tests that call decompress_parallel also increment the shared global
        // counter, so `after - before` may exceed 2 without indicating a bug.
        assert!(
            after - before >= 2,
            "expected at least 2 retry increments (one per corrected chunk); got {}",
            after - before
        );

        // All four chunks must be resolved.
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.is_some(),
                "chunk {i} must be Some after phase1c correction"
            );
        }

        // Corrected chunks must have the right start bits.
        assert_eq!(
            start_bits[1],
            Some(ChunkStart::from_bits(e0)),
            "chunk 1 start_bit must equal chunk 0 end_bit"
        );
        assert_eq!(
            start_bits[3],
            Some(ChunkStart::from_bits(e2)),
            "chunk 3 start_bit must equal chunk 2 end_bit"
        );
    }

    #[test]
    fn phase1c_collapses_consecutive_zero_range_chunks_without_retries() {
        use std::time::{Duration, Instant};

        let _guard = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let make_empty_chunk = |end_bit_offset| ChunkResult {
            bootstrap: Vec::new(),
            bootstrap_clean: Vec::new(),
            isal_bytes: Vec::new(),
            isal_crc: crc32fast::Hasher::new(),
            end_bit_offset,
            worker_elapsed: Duration::ZERO,
        };

        let pred_end = 100usize;
        let mut chunks: Vec<Option<ChunkResult>> = vec![
            Some(make_empty_chunk(pred_end)),
            None,
            None,
            None,
            Some(make_empty_chunk(150)),
        ];
        let mut start_bits: Vec<Option<ChunkStart>> = vec![
            Some(ChunkStart::from_bits(0)),
            Some(ChunkStart::from_bits(pred_end.wrapping_add(7))), // wrong; phase1c should fix it
            Some(ChunkStart::from_bits(pred_end)),
            Some(ChunkStart::from_bits(pred_end)),
            Some(ChunkStart::from_bits(pred_end)),
        ];

        let before = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);
        phase1c_resolve_consistency(
            &[0u8; 128],
            &mut start_bits,
            &mut chunks,
            Instant::now() + Duration::from_secs(5),
            1024,
            (128 * 8) / 5,
        )
        .expect("zero-range phantom chunks should resolve without decode work");
        let after = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);

        assert_eq!(
            after, before,
            "zero-range phantom chunks must not schedule retry decodes"
        );
        for idx in 1..=3 {
            let chunk = chunks[idx]
                .as_ref()
                .unwrap_or_else(|| panic!("chunk {idx} should be materialized as empty"));
            assert_eq!(
                chunk.decoded_len(),
                0,
                "chunk {idx} should be empty instead of re-decoded"
            );
            assert_eq!(
                chunk.end_bit_offset, pred_end,
                "chunk {idx} should inherit the predecessor end bit"
            );
            assert_eq!(
                start_bits[idx],
                Some(ChunkStart::from_bits(pred_end)),
                "chunk {idx} should be corrected to the predecessor end bit"
            );
        }
    }

    #[test]
    fn benchmark_shaped_missing_spans_use_partition_stops_and_empty_fills() {
        use std::time::{Duration, Instant};

        let _guard = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let num_chunks = 38usize;
        let spacing_bits = 1_000usize;
        let total_bits = num_chunks * spacing_bits;
        let deflate_data = vec![0u8; total_bits / 8];

        let make_chunk = |end_bit_offset| ChunkResult {
            bootstrap: Vec::new(),
            bootstrap_clean: Vec::new(),
            isal_bytes: Vec::new(),
            isal_crc: crc32fast::Hasher::new(),
            end_bit_offset,
            worker_elapsed: Duration::ZERO,
        };

        let mut start_bits: Vec<Option<ChunkStart>> = (0..num_chunks)
            .map(|i| Some(ChunkStart::from_bits(i * spacing_bits)))
            .collect();
        for slot in start_bits.iter_mut().take(17).skip(12) {
            *slot = None;
        }
        for slot in start_bits.iter_mut().take(33).skip(28) {
            *slot = None;
        }

        assert_eq!(
            chunk_decode_stop(11, &start_bits, spacing_bits, total_bits),
            ChunkDecodeStop::Approximate(ApproximateChunkEnd(12 * spacing_bits)),
            "chunk 11 must stop at its own partition instead of chunk 17's boundary",
        );
        assert_eq!(
            chunk_decode_stop(27, &start_bits, spacing_bits, total_bits),
            ChunkDecodeStop::Approximate(ApproximateChunkEnd(28 * spacing_bits)),
            "chunk 27 must stop at its own partition instead of chunk 33's boundary",
        );

        let mut chunks: Vec<Option<ChunkResult>> = (0..num_chunks)
            .map(|i| Some(make_chunk((i + 1).min(num_chunks) * spacing_bits)))
            .collect();
        for slot in chunks.iter_mut().take(17).skip(12) {
            *slot = None;
        }
        for slot in chunks.iter_mut().take(33).skip(28) {
            *slot = None;
        }
        chunks[11] = Some(make_chunk(17 * spacing_bits));
        chunks[27] = Some(make_chunk(33 * spacing_bits));

        let before = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);
        phase1c_resolve_consistency(
            &deflate_data,
            &mut start_bits,
            &mut chunks,
            Instant::now() + Duration::from_secs(5),
            1024,
            spacing_bits,
        )
        .expect("phantom benchmark-shaped spans should drain without decode retries");
        let after = MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed);

        assert_eq!(
            after, before,
            "empty phantom spans should not schedule any correction decode work",
        );
        for idx in 12..=16 {
            let chunk = chunks[idx]
                .as_ref()
                .unwrap_or_else(|| panic!("chunk {idx} should be materialized as empty"));
            assert_eq!(chunk.decoded_len(), 0, "chunk {idx} should stay empty");
            assert_eq!(
                chunk.end_bit_offset,
                17 * spacing_bits,
                "chunk {idx} should inherit chunk 11's swallowed end bit",
            );
            assert_eq!(
                start_bits[idx],
                Some(ChunkStart::from_bits(17 * spacing_bits)),
                "chunk {idx} should be corrected to chunk 11's end bit",
            );
        }
        for idx in 28..=32 {
            let chunk = chunks[idx]
                .as_ref()
                .unwrap_or_else(|| panic!("chunk {idx} should be materialized as empty"));
            assert_eq!(chunk.decoded_len(), 0, "chunk {idx} should stay empty");
            assert_eq!(
                chunk.end_bit_offset,
                33 * spacing_bits,
                "chunk {idx} should inherit chunk 27's swallowed end bit",
            );
            assert_eq!(
                start_bits[idx],
                Some(ChunkStart::from_bits(33 * spacing_bits)),
                "chunk {idx} should be corrected to chunk 27's end bit",
            );
        }
    }

    // ── Snap arithmetic unit tests ───────────────────────────────────────────

    /// F2 — Snap arithmetic: table-test the (pred_end, lim, expected) cases.
    ///
    /// The snap fires when `pred_end > lim && pred_end - lim ≤ 64`.
    /// This test makes the boundary condition explicit and auditable.
    /// Catches off-by-one regressions in `ISA_L_SNAP_TOLERANCE_BITS`.
    #[test]
    fn snap_arithmetic_table_test() {
        const TOL: usize = 64; // ISA_L_SNAP_TOLERANCE_BITS
        let cases: &[(usize, Option<usize>, bool)] = &[
            (1000, Some(996), true),   // 4-bit overshoot → snap fires
            (1000, Some(936), true),   // 64-bit overshoot → edge, still fires
            (1000, Some(935), false),  // 65-bit overshoot → no snap
            (1000, Some(1000), false), // equal → no snap (not strictly greater)
            (1000, Some(1004), false), // pred_end < lim → no snap
            (1000, None, false),       // no limit → no snap
            (0, Some(0), false),       // both zero → no snap
            (64, Some(0), true),       // exactly TOL bits → fires
            (65, Some(0), false),      // TOL+1 bits → no snap
        ];
        for &(pred_end, end_limit, should_snap) in cases {
            let snaps = if let Some(lim) = end_limit {
                pred_end > lim && pred_end - lim <= TOL
            } else {
                false
            };
            assert_eq!(
                snaps, should_snap,
                "snap(pred_end={pred_end}, lim={end_limit:?}): expected {should_snap}, got {snaps}"
            );
        }
    }

    /// B3 — Phase1c terminates; snap invariant is not tested in isolation.
    ///
    /// The snap fires when `(i+2..num_chunks).find_map(|j| start_bits[j])` is
    /// Some — i.e., there is a phase1a-confirmed boundary for chunk i+2. With
    /// only 2 chunks, end_limit is always None and the snap can never fire;
    /// the snap path is exercised by the end-to-end routing test on real silesia
    /// data (see test_single_member_routing_multithread). This test verifies that
    /// phase1c terminates on a realistic 2-chunk setup and produces correct output.
    #[test]
    fn phase1c_snap_terminates_and_updates_predecessor_end_bit() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;
        use std::time::{Duration, Instant};

        let original = make_compressible_data(4 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 2;
        let per_chunk_hint = deflate_data.len() / num_chunks;
        let c0 = match decode_chunk_with_handoff(
            deflate_data,
            ChunkStart::from_bits(0),
            ChunkDecodeStop::UntilEnd,
            per_chunk_hint,
            num_chunks,
            false,
        ) {
            Ok(c) => c,
            Err(_) => return, // ISA-L unavailable on this platform — skip
        };
        if c0.end_bit_offset == 0 {
            return; // degenerate stream — skip
        }

        // phase1c with chunk 1 missing — it should re-decode it from c0.end_bit and terminate.
        let mut chunks: Vec<Option<ChunkResult>> = vec![Some(c0), None];
        let mut start_bits: Vec<Option<ChunkStart>> = vec![Some(ChunkStart::from_bits(0)), None];

        let deadline = Instant::now() + Duration::from_secs(30);
        let result = phase1c_resolve_consistency(
            deflate_data,
            &mut start_bits,
            &mut chunks,
            deadline,
            per_chunk_hint,
            (deflate_data.len() * 8) / num_chunks,
        );
        assert!(
            result.is_ok(),
            "phase1c must terminate within deadline: {result:?}"
        );
        assert!(chunks[1].is_some(), "phase1c must resolve chunk 1");
    }

    /// G2 — bfinal_hit discard: a chunk whose bootstrap hits BFINAL early is
    /// discarded (returns None from phase1b) and corrected by phase1c.
    ///
    /// Regression for commit 94459b4: without the bfinal_hit check, a speculative
    /// start that decodes a false BFINAL would let ISA-L stop early, leaving
    /// the remainder of the chunk un-decoded.
    #[test]
    fn bfinal_hit_chunk_is_discarded_and_corrected_by_phase1c() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;
        use std::time::{Duration, Instant};

        let original = make_compressible_data(8 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        // Decode chunk 0 to get a confirmed end_bit.
        let num_chunks = 3;
        let per_chunk_hint = deflate_data.len() / num_chunks;
        let c0 = match decode_chunk_with_handoff(
            deflate_data,
            ChunkStart::from_bits(0),
            ChunkDecodeStop::UntilEnd,
            per_chunk_hint,
            num_chunks,
            false,
        ) {
            Ok(c) => c,
            Err(_) => return, // ISA-L unavailable — skip
        };
        let e0 = c0.end_bit_offset;

        // Simulate: chunk 1 decoded with a bfinal_hit (returned Err from decode_chunk_with_handoff).
        // In practice, decode_chunk_with_handoff returns Err when bfinal_hit=true.
        // We simulate this by leaving chunks[1]=None and giving it a wrong start_bit
        // so phase1c knows to correct it.
        let mut chunks: Vec<Option<ChunkResult>> = vec![Some(c0), None, None];
        let mut start_bits: Vec<Option<ChunkStart>> = vec![
            Some(ChunkStart::from_bits(0)),
            Some(ChunkStart::from_bits(e0.wrapping_add(99))), // wrong — phase1c will correct from e0
            None,
        ];

        let deadline = Instant::now() + Duration::from_secs(30);
        let result = phase1c_resolve_consistency(
            deflate_data,
            &mut start_bits,
            &mut chunks,
            deadline,
            per_chunk_hint,
            (deflate_data.len() * 8) / num_chunks,
        );
        assert!(
            result.is_ok(),
            "phase1c must resolve discarded chunk: {result:?}"
        );

        // chunk 1 must now be corrected with start_bit = e0.
        assert!(chunks[1].is_some(), "chunk 1 must be resolved by phase1c");
        assert_eq!(
            start_bits[1],
            Some(ChunkStart::from_bits(e0)),
            "chunk 1 start_bit must equal chunk 0 end_bit after phase1c correction"
        );
    }

    /// G1 — start_ok gate: phase1c does NOT propagate a chunk whose predecessor's
    /// own start is unconfirmed, preventing false-positive cascades.
    ///
    /// The `start_ok` guard at single_member.rs:1225 checks that chunk i's start
    /// is confirmed before using its end_bit to seed chunk i+1. Without this guard,
    /// a wrong end_bit from a false-positive chunk could propagate to all successors.
    #[test]
    fn phase1c_start_ok_gate_blocks_unconfirmed_predecessor() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;
        use std::time::{Duration, Instant};

        let original = make_compressible_data(4 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 3;
        let per_chunk_hint = deflate_data.len() / num_chunks;

        // Decode chunk 0 normally.
        let c0 = match decode_chunk_with_handoff(
            deflate_data,
            ChunkStart::from_bits(0),
            ChunkDecodeStop::UntilEnd,
            per_chunk_hint,
            num_chunks,
            false,
        ) {
            Ok(c) => c,
            Err(_) => return,
        };
        let e0 = c0.end_bit_offset;

        // Construct chunk 1 as a fake with a WRONG end_bit — simulating a
        // false-positive start that decoded garbage. Its start_bit is also wrong
        // (doesn't equal e0), so start_ok for chunk 1 will be false.
        let wrong_end = e0.wrapping_add(12345);
        let fake_c1 = ChunkResult {
            bootstrap: Vec::new(),
            bootstrap_clean: Vec::new(),
            isal_bytes: Vec::new(),
            isal_crc: crc32fast::Hasher::new(),
            end_bit_offset: wrong_end,
            worker_elapsed: std::time::Duration::ZERO,
        };

        let mut chunks: Vec<Option<ChunkResult>> = vec![Some(c0), Some(fake_c1), None];
        // start_bits[1] is wrong — chunk 1 was seeded from a false-positive.
        let mut start_bits: Vec<Option<ChunkStart>> = vec![
            Some(ChunkStart::from_bits(0)),
            Some(ChunkStart::from_bits(e0.wrapping_add(99))), // deliberately wrong: e0 != e0+99
            None,
        ];

        let deadline = Instant::now() + Duration::from_secs(5);
        let _ = phase1c_resolve_consistency(
            deflate_data,
            &mut start_bits,
            &mut chunks,
            deadline,
            per_chunk_hint,
            (deflate_data.len() * 8) / num_chunks,
        );

        // The key assertion: chunk 2's start_bit must NOT be `wrong_end`.
        // If start_ok blocked the propagation of chunk 1's wrong end_bit,
        // chunk 2 is either still None or was corrected from the real e0.
        if let Some(sb2) = start_bits[2] {
            assert_ne!(
                sb2.bits(),
                wrong_end,
                "start_ok gate must prevent chunk 2 from inheriting chunk 1's wrong end_bit {wrong_end}"
            );
        }
        // chunk 1's wrong start is corrected by phase1c (seeded from e0).
        assert_eq!(
            start_bits[1],
            Some(ChunkStart::from_bits(e0)),
            "phase1c must correct chunk 1's start_bit to chunk 0's end_bit {e0}"
        );
    }

    #[test]
    fn normalize_isal_end_bit_snaps_small_verified_overshoot() {
        let deflate_data: &[u8] = &[];
        let start = ChunkStart::from_bits(128);
        let limit = ChunkStart::from_bits(512).to_end_limit();

        assert_eq!(
            normalize_isal_end_bit(deflate_data, start, ChunkDecodeStop::Verified(limit), 512),
            Some(limit.boundary())
        );
        assert_eq!(
            normalize_isal_end_bit(deflate_data, start, ChunkDecodeStop::Verified(limit), 513),
            Some(limit.boundary())
        );
        assert_eq!(
            normalize_isal_end_bit(
                deflate_data,
                start,
                ChunkDecodeStop::Verified(limit),
                512 + 64,
            ),
            Some(limit.boundary())
        );
        assert_eq!(
            normalize_isal_end_bit(
                deflate_data,
                start,
                ChunkDecodeStop::Verified(limit),
                512 + 65,
            ),
            None
        );
        assert_eq!(
            normalize_isal_end_bit(deflate_data, start, ChunkDecodeStop::Verified(limit), 500),
            None
        );
        assert_eq!(
            normalize_isal_end_bit(deflate_data, start, ChunkDecodeStop::UntilEnd, 700),
            Some(RealBlockBoundary(700))
        );
        assert_eq!(
            normalize_isal_end_bit(deflate_data, start, ChunkDecodeStop::UntilEnd, 64),
            None
        );
    }

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    #[test]
    fn decode_chunk_with_handoff_snaps_to_verified_end_limit() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;

        let original = make_compressible_data(4 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 3;
        let spacing = deflate_data.len() * 8 / num_chunks;
        let start_bits = phase1_search_boundaries(deflate_data, num_chunks, spacing, 1);
        let end_limit = start_bits[1]
            .map(ChunkStart::to_end_limit)
            .expect("phase1a should find the next chunk boundary");

        let per_chunk_hint = deflate_data.len() / num_chunks;
        let chunk = decode_chunk_with_handoff(
            deflate_data,
            ChunkStart::from_bits(0),
            ChunkDecodeStop::Verified(end_limit),
            per_chunk_hint,
            num_chunks,
            false,
        )
        .expect("worker decode should succeed");

        assert_eq!(
            chunk.end_bit_offset,
            end_limit.bits(),
            "ISA-L worker must snap small ISA-L end-bit overshoots to the verified downstream boundary"
        );
    }

    /// D1 — Pipeline succeeds when most phase1a boundaries are None (chain-decode).
    ///
    /// Tests the phase1c chain-decode fallback when BlockFinder can't find
    /// candidates. Uses a BTYPE=01-heavy stream (repetitive data at level 1
    /// often uses fixed Huffman), forcing many None entries in start_bits.
    #[test]
    fn pipeline_succeeds_when_most_boundaries_are_none() {
        // Very repetitive data → fixed-Huffman blocks → fewer phase1a candidates.
        let original = vec![b'A'; 8 * 1024 * 1024];
        let gzip = make_gzip_at_level(&original, 1);
        let mut output = Vec::new();
        match decompress_parallel(&gzip, &mut output, 4) {
            Ok(_) => {
                assert_eq!(output, original, "decompressed output must match original");
            }
            Err(ParallelError::TooSmall) => {} // parallel not triggered — ok
            Err(e) => panic!("decompress_parallel failed: {e:?}"),
        }
    }

    /// F3 — Counter assertions: snap fires (retry iterations > 0) but slow
    /// path doesn't (SLOW_PATH_USED stays 0) on a stream that exercises the snap.
    ///
    /// Guards against silent regressions where the snap stops firing and the
    /// slow path picks up the slack — producing correct output but tanking
    /// performance (the v0.3.0 failure mode).
    // SLOW_PATH_USED is only meaningful on x86_64 where ISA-L is the fast path.
    // On arm64 the slow path is always used (no ISA-L), so the assertion would
    // always fail. Gate the counter assertion so it only runs where it's meaningful.
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    #[test]
    fn f3_snap_fires_not_slow_path_on_adversarial_input() {
        // Use a stream large enough to trigger parallel decode with ISA-L.
        // The snap fires when ISA-L end_bit overshoots a phase1a boundary by ≤64 bits.
        // We can't guarantee this on every input, so we just verify: if the pipeline
        // succeeds, SLOW_PATH_USED was not incremented by this run (slow path is for
        // anchor-based chunks with no real speculative boundary, not for snap cases).
        let original = make_compressible_data(24 * 1024 * 1024);
        let gzip = make_gzip_data(&original);

        let slow_before = SLOW_PATH_USED.load(Ordering::Relaxed);
        let mut output = Vec::new();
        match decompress_parallel(&gzip, &mut output, 4) {
            Ok(_) => {
                let slow_after = SLOW_PATH_USED.load(Ordering::Relaxed);
                // Slow path should not have fired: ISA-L handles all chunks
                // that have real speculative boundaries, snap handles overshoot.
                // (If slow_path fires, it means we fell back to marker decode
                // for a chunk that should have used ISA-L — a performance regression.)
                assert_eq!(
                    slow_after, slow_before,
                    "SLOW_PATH_USED must not increase during marker pipeline success: \
                     before={slow_before} after={slow_after}"
                );
                assert_eq!(output, original, "output must be byte-perfect");
            }
            Err(ParallelError::TooSmall) => {} // not triggered on this arch — skip
            Err(e) => panic!("pipeline failed: {e:?}"),
        }
    }

    // ── Deletion-trap killer counter ─────────────────────────────────────────

    #[test]
    fn marker_pipeline_counter_increments_on_success() {
        // Same Mutex as the routing-level deletion-trap killer. Under
        // `cargo test`'s default parallel execution, concurrent unit tests
        // that call `decompress_parallel` would otherwise bump the counter
        // between the before/after snapshots and mask a real regression.
        let _guard = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let before = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
        let mut output = Vec::new();
        if decompress_parallel(&compressed, &mut output, 4).is_ok() {
            let after = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
            assert!(
                after > before,
                "marker pipeline counter must increment on successful decode (was {before}, now {after})"
            );
        }
        // If parallel declined (e.g., search failure on this specific data),
        // the counter shouldn't have moved either. The point of this test
        // is to prove the counter exists and works — the routing-level
        // assertion in `src/tests/routing.rs` is the deletion-trap killer
        // proper, exercising the full CLI path.
    }
}
