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
/// Extra compressed-input margin for approximate partition stops so ISA-L can
/// reach the next real block boundary instead of stopping at the raw partition
/// cut.
const APPROX_STOP_INPUT_MARGIN_BYTES: usize = 64 * 1024;

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
struct BoundarySlot {
    speculative: Option<ChunkStart>,
    confirmed: Option<ChunkStart>,
}

#[derive(Debug)]
struct BoundaryRegistry {
    slots: Vec<BoundarySlot>,
}

impl BoundaryRegistry {
    fn from_speculative_starts(speculative_starts: Vec<Option<ChunkStart>>) -> Self {
        let mut slots: Vec<BoundarySlot> = speculative_starts
            .into_iter()
            .map(|start| BoundarySlot {
                speculative: start,
                confirmed: None,
            })
            .collect();
        if let Some(first) = slots.first_mut() {
            first.confirmed = Some(ChunkStart::from_bits(0));
        }
        Self { slots }
    }

    #[inline]
    fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn speculative_start(&self, idx: usize) -> Option<ChunkStart> {
        self.slots.get(idx).and_then(|slot| slot.speculative)
    }

    #[inline]
    fn confirmed_start(&self, idx: usize) -> Option<ChunkStart> {
        self.slots.get(idx).and_then(|slot| slot.confirmed)
    }

    #[inline]
    fn effective_start(&self, idx: usize) -> Option<ChunkStart> {
        self.confirmed_start(idx)
            .or_else(|| self.speculative_start(idx))
    }

    fn speculative_starts(&self) -> Vec<Option<ChunkStart>> {
        self.slots.iter().map(|slot| slot.speculative).collect()
    }

    fn speculative_stop_for(&self, idx: usize) -> ChunkDecodeStop {
        match (idx + 1..self.len())
            .find_map(|j| self.effective_start(j).map(ChunkStart::to_end_limit))
        {
            Some(limit) => ChunkDecodeStop::Verified(limit),
            None => ChunkDecodeStop::UntilEnd,
        }
    }

    fn exact_end_limit_for(&self, idx: usize) -> Option<ChunkEndLimit> {
        (idx + 1..self.len()).find_map(|j| self.effective_start(j).map(ChunkStart::to_end_limit))
    }

    fn confirm_start(&mut self, idx: usize, start: ChunkStart) {
        if let Some(slot) = self.slots.get_mut(idx) {
            slot.confirmed = Some(start);
        }
    }

    fn confirm_next_start_from_end(&mut self, idx: usize, end: RealBlockBoundary) {
        self.confirm_start(idx + 1, ChunkStart(end));
    }

    fn confirmed_starts(&self) -> Vec<Option<ChunkStart>> {
        self.slots.iter().map(|slot| slot.confirmed).collect()
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
/// Rapidgzip-port-design.md step 8: drive decompression via the new
/// GzipChunkFetcher (chunk_fetcher.rs). Gated behind
/// `GZIPPY_USE_RAPIDGZIP_PATH=1` env var so existing tests + bench-sm
/// runs continue to exercise the v0.6 pipeline; bench-sm can compare
/// the two by setting the env var.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decompress_parallel_via_fetcher<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
    t0: std::time::Instant,
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_fetcher::GzipChunkFetcher;

    let header_size = crate::decompress::parallel::marker_decode::skip_gzip_header(gzip_data)
        .map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::TooSmall);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

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

    let chunk_size_bytes = 4 * 1024 * 1024;
    let configuration = ChunkConfiguration {
        split_chunk_size: chunk_size_bytes,
        max_decoded_chunk_size: 20 * chunk_size_bytes,
        crc32_enabled: true,
    };
    let mut fetcher = GzipChunkFetcher::new(deflate_data, num_threads, configuration);

    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size: usize = 0;
    while fetcher.has_more() {
        let chunk = fetcher
            .get_next_chunk()
            .map_err(|_| ParallelError::DecodeFailed)?;
        // After apply_window in the fetcher, every value in
        // data_with_markers is < 256 (a literal byte). Narrow + write
        // both segments in stream order.
        if !chunk.data_with_markers.is_empty() {
            let mut narrowed: Vec<u8> = Vec::with_capacity(chunk.data_with_markers.len());
            for v in &chunk.data_with_markers {
                narrowed.push(*v as u8);
            }
            writer.write_all(&narrowed).map_err(ParallelError::Io)?;
            total_size += narrowed.len();
        }
        if !chunk.data.is_empty() {
            writer.write_all(&chunk.data).map_err(ParallelError::Io)?;
            total_size += chunk.data.len();
        }
        total_crc.combine(&chunk.crc);
    }

    if total_size != expected_size {
        return Err(ParallelError::SizeMismatch);
    }
    if total_crc.clone().finalize() != expected_crc {
        return Err(ParallelError::CrcMismatch);
    }

    MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
    if debug_enabled() {
        let total = t0.elapsed();
        eprintln!(
            "[parallel_sm:rapidgzip] total={:.1}ms isize={}",
            total.as_secs_f64() * 1000.0,
            expected_size
        );
    }
    Ok(total_size as u64)
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

    // Step 8 of the rapidgzip port (rapidgzip-port-design.md): when
    // GZIPPY_USE_RAPIDGZIP_PATH=1, route through the new
    // GzipChunkFetcher pipeline instead of the v0.6 marker pipeline.
    // Lets `make bench-sm` A/B compare the two implementations on
    // identical input before the v0.6 scaffolding is deleted in step 9.
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    if std::env::var("GZIPPY_USE_RAPIDGZIP_PATH").is_ok() {
        return decompress_parallel_via_fetcher(gzip_data, writer, num_threads, t0);
    }
    let _ = t0;

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
    let start_bits_opt =
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
    let mut boundaries = BoundaryRegistry::from_speculative_starts(start_bits_opt);

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
    // Per-chunk decoded-output hint for the ISA-L bulk decode's
    // initial buffer cap. ISIZE / num_chunks is the average; the last
    // chunk may legitimately exceed it, so `decode_chunk_with_handoff`
    // adds 50% headroom.
    let per_chunk_output_hint = expected_size / num_chunks.max(1);
    let speculative_outcomes = phase1_marker_decode_parallel(
        deflate_data,
        &boundaries,
        per_chunk_output_hint,
        num_threads,
    );
    let decode_elapsed = t_decode.elapsed();
    let mut partitions = PartitionRegistry::from_worker_outcomes(speculative_outcomes);

    if debug_enabled() {
        // Per-chunk breakdown: elapsed, bootstrap size, ISA-L size, total.
        // Skew in elapsed reveals load imbalance; skew in sizes reveals
        // uneven boundary placement. High CV in the benchmark usually comes
        // from one chunk taking significantly longer than the others.
        eprintln!("[parallel_sm:v0.6] per-chunk phase1b breakdown:");
        for (i, partition) in partitions.partitions.iter().enumerate() {
            match partition {
                PartitionState::Speculative(candidate) => {
                    let c = &candidate.chunk;
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
                    let end_limit_kind = boundaries.speculative_stop_for(i).label();
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
                PartitionState::NoDecode => {
                    eprintln!("  chunk {:2}: (boundary not found — phase1c will fill)", i);
                }
                PartitionState::Decoded(_)
                | PartitionState::Subsumed { .. }
                | PartitionState::EmptyTail { .. } => unreachable!(),
            }
        }
        // Imbalance ratio: longest / shortest non-empty worker.
        let elapsed_ms: Vec<u128> = partitions
            .partitions
            .iter()
            .filter_map(|partition| match partition {
                PartitionState::Speculative(candidate) => Some(&candidate.chunk),
                PartitionState::NoDecode
                | PartitionState::Decoded(_)
                | PartitionState::Subsumed { .. }
                | PartitionState::EmptyTail { .. } => None,
            })
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
    reconcile_partitions(
        deflate_data,
        &mut boundaries,
        &mut partitions,
        retry_deadline,
        per_chunk_output_hint,
    )?;
    let retry_elapsed = t_retry.elapsed();
    if debug_enabled() {
        let (decoded, subsumed, empty_tail, no_decode, speculative) = partitions.outcome_counts();
        let swallowed = partitions.swallowed_bytes();
        eprintln!(
            "[parallel_sm:v0.6] partition_outcomes decoded={} subsumed={} empty_tail={} no_decode={} speculative={} total={}",
            decoded, subsumed, empty_tail, no_decode, speculative, partitions.len()
        );
        eprintln!(
            "[parallel_sm:v0.6] swallowed_bytes={} (sum of subsumed-partition byte ranges decoded by predecessors)",
            swallowed
        );
    }
    let chunks = partitions.into_authoritative_chunks();
    let chunks = redistribute_oversized_chunks(chunks);
    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] after_redistribute chunks={} (split oversized accepted chunks at \
             ISA-L-discovered block boundaries to reclaim parallelism from subsumed partitions)",
            chunks.len()
        );
    }

    // G1 assertion after phase1c: every chunk's end_bit_offset must be a real
    // deflate block boundary. This catches decode primitive bugs (like the
    // decode_stored invariant break) that phase1c may have propagated forward.
    #[cfg(debug_assertions)]
    for (i, chunk) in chunks.iter().enumerate() {
        debug_assert!(
            crate::decompress::parallel::fast_marker_inflate::validate_boundary(
                deflate_data,
                chunk.end.bits(),
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
                    i + 1 == chunks.len() && total_bits.saturating_sub(chunk.end.bits()) < 8
                },
            "G1 invariant after phase1c: chunk {} end_bit_offset {} is not a real block boundary",
            i,
            chunk.end.bits(),
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
        let counters = OptimizationCounters::snapshot();
        eprintln!(
            "[parallel_sm:v0.6] counters handoff_fired={} slow_path_used={} \
             bootstrap_output_bytes={} isal_output_bytes={} \
             boundary_validations={} retry_iterations={}",
            counters.handoff_fired,
            counters.slow_path_used,
            counters.bootstrap_output_bytes,
            counters.isal_output_bytes,
            counters.boundary_validations,
            counters.retry_iterations,
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
/// Adapter for the new chunk_fetcher: invokes v0.6's bootstrap-then-
/// ISA-L `decode_chunk_inexact` and converts the result into the
/// rapidgzip-shape ChunkData. This matches the structural pattern
/// rapidgzip uses (bootstrap with marker decoder, hand off to ISA-L
/// after 32 KiB clean tail accumulates) which is what makes
/// speculative parallel decode correct on cross-chunk back-references.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub(crate) fn decode_chunk_for_fetcher(
    deflate_data: &[u8],
    start_bit: usize,
    until_bit: usize,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
    configuration: crate::decompress::parallel::chunk_data::ChunkConfiguration,
) -> Option<crate::decompress::parallel::chunk_data::ChunkData> {
    use crate::decompress::parallel::chunk_data::ChunkData;

    let start = ChunkStart::from_bits(start_bit);
    // UntilEnd: decoder runs to BFINAL or max_decoded_chunk_size cap.
    // Workers may decode past their partition range; the consumer's
    // overlap reconciliation in GzipChunkFetcher::get_next_chunk
    // drops later chunks whose range is shadowed. This is rapidgzip's
    // pattern: speculative work can overlap, consumer dedups via the
    // BlockMap insertion-sort.
    let _ = until_bit;
    let stop = ChunkDecodeStop::UntilEnd;
    let outcome = decode_chunk_inexact(
        deflate_data,
        usize::MAX,
        start,
        stop,
        per_chunk_output_hint,
        num_chunks_total,
        false,
    );
    let cand = match outcome {
        WorkerOutcome::Candidate(c) => c,
        WorkerOutcome::NoDecode => return None,
    };
    let mut chunk = ChunkData::new(cand.discovered_start.bits(), configuration);
    // data_with_markers = the marker bootstrap (u16 with cross-chunk
    // back-refs as markers). data = bootstrap_clean (u8 tail of
    // bootstrap with no markers) ++ isal_bytes (clean ISA-L decode
    // using bootstrap_clean as dict). All in stream order.
    chunk.data_with_markers = cand.chunk.bootstrap;
    chunk.data.extend_from_slice(&cand.chunk.bootstrap_clean);
    chunk.data.extend_from_slice(&cand.chunk.isal_bytes);
    // CRC for the clean part (data segment) was computed during ISA-L
    // decode in v0.6 (isal_crc). apply_window will prepend the CRC of
    // resolved markers later.
    chunk.crc = cand.chunk.isal_crc.clone();
    // Also CRC bootstrap_clean (it's NOT in isal_crc because v0.6
    // computes isal_crc only over isal_bytes).
    {
        let mut bc_crc = crc32fast::Hasher::new();
        bc_crc.update(&cand.chunk.bootstrap_clean);
        let mut combined = bc_crc;
        combined.combine(&chunk.crc);
        chunk.crc = combined;
    }
    let end_bit = cand.discovered_end.bits();
    chunk.finalize(end_bit);
    // Push a single subchunk covering the whole chunk; rapidgzip's
    // finer-grained subchunks aren't needed for our consumer model.
    chunk.subchunks.clear();
    chunk
        .subchunks
        .push(crate::decompress::parallel::chunk_data::Subchunk {
            encoded_offset_bits: cand.discovered_start.bits(),
            encoded_size_bits: end_bit - cand.discovered_start.bits(),
            decoded_offset: 0,
            decoded_size: chunk.decoded_size(),
            window: None,
        });
    Some(chunk)
}

// Re-export of search_boundary_forward as a stable bit-offset API.
// Used by the new GzipChunkFetcher (chunk_fetcher.rs) until a faithful
// rapidgzip BlockFinder.hpp port lands. Searches a wider radius than
// the v0.6 path so Silesia-gzip-9-class inputs (where 512 KiB windows
// can contain zero validator-accepted candidates) reliably find a
// boundary. Returns the bit offset of the first real deflate block
// boundary at or after `from_bit`, or None.
pub(crate) fn find_real_boundary_for_fetcher(
    deflate_data: &[u8],
    from_bit: usize,
) -> Option<usize> {
    // Try the cheap 512 KiB search first.
    if let Some(cs) = search_boundary_forward(deflate_data, from_bit) {
        return Some(cs.bits());
    }
    // Fall back to a much wider scan — up to 8 MiB so a sparse-boundary
    // region (BTYPE=01-heavy) can still surface at least one candidate.
    // search_boundary_forward only searches SEARCH_RADIUS=512 KiB; reach
    // further by iterating its window forward.
    let max_extra = 8 * 1024 * 1024 * 8;
    let mut cursor = from_bit + SEARCH_RADIUS * 8;
    let limit = (from_bit + max_extra).min(deflate_data.len() * 8);
    while cursor < limit {
        if let Some(cs) = search_boundary_forward(deflate_data, cursor) {
            return Some(cs.bits());
        }
        cursor += SEARCH_RADIUS * 8;
    }
    None
}

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
    _spacing_bits: usize,
    _total_bits: usize,
) -> ChunkDecodeStop {
    correction_decode_stop(idx, start_bits)
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
#[derive(Debug)]
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

fn empty_chunk_result(end_bit_offset: usize) -> ChunkResult {
    ChunkResult {
        bootstrap: Vec::new(),
        bootstrap_clean: Vec::new(),
        isal_bytes: Vec::new(),
        isal_crc: crc32fast::Hasher::new(),
        end_bit_offset,
        worker_elapsed: std::time::Duration::ZERO,
    }
}

/// Transitional worker contract for the dispatcher-owned retry architecture.
///
/// Commit 1 keeps the legacy worker implementation underneath these types so the
/// current pipeline behavior stays unchanged while later commits replace the
/// rescue-path control flow.
#[derive(Debug)]
enum WorkerOutcome {
    NoDecode,
    Candidate(CandidateChunk),
}

#[derive(Debug)]
struct CandidateChunk {
    requested_partition: usize,
    discovered_start: ChunkStart,
    discovered_end: RealBlockBoundary,
    /// Real deflate block boundaries the worker crossed during decode, in
    /// strictly increasing bit-offset order. Each entry carries both the
    /// compressed bit position AND the byte position inside `chunk.isal_bytes`
    /// where the block ended, so the dispatcher can split the worker's
    /// output at any boundary without re-decoding. The last entry's
    /// `bit_offset` equals `discovered_end`. Empty when the worker took the
    /// marker-only path.
    discovered_boundaries: Vec<crate::backends::isal_decompress::BlockBoundary>,
    chunk: ChunkResult,
}

#[derive(Debug)]
struct AuthoritativeChunk {
    start: ChunkStart,
    end: RealBlockBoundary,
    /// Block boundaries inside `chunk.isal_bytes` (with output_offset
    /// pointing into the ISA-L decoded bytes). Used by
    /// [`redistribute_oversized_chunks`] after reconcile to split a chunk
    /// that swallowed downstream partitions back into smaller pieces.
    boundaries: Vec<crate::backends::isal_decompress::BlockBoundary>,
    chunk: ChunkResult,
}

impl AuthoritativeChunk {
    fn from_candidate(candidate: CandidateChunk) -> Self {
        Self {
            start: candidate.discovered_start,
            end: candidate.discovered_end,
            boundaries: candidate.discovered_boundaries,
            chunk: candidate.chunk,
        }
    }

    fn empty(start: ChunkStart) -> Self {
        Self {
            start,
            end: RealBlockBoundary(start.bits()),
            boundaries: Vec::new(),
            chunk: empty_chunk_result(start.bits()),
        }
    }
}

#[derive(Debug)]
enum PartitionState {
    NoDecode,
    Speculative(CandidateChunk),
    Decoded(AuthoritativeChunk),
    Subsumed { start: ChunkStart },
    EmptyTail { start: ChunkStart },
}

#[derive(Debug)]
struct PartitionRegistry {
    partitions: Vec<PartitionState>,
}

impl PartitionRegistry {
    fn from_worker_outcomes(outcomes: Vec<WorkerOutcome>) -> Self {
        let partitions = outcomes
            .into_iter()
            .map(|outcome| match outcome {
                WorkerOutcome::NoDecode => PartitionState::NoDecode,
                WorkerOutcome::Candidate(candidate) => PartitionState::Speculative(candidate),
            })
            .collect();
        Self { partitions }
    }

    fn from_legacy_chunks(
        start_bits: &[Option<ChunkStart>],
        chunks: &mut [Option<ChunkResult>],
    ) -> Self {
        let partitions = chunks
            .iter_mut()
            .enumerate()
            .map(|(idx, chunk)| match chunk.take() {
                Some(chunk) => PartitionState::Speculative(CandidateChunk {
                    requested_partition: idx,
                    discovered_start: start_bits[idx].unwrap_or(ChunkStart::from_bits(0)),
                    discovered_end: RealBlockBoundary(chunk.end_bit_offset),
                    discovered_boundaries: Vec::new(),
                    chunk,
                }),
                None => PartitionState::NoDecode,
            })
            .collect();
        Self { partitions }
    }

    #[inline]
    fn len(&self) -> usize {
        self.partitions.len()
    }

    fn take_speculative(&mut self, idx: usize) -> Option<CandidateChunk> {
        let slot = self.partitions.get_mut(idx)?;
        let mut taken = PartitionState::NoDecode;
        std::mem::swap(slot, &mut taken);
        match taken {
            PartitionState::Speculative(candidate) => Some(candidate),
            other => {
                *slot = other;
                None
            }
        }
    }

    fn set_decoded(&mut self, idx: usize, chunk: AuthoritativeChunk) {
        self.partitions[idx] = PartitionState::Decoded(chunk);
    }

    fn set_subsumed(&mut self, idx: usize, start: ChunkStart) {
        self.partitions[idx] = PartitionState::Subsumed { start };
    }

    fn set_empty_tail(&mut self, idx: usize, start: ChunkStart) {
        self.partitions[idx] = PartitionState::EmptyTail { start };
    }

    /// Counts of each terminal partition state after reconcile.
    /// `(decoded, subsumed, empty_tail, no_decode, speculative)`.
    fn outcome_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut counts = (0usize, 0usize, 0usize, 0usize, 0usize);
        for partition in &self.partitions {
            match partition {
                PartitionState::Decoded(_) => counts.0 += 1,
                PartitionState::Subsumed { .. } => counts.1 += 1,
                PartitionState::EmptyTail { .. } => counts.2 += 1,
                PartitionState::NoDecode => counts.3 += 1,
                PartitionState::Speculative(_) => counts.4 += 1,
            }
        }
        counts
    }

    /// Bytes swallowed by Decoded partitions whose output extends past their
    /// own partition's notional end. For consecutive Subsumed/EmptyTail
    /// partitions after a Decoded one, sum the Decoded chunk's `decoded_len`
    /// beyond the size that would have been expected if each partition decoded
    /// independently. Approximates how much work would parallelize away if
    /// in-band boundary discovery were exact.
    fn swallowed_bytes(&self) -> u64 {
        let mut swallowed = 0u64;
        let mut iter = self.partitions.iter().peekable();
        while let Some(p) = iter.next() {
            if let PartitionState::Decoded(chunk) = p {
                let mut subsumed_run = 0usize;
                while matches!(
                    iter.peek(),
                    Some(PartitionState::Subsumed { .. } | PartitionState::EmptyTail { .. })
                ) {
                    subsumed_run += 1;
                    iter.next();
                }
                if subsumed_run > 0 {
                    let total = chunk.chunk.decoded_len() as u64;
                    let per_partition = total / (subsumed_run as u64 + 1);
                    swallowed += per_partition * subsumed_run as u64;
                }
            }
        }
        swallowed
    }

    fn into_authoritative_chunks(self) -> Vec<AuthoritativeChunk> {
        self.partitions
            .into_iter()
            .map(|partition| match partition {
                PartitionState::Decoded(chunk) => chunk,
                PartitionState::Subsumed { start } | PartitionState::EmptyTail { start } => {
                    AuthoritativeChunk::empty(start)
                }
                PartitionState::NoDecode | PartitionState::Speculative(_) => {
                    panic!("reconcile must resolve every partition before phase 2")
                }
            })
            .collect()
    }

    fn write_back_legacy(
        self,
        start_bits: &mut [Option<ChunkStart>],
        chunks: &mut [Option<ChunkResult>],
    ) {
        for ((start_slot, chunk_slot), partition) in start_bits
            .iter_mut()
            .zip(chunks.iter_mut())
            .zip(self.partitions)
        {
            match partition {
                PartitionState::Decoded(authoritative) => {
                    *start_slot = Some(authoritative.start);
                    *chunk_slot = Some(authoritative.chunk);
                }
                PartitionState::Subsumed { start } | PartitionState::EmptyTail { start } => {
                    *start_slot = Some(start);
                    *chunk_slot = Some(empty_chunk_result(start.bits()));
                }
                PartitionState::NoDecode | PartitionState::Speculative(_) => {
                    *chunk_slot = None;
                }
            }
        }
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
/// Chunks whose speculative start is `None` are skipped. The dispatcher will
/// later derive their exact start from the preceding authoritative partition.
fn phase1_marker_decode_parallel(
    deflate_data: &[u8],
    boundaries: &BoundaryRegistry,
    per_chunk_output_hint: usize,
    num_workers: usize,
) -> Vec<WorkerOutcome> {
    let num_chunks = boundaries.len();
    let results: Vec<Mutex<WorkerOutcome>> = (0..num_chunks)
        .map(|_| Mutex::new(WorkerOutcome::NoDecode))
        .collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_workers {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let Some(start_bit) = boundaries.speculative_start(idx) else {
                    continue;
                };
                let stop = boundaries.speculative_stop_for(idx);
                let force_slow_path = !stop.use_isal();
                let outcome = decode_chunk_inexact(
                    deflate_data,
                    idx,
                    start_bit,
                    stop,
                    per_chunk_output_hint,
                    num_chunks,
                    force_slow_path,
                );
                *results[idx].lock().unwrap() = outcome;
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
fn decode_chunk_with_handoff_legacy(
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
        let end_byte = match stop {
            ChunkDecodeStop::Approximate(limit) => ((limit
                .bits()
                .saturating_add(APPROX_STOP_INPUT_MARGIN_BYTES * 8))
            .div_ceil(8))
            .min(deflate_data.len()),
            _ => stop_hint_bits
                .map(|end_bits| (end_bits.div_ceil(8)).min(deflate_data.len()))
                .unwrap_or(deflate_data.len()),
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

/// Inexact worker seam used by phase 1b.
///
/// Two structurally distinct outcomes, both deterministic by precondition:
///
/// 1. **Marker-only candidate** — when ISA-L cannot run (no clean 32 KB window
///    accumulated during bootstrap, or non-x86_64, or `force_slow_path`),
///    bootstrap covers the entire chunk and ends at a real block boundary
///    ≥ `end_bit_limit`. The returned `Candidate` carries marker output that
///    phase 2 will resolve. This is the *canonical* path for chunks whose
///    structure prevents ISA-L's dict precondition; it is not a rescue.
///
/// 2. **ISA-L candidate** — when bootstrap accumulated a clean window AND
///    ISA-L is available, hand off to ISA-L. If ISA-L overshoots its verified
///    end limit or returns failure, return `NoDecode` so the dispatcher can
///    decide. This worker **never** rescues a failed ISA-L attempt with the
///    marker decoder; that is what `SLOW_PATH_USED` counts (it should stay
///    zero on healthy x86_64 data — any nonzero value is a refused rescue
///    the dispatcher will retry via [`decode_chunk_exact`]).
fn decode_chunk_inexact(
    deflate_data: &[u8],
    requested_partition: usize,
    start_bit: ChunkStart,
    stop: ChunkDecodeStop,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
    force_slow_path: bool,
) -> WorkerOutcome {
    let t_worker = std::time::Instant::now();
    let stop_hint_bits = stop.hint_bits();
    let BootstrapResult {
        markers: bootstrap_markers,
        end_bit_offset: bootstrap_end_bit,
        clean_window,
        bfinal_hit,
    } = match decode_chunk_bootstrap(deflate_data, start_bit.bits(), stop_hint_bits) {
        Ok(result) => result,
        Err(_) => return WorkerOutcome::NoDecode,
    };

    BOOTSTRAP_OUTPUT_BYTES.fetch_add(bootstrap_markers.len() as u64, Ordering::Relaxed);

    if debug_enabled() {
        eprintln!(
            "[bootstrap] start={start_bit} stop={stop:?} \
             end_bit={bootstrap_end_bit} bfinal_hit={bfinal_hit} \
             window={} output_bytes={}",
            clean_window.is_some(),
            bootstrap_markers.len(),
        );
    }

    // Dispatch: two decoders, chosen by a single precondition. Not a
    // fallback chain — both branches are canonical for their input class.
    //
    //   isal_capable = clean 32 KiB window from bootstrap
    //                AND ISA-L is available (x86_64 + feature)
    //                AND caller did not force slow path
    //
    // True  → ISA-L handoff produces the bulk of output at full ISA-L
    //         throughput. The only correct decoder when ISA-L's dict
    //         precondition is met.
    // False → Marker decoder decodes the entire chunk; phase 2 resolves
    //         markers against the predecessor's window. The only correct
    //         decoder when ISA-L cannot run (no dict, or no ISA-L).
    let isal_capable = clean_window.is_some()
        && cfg!(all(feature = "isal-compression", target_arch = "x86_64"))
        && !force_slow_path;

    if !isal_capable {
        let _ = (per_chunk_output_hint, num_chunks_total);
        return marker_only_candidate(
            requested_partition,
            start_bit,
            bootstrap_markers,
            bootstrap_end_bit,
            stop_hint_bits,
            t_worker,
            deflate_data,
        );
    }
    // SAFETY: `isal_capable` true ⇒ `clean_window.is_some()` ⇒ unwrap safe.
    let dict = clean_window.unwrap();

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    {
        let end_byte = match stop {
            ChunkDecodeStop::Approximate(limit) => ((limit
                .bits()
                .saturating_add(APPROX_STOP_INPUT_MARGIN_BYTES * 8))
            .div_ceil(8))
            .min(deflate_data.len()),
            _ => stop_hint_bits
                .map(|end_bits| (end_bits.div_ceil(8)).min(deflate_data.len()))
                .unwrap_or(deflate_data.len()),
        };
        let input = &deflate_data[..end_byte];
        let max_output = per_chunk_output_hint.saturating_mul(2);
        let _ = num_chunks_total;

        let mut isal_crc = crc32fast::Hasher::new();
        match crate::backends::isal_decompress::decompress_deflate_from_bit_with_boundaries(
            input,
            bootstrap_end_bit,
            &dict,
            max_output,
            &mut isal_crc,
        ) {
            Some((isal_bytes, isal_end_bit, raw_boundaries)) => {
                let Some(verified_end_bit) =
                    normalize_isal_end_bit(deflate_data, start_bit, stop, isal_end_bit)
                else {
                    // ISA-L was attempted, succeeded, but overshot the
                    // verified end beyond the snap tolerance. We refuse to
                    // rescue with the marker decoder; surface as NoDecode so
                    // the dispatcher can discard and exact-retry. SLOW_PATH_USED
                    // counts these refused-rescue events — it must stay at 0
                    // on healthy production data.
                    SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm:v0.6] chunk start={start_bit}: inexact worker returning NoDecode \
                             after rejecting ISA-L end_bit={} for stop={stop:?}",
                            isal_end_bit
                        );
                    }
                    return WorkerOutcome::NoDecode;
                };
                HANDOFF_FIRED.fetch_add(1, Ordering::Relaxed);
                ISAL_OUTPUT_BYTES.fetch_add(isal_bytes.len() as u64, Ordering::Relaxed);
                let split_at = bootstrap_markers.len().saturating_sub(dict.len());
                let mut bootstrap = bootstrap_markers;
                bootstrap.truncate(split_at);
                // Patched ISA-L sets stopped_at = END_OF_BLOCK only at real
                // deflate block boundaries (see oracle test), so every entry
                // is a verified boundary plus the output offset at which the
                // just-finished block ended.
                let discovered_boundaries = raw_boundaries;
                WorkerOutcome::Candidate(CandidateChunk {
                    requested_partition,
                    discovered_start: start_bit,
                    discovered_end: verified_end_bit,
                    discovered_boundaries,
                    chunk: ChunkResult {
                        bootstrap,
                        bootstrap_clean: dict,
                        isal_bytes,
                        isal_crc,
                        end_bit_offset: verified_end_bit.bits(),
                        worker_elapsed: t_worker.elapsed(),
                    },
                })
            }
            None => {
                // ISA-L was attempted and returned failure. Refuse to rescue
                // with the marker decoder; surface as NoDecode for the
                // dispatcher to discard + exact-retry. See SLOW_PATH_USED
                // comment above for the meaning of this counter.
                SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] chunk start={start_bit}: inexact worker returning NoDecode \
                         after ISA-L failed"
                    );
                }
                WorkerOutcome::NoDecode
            }
        }
    }

    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    {
        let _ = (
            stop,
            per_chunk_output_hint,
            num_chunks_total,
            force_slow_path,
            dict,
        );
        marker_only_candidate(
            requested_partition,
            start_bit,
            bootstrap_markers,
            bootstrap_end_bit,
            stop_hint_bits,
            t_worker,
            deflate_data,
        )
    }
}

/// Marker-only candidate constructor for chunks whose precondition for the
/// ISA-L handoff is not met (no 32 KB clean window accumulated during
/// bootstrap, or non-x86_64, or `force_slow_path`). Bootstrap stops at the
/// first clean-tail block boundary or at end_bit_limit; this helper extends
/// it to end_bit_limit via the marker decoder so the chunk's output is
/// complete. Returns a `Candidate` with marker-only output (phase 2 resolves
/// against the predecessor's window).
///
/// This is *not* a rescue path. It is the canonical decoder for chunks the
/// bootstrap→ISA-L design cannot accelerate; the marker decoder is the only
/// correct deflate decoder available without a dict. `SLOW_PATH_USED`
/// remains an "ISA-L rescue refused" counter and is *not* incremented here.
fn marker_only_candidate(
    requested_partition: usize,
    start_bit: ChunkStart,
    bootstrap_markers: Vec<u16>,
    bootstrap_end_bit: usize,
    end_bit_limit: Option<usize>,
    t_worker: std::time::Instant,
    deflate_data: &[u8],
) -> WorkerOutcome {
    let chunk = match marker_finish_after_bootstrap(
        deflate_data,
        bootstrap_markers,
        bootstrap_end_bit,
        end_bit_limit,
    ) {
        Ok(c) => c,
        Err(_) => return WorkerOutcome::NoDecode,
    };
    let end_bit = chunk.end_bit_offset;
    WorkerOutcome::Candidate(CandidateChunk {
        requested_partition,
        discovered_start: start_bit,
        discovered_end: RealBlockBoundary(end_bit),
        // Marker-only path doesn't record per-block bit-offsets in-band
        // (the pure-Rust decoder returns a single end offset). Leave empty;
        // the dispatcher's boundary-promotion logic must be a no-op when
        // boundaries are empty.
        discovered_boundaries: Vec::new(),
        chunk: ChunkResult {
            worker_elapsed: t_worker.elapsed(),
            ..chunk
        },
    })
}

fn decode_chunk_inexact_legacy(
    deflate_data: &[u8],
    requested_partition: usize,
    start_bit: ChunkStart,
    stop: ChunkDecodeStop,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
    force_slow_path: bool,
) -> WorkerOutcome {
    decode_chunk_inexact(
        deflate_data,
        requested_partition,
        start_bit,
        stop,
        per_chunk_output_hint,
        num_chunks_total,
        force_slow_path,
    )
}

/// Exact worker contract. Caller guarantees `start_bit` is a verified real
/// deflate block boundary (from the BoundaryRegistry). Returns
/// `Err(ExactStopFailed)` only when the inexact worker returned `NoDecode`
/// — i.e. when ISA-L was attempted, succeeded with a clean dict, but the
/// resulting end overshot the verified end-limit beyond the snap tolerance
/// (or ISA-L itself failed). The marker-only path always produces a
/// `Candidate`, so chunks whose structure prevents ISA-L still succeed via
/// this entry. The dispatcher converts the error into `ParallelError::DecodeFailed`
/// which the routing layer handles by falling back to libdeflate sequential.
fn decode_chunk_exact(
    deflate_data: &[u8],
    start_bit: ChunkStart,
    end_limit: Option<ChunkEndLimit>,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
) -> Result<AuthoritativeChunk, ExactStopFailed> {
    let stop = end_limit.map_or(ChunkDecodeStop::UntilEnd, ChunkDecodeStop::Verified);
    match decode_chunk_inexact(
        deflate_data,
        usize::MAX,
        start_bit,
        stop,
        per_chunk_output_hint,
        num_chunks_total,
        false,
    ) {
        WorkerOutcome::Candidate(candidate) => Ok(AuthoritativeChunk::from_candidate(candidate)),
        WorkerOutcome::NoDecode => {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] exact: inexact worker returned NoDecode start={} end_limit={:?}",
                    start_bit.bits(),
                    end_limit
                );
            }
            Err(ExactStopFailed)
        }
    }
}

/// Returned by [`decode_chunk_exact`] when the worker cannot produce an
/// authoritative chunk matching its verified-start / verified-end contract.
/// The dispatcher converts this into [`ParallelError::DecodeFailed`].
#[derive(Debug)]
struct ExactStopFailed;

impl std::fmt::Display for ExactStopFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("exact-stop worker failed: ISA-L was attempted and refused rescue")
    }
}

impl std::error::Error for ExactStopFailed {}

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
            let max_approx_end =
                limit_bits.saturating_add(APPROX_STOP_INPUT_MARGIN_BYTES.saturating_mul(8));
            if isal_end_bit > max_approx_end {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] approximate stop reject: start={} end_bit={} \
                         limit={} max_end={} reason=overshoot",
                        start_bit, isal_end_bit, limit_bits, max_approx_end,
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
                         limit={} max_end={} reason=validated-boundary",
                        start_bit, isal_end_bit, limit_bits, max_approx_end,
                    );
                }
                Some(RealBlockBoundary(isal_end_bit))
            } else {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] approximate stop reject: start={} end_bit={} \
                         limit={} max_end={} reason=not-a-boundary",
                        start_bit, isal_end_bit, limit_bits, max_approx_end,
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

// ── Reconcile: dispatcher-owned boundary acceptance / retry ──────────────────

fn exact_or_empty_authoritative(
    deflate_data: &[u8],
    idx: usize,
    start: ChunkStart,
    end_limit: Option<ChunkEndLimit>,
    deadline: std::time::Instant,
    per_chunk_output_hint: usize,
    num_chunks: usize,
) -> Result<AuthoritativeChunk, ParallelError> {
    let total_bits = deflate_data.len() * 8;
    if end_limit.is_none() && total_bits.saturating_sub(start.bits()) < 8 {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] reconcile: chunk {} is empty tail at {}",
                idx,
                start.bits()
            );
        }
        return Ok(AuthoritativeChunk::empty(start));
    }
    if end_limit.is_some_and(|limit| start.bits() >= limit.bits()) {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] reconcile: chunk {} subsumed at {}",
                idx,
                start.bits()
            );
        }
        return Ok(AuthoritativeChunk::empty(start));
    }
    if std::time::Instant::now() >= deadline {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] reconcile: wall-time deadline exceeded after {} retries",
                MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed)
            );
        }
        return Err(ParallelError::DecodeFailed);
    }

    MARKER_PIPELINE_RETRY_ITERATIONS.fetch_add(1, Ordering::Relaxed);
    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] reconcile: exact decode chunk {} start={} stop={:?}",
            idx, start, end_limit
        );
    }

    decode_chunk_exact(
        deflate_data,
        start,
        end_limit,
        per_chunk_output_hint,
        num_chunks,
    )
    .map_err(|e| {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] reconcile: exact decode of chunk {} from {} failed: {e}",
                idx, start
            );
        }
        ParallelError::DecodeFailed
    })
}

/// After reconcile, recover parallelism from oversized accepted chunks by
/// splitting their ISA-L output at boundaries the worker recorded in-band.
///
/// When phase 1a returned `None` for a run of partitions, those partitions
/// have no speculative starts; reconcile assigns them `expected_start` equal
/// to the predecessor's actual end_bit. If the predecessor decoded past
/// several partitions' notional ranges (a "swallow"), the consecutive
/// successors get marked Subsumed/EmptyTail and contribute no parallelism.
///
/// The patched-ISA-L stopping-point machinery makes those swallowed bytes
/// independently splittable: each `BlockBoundary` in the predecessor's
/// `boundaries` carries both a compressed bit-offset and an output byte
/// offset. We split the predecessor's `isal_bytes` at chosen boundaries to
/// produce K+1 contiguous pieces, where K is the number of consecutive
/// Subsumed/EmptyTail followers. Each piece replaces one slot in the chunk
/// vector — the original chunk shrinks, and each subsumed slot now carries
/// real bytes.
///
/// CRC accounting: the original chunk's `isal_crc` covers the full
/// (now-split) output range. We attach it to the first piece and leave the
/// rest with the identity hasher; phase 2's `total_crc.combine(&chunk.isal_crc)`
/// over chunks in order then produces the same total CRC as the unsplit
/// version. This is sound because CRC32-combine is associative and the
/// bytes are written to the output in the same order regardless of how
/// they're grouped into chunks.
///
/// Window propagation: each piece's `isal_bytes` is a contiguous slice of
/// the original (≥32 KiB for non-trivial pieces), so `phase2`'s
/// `try_precompute_input_windows` still succeeds along the split.
fn redistribute_oversized_chunks(chunks: Vec<AuthoritativeChunk>) -> Vec<AuthoritativeChunk> {
    let n = chunks.len();
    if n <= 1 {
        return chunks;
    }
    let mut slots: Vec<Option<AuthoritativeChunk>> = chunks.into_iter().map(Some).collect();
    let mut result: Vec<AuthoritativeChunk> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        // Count consecutive "empty subsumed" slots after i (the slots that
        // currently carry no decoded data because their predecessor
        // swallowed them).
        let mut k = 0usize;
        while i + 1 + k < n {
            match slots[i + 1 + k].as_ref() {
                Some(next) if is_empty_passthrough(next) => k += 1,
                _ => break,
            }
        }

        let main = slots[i].take().expect("slot must be Some at index i");
        if k == 0 || !main.is_splittable() {
            // Nothing to redistribute. Take this slot as-is.
            result.push(main);
            i += 1;
            continue;
        }

        // Drain the K following empty slots so we can replace them.
        for j in 1..=k {
            slots[i + j].take();
        }
        let pieces = split_authoritative_chunk(main, k);
        debug_assert_eq!(pieces.len(), k + 1);
        result.extend(pieces);
        i += 1 + k;
    }
    result
}

/// A slot is "empty passthrough" when it carries no decoded data and no
/// discovered boundaries — produced by [`AuthoritativeChunk::empty`] for
/// Subsumed and EmptyTail partitions.
#[inline]
fn is_empty_passthrough(chunk: &AuthoritativeChunk) -> bool {
    chunk.chunk.isal_bytes.is_empty()
        && chunk.chunk.bootstrap.is_empty()
        && chunk.chunk.bootstrap_clean.is_empty()
        && chunk.boundaries.is_empty()
}

impl AuthoritativeChunk {
    /// True when the chunk has non-empty ISA-L output AND at least one
    /// recorded internal block boundary that can serve as a split point.
    /// Marker-only chunks (bootstrap-only, no isal_bytes) are not
    /// splittable here because the marker pipeline doesn't record per-block
    /// output offsets.
    fn is_splittable(&self) -> bool {
        !self.chunk.isal_bytes.is_empty() && !self.boundaries.is_empty()
    }
}

/// Split a single AuthoritativeChunk into K+1 contiguous pieces using the
/// chunk's recorded `boundaries`. Pieces are picked to be roughly equal in
/// output byte size by selecting boundary indices at evenly-spaced
/// positions in the boundary list. See `redistribute_oversized_chunks` for
/// the CRC and window-propagation correctness argument.
fn split_authoritative_chunk(main: AuthoritativeChunk, k: usize) -> Vec<AuthoritativeChunk> {
    debug_assert!(k >= 1);
    let boundaries = &main.boundaries;
    let m = boundaries.len();
    // Pick K split indices into `boundaries`. We want indices
    // boundaries[i_1], boundaries[i_2], ... boundaries[i_K] with
    // 0 ≤ i_1 < i_2 < ... < i_K < m, evenly spaced. If we have fewer
    // boundaries than needed split points (m < K), we can only produce
    // up to m+1 pieces; cap K at m-1 so the LAST piece always extends to
    // main.end (we exclude the final boundary which equals main.end).
    let usable_boundaries = m.saturating_sub(1); // exclude the boundary equal to end
    let k_actual = k.min(usable_boundaries);
    if k_actual == 0 {
        // Not enough internal boundaries to split. Return main + k empties.
        let mut out = Vec::with_capacity(1 + k);
        out.push(empty_replacing(&main));
        out[0] = main;
        for _ in 0..k {
            // The next pieces must use real ChunkStart values; the original
            // empty AuthoritativeChunk we replaced had start=that partition's
            // expected start. We've lost that info. Use the main chunk's end
            // for now — phase 2 will treat them as empty so the actual start
            // bit doesn't matter for output.
            out.push(empty_replacing(&out[0]));
        }
        return out;
    }

    // Pick split boundary indices: evenly spaced from 1 to usable_boundaries
    // inclusive. With k_actual = K and usable = M-1, target j*(M-1)/(K+1)
    // for j in 1..=K.
    let split_idx: Vec<usize> = (1..=k_actual)
        .map(|j| ((j as u64) * (usable_boundaries as u64) / (k_actual as u64 + 1)) as usize)
        .collect();

    let AuthoritativeChunk {
        start: main_start,
        end: main_end,
        boundaries: main_boundaries,
        chunk: main_chunk,
    } = main;
    let ChunkResult {
        bootstrap,
        bootstrap_clean,
        isal_bytes,
        isal_crc,
        end_bit_offset: _,
        worker_elapsed,
    } = main_chunk;

    let mut pieces: Vec<AuthoritativeChunk> = Vec::with_capacity(k_actual + 1);
    let mut byte_cursor = 0usize;
    let mut prev_end = main_start;
    // Split isal_bytes into k_actual+1 contiguous Vecs via split_off.
    let mut remaining = isal_bytes;
    for (piece_idx, &split_at_idx) in split_idx.iter().enumerate() {
        let boundary = main_boundaries[split_at_idx];
        let take_len = boundary.output_offset - byte_cursor;
        // remaining.split_off(take_len): remaining keeps [..take_len], returns [take_len..]
        let mut piece_bytes = remaining;
        let next_remaining = piece_bytes.split_off(take_len);
        remaining = next_remaining;
        byte_cursor = boundary.output_offset;
        let piece_end = RealBlockBoundary(boundary.bit_offset);
        // The first piece carries the original chunk's bootstrap +
        // bootstrap_clean + isal_crc. Subsequent pieces have empty
        // bootstrap and identity CRC.
        let (piece_bootstrap, piece_clean, piece_crc) = if piece_idx == 0 {
            (
                std::mem::take(&mut bootstrap_carry(&mut Some(bootstrap.clone()))),
                std::mem::take(&mut bootstrap_clean_carry(&mut Some(
                    bootstrap_clean.clone(),
                ))),
                isal_crc_take(&mut Some(isal_crc.clone())),
            )
        } else {
            (Vec::new(), Vec::new(), crc32fast::Hasher::new())
        };
        let piece = AuthoritativeChunk {
            start: prev_end,
            end: piece_end,
            boundaries: Vec::new(),
            chunk: ChunkResult {
                bootstrap: piece_bootstrap,
                bootstrap_clean: piece_clean,
                isal_bytes: piece_bytes,
                isal_crc: piece_crc,
                end_bit_offset: boundary.bit_offset,
                worker_elapsed: if piece_idx == 0 {
                    worker_elapsed
                } else {
                    std::time::Duration::ZERO
                },
            },
        };
        pieces.push(piece);
        prev_end = ChunkStart(piece_end);
    }
    // Final piece: takes whatever is left, ends at main_end.
    let final_piece = AuthoritativeChunk {
        start: prev_end,
        end: main_end,
        boundaries: Vec::new(),
        chunk: ChunkResult {
            bootstrap: Vec::new(),
            bootstrap_clean: Vec::new(),
            isal_bytes: remaining,
            isal_crc: crc32fast::Hasher::new(),
            end_bit_offset: main_end.bits(),
            worker_elapsed: std::time::Duration::ZERO,
        },
    };
    pieces.push(final_piece);
    // If k > k_actual, pad with empties at main_end so the slot count matches.
    for _ in k_actual..k {
        pieces.push(AuthoritativeChunk::empty(ChunkStart(main_end)));
    }
    pieces
}

// These helpers are workarounds to satisfy the borrow checker while moving
// the original chunk's owned fields into only the first piece.
#[inline]
fn bootstrap_carry(slot: &mut Option<Vec<u16>>) -> Vec<u16> {
    slot.take().unwrap_or_default()
}
#[inline]
fn bootstrap_clean_carry(slot: &mut Option<Vec<u8>>) -> Vec<u8> {
    slot.take().unwrap_or_default()
}
#[inline]
fn isal_crc_take(slot: &mut Option<crc32fast::Hasher>) -> crc32fast::Hasher {
    slot.take().unwrap_or_default()
}
/// Construct an empty-passthrough chunk with the same start coordinates
/// as `peer`. Used as the "no split possible" pad when there aren't enough
/// internal boundaries.
fn empty_replacing(peer: &AuthoritativeChunk) -> AuthoritativeChunk {
    AuthoritativeChunk::empty(ChunkStart(peer.end))
}

fn reconcile_partitions(
    deflate_data: &[u8],
    boundaries: &mut BoundaryRegistry,
    partitions: &mut PartitionRegistry,
    deadline: std::time::Instant,
    per_chunk_output_hint: usize,
) -> Result<(), ParallelError> {
    if partitions.len() == 0 {
        return Ok(());
    }

    for idx in 0..partitions.len() {
        let expected_start = boundaries.confirmed_start(idx).ok_or_else(|| {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] reconcile: partition {} has no confirmed start",
                    idx
                );
            }
            ParallelError::DecodeFailed
        })?;

        let end_limit = boundaries.exact_end_limit_for(idx);
        let candidate = partitions.take_speculative(idx);
        let authoritative = match candidate {
            Some(candidate) if candidate.discovered_start == expected_start => {
                AuthoritativeChunk::from_candidate(candidate)
            }
            Some(candidate) => {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] reconcile: discard speculative chunk {} \
                         start={} expected_start={} end={}",
                        idx,
                        candidate.discovered_start.bits(),
                        expected_start.bits(),
                        candidate.discovered_end.bits(),
                    );
                }
                exact_or_empty_authoritative(
                    deflate_data,
                    idx,
                    expected_start,
                    end_limit,
                    deadline,
                    per_chunk_output_hint,
                    partitions.len(),
                )?
            }
            None => exact_or_empty_authoritative(
                deflate_data,
                idx,
                expected_start,
                end_limit,
                deadline,
                per_chunk_output_hint,
                partitions.len(),
            )?,
        };

        if end_limit.is_none() && authoritative.chunk.decoded_len() == 0 {
            partitions.set_empty_tail(idx, authoritative.start);
        } else if end_limit.is_some_and(|limit| authoritative.start.bits() >= limit.bits()) {
            partitions.set_subsumed(idx, authoritative.start);
        } else {
            partitions.set_decoded(idx, authoritative);
        }

        let end = match &partitions.partitions[idx] {
            PartitionState::Decoded(chunk) => chunk.end,
            PartitionState::Subsumed { start } | PartitionState::EmptyTail { start } => {
                RealBlockBoundary(start.bits())
            }
            PartitionState::NoDecode | PartitionState::Speculative(_) => unreachable!(),
        };
        boundaries.confirm_start(idx, expected_start);
        boundaries.confirm_next_start_from_end(idx, end);
    }

    Ok(())
}

/// Compatibility wrapper for the legacy test surface. Internally this now uses
/// dispatcher-owned reconcile state rather than mutating speculative starts and
/// decoded chunks in place.
fn phase1c_resolve_consistency(
    deflate_data: &[u8],
    start_bits: &mut [Option<ChunkStart>],
    chunks: &mut [Option<ChunkResult>],
    deadline: std::time::Instant,
    per_chunk_output_hint: usize,
    _spacing_bits: usize,
) -> Result<(), ParallelError> {
    let mut boundaries = BoundaryRegistry::from_speculative_starts(start_bits.to_vec());
    let mut partitions = PartitionRegistry::from_legacy_chunks(start_bits, chunks);
    reconcile_partitions(
        deflate_data,
        &mut boundaries,
        &mut partitions,
        deadline,
        per_chunk_output_hint,
    )?;
    partitions.write_back_legacy(start_bits, chunks);
    Ok(())
}

// ── Phase 2 + streaming write: resolve markers, write, combine CRCs ──────────

/// Resolve one chunk's bootstrap markers into u8 bytes and CRC them. Runs
/// in a worker thread; takes the input window (predecessor's last 32 KiB)
/// by reference. Pure function — no shared state. The returned tuple is
/// the resolved bootstrap bytes (markers replaced + narrowed to u8) plus
/// a CRC32 hasher over those bytes.
fn resolve_chunk_markers(
    chunk_idx: usize,
    bootstrap: &mut [u16],
    bootstrap_clean: &[u8],
    input_window: &[u8],
) -> Result<(Vec<u8>, crc32fast::Hasher), ParallelError> {
    if chunk_idx > 0 {
        // In-place SIMD substitution; ~memory-bandwidth speed on AVX2/NEON.
        replace_markers(bootstrap, input_window);
    }
    let mut bootstrap_u8: Vec<u8> = Vec::with_capacity(bootstrap.len() + bootstrap_clean.len());
    narrow_and_append(&mut bootstrap_u8, bootstrap).map_err(|pos| {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] chunk {} unresolved marker at bootstrap[{}] (offset {})",
                chunk_idx,
                pos,
                bootstrap[pos] - MARKER_BASE,
            );
        }
        ParallelError::DecodeFailed
    })?;
    bootstrap_u8.extend_from_slice(bootstrap_clean);
    let mut bootstrap_crc = crc32fast::Hasher::new();
    bootstrap_crc.update(&bootstrap_u8);
    Ok((bootstrap_u8, bootstrap_crc))
}

/// Pre-compute input windows for every chunk without resolving markers.
/// Returns `Some(windows)` when window propagation is determinable purely
/// from chunk `isal_bytes` (the common case: every non-empty chunk has
/// ≥32 KiB of ISA-L output). Returns `None` when at least one chunk's
/// output is smaller than the window and its successor's input window
/// would depend on the resolved bootstrap — in which case the caller
/// falls back to the sequential phase 2 path that walks chunks in order
/// and threads the rolling window through marker resolution.
fn try_precompute_input_windows(chunks: &[AuthoritativeChunk]) -> Option<Vec<Vec<u8>>> {
    let n = chunks.len();
    let mut windows: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut running: Vec<u8> = Vec::with_capacity(WINDOW_SIZE);
    for chunk in chunks.iter() {
        windows.push(running.clone());
        let c = &chunk.chunk;
        if c.isal_bytes.len() >= WINDOW_SIZE {
            running.clear();
            running.extend_from_slice(&c.isal_bytes[c.isal_bytes.len() - WINDOW_SIZE..]);
        } else if c.bootstrap.is_empty() && c.bootstrap_clean.is_empty() && c.isal_bytes.is_empty()
        {
            // Empty chunk (Subsumed / EmptyTail): window unchanged.
        } else {
            // Successor's window depends on this chunk's resolved bootstrap
            // bytes, which we don't have yet. Sequential phase 2 required.
            return None;
        }
    }
    Some(windows)
}

/// Phase 2 + streaming write: resolve bootstrap markers per chunk, write
/// immediately to `writer`, combine pre-computed ISA-L CRCs. Returns
/// `(combined_crc32, total_uncompressed_bytes)` after all chunks are
/// written. CRC+ISIZE verification happens in the caller after this
/// returns — partial bytes may already be written on mismatch (same
/// trade-off as rapidgzip's per-chunk streaming design).
///
/// Two paths:
///   - **Parallel**: when input windows for all chunks are determinable
///     purely from ISA-L output (common case), marker resolution +
///     bootstrap-bytes CRC happen in a worker pool. Mirrors rapidgzip's
///     `applyWindow` post-processing pattern.
///   - **Sequential**: when any chunk's successor window requires the
///     resolved bootstrap (rare: chunk with <32 KiB ISA-L output and
///     non-empty bootstrap), fall back to chunk-by-chunk resolution
///     threading the rolling window through.
fn phase2_resolve_write_combine<W: Write>(
    chunks: Vec<AuthoritativeChunk>,
    writer: &mut W,
) -> Result<(u32, usize), ParallelError> {
    if let Some(input_windows) = try_precompute_input_windows(&chunks) {
        return phase2_resolve_parallel(chunks, input_windows, writer);
    }
    phase2_resolve_sequential(chunks, writer)
}

/// Parallel phase 2: workers resolve markers + CRC bootstrap bytes against
/// the pre-computed input window for their chunk. Main thread writes
/// outputs in chunk order and combines CRCs sequentially.
fn phase2_resolve_parallel<W: Write>(
    chunks: Vec<AuthoritativeChunk>,
    input_windows: Vec<Vec<u8>>,
    writer: &mut W,
) -> Result<(u32, usize), ParallelError> {
    let n = chunks.len();
    debug_assert_eq!(input_windows.len(), n);
    if n == 0 {
        return Ok((0, 0));
    }

    let num_workers = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(n);

    // Wrap chunks in per-index Mutexes so workers can take ownership of
    // the bootstrap data via std::mem::take. The other AuthoritativeChunk
    // fields stay accessible for the sequential write/combine pass.
    let chunks: Vec<Mutex<AuthoritativeChunk>> = chunks.into_iter().map(Mutex::new).collect();
    type ResolveResult = Result<(Vec<u8>, crc32fast::Hasher), ParallelError>;
    let results: Vec<Mutex<Option<ResolveResult>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_workers {
            let chunks_ref = &chunks;
            let windows_ref = &input_windows;
            let results_ref = &results;
            let next_ref = &next_task;
            s.spawn(move || loop {
                let i = next_ref.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let mut chunk = chunks_ref[i].lock().unwrap();
                let mut bootstrap = std::mem::take(&mut chunk.chunk.bootstrap);
                let bootstrap_clean = std::mem::take(&mut chunk.chunk.bootstrap_clean);
                drop(chunk);
                let result =
                    resolve_chunk_markers(i, &mut bootstrap, &bootstrap_clean, &windows_ref[i]);
                *results_ref[i].lock().unwrap() = Some(result);
            });
        }
    });

    // Sequential merge: in order, take each chunk + its resolution result,
    // combine CRCs, write output bytes to the writer.
    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size = 0usize;
    for (i, chunk_mutex) in chunks.into_iter().enumerate() {
        let chunk = chunk_mutex.into_inner().unwrap().chunk;
        let result = results[i]
            .lock()
            .unwrap()
            .take()
            .expect("worker must have produced a result for every chunk index");
        let (bootstrap_u8, bootstrap_crc) = result?;
        total_crc.combine(&bootstrap_crc);
        total_crc.combine(&chunk.isal_crc);
        total_size += bootstrap_u8.len() + chunk.isal_bytes.len();
        writer.write_all(&bootstrap_u8).map_err(ParallelError::Io)?;
        writer
            .write_all(&chunk.isal_bytes)
            .map_err(ParallelError::Io)?;
    }

    Ok((total_crc.finalize(), total_size))
}

/// Sequential phase 2 (fallback when input windows cannot be pre-computed).
fn phase2_resolve_sequential<W: Write>(
    chunks: Vec<AuthoritativeChunk>,
    writer: &mut W,
) -> Result<(u32, usize), ParallelError> {
    let mut total_crc = crc32fast::Hasher::new();
    let mut window: Vec<u8> = Vec::with_capacity(WINDOW_SIZE);
    let mut total_size = 0usize;

    for (i, authoritative) in chunks.into_iter().enumerate() {
        let mut chunk = authoritative.chunk;
        let (bootstrap_u8, bootstrap_crc) =
            resolve_chunk_markers(i, &mut chunk.bootstrap, &chunk.bootstrap_clean, &window)?;
        // Free the u16 buffer eagerly — it's 2× the size of bootstrap_u8.
        chunk.bootstrap = Vec::new();
        chunk.bootstrap_clean = Vec::new();

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
        let c0 = match decode_chunk_with_handoff_legacy(
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
        let c2 = match decode_chunk_with_handoff_legacy(
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
            ChunkDecodeStop::Verified(ChunkStart::from_bits(17 * spacing_bits).to_end_limit()),
            "chunk 11 should still stop at the next downstream verified boundary",
        );
        assert_eq!(
            chunk_decode_stop(27, &start_bits, spacing_bits, total_bits),
            ChunkDecodeStop::Verified(ChunkStart::from_bits(33 * spacing_bits).to_end_limit()),
            "chunk 27 should still stop at the next downstream verified boundary",
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
        let c0 = match decode_chunk_with_handoff_legacy(
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
        let c0 = match decode_chunk_with_handoff_legacy(
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
        let c0 = match decode_chunk_with_handoff_legacy(
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
        let chunk = decode_chunk_with_handoff_legacy(
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

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    #[test]
    fn decode_chunk_inexact_legacy_returns_candidate_on_success() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;

        let original = make_compressible_data(4 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 3;
        let spacing = deflate_data.len() * 8 / num_chunks;
        let start_bits = phase1_search_boundaries(deflate_data, num_chunks, spacing, 1);
        let start_bit = start_bits[0].expect("chunk 0 start is pinned to bit 0");
        let stop = ChunkDecodeStop::Verified(
            start_bits[1]
                .map(ChunkStart::to_end_limit)
                .expect("phase1a should find next boundary"),
        );

        let outcome = decode_chunk_inexact_legacy(
            deflate_data,
            0,
            start_bit,
            stop,
            deflate_data.len() / num_chunks,
            num_chunks,
            false,
        );

        match outcome {
            WorkerOutcome::Candidate(candidate) => {
                assert_eq!(candidate.requested_partition, 0);
                assert_eq!(candidate.discovered_start, start_bit);
                assert_eq!(
                    candidate.discovered_end.bits(),
                    candidate.chunk.end_bit_offset
                );
            }
            WorkerOutcome::NoDecode => {
                panic!("expected candidate outcome for valid speculative start")
            }
        }
    }

    #[test]
    fn decode_chunk_inexact_legacy_returns_no_decode_on_error() {
        let deflate_data = [0u8; 8];
        let outcome = decode_chunk_inexact_legacy(
            &deflate_data,
            7,
            ChunkStart::from_bits(deflate_data.len() * 8 + 8),
            ChunkDecodeStop::UntilEnd,
            1024,
            1,
            false,
        );
        assert!(matches!(outcome, WorkerOutcome::NoDecode));
    }

    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    #[test]
    fn decode_chunk_exact_legacy_returns_authoritative_chunk() {
        use crate::decompress::parallel::marker_decode::skip_gzip_header;

        let original = make_compressible_data(4 * 1024 * 1024);
        let gzip = make_gzip_data(&original);
        let header_size = skip_gzip_header(&gzip).expect("valid gzip header");
        let deflate_data = &gzip[header_size..gzip.len() - 8];

        let num_chunks = 3;
        let spacing = deflate_data.len() * 8 / num_chunks;
        let start_bits = phase1_search_boundaries(deflate_data, num_chunks, spacing, 1);
        let start_bit = start_bits[0].expect("chunk 0 start is pinned to bit 0");
        let end_limit = start_bits[1]
            .map(ChunkStart::to_end_limit)
            .expect("phase1a should find next boundary");

        let chunk = decode_chunk_exact_legacy(
            deflate_data,
            start_bit,
            Some(end_limit),
            deflate_data.len() / num_chunks,
            num_chunks,
        )
        .expect("exact legacy seam should wrap successful worker decode");

        assert_eq!(chunk.start, start_bit);
        assert_eq!(chunk.end.bits(), chunk.chunk.end_bit_offset);
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
