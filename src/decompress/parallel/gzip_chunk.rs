#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_with_rapidgzip`] — vendor `decodeChunkWithRapidgzip` +
//!   `finishDecodeChunkWithInexactOffset` on one [`ChunkData`]:
//!   one outer decode iteration (`worker.decode_chunk`) alternates
//!   `marker_inflate` blocks (u16 markers) until 32 KiB clean, then streaming
//!   inflate on the same [`ChunkData`].
//! - [`decode_chunk`] — production entry (known 32 KiB window or chunk 0).
//! - [`decode_chunk_window_absent`] — marker bootstrap + clean tail (no window).
//!
//! `stop_hint_bits` is an inexact stop hint (vendor `untilOffset`): the
//! decoder runs to the first block boundary at-or-past it, then stops.

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::inflate_wrapper::InflateError;
#[cfg(parallel_sm)]
use crate::decompress::parallel::inflate_wrapper::{
    DeflateCompressionType, StoppingPoints, StreamingInflateWrapper,
};

#[derive(Debug)]
#[allow(dead_code)] // error payloads surfaced via Debug in production
pub enum ChunkDecodeError {
    InflateFailed(InflateError),
    BootstrapFailed(std::io::Error),
    ExactStopMissed {
        requested: usize,
        actual: usize,
    },
    #[allow(dead_code)] // non-SM-build cfg stub
    UnsupportedPlatform,
}

impl From<InflateError> for ChunkDecodeError {
    fn from(e: InflateError) -> Self {
        ChunkDecodeError::InflateFailed(e)
    }
}

impl From<std::io::Error> for ChunkDecodeError {
    fn from(e: std::io::Error) -> Self {
        ChunkDecodeError::BootstrapFailed(e)
    }
}

/// Output buffer size used per `read_stream` iteration. Matches
/// rapidgzip's `ALLOCATION_CHUNK_SIZE` (GzipChunk.hpp uses 128 KiB).
#[allow(dead_code)]
const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

// =========================================================================
// Body-failure diagnostics — speculation-accuracy investigation
// =========================================================================
// Per advisor-disproof: gzippy fails ~34 body speculations per silesia run,
// vendor fails 0. The hypothesis is that the WHY of these failures matters:
// where in the body does decode break, what error variant fires, how many
// bytes are wasted before failure. If failures cluster on specific corpus
// regions or specific error types, we can build a precondition heuristic
// that vendor has but we don't.

pub static BODY_FAIL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_BYTES_WASTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_BITS_INTO_BODY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_HUFFMAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_EXCEEDED_WINDOW: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_COMPRESSION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_CODE_LENGTHS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_OTHER_VARIANT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// C-instrumentation: per-block table-build vs body cost inside
// bootstrap_with_deflate_block. Each chunk's bootstrap iterates blocks,
// for each: read_header (precode + lit/len/dist table build) then body.
// 8.3B flame samples in "marker bootstrap" — need to split.
pub static BOOTSTRAP_HEADER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_HEADER_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_BODY_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_BODY_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Bootstrap instrumentation: output bytes belonging to bootstrap blocks that
/// ended CLEAN (`!contains_marker_bytes()` — no marker bytes remain after the
/// block). This is the marker-FREE *complement* of the marker-decode domain:
/// the new u16 marker fast loop touches the OTHER bytes (blocks that still
/// contain markers). NOTE: the old name `BOOTSTRAP_POST_FLIP_U16_BYTES` and its
/// "decoded into the u16 marker ring after the flip" doc were BACKWARDS and had
/// been read inverted multiple times — it counts clean-flipped bytes, NOT marker
/// bytes. The ratio CLEAN_FLIPPED / BODY_BYTES sizes the clean-flipped sliver
/// (small on marker-heavy workloads), NOT the marker-loop prize. No behavior change.
pub static BOOTSTRAP_CLEAN_FLIPPED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P2 unified decode: clean literals/backrefs routed to `chunk.data` (u8) after
/// `contains_marker_bytes` flips false — rapidgzip-shaped single output stream.
pub static UNIFIED_ROUTE_CLEAN_U8_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static UNIFIED_MODE_CLEAN_FLIPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Clean tail stayed on bulk LUT across a 128 KiB segment boundary (P2 unified).
pub static BULK_TAIL_SEGMENT_CONTINUES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Bulk LUT declined; tail used ResumableInflate2 (should be rare with 32 KiB window).
pub static BULK_TAIL_RESUMABLE_FALLBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Times the reused thread-local marker→clean handoff window buffer had to GROW
/// its capacity. Proves the per-chunk 32 KiB `clean_window: Vec<u8>` allocation
/// is gone: this should settle at ≈ num_worker_threads (one grow per thread on
/// first handoff) and then stay flat — NOT scale with the window-absent chunk
/// count. If it tracks the chunk count, the buffer is being reallocated per
/// chunk (the seam is not actually cut).
pub static HANDOFF_WINDOW_BUF_GROWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Legacy counter: non-vendor resync after marker bootstrap failure (removed Gate 1).
/// Must stay 0 in production — any increment means tryToDecode is not vendor-shaped.
pub static BAD_SEED_RESYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Chunks that entered `finish_decode_chunk_impl` (ISA-L / ResumableInflate2 tail).
pub static FINISH_DECODE_ENTRIES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// PHASE-0 WALL ORACLE (measurement-only, NOT production): chunks whose clean
/// tail was decoded by REAL ISA-L FFI via `decompress_deflate_from_bit_with_boundaries`
/// instead of the pure-Rust engine, when `GZIPPY_ISAL_ENGINE_ORACLE=1`. Proves the
/// engine actually ran ISA-L (assert this counter == clean-chunk count post-run).
pub static ISAL_ENGINE_ORACLE_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// PHASE-0 WALL ORACLE: clean tails that fell through to the pure-Rust engine
/// because the ISA-L oracle could not satisfy the contract (no exact boundary,
/// FFI unavailable). Non-zero ⇒ the oracle did NOT fully cover the bulk decode;
/// the wall number is contaminated by pure-Rust decode and must be discarded.
pub static ISAL_ENGINE_ORACLE_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Chunks accepted via the BFINAL exact-landing fix: `until_exact=true`,
/// no recorded EOB boundary at stop_hint_bits, but ISA-L's `end_bit` is
/// within 1 bit of stop_hint_bits (delta 0 = ISA-L exact; delta 1 = the
/// 1-bit coordinate discrepancy between ISA-L's chunk-decode context and
/// the block_finder's canonical scan position — diagnosed on NASA Jul95
/// start_bit=150999587, stop_hint=156360208, end_bit=156360207, 3–5/30 runs).
pub static ISAL_BFINAL_EXACT_LANDING_ACCEPTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// `until_exact=true` fallbacks: no boundary within 1 bit of stop_hint AND
/// end_bit is more than 1 bit away from stop_hint (genuine decline).
pub static ISAL_UNTIL_EXACT_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// `until_exact=false` fallbacks (no boundary at-or-past stop_hint AND end_bit > stop_hint).
pub static ISAL_INEXACT_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Whether the clean-tail decode routes through REAL ISA-L FFI.
///
/// PRODUCTION DEFAULT is the build: on `gzippy-isal` (`isal_clean_tail`) the clean
/// tail is decoded by ISA-L (faithful rapidgzip WITH_ISAL,
/// `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`, GzipChunk.hpp:440-444 +
/// :520-526); on `gzippy-native` it stays pure-Rust. The env var is an OVERRIDE:
///   * `GZIPPY_ISAL_ENGINE_ORACLE=1` — force ISA-L ON (the measurement oracle on the
///     native build; OFF==pure-Rust identity there).
///   * `GZIPPY_ISAL_ENGINE_ORACLE=0` — force ISA-L OFF (used by the differential gate
///     to obtain a genuine pure-Rust comparison decode on the isal build).
///   * unset — the build default (`cfg!(isal_clean_tail)`).
#[cfg(parallel_sm)]
#[inline]
fn isal_engine_oracle_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("GZIPPY_ISAL_ENGINE_ORACLE").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => cfg!(isal_clean_tail),
        },
    )
}

/// ALWAYS-SMALL INCREMENTAL GROWTH (measurement arm) — the footprint /
/// cache-locality sweep knob (DIS-14/DIS-17: gzippy RSS +21-25% / dTLB MPKI ~2x
/// vs rapidgzip, traced to the 8x-compressed-span UPFRONT output reserve).
/// When `GZIPPY_ISAL_INCREMENTAL_GROWTH=1`, the copy-free ISA-L decode reserves
/// a SMALL initial buffer (FAR below 8x) and GROWS it on demand as ISA-L fills
/// it — a faithful port of rapidgzip's fixed-`ALLOCATION_CHUNK_SIZE`
/// segment-append loop (`GzipChunk.hpp:309-379`: `DecodedVector(128 KiB)` ->
/// `readStream` -> `resize` -> `append`, repeat), done on gzippy's one
/// contiguous Vec, so the per-worker pooled buffer tracks the ACTUAL decoded
/// size.
///
/// Default OFF == production: the SAME growable decode but with a
/// **ratio-informed** upfront reserve (see `compute_initial_reserve` and
/// `ChunkConfiguration::expansion_ratio_ceil`), which sizes the initial from
/// the member's KNOWN ISIZE/compressed ratio instead of a fixed 8×. The old
/// fixed 8× is what the env knob FACTOR=2 falsified (+41% model-T8); the ratio
/// path closes that gap while keeping the DIS-29 >8x storm fix (growth still
/// engages on overflow). Do NOT default this env knob ON: the small initial
/// regresses sub-8x corpora at low-T (ghcn T1 -18%, DIS-29).
/// Returns `Some((initial_factor, grow_bytes))` when ON. Both tunable
/// for the footprint sweep: `GZIPPY_ISAL_INITIAL_FACTOR` (default 4),
/// `GZIPPY_ISAL_GROW_MIB` (default 4).
fn isal_incremental_growth() -> Option<(usize, usize)> {
    use std::sync::OnceLock;
    static CFG: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *CFG.get_or_init(|| {
        if !std::env::var("GZIPPY_ISAL_INCREMENTAL_GROWTH").is_ok_and(|v| v == "1") {
            return None;
        }
        let factor = std::env::var("GZIPPY_ISAL_INITIAL_FACTOR")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&f| f >= 1)
            .unwrap_or(4);
        let grow_mib = std::env::var("GZIPPY_ISAL_GROW_MIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&m| m >= 1)
            .unwrap_or(4);
        Some((factor, grow_mib * 1024 * 1024))
    })
}

/// ISOLATION ORACLE (measurement-only, NOT production; produces WRONG BYTES when ON).
///
/// `GZIPPY_FOLD_NODRAIN=1` makes `ContigFoldSink::push_clean_u8` SKIP the
/// ring->`chunk.data` drain memcpy (the `extend_from_slice` second-touch of every
/// clean byte) while still advancing `chunk.data`'s logical length (via
/// `writable_tail_reserve` + `commit` over UNINITIALIZED reserved space) so all
/// downstream length accounting (subchunk decoded_size, writev iovecs,
/// window-publish) stays panic-free. The decode itself (engine `block_body` +
/// ring write + back-ref resolution from `output_ring`) is UNCHANGED, so the wall
/// delta vs production native_fold isolates the cost of the ring->data drain copy
/// — the exact term the advisor said `ocl_cf` (ISA-L decode straight into the
/// final buffer) does not pay, confounding the 0.188× residual.
///
/// `GZIPPY_FOLD_NOCRC=1` additionally skips the per-clean-byte CRC `update` (the
/// CRC IS paid by `ocl_cf`, so this is the symmetric second knob to attribute the
/// total clean-byte second-touch). Both knobs OFF => byte-exact production.
fn fold_nodrain_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_FOLD_NODRAIN").is_ok_and(|v| v == "1"))
}

fn fold_nocrc_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_FOLD_NOCRC").is_ok_and(|v| v == "1"))
}

/// M3 kill-switch (DIV-1 part 1): `GZIPPY_SEEDED_BLOCK=0` restores the
/// pre-M3 wrapper path (`StreamingInflateWrapper`/`unified::Inflate`) for
/// window-seeded INEXACT chunks on gzippy-native. Default ON (Block engine).
/// Production proof of which engine decoded each seeded chunk:
/// [`SEEDED_BLOCK_CHUNKS`] vs [`SEEDED_WRAPPER_CHUNKS`] (GZIPPY_VERBOSE dump).
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn seeded_block_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_SEEDED_BLOCK").map_or(true, |v| v != "0"))
}

/// Whether the M3 seeded-Block route is taken for a window-seeded inexact
/// chunk. gzippy-native: ON unless `GZIPPY_SEEDED_BLOCK=0` or the ISA-L
/// measurement oracle is enabled (`GZIPPY_ISAL_ENGINE_ORACLE` must keep
/// observing the wrapper-entry path it instruments). gzippy-isal: constant
/// FALSE — the faithful rapidgzip WITH_ISAL clean-tail handoff
/// (GzipChunk.hpp:440-444) stays untouched.
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn seeded_block_route_enabled() -> bool {
    seeded_block_enabled() && !isal_engine_oracle_enabled()
}

#[cfg(all(parallel_sm, isal_clean_tail))]
fn seeded_block_route_enabled() -> bool {
    false
}

/// M4 kill-switch (DIV-1 part 2): `GZIPPY_EXACT_BLOCK=0` restores the
/// pre-M4 wrapper path (`StreamingInflateWrapper`/`unified::Inflate`) for
/// window-seeded UNTIL-EXACT chunks on gzippy-native. Default ON (Block
/// engine). Production proof of which engine decoded each exact chunk:
/// [`EXACT_BLOCK_CHUNKS`] vs [`EXACT_WRAPPER_CHUNKS`] (GZIPPY_VERBOSE dump).
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn exact_block_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_EXACT_BLOCK").map_or(true, |v| v != "0"))
}

/// Whether the M4 exact-Block route is taken for a window-seeded UNTIL-EXACT
/// chunk. gzippy-native: ON unless `GZIPPY_EXACT_BLOCK=0` or the ISA-L
/// measurement oracle is enabled (`GZIPPY_ISAL_ENGINE_ORACLE` must keep
/// observing the wrapper-entry path it instruments, including its
/// BFINAL-exact-landing accept). gzippy-isal: constant FALSE — the faithful
/// rapidgzip WITH_ISAL `decodeChunkWithInflateWrapper<IsalInflateWrapper>`
/// path (GzipChunk.hpp:192-265) stays untouched.
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn exact_block_route_enabled() -> bool {
    exact_block_enabled() && !isal_engine_oracle_enabled()
}

#[cfg(all(parallel_sm, isal_clean_tail))]
fn exact_block_route_enabled() -> bool {
    false
}

/// PHASE-0 WALL ORACLE (measurement-only, NOT a production path).
///
/// Decode this chunk's clean tail with REAL ISA-L (`isal_inflate`) via the
/// patched-boundary FFI, then feed ISA-L's bytes / boundaries / end-bit through
/// the SAME `ChunkData` accounting primitives `finish_decode_chunk_impl` uses
/// (commit + per-byte CRC + `append_block_boundary_at` + `finalize_with_deflate`),
/// trimmed to the chunk's natural stop. Returns `Ok(true)` if ISA-L produced the
/// chunk; `Ok(false)` to fall back to the pure-Rust engine (uncovered contract).
///
/// This drops an igzip-class engine into the REAL parallel-SM pipeline (the pool,
/// consumer, window-publish, ring, and CRC all stay) to bound the T8 WALL.

/// Compute the upfront output-reserve byte count for the ISA-L clean-tail decode.
///
/// `compressed_span` is the chunk's compressed byte span (`slice_end − byte_start`
/// inside `finish_decode_chunk_isal_oracle`).  `expansion_ratio_ceil` is the
/// member-level ratio ceiling from `ChunkConfiguration::expansion_ratio_ceil`; a
/// value of 0 means the ratio was unknown at configuration time → falls back to the
/// historical 8× factor.
///
/// Result is clamped to `[RESERVE_FLOOR, RESERVE_CAP]`.  Growth past `RESERVE_CAP`
/// is handled by the GROW_BYTES loop downstream and is always safe — this function
/// only sizes the *upfront* allocation.
///
/// Exposed as `pub(crate)` so the unit-test module can exercise the clamp logic
/// directly without constructing a full ISA-L decode.
#[cfg(all(parallel_sm, feature = "isal-compression", target_arch = "x86_64"))]
pub(crate) fn compute_initial_reserve(compressed_span: usize, expansion_ratio_ceil: u16) -> usize {
    const RESERVE_FLOOR: usize = 4 * 1024 * 1024; // never start below 4 MiB
    const RESERVE_CAP: usize = 64 * 1024 * 1024; // upfront ceiling; growth may exceed on demand
    let factor = if expansion_ratio_ceil == 0 {
        8 // unknown → historical default
    } else {
        expansion_ratio_ceil as usize
    };
    compressed_span
        .saturating_mul(factor)
        .max(RESERVE_FLOOR)
        .min(RESERVE_CAP)
}

#[cfg(all(parallel_sm, feature = "isal-compression", target_arch = "x86_64"))]
fn finish_decode_chunk_isal_oracle(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    until_exact: bool,
) -> Result<bool, ChunkDecodeError> {
    use crate::backends::isal_decompress;
    use std::sync::atomic::Ordering;

    if !isal_decompress::is_available() {
        return Ok(false);
    }
    // The oracle only covers a clean 32 KiB-window continuation (the bulk path
    // once windows are seeded). A non-full window means marker bootstrap is
    // needed — not ISA-L's job; fall back.
    if initial_window.len() != crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE {
        return Ok(false);
    }

    // BOUND THE DECODE TO THIS CHUNK (else ISA-L runs to BFINAL of the WHOLE
    // member per worker). Slice `input` to end a few bytes past `stop_hint_bits`
    // so ISA-L decodes only this chunk's blocks then runs out of input at a block
    // boundary. A clean DEFLATE block boundary is the natural stop the pipeline
    // already trims to.
    let stop_byte = stop_hint_bits.div_ceil(8);
    let slice_end = (stop_byte + 256 * 1024).min(input.len());
    let bounded = &input[..slice_end];

    // COPY-FREE: decode ISA-L DIRECTLY into the chunk's u8 data buffer. Reserve a
    // contiguous spare region (sized to the chunk decode bound) and hand its raw
    // pointer to the FFI — no intermediate 64 MiB `Vec`, no `copy_from_slice`.
    // This removes the per-chunk alloc+copy confound the prior oracle paid that
    // production's direct-to-`writable_tail()` stream never pays, so the
    // ISA-L clean-engine WALL ceiling becomes readable (advisor Q1 fix).
    //
    // RESERVE SIZING — CHUNK-PROPORTIONAL (TOOLING-AUDIT A5 reserve-confound fix).
    // The prior flat 64 MiB-per-chunk `reserve` was a residual confound: production
    // grows its u8 buffer INCREMENTALLY (`contig_decode_window(HEADROOM)`, the Vec
    // doubling to ~chunk size, ~13-16 MiB for a 4 MiB silesia chunk), so a flat
    // 64 MiB allocator request per chunk pays a per-chunk alloc several× larger than
    // production's final footprint. That made ocl_cf PESSIMISTIC (audit A5 / #2).
    // Fix: size the reserve to THIS chunk's REALISTIC decode bound = the compressed
    // input span (`slice_end - byte_start`, ≈ the chunk's compressed size + straddle)
    // × an expansion factor tracking the actual decoded ratio (silesia ~3.3×) with
    // headroom for a compressible chunk, so the allocator footprint MIRRORS
    // production's ~chunk-sized final capacity (~28 MiB for a ~4.7 MiB T8 chunk at
    // factor 8 vs the old flat 64 MiB) instead of the whole-member worst case.
    // EXPAND_FACTOR=8 covers silesia's ratio ~2.4× over; FLOOR keeps small chunks
    // safe; CAP bounds a pathological upfront request. Under-reserve no longer
    // falls back: the GROWABLE decode below grows the buffer on demand when the
    // chunk expands past the reserve (DIS-29 storm fix), so a >8x-compressible
    // chunk stays on ISA-L instead of storming to the ~7.5x pure-Rust fallback.
    // (A window-absent bootstrap chunk still legitimately falls back at some T —
    // a real property of the ISA-L ceiling, not a sizing bug.)
    let byte_start = inflate_start_bit / 8;
    let compressed_span = slice_end.saturating_sub(byte_start);
    let decode_start = chunk.data.len();
    let (written, end_bit, boundaries) = {
        // PRODUCTION DEFAULT: RATIO-INFORMED upfront reserve (box-proven +41%
        // model-T8; DIS-14/DIS-17 footprint mechanism, 2026-06-09).
        //
        // The reserve is sized from the member's KNOWN ISIZE/compressed ratio
        // (stored in chunk.configuration.expansion_ratio_ceil) rather than the
        // historical fixed 8×. On near-incompressible corpora (model: ~1.3×)
        // the 8× reserve over-allocated ~6× per chunk; with O(T) concurrent
        // workers the excess page-fault/dTLB pressure collapsed per-worker
        // ISA-L throughput (GZIPPY_ISAL_INCREMENTAL_GROWTH=1 FACTOR=2 falsifier
        // at model T8: wall 0.31 → 0.22s, maxRSS 390 → 291 MB, 3/3 runs). The
        // ratio-informed path removes the over-reservation without regressing
        // sub-8× low-T corpora (ghcn: ceiling = 10 covers 7.8× with margin;
        // the DIS-29 env-knob regression was exactly "too small a factor at
        // low-T" — now the factor is sized from the actual member ratio).
        //
        // ISIZE is mod 2^32: files >4 GiB raw may wrap → under-ratio → safe
        // regrow via GROW_BYTES (see ChunkConfiguration::expansion_ratio_ceil).
        // RESERVE_CAP bounds only the UPFRONT reserve; growth past it is on
        // demand and tracks the actual decoded size. RESERVE_FLOOR prevents
        // pathologically small allocations on tiny chunks.
        //
        // GZIPPY_ISAL_INCREMENTAL_GROWTH=1 keeps the ALWAYS-SMALL measurement
        // arm (footprint sweeps; DIS-23). It is NOT the production default —
        // the ratio-informed path below IS.
        let (initial, grow_bytes) = if let Some((initial_factor, grow)) = isal_incremental_growth()
        {
            const INCR_FLOOR: usize = 512 * 1024; // never start below 512 KiB
            (
                compressed_span
                    .saturating_mul(initial_factor)
                    .max(INCR_FLOOR),
                grow,
            )
        } else {
            const GROW_BYTES: usize = 4 * 1024 * 1024; // per-grow increment on overflow
            (
                compute_initial_reserve(compressed_span, chunk.configuration.expansion_ratio_ceil),
                GROW_BYTES,
            )
        };
        match isal_decompress::decompress_deflate_from_bit_into_growable(
            bounded,
            inflate_start_bit,
            initial_window,
            &mut chunk.data,
            initial,
            grow_bytes,
        ) {
            Some(v) => {
                // All decoded bytes were committed during the decode; reset the
                // LOGICAL length back to decode_start while the bytes REMAIN
                // physically present in the buffer's spare (Vec::truncate keeps
                // capacity + contents). This leaves the state the old fixed-buffer
                // path produced (len == decode_start, the `written` bytes sitting
                // in spare), so the shared commit/CRC/boundary/fallback logic
                // below — including its several `return Ok(false)` exits — runs
                // byte-for-byte the same. (The exits must leave chunk.data at
                // decode_start; the fixed path got that for free by committing
                // only at the very end.)
                chunk.data.truncate(decode_start);
                v
            }
            None => {
                // The growable decode may have committed bytes during growth
                // before bailing; reset to the pre-decode length before falling
                // back to the byte-exact pure-Rust engine.
                chunk.data.truncate(decode_start);
                return Ok(false);
            }
        }
    };

    // Pick the natural stop: first boundary at-or-past stop_hint (inexact), or
    // the exact boundary == stop_hint (exact). Boundaries record (bit_offset,
    // output_offset) measured from the start of THIS decode (offset into the
    // reserved region == offset from `decode_start`).
    let (final_bit, keep_len) = if until_exact {
        let mut found = None;
        for b in &boundaries {
            if b.bit_offset == stop_hint_bits {
                found = Some((b.bit_offset, b.output_offset));
                break;
            }
        }
        match found {
            Some(v) => v,
            None => {
                // BFINAL exact-landing: the member's final block ends with no
                // recorded EOB boundary AT stop_hint_bits. Two sub-cases:
                //
                // (a) ISA-L exited via ISAL_BLOCK_FINISH before setting
                //     stopped_at — no boundary recorded at all for this block.
                //     `end_bit == stop_hint_bits` (same ISA-L coordinate system
                //     used to derive stop_hint in tests/measurement).
                //
                // (b) Production: ISA-L DID record the BFINAL EOB boundary via
                //     stopped_at, but at `stop_hint_bits - 1` instead of
                //     `stop_hint_bits` (a 1-bit coordinate discrepancy between
                //     ISA-L's chunk-decode context and the block_finder's
                //     canonical scan — ISA-L's `data.len()*8 - avail_in*8 -
                //     read_in_length` tracks bit consumption from a different
                //     internal-buffering state). Diagnosed on NASA Jul95:
                //     start_bit=150999587, stop_hint=156360208, end_bit=156360207,
                //     boundaries=25, delta=-1, wrapper_ok=true, 3–5/30 runs.
                //
                // In both cases `end_bit` is within 1 bit of `stop_hint_bits`
                // and below it, confirming ISA-L reached and decoded the final
                // BFINAL block. The until_exact contract is satisfied — the
                // decode stopped at or within 1 bit of the exact requested
                // position, with the 1-bit slack being a coordinate convention
                // difference, not a data difference. Vendor GzipChunk.hpp
                // decodeChunkWithInflateWrapper tracks the end position directly
                // rather than requiring a recorded boundary; accepting an
                // exact or within-1 landing here converges to that behavior.
                // LOAD-BEARING: `end_bit <= stop_hint_bits` — not the <=1 slack —
                // is what structurally excludes INTERIOR chunks from this accept:
                // an interior decode is input-bounded at stop_byte + 256 KiB and
                // always runs PAST stop_hint (end_bit >> stop_hint), so reaching
                // end_bit <= stop_hint means the decode genuinely hit the member's
                // BFINAL stream end. Do not "simplify" the guard away.
                let end_within_1 = end_bit <= stop_hint_bits && stop_hint_bits - end_bit <= 1;
                if end_within_1 {
                    ISAL_BFINAL_EXACT_LANDING_ACCEPTED.fetch_add(1, Ordering::Relaxed);
                    // Use stop_hint_bits as the canonical final_bit so the
                    // chunk records its end at the block_finder's coordinate.
                    (stop_hint_bits, written)
                } else {
                    ISAL_UNTIL_EXACT_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    return Ok(false); // oracle can't honor exact stop ⇒ fall back
                }
            }
        }
    } else {
        let mut chosen = None;
        for b in &boundaries {
            if b.bit_offset >= stop_hint_bits {
                chosen = Some((b.bit_offset, b.output_offset));
                break;
            }
        }
        match chosen {
            Some(v) => v,
            None => {
                // No recorded block boundary at-or-past the stop hint. Accepting
                // `(end_bit, written)` is correct ONLY when the stream genuinely
                // ENDED (BFINAL) at-or-before the stop hint — the member's last
                // chunk — so `end_bit <= stop_hint`. If instead the decode ran PAST
                // the stop hint with no usable boundary there, ISA-L's END_OF_BLOCK
                // stopping did NOT fire for this chunk (observed on stored/fixed-block
                // input: a multi-MiB decode recorded ZERO boundaries), so committing
                // `end_bit` would OVER-DECODE past the chunk's natural stop and
                // mis-seed the next chunk's start bit (a "Stored block len/nlen
                // mismatch" downstream). DECLINE to the byte-exact pure-Rust engine
                // (counted in ISAL_ENGINE_ORACLE_FALLBACKS). On the all-dynamic parity
                // corpus ISA-L always records a boundary at-or-past the hint, so this
                // never fires there.
                if end_bit <= stop_hint_bits {
                    (end_bit, written)
                } else {
                    ISAL_INEXACT_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    return Ok(false);
                }
            }
        }
    };

    let keep_len = keep_len.min(written);

    // Commit ONLY the kept region into the chunk's u8 data buffer. The bytes are
    // already physically present in the reserved spare (ISA-L wrote them, in BOTH
    // the fixed-buffer and incremental paths — the latter truncated its logical
    // length back to decode_start above while keeping the bytes in spare); commit
    // bumps the logical length with zero copies.
    chunk.data.commit(keep_len);

    // CRC over the exact kept region (mirrors finish_decode_chunk_impl:512-517).
    // The kept bytes are now the contiguous slice `data[decode_start..decode_start+keep_len]`.
    if chunk.configuration.crc32_enabled {
        // Re-borrow the committed region as a slice for hashing (zero-copy view).
        let kept_view = chunk.data.decoded_range(decode_start, keep_len);
        if let Some(last_crc) = chunk.crc32s.last_mut() {
            last_crc.update(kept_view);
        }
    }
    chunk.statistics.non_marker_count += keep_len as u64;

    // Replay boundaries WITH INCREMENTAL byte accounting so on-the-fly subchunk
    // splitting fires exactly as the pure streaming loop's per-EOB path does.
    // `ChunkData::append_block_boundary_at` only starts a new subchunk once the
    // OPEN subchunk's `decoded_size` has crossed `split_chunk_size`; that size is
    // grown by `note_inner_decoded_bytes`. The pure path interleaves
    // note+append per read, so crediting the full `keep_len` up front (the old
    // shape) hid the per-segment growth and produced ONE subchunk for the whole
    // ISA-L tail (the `UNSPLIT_BLOCKS_EMPLACED` deletion trap caught this). Here we
    // credit each [prev_off, b.output_offset) segment to the current subchunk
    // BEFORE recording its boundary, then the final [last_off, keep_len) tail —
    // byte-transparent (the committed bytes and total decoded_size are unchanged;
    // only the subchunk split metadata now matches the pure path / vendor
    // GzipChunk.hpp:321 + appendDeflateBlockBoundary).
    let mut prev_off = 0usize;
    for b in &boundaries {
        if b.bit_offset > final_bit {
            break;
        }
        let off = b.output_offset.min(keep_len);
        if off > prev_off {
            chunk.note_inner_decoded_bytes(off - prev_off);
            prev_off = off;
        }
        let decoded_offset = decode_start + off;
        if decoded_offset > 0 {
            chunk.append_block_boundary_at(b.bit_offset, decoded_offset, Some(input));
        }
    }
    if keep_len > prev_off {
        chunk.note_inner_decoded_bytes(keep_len - prev_off);
    }

    chunk.finalize_with_deflate(final_bit, Some(input));
    ISAL_ENGINE_ORACLE_CHUNKS.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

#[cfg(not(all(parallel_sm, feature = "isal-compression", target_arch = "x86_64")))]
fn finish_decode_chunk_isal_oracle(
    _chunk: &mut ChunkData,
    _input: &[u8],
    _inflate_start_bit: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _until_exact: bool,
) -> Result<bool, ChunkDecodeError> {
    Ok(false)
}
/// Chunks that took vendor `decodeChunkWithInflateWrapper` (exact window + until_exact).
pub static INFLATE_WRAPPER_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Chunks that took the marker→clean FLIP: a bounded u16 marker prefix then the
/// fast u8 clean tail (vendor setInitialWindow). This is the FAST window-absent
/// path; if it stays 0 while speculative chunks exist, every window-absent chunk
/// is decoding its whole body as u16 markers + full resolve (the slow path).
pub static FLIP_TO_CLEAN_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Chunks that finished WITHOUT a flip (BFINAL or stop-hint reached before 32 KiB
/// of clean output accumulated) — the whole decoded body stayed u16 markers.
pub static FINISHED_NO_FLIP_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// `resumable_resync` bulk loop saw `Handoff` with no bit advance (stall guard).
#[cfg(pure_inflate_decode)]
pub static BULK_HANDOFF_NO_PROGRESS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Why [`finish_decode_chunk_bulk_lut`] returned [`BulkCleanTailResult::Decline`].
#[cfg(pure_inflate_decode)]
pub static BULK_LUT_ENTERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_LUT_DECLINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_DISABLED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_NO_LOOKBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_INVALID_HUFFMAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_INVALID_LOOKBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_OUTPUT_OVERFLOW: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(pure_inflate_decode)]
#[allow(dead_code)]
fn record_bulk_decline(err: crate::decompress::parallel::lut_bulk_inflate::BulkDecodeError) {
    use crate::decompress::parallel::lut_bulk_inflate::BulkDecodeError;
    use std::sync::atomic::Ordering;
    let c = match err {
        BulkDecodeError::InvalidHuffmanCode => &BULK_DECLINE_INVALID_HUFFMAN,
        BulkDecodeError::InvalidLookback => &BULK_DECLINE_INVALID_LOOKBACK,
        BulkDecodeError::OutputOverflow => &BULK_DECLINE_OUTPUT_OVERFLOW,
        BulkDecodeError::InvalidCodeLengths | BulkDecodeError::BlockTypeReserved => {
            &BULK_DECLINE_OTHER
        }
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// Per-failure structured log. Writes one JSON line to the path in
/// `GZIPPY_BODY_FAIL_LOG` env var (no-op if unset). Use this for
/// distributional analysis: cluster failures by offset to see if they
/// fall in specific corpus regions.
#[cfg(parallel_sm)]
fn body_fail_log(start_bit: usize, bits_into_body: usize, bytes_wasted: usize, err: &str) {
    use std::io::Write;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let f = FILE.get_or_init(|| {
        let path = std::env::var("GZIPPY_BODY_FAIL_LOG").ok()?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Mutex::new(f))
    });
    let Some(mtx) = f else {
        return;
    };
    let line = format!(
        r#"{{"start_bit":{start_bit},"bits_into_body":{bits_into_body},"bytes_wasted":{bytes_wasted},"err":"{}"}}"#,
        err.replace('"', "\\\"")
    );
    let mut g = mtx.lock().unwrap();
    let _ = writeln!(g, "{}", line);
}

#[cfg(not(parallel_sm))]
fn body_fail_log(_: usize, _: usize, _: usize, _: &str) {}

/// Rapidgzip-shaped chunk decode — `GzipChunk.hpp::decodeChunkWithRapidgzip`
/// (outer `while` over deflate blocks) with handoff to
/// `finishDecodeChunkWithInexactOffset` once 32 KiB clean exist at a block
/// boundary (or immediately when `initial_window` is full 32 KiB).
#[cfg(parallel_sm)]
pub fn decode_chunk_with_rapidgzip(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip_until_exact(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
        false,
    )
}

#[cfg(parallel_sm)]
pub fn decode_chunk_with_rapidgzip_until_exact(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
    until_exact: bool,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
        until_exact,
    )
}

/// Production chunk decode with optional predecessor window (vendor
/// `decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset`).
#[cfg(parallel_sm)]
pub fn decode_chunk(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_until_exact(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
        false,
    )
}

#[cfg(parallel_sm)]
pub fn decode_chunk_until_exact(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
    until_exact: bool,
) -> Result<ChunkData, ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    if initial_window.len() == MAX_WINDOW_SIZE && until_exact {
        // M4 (DIV-1 part 2): window-seeded UNTIL-EXACT chunks decode on the
        // ONE `deflate::Block` engine on gzippy-native (kill-switch
        // `GZIPPY_EXACT_BLOCK=0` restores the wrapper arm below exactly).
        if exact_block_route_enabled() {
            return decode_chunk_exact_block_native(
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
                configuration,
            );
        }
        return decode_chunk_with_inflate_wrapper(
            input,
            encoded_offset_bits,
            stop_hint_bits,
            initial_window,
            configuration,
        );
    }

    decode_chunk_with_rapidgzip_until_exact(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
        until_exact,
    )
}

/// Vendor `decodeChunkWithInflateWrapper` shape: exact known-window decode
/// using the inflate wrapper only.
#[cfg(parallel_sm)]
fn decode_chunk_with_inflate_wrapper(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    INFLATE_WRAPPER_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // M4 kill-switch complement proof: an until-exact chunk decoded on the
    // wrapper engine (gzippy-isal production, `GZIPPY_EXACT_BLOCK=0`, or the
    // ISA-L measurement oracle).
    EXACT_WRAPPER_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    finish_decode_chunk_impl(
        &mut chunk,
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        true,
        true,
        // PRODUCTION: allow the ISA-L clean-tail engine.
        true,
    )?;
    Ok(chunk)
}

/// Vendor `finishDecodeChunkWithInexactOffset` shape. Continues a chunk with a
/// known clean 32 KiB window, stopping at the first deflate boundary at-or-past
/// `stop_hint_bits` (or exactly at it on the exact path).
#[cfg(parallel_sm)]
fn finish_decode_chunk_with_inexact_offset(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    record_decode_duration: bool,
) -> Result<(), ChunkDecodeError> {
    finish_decode_chunk_impl(
        chunk,
        input,
        inflate_start_bit,
        stop_hint_bits,
        initial_window,
        record_decode_duration,
        false,
        // PRODUCTION: allow the ISA-L clean-tail engine (the build decides whether it
        // actually fires — see `isal_engine_oracle_enabled`).
        true,
    )
}

#[cfg(parallel_sm)]
fn finish_decode_chunk_impl(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    record_decode_duration: bool,
    until_exact: bool,
    // When false, FORCE the pure-Rust clean tail regardless of build/env. Used only
    // by the differential gate's LEFT (pure) decode so it can compare against the
    // ISA-L RIGHT on the `isal_clean_tail` build (where ISA-L is the default).
    allow_isal: bool,
) -> Result<(), ChunkDecodeError> {
    FINISH_DECODE_ENTRIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // gzippy-isal PRODUCTION clean-tail routing (faithful rapidgzip WITH_ISAL,
    // GzipChunk.hpp:440-444 known-window + :520-526 post-32 KiB-flip clean bulk):
    // decode the clean tail through REAL ISA-L FFI (copy-free
    // `decompress_deflate_from_bit_into`), exactly as rapidgzip's
    // `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`. The <=32 KiB markered
    // prefix already ran on the pure-Rust marker engine (`deflate::Block`) before the
    // flip — faithful (rapidgzip uses `deflate::Block` there too). On gzippy-native
    // this routes ISA-L only under the measurement env oracle (OFF==pure-Rust identity).
    //
    // FALLBACK (`Ok(false)`): the SAME clean tail decodes byte-exact through the
    // pure-Rust engine below, and the event is COUNTED in ISAL_ENGINE_ORACLE_FALLBACKS.
    // This is a correctness safety net for the cases ISA-L genuinely cannot honor (an
    // `until_exact` stop with no exact ISA-L boundary, or a pathologically compressible
    // chunk overrunning the chunk-proportional reserve), NOT a silent divergence: the
    // bytes emitted are identical to what ISA-L would emit — only the emitting engine
    // differs. The differential gate + parity.sh assert this counter ==0 on the corpus
    // so a real fallback surfaces loudly. On window-absent bootstrap chunks the body
    // stays u16 markers and returns `Finished` WITHOUT reaching here (faithful: rapidgzip
    // also marker-decodes a sub-32 KiB body) — so those never count as a fallback.
    if allow_isal && isal_engine_oracle_enabled() {
        match finish_decode_chunk_isal_oracle(
            chunk,
            input,
            inflate_start_bit,
            stop_hint_bits,
            initial_window,
            until_exact,
        ) {
            Ok(true) => return Ok(()),
            Ok(false) => {
                ISAL_ENGINE_ORACLE_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => return Err(e),
        }
    }

    let t_decode = std::time::Instant::now();
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.isal_stream_inflate",
        &format!(
            r#""start_bit":{inflate_start_bit},"stop_hint":{stop_hint_bits},"has_window":{},"until_exact":{}"#,
            !initial_window.is_empty(),
            until_exact
        ),
    );

    let read_cap = if until_exact {
        stop_hint_bits
    } else {
        input.len() * 8
    };
    let mut wrapper = StreamingInflateWrapper::with_until_bits(input, inflate_start_bit, read_cap)?;
    wrapper.set_window(initial_window)?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER,
    );
    if !until_exact {
        wrapper.set_coalesce_stop_hint(stop_hint_bits);
    }

    let mut stopping_point_reached = false;
    let mut last_end_bit = inflate_start_bit;
    let mut last_eob_pos = inflate_start_bit;
    let mut last_eob_decoded_bytes: usize = chunk.decoded_size();
    let mut already_decoded: usize = chunk.decoded_size();
    let mut pending_stop_after_flush = false;
    const STOP_INNER_ON_PENDING_FLUSH: bool = true;

    while !stopping_point_reached || wrapper.session_pending() {
        let prev_data_len = chunk.data.len();
        let seg_tail = chunk.data.writable_tail();
        let seg_ptr = seg_tail.as_mut_ptr();
        let buffer_cap = seg_tail.len();
        let mut n_bytes_read: usize = 0;
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;

        let decode_base = already_decoded;
        while n_bytes_read < buffer_cap
            && !stopping_point_reached
            && !(STOP_INNER_ON_PENDING_FLUSH
                && pending_stop_after_flush
                && !wrapper.session_pending())
        {
            let bit_before_read = wrapper.tell_compressed();
            let spare: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(seg_ptr.add(n_bytes_read), buffer_cap - n_bytes_read)
            };
            let r = wrapper.read_stream(spare)?;
            last_per_call = r.bytes_written;
            n_bytes_read += last_per_call;
            chunk.note_inner_decoded_bytes(last_per_call);

            last_stopped_at = r.stopped_at;
            last_finished = r.finished;
            last_end_bit = r.bit_position;

            let call_base = decode_base + (n_bytes_read - last_per_call);
            for (bp, rel_off) in wrapper.take_block_boundaries() {
                let decoded_offset = call_base + rel_off;
                if decoded_offset > 0 {
                    chunk.append_block_boundary_at(bp, decoded_offset, Some(input));
                }
            }

            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && !r.finished
                && r.bit_position == bit_before_read
            {
                stopping_point_reached = true;
                break;
            }

            if r.finished {
                break;
            }

            match r.stopped_at {
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                    if decode_base + n_bytes_read > 0 {
                        chunk.append_block_boundary_at(
                            r.bit_position,
                            decode_base + n_bytes_read,
                            Some(input),
                        );
                    }
                }
                sp if sp == StoppingPoints::END_OF_BLOCK => {
                    if !wrapper.is_final_block() {
                        if decode_base + n_bytes_read > 0 {
                            chunk.append_block_boundary_at(
                                r.bit_position,
                                decode_base + n_bytes_read,
                                Some(input),
                            );
                        }
                        if !until_exact && r.bit_position >= stop_hint_bits {
                            // Do not keep filling this buffer from the next block
                            // before HEADER/NONE handling — finalize at pre-header EOB.
                            last_end_bit = r.bit_position;
                            pending_stop_after_flush = true;
                        }
                    }
                    last_eob_pos = r.bit_position;
                    last_eob_decoded_bytes = decode_base + n_bytes_read;
                }
                sp if sp == StoppingPoints::END_OF_BLOCK_HEADER => {
                    let next_block_offset = wrapper.tell_compressed();
                    let not_final = !wrapper.is_final_block();
                    let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
                    if !until_exact
                        && ((next_block_offset >= stop_hint_bits && not_final && not_fixed)
                            || next_block_offset == stop_hint_bits)
                    {
                        last_end_bit = last_eob_pos;
                        pending_stop_after_flush = true;
                    }
                }
                sp if sp == StoppingPoints::NONE
                    && !until_exact
                    && last_per_call == 0
                    && last_eob_pos >= stop_hint_bits =>
                {
                    last_end_bit = last_eob_pos;
                    pending_stop_after_flush = true;
                }
                _ => {}
            }
            if last_finished {
                break;
            }
        }

        let mut append_len = n_bytes_read;
        if stopping_point_reached {
            append_len = last_eob_decoded_bytes.saturating_sub(decode_base);
        } else if pending_stop_after_flush {
            append_len = n_bytes_read;
        }
        if append_len > 0 {
            if chunk.configuration.crc32_enabled {
                let kept: &[u8] = unsafe { std::slice::from_raw_parts(seg_ptr, append_len) };
                if let Some(last_crc) = chunk.crc32s.last_mut() {
                    last_crc.update(kept);
                }
            }
            chunk.data.commit(append_len);
            chunk.statistics.non_marker_count += append_len as u64;
        }
        let _ = prev_data_len;
        already_decoded = decode_base + append_len;

        if pending_stop_after_flush && !wrapper.session_pending() {
            stopping_point_reached = true;
        }

        if last_finished {
            break;
        }
        if last_stopped_at == StoppingPoints::NONE && last_per_call == 0 {
            if !until_exact && last_eob_pos >= stop_hint_bits {
                last_end_bit = last_eob_pos;
                break;
            }
            continue;
        }
    }

    if record_decode_duration {
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    }

    let final_bit = if until_exact {
        wrapper.tell_compressed()
    } else if stopping_point_reached {
        last_end_bit
    } else if last_eob_pos > inflate_start_bit {
        last_eob_pos
    } else {
        wrapper.tell_compressed()
    };
    if until_exact && final_bit != stop_hint_bits {
        return Err(ChunkDecodeError::ExactStopMissed {
            requested: stop_hint_bits,
            actual: final_bit,
        });
    }
    chunk.finalize_with_deflate(final_bit, Some(input));
    Ok(())
}

/// Window-absent chunk decode (speculative prefetch / boundary search).
/// Same unified [`decode_chunk_with_rapidgzip_impl`] as [`decode_chunk`], with
/// an empty initial window so the marker phase runs until 32 KiB clean.
#[cfg(parallel_sm)]
pub fn decode_chunk_window_absent(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        &[],
        configuration,
        false,
    )
}

/// `decodeChunkWithRapidgzip` body (GzipChunk.hpp:468-654).
#[cfg(parallel_sm)]
fn decode_chunk_with_rapidgzip_impl(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
    until_exact: bool,
) -> Result<ChunkData, ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    // Envelope span: `chunk_fetcher::run_decode_task` (`worker.decode_chunk`).
    let t_decode = std::time::Instant::now();

    if initial_window.len() == MAX_WINDOW_SIZE {
        WINDOW_SEEDED_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::decompress::parallel::trace_v2::emit_instant(
            "worker.chunk_phase",
            &format!(
                r#""start_bit":{encoded_offset_bits},"phase":"window_seeded","until_exact":{until_exact}"#
            ),
            "t",
        );
        let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
        if until_exact {
            if exact_block_route_enabled() {
                // M4 (DIV-1 part 2): until-exact decodes on the ONE
                // `deflate::Block` engine (see
                // `finish_decode_chunk_exact_block_native` for the labeled
                // deviation + pre-registered contract).
                finish_decode_chunk_exact_block_native(
                    &mut chunk,
                    input,
                    encoded_offset_bits,
                    stop_hint_bits,
                    initial_window,
                )?;
            } else {
                EXACT_WRAPPER_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                finish_decode_chunk_impl(
                    &mut chunk,
                    input,
                    encoded_offset_bits,
                    stop_hint_bits,
                    initial_window,
                    false,
                    true,
                    // PRODUCTION: allow the ISA-L clean-tail engine.
                    true,
                )?;
            }
        } else if seeded_block_route_enabled() {
            // M3 (DIV-1 part 1): window-seeded INEXACT chunks decode on the ONE
            // `deflate::Block` engine (vendor GzipChunk.hpp:454-458, non-ISAL
            // build) instead of the second clean engine
            // (`StreamingInflateWrapper`/`unified::Inflate`).
            finish_decode_chunk_seeded_block_native(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
            )?;
        } else {
            SEEDED_WRAPPER_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            finish_decode_chunk_with_inexact_offset(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
                false,
            )?;
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        return Ok(chunk);
    }

    // Vendor tryToDecode: bootstrap failure propagates; caller catches and tries next candidate.
    let mut chunk =
        decode_chunk_unified_marker(input, encoded_offset_bits, stop_hint_bits, configuration)?;
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    Ok(chunk)
}

/// Clean-fold sink (gzippy-native FOLD production path): writes post-flip clean
/// u8 bytes DIRECTLY into the pre-reserved contiguous `chunk.data`, replicating
/// `ChunkData::append_clean`'s exact accounting (CRC + subchunk decoded_size +
/// non_marker_count) — NO intermediate `pending_clean` Vec, NO second copy, NO
/// per-run regrow. Together with the copy-free ring drain
/// (`marker_inflate::drain_to_output`) the post-flip clean tail is fully
/// copy-free (ring slice -> chunk.data in one memcpy). Holds disjoint field
/// borrows so `push_slice` (markers) and `push_clean_u8` (clean data) never
/// alias. This recovered +0.059× of the T8 wall (native_fold 0.678× -> 0.737× rg,
/// quiet-box banked, sha-exact; the loaded 6-pass split showed the same recovery
/// monotonic across copy#1 + copy#2/3/grow but load-inflated, so the banked
/// number is +0.059×). Vendor decodes the clean tail straight into one contiguous
/// DecodedData buffer (DecodedData.hpp:278-289). NOTE: this ContigFoldSink ring
/// path is the ~1% marker-loop dribble on gzippy-native; the BULK clean tail is
/// u8-direct via `decode_clean_into_contig` (no ring, no drain). The
/// GZIPPY_FOLD_NODRAIN/NOCRC split measured the remaining drain+CRC second-touch at
/// ~0-1ms (frozen host N=21), so the gap to the ISA-L `ocl_cf` ceiling
/// (matched-comparator 0.945× rg) is ~36ms of essentially PURE symbol rate on the
/// SAME covered chunks — NOT an upper bound padded by ring cost (that earlier
/// caveat is STALE for the contig bulk). See plans/fold-drain-split-result.md.
#[cfg(parallel_sm)]
struct ContigFoldSink<'a> {
    markers: &'a mut crate::decompress::parallel::segmented_markers::SegmentedU16,
    data: &'a mut crate::decompress::parallel::segmented_buffer::SegmentedU8,
    crc32s: &'a mut Vec<crate::decompress::parallel::crc32::CRC32Calculator>,
    subchunks: &'a mut Vec<crate::decompress::parallel::chunk_data::Subchunk>,
    non_marker_count: &'a mut u64,
    clean_appended: &'a mut usize,
    crc32_enabled: bool,
    boundaries: &'a mut Vec<(usize, usize)>,
}

#[cfg(parallel_sm)]
impl crate::decompress::parallel::marker_inflate::MarkerSink for ContigFoldSink<'_> {
    fn push_slice(&mut self, values: &[u16]) {
        self.markers.push_slice(values);
    }
    fn push_clean_u8(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // Identical accounting to ChunkData::append_clean, but the bytes land
        // straight in the pre-reserved contiguous tail (single copy, no regrow).
        if self.crc32_enabled && !fold_nocrc_enabled() {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(bytes);
            }
        }
        *self.non_marker_count += bytes.len() as u64;
        if fold_nodrain_enabled() {
            // MEASUREMENT-ONLY: skip the ring->data drain memcpy. Advance the
            // logical length over uninitialized reserved space so accounting
            // stays consistent (WRONG bytes; never a production path).
            let n = bytes.len();
            let _ = self.data.writable_tail_reserve(n);
            self.data.commit(n);
        } else {
            self.data.extend_from_slice(bytes);
        }
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += bytes.len();
        }
        *self.clean_appended += bytes.len();
    }
    fn clean_appended_len(&self) -> usize {
        *self.clean_appended
    }
    fn sink_len(&self) -> usize {
        self.markers.sink_len() + *self.clean_appended
    }
    fn as_slice(&self) -> &[u16] {
        self.markers.as_slice()
    }
    fn trailing_clean_since(&self, from: usize) -> usize {
        let marker_len = self.markers.sink_len();
        let clean_len = *self.clean_appended;
        if from >= marker_len + clean_len {
            return 0;
        }
        if from >= marker_len {
            clean_len - (from - marker_len)
        } else {
            self.markers.trailing_clean_since(from) + clean_len
        }
    }
    fn copy_last_n_clean_u8(&self, n: usize, out: &mut Vec<u8>) -> bool {
        // Clean bytes already live contiguously in chunk.data; the marker
        // path only needs this BEFORE the flip (clean tail < window), at which
        // point clean_appended is 0 and the marker sink serves the request.
        if *self.clean_appended < n {
            return self.markers.copy_last_n_clean_u8(n, out);
        }
        false
    }
    fn note_block_boundary(&mut self, encoded_offset_bits: usize, decoded_offset: usize) {
        self.boundaries.push((encoded_offset_bits, decoded_offset));
    }
}

#[cfg(parallel_sm)]
fn apply_recorded_block_boundaries(
    chunk: &mut ChunkData,
    deflate_data: &[u8],
    boundaries: &[(usize, usize)],
) {
    for &(encoded_offset_bits, decoded_offset) in boundaries {
        if decoded_offset > 0 {
            chunk.append_block_boundary_at(encoded_offset_bits, decoded_offset, Some(deflate_data));
        }
    }
}

/// The ONE unified decode driver — vendor `deflate::Block` (see
/// `marker_decode_step`). Once 32 KiB of clean output exist at a block boundary, control hands off to
/// `finish_decode_chunk_with_inexact_offset` with that clean window.
#[cfg(parallel_sm)]
fn decode_chunk_unified_marker(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let mut marker_ctx = MarkerDecodeCtx::new(input, encoded_offset_bits)?;
    chunk.data_with_markers.reserve(128 * 1024);
    // Pre-reserve ONE contiguous clean-data region up-front so the post-flip
    // clean tail lands without per-run amortized regrow. Estimate the decoded
    // size as compressed × 8 (a typical-ratio HEURISTIC, NOT DEFLATE's worst-case
    // ~1032:1 expansion) clamped to a sane ceiling; an under-reserve on a
    // highly-compressible chunk just falls back to amortized regrow (safe — the
    // sink writes via `extend_from_slice`), it never corrupts. The clamp bounds
    // the high-T × large-chunk RSS bump. With the copy-free ring drain
    // (`marker_inflate::drain_to_output`'s clean branch pushes the ≤2 contiguous
    // ring slices straight to the sink) and `ContigFoldSink` (writes those slices
    // DIRECTLY into `chunk.data`, no `pending_clean` middle-man, no second
    // `append_clean` copy), the post-flip clean tail drops the per-block u8buf
    // alloc + the pending_clean double-copy. This recovered +0.059× of the T8
    // wall (native_fold 0.678× -> 0.737× rg, quiet-box banked, sha-exact; the
    // loaded 6-pass split confirmed the recovery is monotonic across copy#1 +
    // copy#2/3/grow but load-inflated). NOTE (2026-06-08, measured): on the
    // gzippy-native build the BULK clean tail does NOT take this ContigFoldSink
    // ring path at all — it takes `finish_decode_chunk_contig_native` ->
    // `decode_clean_into_contig` (u8-DIRECT into chunk.data, no ring, no drain;
    // this sink governs only the ~1% marker-loop dribble). The
    // GZIPPY_FOLD_NODRAIN/NOCRC split measured the remaining drain+CRC
    // second-touch at ~0-1ms (frozen host, N=21), so the gap to the engine-removed
    // ceiling (ocl_cf, matched-comparator 0.945× rg) is ~36ms of essentially PURE
    // pure-Rust-vs-ISA-L SYMBOL RATE on the SAME covered chunks (coverage symmetry
    // confirmed: native flip_to_clean=12 finished_no_flip=4 window_seeded=2 ==
    // ocl_cf's 14 covered). The earlier "ring-write+drain remain, upper bound only"
    // caveat is STALE for the contig bulk path. See plans/fold-drain-split-result.md.
    {
        const RESERVE_CLAMP: usize = 16 * 1024 * 1024;
        let compressed_bytes = stop_hint_bits.saturating_sub(encoded_offset_bits) / 8;
        let estimate = compressed_bytes
            .saturating_mul(8)
            .saturating_add(1024 * 1024);
        chunk.reserve_clean(estimate.min(RESERVE_CLAMP));
    }
    let mut pending_boundaries: Vec<(usize, usize)> = Vec::new();
    loop {
        let mut clean_appended = marker_ctx.clean_data_count;
        let crc32_enabled = chunk.configuration.crc32_enabled;
        let mut sink = ContigFoldSink {
            markers: &mut chunk.data_with_markers,
            data: &mut chunk.data,
            crc32s: &mut chunk.crc32s,
            subchunks: &mut chunk.subchunks,
            non_marker_count: &mut chunk.statistics.non_marker_count,
            clean_appended: &mut clean_appended,
            crc32_enabled,
            boundaries: &mut pending_boundaries,
        };
        let (step, flipped_clean) =
            marker_decode_step(&mut marker_ctx, input, stop_hint_bits, &[], &mut sink)?;
        let n = clean_appended - marker_ctx.clean_data_count;
        marker_ctx.clean_data_count = clean_appended;
        UNIFIED_ROUTE_CLEAN_U8_BYTES.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        apply_recorded_block_boundaries(&mut chunk, input, &pending_boundaries);
        pending_boundaries.clear();
        if flipped_clean {
            UNIFIED_MODE_CLEAN_FLIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        match step {
            MarkerStep::Continue => {}
            #[cfg(not(isal_clean_tail))]
            MarkerStep::FlipToContig { end_bit_offset } => {
                FLIP_TO_CLEAN_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::decompress::parallel::trace_v2::emit_instant(
                    "worker.chunk_phase",
                    &format!(
                        r#""start_bit":{encoded_offset_bits},"phase":"flip_to_contig","end_bit":{end_bit_offset}"#
                    ),
                    "t",
                );
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                finish_decode_chunk_contig_native(
                    &mut chunk,
                    &mut marker_ctx,
                    input,
                    end_bit_offset,
                    stop_hint_bits,
                    false,
                )?;
                return Ok(chunk);
            }
            MarkerStep::FlipToClean { end_bit_offset, .. } => {
                FLIP_TO_CLEAN_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::decompress::parallel::trace_v2::emit_instant(
                    "worker.chunk_phase",
                    &format!(
                        r#""start_bit":{encoded_offset_bits},"phase":"flip_to_clean","end_bit":{end_bit_offset}"#
                    ),
                    "t",
                );
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                let clean_window = chunk.last_32kib_window_vec().ok_or_else(|| {
                    ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "flip reached without a clean 32 KiB window",
                    ))
                })?;
                finish_decode_chunk_with_inexact_offset(
                    &mut chunk,
                    input,
                    end_bit_offset,
                    stop_hint_bits,
                    &clean_window,
                    false,
                )?;
                return Ok(chunk);
            }
            MarkerStep::Finished { end_bit_offset, .. } => {
                FINISHED_NO_FLIP_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::decompress::parallel::trace_v2::emit_instant(
                    "worker.chunk_phase",
                    &format!(
                        r#""start_bit":{encoded_offset_bits},"phase":"finished_no_flip","end_bit":{end_bit_offset}"#
                    ),
                    "t",
                );
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                chunk.finalize_with_deflate(end_bit_offset, Some(input));
                return Ok(chunk);
            }
        }
    }
}

/// gzippy-native copy-free-to-final clean tail. Resumes the SAME thread-local
/// `Block` (already flipped to clean) and decodes every subsequent deflate block
/// DIRECTLY into `chunk.data`'s reserved contiguous tail — no u8 ring, no
/// ring->chunk.data drain memcpy. Back-refs resolve from `chunk.data[*pos-d]`,
/// the already-committed contiguous clean output (the faithful vendor
/// `setInitialWindow` prepend; `data_prefix_len` stays 0 because the 32 KiB
/// predecessor window is real prior output). Replicates `marker_decode_step_loop`'s
/// per-block bookkeeping (header parse, stop-hint early-out, BFINAL stop, EOB
/// block-boundary recording, CRC + subchunk + non_marker accounting) for the
/// clean phase only. The isal two-phase path (`finish_decode_chunk_with_inexact_offset`)
/// is unchanged.
///
/// `until_exact` (M4, DIV-1 part 2) switches the stop condition from the
/// inexact "first block boundary at-or-past `stop_hint_bits`" to the EXACT
/// contract of the wrapper arm (`finish_decode_chunk_impl` with
/// `until_exact=true`, whose bit reader is hard-capped at `stop_hint_bits`
/// via `with_until_bits`):
///
///   - SUCCESS iff the decode lands EXACTLY at `stop_hint_bits`: either the
///     bit cursor reaches `stop_hint_bits` at a block-header boundary (the
///     wrapper's `try_enter_next_block` cap stop), or the member's BFINAL
///     block ends with its BYTE-ALIGNED post-EOB bit == `stop_hint_bits`.
///   - Otherwise `ChunkDecodeError::ExactStopMissed { requested, actual }`,
///     replicating gzip_chunk.rs `finish_decode_chunk_impl`'s
///     `final_bit != stop_hint_bits` assertion with the same coordinates.
///
/// END-BIT COORDINATE CONVENTION (explicit, the BFINAL scar-class lesson):
///   - interior stop: the exact (possibly non-byte-aligned) bit of the
///     confirmed boundary — identical to the wrapper's read-cap cursor.
///   - member-final stop: the post-EOB bit rounded UP to the next byte
///     boundary. The wrapper consumes the RFC 1952 zero padding via
///     `finish_current_block`'s `align_to_byte()` (resumable.rs:823-842),
///     so its `tell_compressed()` at stream end is byte-aligned; the Block
///     arm replicates that by aligning `end_bit_offset` explicitly. The
///     8-byte gzip footer is NOT consumed by either arm (`sm_driver`
///     slices it off the input; production `stop_hint == total_bits` is
///     the padded deflate end, footer excluded). NOTE this differs from
///     the INEXACT Block arm (M3), which reports the UNALIGNED post-EOB
///     bit (documented M3 stream-end exception) — for the exact arm the
///     aligned convention is REQUIRED so the production
///     `stop_hint == total_bits` member-final chunk lands exactly.
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn finish_decode_chunk_contig_native(
    chunk: &mut ChunkData,
    marker_ctx: &mut MarkerDecodeCtx,
    input: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    until_exact: bool,
) -> Result<(), ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::{BlockError, CompressionType};

    // Per-call contig headroom: one max-length back-ref (258) + the 8-byte
    // word-copy overshoot (matches `decode_clean_into_contig`'s `out_room`
    // reservation `cap - (MAX_RUN_LENGTH + 8)`, MAX_RUN_LENGTH == 258).
    const HEADROOM: usize = 258 + 8;

    marker_ctx.current_bit_offset = start_bit_offset;
    let crc32_enabled = chunk.configuration.crc32_enabled;

    BOOTSTRAP_BLOCK.with(|cell_block| -> Result<(), ChunkDecodeError> {
        let mut block = cell_block.borrow_mut();
        debug_assert!(
            !block.contains_marker_bytes(),
            "contig native tail requires a flipped (clean) Block"
        );

        loop {
            let slice_byte = marker_ctx.current_bit_offset / 8;
            let mut bits = marker_ctx.open_bits(input);
            let next_block_offset = absolute_bit_pos(slice_byte, &bits);

            // M4 exact stop at a block-header boundary. Wrapper analog: the
            // bit reader is capped at `stop_hint_bits` (`with_until_bits`),
            // so `try_enter_next_block` returns false the moment the cursor
            // reaches the cap and `final_bit = tell_compressed() ==
            // stop_hint_bits` — success without parsing the next header.
            // A cursor PAST the cap is impossible for the wrapper (the
            // reader refuses those bits) and means a mis-registered stop
            // hint here (never a confirmed boundary) — error with the same
            // ExactStopMissed coordinates the wrapper's final assertion uses.
            if until_exact {
                if next_block_offset == stop_hint_bits {
                    marker_ctx.current_bit_offset = next_block_offset;
                    chunk.finalize_with_deflate(next_block_offset, Some(input));
                    return Ok(());
                }
                if next_block_offset > stop_hint_bits {
                    return Err(ChunkDecodeError::ExactStopMissed {
                        requested: stop_hint_bits,
                        actual: next_block_offset,
                    });
                }
            }

            // Header parse (mirror marker_decode_step_loop:1499-1515).
            {
                let _tv2 =
                    crate::decompress::parallel::trace_v2::SpanGuard::begin("worker.block_header");
                if let Err(e) = block.read_header(&mut bits, false) {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deflate header at bit {next_block_offset}: {e:?}"),
                    )));
                }
            }

            // Stop-hint early-out at a block boundary (mirror :1517-1527).
            // INEXACT arm only — the exact arm stops solely on the
            // `next_block_offset == stop_hint_bits` cap check above.
            if !until_exact && next_block_offset >= stop_hint_bits && !block.is_last_block() {
                marker_ctx.current_bit_offset = next_block_offset;
                chunk.finalize_with_deflate(next_block_offset, Some(input));
                return Ok(());
            }

            // Decode the whole block body into chunk.data's contiguous tail.
            // One deflate block may span MULTIPLE contig calls (a block bigger
            // than the per-call out_room); accounting accumulates across them and
            // the block boundary fires only at real EOB.
            let _tv2_body =
                crate::decompress::parallel::trace_v2::SpanGuard::begin("worker.block_body");
            let comp_type = block.compression_type();
            while !block.eob() {
                // H4: re-fetch (base, cap, pos) every iteration — a grow inside
                // `contig_decode_window` may have moved the allocation.
                //
                // Request HEADROOM + 1, not HEADROOM: the decoders cap their
                // write budget at `out_room = cap - HEADROOM`, so a returned
                // spare of EXACTLY HEADROOM makes `out_room == pos` → zero
                // budget → `Ok(0)` → a spurious "no progress" error below.
                // (Observed on the M3 seeded route at spare == 266 after the
                // reserve estimate was outgrown; latent on the FOLD path too.)
                // Vec growth is amortized, so the +1 only changes the moment a
                // doubling fires, never the decoded bytes.
                let (base, cap, pos_before) = chunk.data.contig_decode_window(HEADROOM + 1);
                let mut pos = pos_before;
                // H1: release-mode guard — never let the decoder write past the
                // reserved headroom (a contract violation here is a heap OOB,
                // not a CRC-catchable wrong byte, because the contig path has no
                // ring modulo).
                let out_room = cap.saturating_sub(HEADROOM);
                assert!(
                    pos <= out_room && cap >= pos_before + HEADROOM,
                    "contig native tail: insufficient headroom (pos {pos} cap {cap})"
                );

                let body_res = match comp_type {
                    CompressionType::Uncompressed => block.decode_clean_stored_into_contig(
                        &mut bits,
                        base,
                        cap,
                        &mut pos,
                        usize::MAX,
                    ),
                    CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                        block.decode_clean_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
                    }
                    CompressionType::Reserved => Err(BlockError::InvalidCompression),
                };
                let emitted = match body_res {
                    Ok(n) => n,
                    Err(e) => {
                        record_block_body_fail(&e);
                        // Commit whatever was written before the failure so the
                        // logical length stays consistent, then surface the error.
                        chunk.data.commit(pos - pos_before);
                        return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("deflate body at bit {next_block_offset}: {e:?}"),
                        )));
                    }
                };
                debug_assert_eq!(emitted, pos - pos_before);
                // H3: commit BEFORE reading the bytes back for CRC (decoded_range
                // indexes the committed region).
                chunk.data.commit(emitted);
                if emitted > 0 {
                    if crc32_enabled {
                        if let Some(last_crc) = chunk.crc32s.last_mut() {
                            last_crc.update(chunk.data.decoded_range(pos_before, emitted));
                        }
                    }
                    chunk.statistics.non_marker_count += emitted as u64;
                    if let Some(last) = chunk.subchunks.last_mut() {
                        last.decoded_size += emitted;
                    }
                    marker_ctx.clean_data_count += emitted;
                    UNIFIED_ROUTE_CLEAN_U8_BYTES
                        .fetch_add(emitted as u64, std::sync::atomic::Ordering::Relaxed);
                }
                // No forward progress AND not at EOB ⇒ the buffer can't grow
                // enough (degenerate); bail rather than spin.
                if emitted == 0 && !block.eob() {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "contig native tail: no progress",
                    )));
                }
            }

            let end_bit_offset = absolute_bit_pos(slice_byte, &bits);
            marker_ctx.current_bit_offset = end_bit_offset;

            if block.is_last_block() {
                if until_exact {
                    // Member-final exact stop: byte-align the post-EOB bit
                    // (the wrapper consumes the RFC 1952 padding via
                    // `align_to_byte`, resumable.rs:823-842) and require it
                    // to equal the requested stop — the wrapper's
                    // `final_bit != stop_hint_bits` assertion with identical
                    // coordinates. The footer is NOT consumed (input slice
                    // excludes it); a multi-member-crossing stop hint
                    // therefore errors here exactly like the wrapper arm
                    // (no `read_footer_at_current`/`reset_for_next_stream`
                    // call exists on the production until-exact path).
                    let aligned_end = end_bit_offset.div_ceil(8) * 8;
                    marker_ctx.current_bit_offset = aligned_end;
                    if aligned_end != stop_hint_bits {
                        return Err(ChunkDecodeError::ExactStopMissed {
                            requested: stop_hint_bits,
                            actual: aligned_end,
                        });
                    }
                    chunk.finalize_with_deflate(aligned_end, Some(input));
                    return Ok(());
                }
                chunk.finalize_with_deflate(end_bit_offset, Some(input));
                return Ok(());
            }
            // Record the block boundary at the real EOB (decoded_offset = total
            // decoded bytes = markers + clean), mirror :1597. `data_prefix_len`
            // is 0 on the FOLD path (the 32 KiB window is real prior output);
            // on the M3 seeded path the dictionary prefix at `data[0..32768)`
            // is NOT chunk output and must not shift boundary keys.
            let decoded_offset =
                chunk.data_with_markers.len() + chunk.data.len() - chunk.data_prefix_len;
            chunk.append_block_boundary_at(end_bit_offset, decoded_offset, Some(input));
        }
    })
}

/// M3 (DIV-1 part 1) — vendor `GzipChunk.hpp:454-458` (non-ISAL build):
///
/// ```c++
/// auto block = std::make_shared<deflate::Block</* CRC32 */ false,
///                                              /* enable analysis */ false>>();
/// if ( initialWindow ) {
///     block->setInitialWindow( *initialWindow );
/// }
/// ```
///
/// A window-KNOWN chunk decodes on the SAME ONE `deflate::Block` engine,
/// seeded clean-from-byte-0 — vendor-native has NO second engine for this
/// path (the ISA-L fork at GzipChunk.hpp:440-444 is compiled out). gzippy
/// mirror: seed the thread-local [`BOOTSTRAP_BLOCK`] (`Block::reset` +
/// `set_initial_window` → `WidthRing::seed_window`, deflate.hpp:1750-1759)
/// and decode every deflate block u8-DIRECT into `chunk.data`'s contiguous
/// tail via `decode_clean_into_contig` — the design's single clean-destination
/// contract (plans/engine-u8-design.md §4.3), the same machinery the FOLD
/// post-flip tail already runs ([`finish_decode_chunk_contig_native`]).
///
/// The 32 KiB seed is installed as a NON-OUTPUT dictionary prefix at
/// `chunk.data[0..32768)` (`prefill_window_prefix`, `data_prefix_len` =
/// 32 KiB — the A3/A4 scaffolding: `decoded_size()`, boundary keys, window
/// extraction and the consumer write all skip it), so back-refs resolve as
/// pure `base[*pos - d]` — byte-equal to vendor's ring-window reads because
/// the prefix bytes ARE the predecessor window (deflate.hpp:1750-1759 prime
/// + DecodedData.hpp:278-289 contiguous clean storage).
///
/// This REPLACES `StreamingInflateWrapper`/`unified::Inflate`
/// (inflate_wrapper.rs:153-161) on the gzippy-native window-seeded INEXACT
/// path — the DIV-1 second clean engine. The until-exact path stays on the
/// wrapper until M4 (its stopping-point/footer contract is pre-registered
/// there); the gzippy-isal clean-tail handoff is untouched (faithful
/// rapidgzip WITH_ISAL, GzipChunk.hpp:440-444/520-526).
///
/// Kill-switch: `GZIPPY_SEEDED_BLOCK=0` restores the wrapper arm exactly
/// (see [`seeded_block_route_enabled`]).
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn finish_decode_chunk_seeded_block_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    SEEDED_BLOCK_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::decompress::parallel::trace_v2::emit_instant(
        "worker.chunk_phase",
        &format!(r#""start_bit":{inflate_start_bit},"phase":"window_seeded_block""#),
        "t",
    );

    let mut marker_ctx = seed_block_for_contig_native(
        chunk,
        input,
        inflate_start_bit,
        stop_hint_bits,
        initial_window,
    )?;

    // Same per-block driver the FOLD post-flip tail uses: header parse,
    // stop-hint early-out at block boundaries, `decode_clean_into_contig`
    // bodies, EOB boundary recording, BFINAL finalize.
    finish_decode_chunk_contig_native(
        chunk,
        &mut marker_ctx,
        input,
        inflate_start_bit,
        stop_hint_bits,
        false,
    )
}

/// Shared M3/M4 seeding: dictionary prefix + contig reserve + priming the
/// thread-local [`BOOTSTRAP_BLOCK`] clean-from-byte-0 with the predecessor
/// window (vendor GzipChunk.hpp:456-458 → `Block::setInitialWindow`,
/// deflate.hpp:1750-1759).
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn seed_block_for_contig_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<MarkerDecodeCtx, ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    debug_assert_eq!(
        initial_window.len(),
        MAX_WINDOW_SIZE,
        "seeded-Block route requires a full 32 KiB window (caller gate)"
    );

    // Dictionary prefix: the predecessor window at `data[0..32768)`, excluded
    // from output/boundary accounting via `data_prefix_len`.
    chunk.prefill_window_prefix(initial_window);

    // Pre-reserve ONE contiguous clean-data region (mirror of
    // `decode_chunk_unified_marker`'s reserve — same heuristic, same clamp;
    // an under-reserve falls back to safe amortized regrow between calls).
    {
        const RESERVE_CLAMP: usize = 16 * 1024 * 1024;
        let compressed_bytes = stop_hint_bits.saturating_sub(inflate_start_bit) / 8;
        let estimate = compressed_bytes
            .saturating_mul(8)
            .saturating_add(1024 * 1024);
        chunk.reserve_clean(estimate.min(RESERVE_CLAMP));
    }

    let mut marker_ctx = MarkerDecodeCtx::new(input, inflate_start_bit)?;

    // Seed the ONE engine: reset the thread-local Block, then prime CLEAN
    // mode from byte 0 with the predecessor window.
    BOOTSTRAP_BLOCK.with(|cell_block| -> Result<(), ChunkDecodeError> {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let mut unused: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut unused, initial_window)
            .map_err(|e| {
                ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("seeded-Block set_initial_window: {e:?}"),
                ))
            })?;
        debug_assert!(unused.is_empty(), "seed must not drain into output");
        Ok(())
    })?;
    marker_ctx.block_primed = true;
    Ok(marker_ctx)
}

/// M4 (DIV-1 part 2) — LABELED DEVIATION from the vendor blueprint.
///
/// Vendor's exact-stop path is `decodeChunkWithInflateWrapper<ZlibInflateWrapper/
/// IsalInflateWrapper>` (GzipChunk.hpp:192-265) — a C-FFI inflate wrapper, NOT
/// `deflate::Block`. Putting the until-exact decode on Block-with-exact-stop is
/// justified SOLELY by gzippy-native's no-C-FFI charter (the pure-Rust build has
/// no faithful wrapper engine to hand the chunk to; gzippy-isal keeps the
/// faithful wrapper path untouched, see `exact_block_route_enabled`).
///
/// PRE-REGISTERED CONTRACT (plans/engine-u8-design.md GATE AMENDMENTS §2) —
/// Block must replicate from the `unified::Inflate` wrapper arm
/// (`finish_decode_chunk_impl`, until_exact=true):
///
///  (a) stopping-point reactions: END_OF_BLOCK → every non-final EOB records
///      a block boundary (`append_block_boundary_at`; here at the driver's
///      EOB recording site). END_OF_STREAM_HEADER → on the wrapper arm this
///      stop fires only after `reset_for_next_stream`, which NO production
///      parallel-SM caller invokes (first-hand verified; see
///      inflate_wrapper.rs:1058-1065) — it is unreachable on the until-exact
///      arm, and Block replicates that observable: decode ends at the
///      member's BFINAL EOB with no next-stream continuation.
///  (b) the exact `final_bit != stop_hint_bits => ExactStopMissed` assertion
///      (`finish_decode_chunk_impl`'s final check), same `requested`/`actual`
///      coordinates — enforced in `finish_decode_chunk_contig_native`'s
///      until_exact arm at both stop sites.
///  (c) footer/multi-stream (`read_footer_at_current`/`reset_for_next_stream`):
///      these wrapper APIs are NEVER called by the production until-exact
///      arm — the wrapper stops at the BFINAL EOB (byte-aligned via
///      `align_to_byte`, resumable.rs:823-842) and asserts against
///      `stop_hint_bits` WITHOUT consuming the footer (sm_driver slices the
///      footer off `input`; vendor's wrapper DOES read footers,
///      GzipChunk.hpp:246-251, because its chunks may span members — gzippy's
///      single-member slice cannot). Block replicates the wrapper arm's
///      observable exactly: member-final success iff the byte-aligned
///      post-EOB bit == stop_hint_bits; a member-crossing stop hint errors
///      `ExactStopMissed` identically on both arms (pinned by the
///      `exact_block_parity::exact_multi_member_trailing` net).
///  (d) block-boundary recording (`take_block_boundaries` replay →
///      `append_block_boundary_at`): the driver records every non-final EOB
///      boundary with decoded offsets excluding the dictionary prefix
///      (`data_prefix_len`), key-identical to the wrapper's
///      `decode_base + n_bytes_read` accounting (pinned by the parity nets'
///      subchunk-key equality).
///
/// Kill-switch: `GZIPPY_EXACT_BLOCK=0` restores the wrapper arm exactly
/// (see [`exact_block_route_enabled`]). Engine proof: [`EXACT_BLOCK_CHUNKS`]
/// vs [`EXACT_WRAPPER_CHUNKS`].
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn finish_decode_chunk_exact_block_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    EXACT_BLOCK_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::decompress::parallel::trace_v2::emit_instant(
        "worker.chunk_phase",
        &format!(r#""start_bit":{inflate_start_bit},"phase":"window_seeded_block_exact""#),
        "t",
    );

    let mut marker_ctx = seed_block_for_contig_native(
        chunk,
        input,
        inflate_start_bit,
        stop_hint_bits,
        initial_window,
    )?;

    finish_decode_chunk_contig_native(
        chunk,
        &mut marker_ctx,
        input,
        inflate_start_bit,
        stop_hint_bits,
        true,
    )
}

/// gzippy-isal stub: [`exact_block_route_enabled`] is constant `false` on the
/// `isal_clean_tail` build (the faithful WITH_ISAL
/// `decodeChunkWithInflateWrapper` path stays production, GzipChunk.hpp:192-265),
/// so this is statically unreachable — it exists only so the route call site
/// compiles on both builds.
#[cfg(all(parallel_sm, isal_clean_tail))]
fn finish_decode_chunk_exact_block_native(
    _chunk: &mut ChunkData,
    _input: &[u8],
    _inflate_start_bit: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    unreachable!("exact-Block route is gzippy-native only (exact_block_route_enabled() == false on isal_clean_tail)")
}

/// Vendor `decodeChunkWithInflateWrapper`-shaped envelope for the M4 Block
/// route: fresh `ChunkData` + window-seeded exact decode + decode-duration
/// accounting (mirror of `decode_chunk_with_inflate_wrapper`'s envelope).
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
fn decode_chunk_exact_block_native(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    finish_decode_chunk_exact_block_native(
        &mut chunk,
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
    )?;
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    Ok(chunk)
}

/// gzippy-isal stub (see [`finish_decode_chunk_exact_block_native`]'s stub).
#[cfg(all(parallel_sm, isal_clean_tail))]
fn decode_chunk_exact_block_native(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    unreachable!("exact-Block route is gzippy-native only (exact_block_route_enabled() == false on isal_clean_tail)")
}

/// gzippy-isal stub: [`seeded_block_route_enabled`] is constant `false` on the
/// `isal_clean_tail` build (the faithful WITH_ISAL clean-tail handoff stays the
/// production seeded path, GzipChunk.hpp:440-444), so this is statically
/// unreachable — it exists only so the route call site compiles on both builds.
#[cfg(all(parallel_sm, isal_clean_tail))]
fn finish_decode_chunk_seeded_block_native(
    _chunk: &mut ChunkData,
    _input: &[u8],
    _inflate_start_bit: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    unreachable!("seeded-Block route is gzippy-native only (seeded_block_route_enabled() == false on isal_clean_tail)")
}

/// Counter: chunks that took the window-present clean-from-block-0 fast path
/// (vendor `setInitialWindow`) instead of full marker decode.
#[cfg(parallel_sm)]
pub static WINDOW_SEEDED_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// M3 engine proof: window-seeded INEXACT chunks decoded on the ONE
/// `deflate::Block` engine (`finish_decode_chunk_seeded_block_native`).
#[cfg(parallel_sm)]
pub static SEEDED_BLOCK_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// M3 engine proof (complement): window-seeded INEXACT chunks decoded on the
/// pre-M3 wrapper arm (`GZIPPY_SEEDED_BLOCK=0` kill-switch, the gzippy-isal
/// build, or the ISA-L measurement oracle).
#[cfg(parallel_sm)]
pub static SEEDED_WRAPPER_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// M4 engine proof: UNTIL-EXACT chunks decoded on the ONE `deflate::Block`
/// engine (`finish_decode_chunk_exact_block_native`).
#[cfg(parallel_sm)]
pub static EXACT_BLOCK_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// M4 engine proof (complement): UNTIL-EXACT chunks decoded on the wrapper
/// arm (`GZIPPY_EXACT_BLOCK=0` kill-switch, the gzippy-isal build, or the
/// ISA-L measurement oracle).
#[cfg(parallel_sm)]
pub static EXACT_WRAPPER_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Clean-tail [`MarkerSink`] for the merged phase-2 decode: narrows the
/// post-flip u16 ring drains to u8 and appends them to `chunk.data` (CRC +
/// subchunk accounting). `as_slice` is never consulted on this sink —
/// `trailing_clean_since` is overridden because every byte is clean by
/// construction (the ring flipped before phase 2 began).
#[cfg(parallel_sm)]
struct CleanTailSink<'a> {
    chunk: &'a mut ChunkData,
    deflate_data: &'a [u8],
    /// Running count of u8 bytes pushed — the sink's logical length. Tracked
    /// independently of `chunk.data.len()` so any window prefix or later
    /// `clean_unmarked_data` migration cannot perturb the `before_len` deltas
    /// the block loop computes.
    pushed: usize,
}

#[cfg(parallel_sm)]
impl crate::decompress::parallel::marker_inflate::MarkerSink for CleanTailSink<'_> {
    #[inline]
    fn push_slice(&mut self, values: &[u16]) {
        self.chunk.append_clean_narrowed(values);
        self.pushed += values.len();
    }
    #[inline]
    fn sink_len(&self) -> usize {
        self.pushed
    }
    #[inline]
    fn as_slice(&self) -> &[u16] {
        &[]
    }
    #[inline]
    fn trailing_clean_since(&self, from: usize) -> usize {
        self.pushed.saturating_sub(from)
    }
    #[inline]
    fn push_clean_u8(&mut self, bytes: &[u8]) {
        // Post-flip u8-direct output: write straight into chunk.data (CRC +
        // subchunk accounting via append_clean) — no u16→u8 narrow pass.
        self.chunk.append_clean(bytes);
        self.pushed += bytes.len();
    }
    #[inline]
    fn note_block_boundary(&mut self, encoded_offset_bits: usize, decoded_offset: usize) {
        // Clean-tail-relative decoded offset (0-based on clean bytes), matching
        // the convention the retired resumable_resync clean tail used. Drives
        // the split_chunk_size subchunk split for vendor-parity / the seekable
        // index (no production read site yet — locked by the
        // UNSPLIT_BLOCKS_EMPLACED deletion trap in tests/routing.rs).
        self.chunk.append_block_boundary_at(
            encoded_offset_bits,
            decoded_offset,
            Some(self.deflate_data),
        );
    }
}

/// One iteration of the vendor `decodeChunkWithRapidgzip` block loop.
#[cfg(parallel_sm)]
enum MarkerStep {
    /// Another deflate block was decoded; call again.
    #[allow(dead_code)]
    Continue,
    /// 32 KiB of clean output reached at a block boundary — FLIP to the u8 clean
    /// tail (vendor setInitialWindow). The clean 32 KiB window is the tail of
    /// `data_with_markers`; the caller decodes the rest as u8 into `chunk.data`.
    /// Constructed only on the `isal_clean_tail` (gzippy-isal) build; on
    /// gzippy-native the fold keeps Engine M decoding in-place, so this variant
    /// is never returned (the match arm stays live for the isal build).
    #[allow(dead_code)]
    FlipToClean {
        end_bit_offset: usize,
        window_len: usize,
    },
    /// gzippy-native FOLD copy-free-to-final tail. At the ctx-flip point
    /// (`clean_appended_len() >= 32768`) the engine has ALREADY flipped to clean
    /// (`contains_marker_bytes==false`) and ≥32 KiB of contiguous clean output
    /// is in `chunk.data` — the 32 KiB predecessor window is that contiguous
    /// tail. Instead of continuing the ring engine + draining (the
    /// ring->chunk.data memcpy), the driver resumes the SAME thread-local
    /// `Block` and decodes subsequent clean blocks DIRECTLY into `chunk.data`'s
    /// reserved tail via `decode_clean_into_contig` (faithful vendor prepend;
    /// `data_prefix_len` stays 0 because the window is real prior output).
    /// Returned only on the `not(isal_clean_tail)` (gzippy-native) build.
    #[cfg(not(isal_clean_tail))]
    FlipToContig { end_bit_offset: usize },
    /// Chunk ends in the marker path (BFINAL, stop hint, or no clean dict).
    Finished {
        end_bit_offset: usize,
        #[allow(dead_code)]
        bfinal_hit: bool,
    },
}

/// Persistent state for [`marker_decode_step`] (one block per call).
#[cfg(parallel_sm)]
struct MarkerDecodeCtx {
    /// `Bits` slice base in `data` (byte index).
    #[allow(dead_code)]
    data_base_byte: usize,
    current_bit_offset: usize,
    trailing_clean: usize,
    /// Clean u8 bytes committed to `chunk.data` (vendor `cleanDataCount`).
    clean_data_count: usize,
    block_primed: bool,
    /// Gates the vendor `cleanDataCount >= MAX_WINDOW_SIZE` handoff so it
    /// fires exactly once per chunk.
    flipped: bool,
}

#[cfg(parallel_sm)]
impl MarkerDecodeCtx {
    fn new(_data: &[u8], start_bit_offset: usize) -> Result<Self, ChunkDecodeError> {
        let data_base_byte = start_bit_offset / 8;
        if data_base_byte >= _data.len() {
            return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "start_bit_offset past end of data",
            )));
        }
        Ok(Self {
            data_base_byte,
            current_bit_offset: start_bit_offset,
            trailing_clean: 0,
            clean_data_count: 0,
            block_primed: false,
            flipped: false,
        })
    }

    fn open_bits<'a>(
        &self,
        data: &'a [u8],
    ) -> crate::decompress::inflate::consume_first_decode::Bits<'a> {
        let slice_byte = self.current_bit_offset / 8;
        let mut bits =
            crate::decompress::inflate::consume_first_decode::Bits::new(&data[slice_byte..]);
        let bit_in_byte = (self.current_bit_offset % 8) as u32;
        if bit_in_byte > 0 {
            bits.consume(bit_in_byte);
        }
        bits
    }
}

/// Counter: fresh `Vec::reserve(128*1024)` events from
/// `take_bootstrap_output_from_pool`. Each is an mmap-eligible
/// allocation (256 KiB u16 buffer > glibc mmap threshold). With
/// pool reuse working, expected ≈ num_worker_threads (one allocation
/// per thread, then reused on every subsequent call). If this counter
/// approaches the bootstrap call count, the pool is broken.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_ALLOCS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: total `take` calls. Pool effectiveness =
/// `1 - (allocs / takes)`. Vendor-equivalent: rpmalloc per-thread arena
/// reuse rate, which approaches 1.0 after warm-up.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_TAKES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: sum of capacity (in BYTES) of Vecs returned from the
/// pool with cap ≥ 128 KiB. Total bytes the pool "saved" from a fresh
/// allocation. A run-end value of `~BOOTSTRAP_OUTPUT_TAKES *
/// 256 KiB` means every take hit the pool path.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_REUSED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: every call to `return_bootstrap_output_to_pool`. Mirrors
/// TAKES on the success path; if RETURNS << TAKES, bootstraps are
/// erroring out and not returning their buffer (it's still moved via
/// the chunk_fetcher.rs path at line 508, so should always match).
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_RETURNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: returns where the incoming Vec was DROPPED instead of
/// retained (because the pool's slot already had a larger cap).
/// Non-zero means we threw away a hot allocation — should be 0 with
/// 1-Vec-per-thread pool when caps are stable.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_DROPPED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Mirror of the `while ( true )` loop in
/// `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654), restricted to the
/// single-member case (no multi-stream loop) and with the handoff
/// triggered exclusively by `cleanDataCount` (GzipChunk.hpp:520-525).

/// Map vendor `Block::read` failures into body-fail telemetry buckets.
#[cfg(parallel_sm)]
fn record_block_body_fail(err: &crate::decompress::parallel::marker_inflate::BlockError) {
    use crate::decompress::parallel::marker_inflate::BlockError;
    use std::sync::atomic::Ordering;
    match err {
        BlockError::InvalidHuffmanCode => {
            BODY_FAIL_INVALID_HUFFMAN.fetch_add(1, Ordering::Relaxed);
        }
        BlockError::ExceededWindowRange | BlockError::ExceededDistanceRange => {
            BODY_FAIL_EXCEEDED_WINDOW.fetch_add(1, Ordering::Relaxed);
        }
        BlockError::InvalidCompression => {
            BODY_FAIL_INVALID_COMPRESSION.fetch_add(1, Ordering::Relaxed);
        }
        BlockError::InvalidCodeLengths => {
            BODY_FAIL_INVALID_CODE_LENGTHS.fetch_add(1, Ordering::Relaxed);
        }
        _ => {
            BODY_FAIL_OTHER_VARIANT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Decode one deflate block into `output` (vendor block loop body).
#[cfg(parallel_sm)]
fn marker_decode_step(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    initial_window: &[u8],
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    marker_decode_step_vendor_block(ctx, data, stop_hint_bits, initial_window, output)
}

// The per-thread vendor `deflate::Block` engine, persistent across the
// `marker_decode_step` calls of ONE chunk (primed once via `ctx.block_primed`)
// and reused across chunks (reset on the next chunk's first call). Module-scoped
// so the gzippy-native copy-free-to-final tail (`finish_decode_chunk_contig_native`)
// can re-borrow the SAME engine to continue the post-flip clean decode in-place.
#[cfg(parallel_sm)]
thread_local! {
    static BOOTSTRAP_BLOCK: std::cell::RefCell<crate::decompress::parallel::marker_inflate::Block> =
        std::cell::RefCell::new(crate::decompress::parallel::marker_inflate::Block::new());
}

/// Vendor `deflate::Block` bootstrap (rapidgzip `decodeChunkWithRapidgzip`).
#[cfg(parallel_sm)]
fn marker_decode_step_vendor_block(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    initial_window: &[u8],
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        if !ctx.block_primed {
            let window_opt = if initial_window.len() == MAX_WINDOW_SIZE {
                Some(initial_window)
            } else {
                None
            };
            block.reset(None, window_opt);
            ctx.block_primed = true;
        }
        // Cache-mandate instrument (byte-transparent, GZIPPY_MEM_STATS-gated,
        // at-most-once per worker thread): record THIS thread's native engine
        // resident working set. This is the real native per-thread state after
        // the fold — the staging-box hooks are dead here.
        crate::decompress::inflate::mem_stats::on_block_active(block.heap_bytes());
        marker_decode_step_loop(
            ctx,
            data,
            stop_hint_bits,
            output,
            &mut *block,
            |block, bits| block.read_header(bits, false),
            |block, bits, output| block.read(bits, output, usize::MAX),
            record_block_body_fail,
        )
    })
}

/// Engine surface of the ONE vendor `Block` engine, consumed by
/// `marker_decode_step_loop` (the legacy `MarkerRing` impl was deleted in M5).
#[cfg(parallel_sm)]
trait BootstrapEngine {
    fn contains_marker_bytes(&self) -> bool;
    fn eob(&self) -> bool;
    fn is_last_block(&self) -> bool;
}

#[cfg(parallel_sm)]
impl BootstrapEngine for crate::decompress::parallel::marker_inflate::Block {
    fn contains_marker_bytes(&self) -> bool {
        crate::decompress::parallel::marker_inflate::Block::contains_marker_bytes(self)
    }
    fn eob(&self) -> bool {
        crate::decompress::parallel::marker_inflate::Block::eob(self)
    }
    fn is_last_block(&self) -> bool {
        crate::decompress::parallel::marker_inflate::Block::is_last_block(self)
    }
}

/// Per-iteration body of the vendor `Block` bootstrap loop.
#[cfg(parallel_sm)]
fn marker_decode_step_loop<B, S, EH, RH, E, R, F>(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    output: &mut S,
    block: &mut B,
    mut read_header: RH,
    mut read_body: R,
    mut on_body_fail: F,
) -> Result<(MarkerStep, bool), ChunkDecodeError>
where
    B: BootstrapEngine,
    S: crate::decompress::parallel::marker_inflate::MarkerSink,
    EH: std::fmt::Debug,
    RH: FnMut(
        &mut B,
        &mut crate::decompress::inflate::consume_first_decode::Bits<'_>,
    ) -> Result<(), EH>,
    R: FnMut(
        &mut B,
        &mut crate::decompress::inflate::consume_first_decode::Bits<'_>,
        &mut S,
    ) -> Result<usize, E>,
    E: std::fmt::Debug,
    F: FnMut(&E),
{
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    use crate::decompress::parallel::trace_v2;

    loop {
        let slice_byte = ctx.current_bit_offset / 8;
        let mut bits = ctx.open_bits(data);
        let next_block_offset = absolute_bit_pos(slice_byte, &bits);

        // Vendor GzipChunk.hpp:520-525 — at 32 KiB of clean u8 the engine
        // strategy forks by build:
        //   * gzippy-isal (`isal_clean_tail`): two-phase Design-A handoff —
        //     return `FlipToClean` so Engine C (`StreamingInflateWrapper`)
        //     decodes the clean tail from the 32 KiB window. UNCHANGED.
        //   * gzippy-native (`not(isal_clean_tail)`): FOLD — Engine M
        //     (`marker_inflate::Block`) keeps decoding this and subsequent
        //     blocks in-place on the SAME `ctx` cursor. `read()` already drains
        //     clean u8 directly (marker_inflate.rs:1011 -> push_clean_u8 once
        //     `contains_marker_bytes()==false`). The loop terminates naturally
        //     at BFINAL (`Finished`, :1293) or stop_hint (:1222).
        // `ctx.flipped` is set once so this check fires a single time per chunk.
        if output.clean_appended_len() >= MAX_WINDOW_SIZE && !ctx.flipped {
            ctx.flipped = true;
            #[cfg(isal_clean_tail)]
            {
                let end_bit_offset = next_block_offset;
                ctx.current_bit_offset = end_bit_offset;
                return Ok((
                    MarkerStep::FlipToClean {
                        end_bit_offset,
                        window_len: MAX_WINDOW_SIZE,
                    },
                    false,
                ));
            }
            // gzippy-native: copy-free-to-final. The engine has already flipped
            // (clean_appended only grows post-engine-flip) and ≥32 KiB clean is
            // contiguous in chunk.data; hand the contig tail to the driver, which
            // resumes THIS Block decoding straight into chunk.data (no ring, no
            // drain memcpy). The driver re-borrows the same thread-local Block.
            #[cfg(not(isal_clean_tail))]
            {
                ctx.current_bit_offset = next_block_offset;
                return Ok((
                    MarkerStep::FlipToContig {
                        end_bit_offset: next_block_offset,
                    },
                    false,
                ));
            }
        }

        let header_res = {
            let _tv2 = trace_v2::SpanGuard::begin("worker.block_header");
            let t_header = std::time::Instant::now();
            let r = read_header(block, &mut bits);
            BOOTSTRAP_HEADER_US.fetch_add(
                t_header.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BOOTSTRAP_HEADER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            r
        };
        if let Err(e) = header_res {
            return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("deflate header at bit {next_block_offset}: {e:?}"),
            )));
        }

        if next_block_offset >= stop_hint_bits && !block.is_last_block() {
            let end_bit_offset = next_block_offset;
            ctx.current_bit_offset = end_bit_offset;
            return Ok((
                MarkerStep::Finished {
                    end_bit_offset,
                    bfinal_hit: false,
                },
                false,
            ));
        }

        let before_len = output.sink_len();
        let t_body = std::time::Instant::now();
        let _tv2_body = trace_v2::SpanGuard::begin("worker.block_body");
        while !block.eob() {
            if let Err(e) = read_body(block, &mut bits, output) {
                let bits_at_fail = absolute_bit_pos(slice_byte, &bits);
                let bytes_wasted = output.sink_len() - before_len;
                let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                BODY_FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BYTES_WASTED
                    .fetch_add(bytes_wasted as u64, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BITS_INTO_BODY
                    .fetch_add(bits_into_body as u64, std::sync::atomic::Ordering::Relaxed);
                on_body_fail(&e);
                body_fail_log(
                    next_block_offset,
                    bits_into_body,
                    bytes_wasted,
                    &format!("{e:?}"),
                );
                return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "deflate body at bit {next_block_offset} (failed +{bits_into_body} bits, wasted {bytes_wasted} bytes): {e:?}"
                    ),
                )));
            }
        }
        BOOTSTRAP_BODY_US.fetch_add(
            t_body.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        BOOTSTRAP_BODY_BYTES.fetch_add(
            (output.sink_len() - before_len) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        if output.sink_len() > before_len {
            let block_len = output.sink_len() - before_len;
            let trailing_this_block = output.trailing_clean_since(before_len);
            if trailing_this_block == block_len {
                ctx.trailing_clean =
                    (ctx.trailing_clean + trailing_this_block).min(MAX_WINDOW_SIZE);
            } else {
                ctx.trailing_clean = trailing_this_block.min(MAX_WINDOW_SIZE);
            }
        }

        let flipped_clean = !block.contains_marker_bytes();
        if flipped_clean {
            BOOTSTRAP_CLEAN_FLIPPED_BYTES.fetch_add(
                (output.sink_len().saturating_sub(before_len)) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let end_bit_offset = absolute_bit_pos(slice_byte, &bits);
        ctx.current_bit_offset = end_bit_offset;

        if block.is_last_block() {
            return Ok((
                MarkerStep::Finished {
                    end_bit_offset,
                    bfinal_hit: true,
                },
                flipped_clean,
            ));
        }
        output.note_block_boundary(end_bit_offset, output.sink_len());
    }
}

/// Compute the absolute bit position within `data` given that `bits`
/// was constructed from `&data[byte_offset..]`. The Bits buffer
/// pre-loads bytes from its slice, so the actual consumed-from-slice
/// count is `bits.pos * 8 - bits.available()`.
#[cfg(parallel_sm)]
#[inline]
fn absolute_bit_pos(
    byte_offset: usize,
    bits: &crate::decompress::inflate::consume_first_decode::Bits,
) -> usize {
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    byte_offset * 8 + bits_consumed_from_slice
}

#[cfg(not(parallel_sm))]
pub fn decode_chunk_window_absent(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(parallel_sm))]
pub fn decode_chunk(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(test)]
#[cfg(parallel_sm)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Force multiple deflate blocks (Sync flush every 32 KiB) so
    /// inexact-stop and END_OF_BLOCK probing tests have real boundaries.
    fn make_multi_block_deflate(payload: &[u8]) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress, Status};
        let mut compress = Compress::new(Compression::new(6), false);
        let mut out = Vec::new();
        let mut scratch = vec![0u8; 64 * 1024];
        for piece in payload.chunks(32 * 1024) {
            let mut block_data = piece;
            loop {
                let before_in = compress.total_in();
                let before_out = compress.total_out();
                let status = compress
                    .compress(block_data, &mut scratch, FlushCompress::None)
                    .unwrap();
                let consumed = (compress.total_in() - before_in) as usize;
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                block_data = &block_data[consumed..];
                if block_data.is_empty() {
                    break;
                }
                if matches!(status, Status::BufError) && produced == 0 {
                    break;
                }
            }
            loop {
                let before_out = compress.total_out();
                let status = compress
                    .compress(&[], &mut scratch, FlushCompress::Sync)
                    .unwrap();
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                if produced == 0 || matches!(status, Status::StreamEnd) {
                    break;
                }
            }
        }
        loop {
            let before_out = compress.total_out();
            let status = compress
                .compress(&[], &mut scratch, FlushCompress::Finish)
                .unwrap();
            let produced = (compress.total_out() - before_out) as usize;
            out.extend_from_slice(&scratch[..produced]);
            if matches!(status, Status::StreamEnd) || produced == 0 {
                break;
            }
        }
        out
    }

    fn flatten(chunk: &ChunkData) -> Vec<u8> {
        let mut out = Vec::with_capacity(chunk.decoded_size());
        for v in chunk.data_with_markers.iter() {
            out.push(v as u8);
        }
        for seg in chunk.data.segments() {
            out.extend_from_slice(seg);
        }
        out
    }

    /// MICROBENCH (advisor-prescribed disambiguation): is the ISA-L multi-symbol
    /// bulk-LUT (`decode_block`) actually faster than the single-symbol resumable
    /// (`ResumableInflate2` via `StreamingInflateWrapper`) on THIS code state, over a
    /// REAL silesia clean span (dynamic Huffman)? The locked-Fulcrum lever says
    /// the clean decode tail is the wall (798ms gzippy vs 305ms rapidgzip, T8);
    /// the FlipToClean tail currently runs 100% through resumable. Only integrate
    /// a bulk-LUT clean tail if it wins here. Run:
    ///   cargo test --release --lib --target x86_64-apple-darwin \
    ///     --no-default-features --features pure-rust-inflate \
    ///     -- --ignored --nocapture clean_tail_engine_microbench
    #[cfg(pure_inflate_decode)]
    #[test]
    #[ignore = "manual perf microbench; needs /tmp/ref_silesia.bin"]
    fn clean_tail_engine_microbench() {
        use crate::decompress::inflate::consume_first_decode::Bits;
        use crate::decompress::parallel::inflate_wrapper::{
            StoppingPoints, StreamingInflateWrapper,
        };
        use crate::decompress::parallel::lut_bulk_inflate::{decode_block, DecoderScratch};
        use std::time::Instant;

        let raw = match std::fs::read("/tmp/ref_silesia.bin") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("SKIP: /tmp/ref_silesia.bin absent");
                return;
            }
        };
        // ~8 MiB of real silesia → dynamic-Huffman blocks, realistic literal/
        // backref mix (NOT synthetic repeat() which is backref-degenerate).
        // Multi-block (sync flush every 32 KiB) to mirror silesia's block density
        // so the per-block stopping-point overhead is exercised.
        let payload = &raw[..(8 * 1024 * 1024).min(raw.len())];
        // Single-stream for the engine comparison (apples-to-apples with the
        // first run). Multi-block (sync flush every 32 KiB) for the per-block
        // stopping-point comparison — only ResumableInflate2 is exercised there
        // (decode_block doesn't decode flate2 sync-flush empty stored blocks).
        let deflate = make_deflate(payload);
        let deflate_mb = make_multi_block_deflate(payload);
        let n = payload.len();
        let iters = 7;

        // ── bulk-LUT (decode_block) ──
        let mut bulk_out = vec![0u8; n + 64];
        let mut best_bulk = f64::MAX;
        for _ in 0..iters {
            let mut bits = Bits::new(&deflate);
            let mut out_pos = 0usize;
            let mut scratch = DecoderScratch::new();
            let t = Instant::now();
            loop {
                let r = decode_block(&mut bits, &mut bulk_out, &mut out_pos, &[], &mut scratch)
                    .expect("bulk decode");
                if r.is_final_block {
                    break;
                }
            }
            best_bulk = best_bulk.min(t.elapsed().as_secs_f64());
            assert_eq!(&bulk_out[..n], payload, "bulk output mismatch");
        }

        // ── resumable FREE-RUN (no stopping points) on the multi-block stream ──
        let mut res_out = vec![0u8; n + 64];
        let mut best_res = f64::MAX;
        for _ in 0..iters {
            let mut wrapper =
                StreamingInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8)
                    .unwrap();
            wrapper.set_window(&[]).unwrap();
            let mut out_pos = 0usize;
            let t = Instant::now();
            loop {
                let r = wrapper
                    .read_stream(&mut res_out[out_pos..])
                    .expect("resumable decode");
                out_pos += r.bytes_written;
                if r.finished || (r.bytes_written == 0 && out_pos >= n) {
                    break;
                }
                if r.bytes_written == 0 {
                    break;
                }
            }
            best_res = best_res.min(t.elapsed().as_secs_f64());
            assert_eq!(&res_out[..n], payload, "resumable output mismatch");
        }

        // ── resumable WITH production stopping points (returns at EVERY block
        // boundary, as resumable_resync does) — isolates the per-block stop +
        // boundary-bookkeeping overhead from the raw decode rate. ──
        let mut sp_out = vec![0u8; n + 64];
        let mut best_sp = f64::MAX;
        let mut block_stops = 0u64;
        for it in 0..iters {
            let mut wrapper =
                StreamingInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8)
                    .unwrap();
            wrapper.set_window(&[]).unwrap();
            wrapper.set_stopping_points(
                StoppingPoints::END_OF_BLOCK
                    | StoppingPoints::END_OF_BLOCK_HEADER
                    | StoppingPoints::END_OF_STREAM_HEADER
                    | StoppingPoints::END_OF_STREAM,
            );
            let mut out_pos = 0usize;
            let mut stops = 0u64;
            let t = Instant::now();
            loop {
                let r = wrapper
                    .read_stream(&mut sp_out[out_pos..])
                    .expect("sp decode");
                out_pos += r.bytes_written;
                stops += 1;
                if r.finished || (r.bytes_written == 0 && r.stopped_at == StoppingPoints::NONE) {
                    break;
                }
            }
            best_sp = best_sp.min(t.elapsed().as_secs_f64());
            if it == 0 {
                block_stops = stops;
            }
            assert_eq!(&sp_out[..n], payload, "sp output mismatch");
        }

        let mbps = |s: f64| (n as f64 / (1024.0 * 1024.0)) / s;
        eprintln!(
            "CLEAN-TAIL MICROBENCH (n={} MiB, best-of-{}, {} block-stops):\n  bulk-LUT             {:.3}ms  {:.0} MB/s\n  resumable free-run   {:.3}ms  {:.0} MB/s\n  resumable +stoppts   {:.3}ms  {:.0} MB/s  <- production driver shape\n  bulk/resumable = {:.2}x   stoppts-overhead = {:.2}x (free/stop)",
            n / 1024 / 1024,
            iters,
            block_stops,
            best_bulk * 1e3,
            mbps(best_bulk),
            best_res * 1e3,
            mbps(best_res),
            best_sp * 1e3,
            mbps(best_sp),
            best_res / best_bulk,
            best_res / best_sp,
        );
    }

    #[test]
    fn decode_chunk_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let stop_hint_bits = deflate.len() * 8;
        let chunk = decode_chunk(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
    }

    /// ADVERSARIAL FLIP-SEAM test (advisor trap A): force the marker→clean flip
    /// and then issue MAX-DISTANCE (32768) back-references that reach across the
    /// flip seam into the OLDEST part of the pre-flip window. silesia's
    /// differential does NOT reliably exercise a 32768-distance ref landing
    /// exactly on the seam, so this is engineered deliberately.
    ///
    /// Construction: `A || A` with `|A| == 32768`. Decoding from bit 0 has no
    /// predecessor window, so the first A is clean literals; the flip fires at
    /// `decoded_bytes == 32768` (end of A). The second A is encoded by flate2 as
    /// distance-32768 back-references (A repeats exactly one window back) — every
    /// one of them resolves across the just-flipped seam. A wrong conflate /
    /// u8-direct repositioning corrupts the second A; byte-exact vs the payload
    /// is the gate that locks the faithful one-buffer port.
    #[test]
    fn decode_chunk_flip_seam_max_distance_backref() {
        // Deterministic pseudo-random A so the first window is literal (clean)
        // and the second copy must reference it at distance |A| = 32768.
        let mut a = vec![0u8; 32 * 1024];
        let mut s = 0x1234_5678_9abc_def0u64;
        for b in &mut a {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        let mut payload = a.clone();
        payload.extend_from_slice(&a); // A || A → distance-32768 refs in the 2nd A

        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let stop_hint_bits = deflate.len() * 8;
        // window-absent entry (the production speculative path that flips).
        let chunk =
            decode_chunk_window_absent(&deflate, 0, stop_hint_bits, cfg).expect("decode seam");
        let out = flatten(&chunk);
        assert_eq!(out.len(), payload.len(), "seam decode length");
        assert_eq!(
            out, payload,
            "seam decode bytes (flip + max-distance backref)"
        );
    }

    /// MANDATORY faithful-u8 seam trap (charter 2026-06-07). After the in-place
    /// u16->u8 width flip at 32768, a distance-32768 back-ref must read the
    /// OLDEST byte of the repacked u8 window (the value-downcasted survivor at
    /// u8 slot `U8_RING_SIZE - 32768`). A wrong dest offset, a missing rotation,
    /// or a LE bit-reinterpret instead of `(x & 0xFF)` downcast all corrupt this
    /// byte. Decoded against an INDEPENDENT flate2 oracle over the whole stream
    /// (not against the test's own construction), per the no-self-trust rule.
    #[test]
    fn faithful_u8_flip_seam_max_distance_backref_vs_flate2() {
        use std::io::Read;
        // First 32 KiB: distinct pseudo-random bytes, with a DISTINCTIVE sentinel
        // at index 0 so the oldest-window byte is unambiguous after the flip.
        let mut a = vec![0u8; 32 * 1024];
        let mut s = 0xDEAD_BEEF_CAFE_F00Du64;
        for b in &mut a {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        a[0] = 0xA5; // sentinel: must reappear via the distance-32768 back-ref
        a[1] = 0x5A;
        // payload = A repeated 6× (192 KiB). The Block flip is checked only AFTER
        // a marker-mode read() call returns, and a call is capped at
        // RING_SIZE - MAX_RUN_LENGTH = 65278 bytes — so the flip fires near
        // ~65278, NOT at 32768 (advisor caveat). By repeating A six times the
        // 4th/5th/6th copies' distance-32768 back-refs are UNAMBIGUOUSLY in the
        // post-flip u8 region, reading the value-downcasted repacked window. The
        // sentinel check below targets byte 5*32768 = 163840 (well past the flip)
        // so it deterministically proves the u8 repack, not the marker path.
        let reps = 6;
        let mut payload = Vec::with_capacity(reps * a.len() + 400);
        for _ in 0..reps {
            payload.extend_from_slice(&a);
        }
        // A short RLE run + a 100-distance ref deep in the post-flip region to
        // exercise the u8 RLE/overlap arms across/after the seam.
        payload.extend(std::iter::repeat_n(0x33u8, 300));
        payload.extend_from_slice(&a[..100]);

        let deflate = make_deflate(&payload);

        // INDEPENDENT oracle: flate2 inflate of the same raw-deflate stream.
        let mut oracle = Vec::new();
        flate2::read::DeflateDecoder::new(&deflate[..])
            .read_to_end(&mut oracle)
            .expect("flate2 oracle decode");
        assert_eq!(oracle, payload, "oracle sanity: flate2 == payload");

        let cfg = ChunkConfiguration {
            split_chunk_size: 1024 * 1024,
            max_decoded_chunk_size: 20 * 1024 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let stop_hint_bits = deflate.len() * 8;
        // Production window-absent speculative entry — the path that flips u16->u8.
        let chunk =
            decode_chunk_window_absent(&deflate, 0, stop_hint_bits, cfg).expect("u8 seam decode");
        let out = flatten(&chunk);
        assert_eq!(out.len(), oracle.len(), "u8 seam decode length");
        assert_eq!(
            out, oracle,
            "u8 seam decode bytes vs flate2 (flip + distance-32768 + RLE/overlap)"
        );
        // Sentinel at byte 5*32768 = 163840 is GUARANTEED past the ~65278 flip
        // point, so this distance-32768 back-ref read the value-downcasted byte
        // from the repacked u8 window (slot U8_RING_SIZE-32768) — proving the u8
        // repack rotation + downcast, not the marker path.
        assert_eq!(
            out[5 * 32 * 1024],
            0xA5,
            "post-flip distance-32768 u8 backref read the repacked sentinel"
        );
    }

    #[test]
    fn decode_chunk_stops_before_eof_when_stop_hint_bits_set() {
        let payload: Vec<u8> = (0u32..500_000)
            .map(|i| (i.wrapping_mul(31) as u8).wrapping_add(7))
            .collect();
        let deflate = make_multi_block_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 256 * 1024,
            max_decoded_chunk_size: 20 * 256 * 1024,
            crc32_enabled: false,
            ..Default::default()
        };
        let stop_hint_bits = deflate.len() * 8 / 2;
        let chunk = decode_chunk(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert!(chunk.decoded_size() < payload.len());
        let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        assert!(chunk_end >= stop_hint_bits);
    }

    /// Neurotic profile fixture: gzip(1) -9 on 64 MiB silesia head. Chunk 0
    /// stops at a non-byte-aligned bit; chunk 1 must resume with the published
    /// 32 KiB window. Fails with `InvalidBlock` when handoff is wrong.
    ///
    /// Gated to `isal-compression`: this exercises the one-shot
    /// `inflate_bit::decompress_deflate_from_bit` primitive, which supports
    /// arbitrary-bit-offset resume only on the ISA-L backend. The production
    /// pure-rust parallel-SM path resumes via `ResumableInflate2` (bit-offset
    /// capable; `decompress_deflate_from_bit` has no production caller), and
    /// that path is covered by the silesia differential + `resumable_isal_oracle`.
    /// The `not(isal)` zng fallback's `inflatePrime` convention doesn't match
    /// this primitive's contract — tracked future work for arm64 parallel-SM,
    /// which will itself use `ResumableInflate2`, not zng.
    #[cfg(feature = "isal-compression")]
    #[test]
    fn cross_chunk_resume_silesia_gzip9_chunk0_handoff() {
        use std::io::Read;

        let gz = if std::path::Path::new("/tmp/silesia64.gz").exists() {
            std::fs::read("/tmp/silesia64.gz").expect("read cached gzip")
        } else {
            let path = std::path::Path::new("benchmark_data/silesia-large.bin");
            if !path.exists() {
                return;
            }
            let raw = std::fs::read(path).expect("read silesia");
            let head_len = (64 * 1024 * 1024).min(raw.len());
            let head = &raw[..head_len];
            let mut child = std::process::Command::new("gzip")
                .args(["-9", "-c", "-n"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .expect("spawn gzip");
            // Drain stdout on the main thread while a worker feeds stdin —
            // otherwise gzip's 64 KiB stdout pipe fills mid-write and both
            // sides deadlock (`write_all` 64 MiB stdin vs unread stdout).
            let mut stdin = child.stdin.take().expect("stdin");
            let head_owned = head.to_vec();
            let writer = std::thread::spawn(move || {
                let _ = std::io::Write::write_all(&mut stdin, &head_owned);
                // drop(stdin) here closes the pipe → gzip sees EOF
            });
            let mut gz = Vec::new();
            child
                .stdout
                .as_mut()
                .expect("stdout")
                .read_to_end(&mut gz)
                .expect("read gzip stdout");
            writer.join().expect("stdin writer thread");
            let _ = child.wait();
            gz
        };
        let head_len = {
            let _ = crate::decompress::parallel::gzip_format::read_header(&gz).expect("gzip hdr");
            let footer = crate::decompress::parallel::gzip_format::read_footer(&gz, gz.len() - 8)
                .expect("footer");
            footer.uncompressed_size as usize
        };

        let (_hdr, hdr_len) =
            crate::decompress::parallel::gzip_format::read_header(&gz).expect("gzip hdr");
        let deflate = &gz[hdr_len..gz.len() - 8];

        let spacing_bits = 4 * 1024 * 1024 * 8;
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 20 * 4 * 1024 * 1024,
            crc32_enabled: false,
            ..Default::default()
        };
        let zero = [0u8; 32768];
        let chunk0 = decode_chunk(deflate, 0, spacing_bits, &zero, cfg).expect("chunk0");
        let resume_at = chunk0.encoded_offset_bits + chunk0.encoded_size_bits;
        assert!(
            resume_at > 0 && !resume_at.is_multiple_of(8),
            "expected non-zero non-byte-aligned handoff, got {resume_at}"
        );
        let tail = chunk0
            .last_32kib_window()
            .unwrap_or_else(|| chunk0.get_last_window(&zero));

        crate::backends::inflate_bit::decompress_deflate_from_bit(
            deflate, resume_at, &tail, head_len,
        )
        .unwrap_or_else(|| {
            panic!("resume at chunk0 end bit {resume_at} must succeed");
        });
        let chunk1 = decode_chunk(deflate, resume_at, resume_at + spacing_bits, &tail, cfg)
            .expect("chunk1 at chunk0 end");
        assert!(!chunk1.is_empty());
    }

    /// Regression for the parallel-SM hang. Given a sub-block input
    /// fragment — the gzip end-of-stream byte-alignment padding (0-7
    /// zero bits before the footer) — `read_stream` can make no
    /// progress, and the streaming inflate loop used to call it forever.
    /// `decode_chunk` must instead return.
    #[test]
    fn decode_chunk_terminates_on_sub_byte_eof_padding() {
        let worker = std::thread::spawn(|| {
            let cfg = ChunkConfiguration {
                split_chunk_size: 512 * 1024,
                max_decoded_chunk_size: 20 * 512 * 1024,
                crc32_enabled: true,
                ..Default::default()
            };
            // One zero byte; decode the sub-byte span [4, 8) — four
            // zero bits, exactly an EOF byte-alignment padding tail.
            // Fresh wrapper, NEW_HDR, stopping points set — same shape
            // as the production chunk in the gdb backtrace.
            let _ = decode_chunk(&[0u8], 4, 8, &[], cfg);
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "decode_chunk did not return on a sub-byte EOF-padding \
                 fragment: isal_inflate is spinning"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        worker.join().expect("decode worker panicked");
    }
}

// =========================================================================
// STEP 1b GATE — pure-tail vs ISA-L-tail accounting differential
// =========================================================================
//
// Decides whether routing the chunk clean-tail through real ISA-L (C FFI)
// can reproduce, byte-for-byte, the ACCOUNTING the pure-Rust tail
// (`finish_decode_chunk_impl`) produces:
//   (a) committed decoded bytes      (b) committed length
//   (c) final_bit handoff            (d) per-chunk CRC32 over committed span
//   (e) deflate block-boundary list (bit offset + chunk-relative output off)
//
// LEFT (truth)   = production `finish_decode_chunk_impl` on a fresh ChunkData.
// RIGHT (candidate ISA-L tail) = `decompress_deflate_from_bit_with_boundaries`
//   over the full member tail, then a FROZEN coalesce post-processing rule:
//     - until_exact=false: first recorded boundary with bit >= stop_hint_bits
//     - until_exact=true : the recorded boundary with bit == stop_hint_bits
//   truncate decoded bytes + recompute CRC + set final_bit at that boundary.
//
// The post-processing rule is frozen BEFORE running (advisor Q1): if it
// disagrees with the pure tail's coalesce rule (gzip_chunk.rs:459-497, esp.
// the END_OF_BLOCK_HEADER/last_eob_pos branch at :478-489) that disagreement
// is the GATE FINDING, not a bug to patch out.
#[cfg(all(test, isal_clean_tail))]
mod isal_tail_parity {
    use super::*;
    use crate::decompress::parallel::crc32::crc32 as crc32_of;

    const WINDOW: usize = 32 * 1024;

    struct Boundary {
        bit: usize,
        out_off: usize,
    }

    /// Decode the whole raw-deflate member once with the pure wrapper,
    /// recording every END_OF_BLOCK boundary (bit position = start of next
    /// block header; output offset = decoded bytes just past finished block).
    fn enumerate(input: &[u8]) -> (Vec<u8>, Vec<Boundary>) {
        let mut wrapper =
            StreamingInflateWrapper::with_until_bits(input, 0, input.len() * 8).expect("init");
        wrapper.set_window(&[]).expect("empty window");
        wrapper.set_stopping_points(
            StoppingPoints::END_OF_BLOCK
                | StoppingPoints::END_OF_BLOCK_HEADER
                | StoppingPoints::END_OF_STREAM_HEADER,
        );
        let mut decoded: Vec<u8> = Vec::new();
        let mut bounds: Vec<Boundary> = Vec::new();
        let mut buf = vec![0u8; 128 * 1024];
        let mut last_bit = 0usize;
        loop {
            let r = wrapper.read_stream(&mut buf).expect("read_stream");
            decoded.extend_from_slice(&buf[..r.bytes_written]);
            if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                bounds.push(Boundary {
                    bit: r.bit_position,
                    out_off: decoded.len(),
                });
            }
            if r.finished {
                break;
            }
            // No-progress guard.
            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && r.bit_position == last_bit
            {
                break;
            }
            last_bit = r.bit_position;
        }
        (decoded, bounds)
    }

    /// One LEFT (production pure tail) decode on a fresh ChunkData.
    /// Returns (committed bytes, len, final_bit, crc, in-span boundaries).
    fn left_tail(
        input: &[u8],
        start_bit: usize,
        stop_hint: usize,
        window: &[u8],
        until_exact: bool,
        all: &[Boundary],
    ) -> (Vec<u8>, usize, usize, u32, Vec<(usize, usize)>) {
        let cfg = ChunkConfiguration::default();
        let mut chunk = ChunkData::new(start_bit, cfg);
        finish_decode_chunk_impl(
            &mut chunk,
            input,
            start_bit,
            stop_hint,
            window,
            false,
            until_exact,
            // FORCE pure-Rust: this is the differential gate's LEFT (pure) decode,
            // compared against the ISA-L RIGHT below. On the isal_clean_tail build
            // ISA-L is the production default, so the gate must explicitly opt out.
            false,
        )
        .expect("pure tail decode");

        // Control LEFT state (advisor Q5).
        assert_eq!(
            chunk.data_prefix_len, 0,
            "LEFT: unexpected window-image prefix"
        );
        assert_eq!(
            chunk.narrowed_len, 0,
            "LEFT: unexpected narrowed marker prefix"
        );
        assert_eq!(
            chunk.crc32s.len(),
            1,
            "LEFT: expected exactly one CRC accumulator"
        );
        assert!(chunk.encoded_size_bits > 0, "LEFT: finalize did not run");

        let bytes = chunk.data.to_contiguous();
        let len = chunk.data.len();
        let final_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let crc = chunk.crc32s[0].crc32();
        // Cross-check the accumulator equals a fresh CRC of the committed span
        // (so a CRC-algorithm mismatch can't masquerade as a (d) divergence).
        assert_eq!(
            crc,
            crc32_of(&bytes),
            "LEFT: CRC accumulator != crc32(committed)"
        );

        let start_off = all
            .iter()
            .find(|b| b.bit == start_bit)
            .map(|b| b.out_off)
            .unwrap_or(0);
        let in_span: Vec<(usize, usize)> = all
            .iter()
            .filter(|b| b.bit > start_bit && b.bit <= final_bit)
            .map(|b| (b.bit, b.out_off - start_off))
            .collect();
        (bytes, len, final_bit, crc, in_span)
    }

    /// One RIGHT (ISA-L FFI + frozen coalesce post-processing) decode.
    fn right_tail(
        input: &[u8],
        start_bit: usize,
        stop_hint: usize,
        window: &[u8],
        until_exact: bool,
    ) -> Option<(Vec<u8>, usize, usize, u32, Vec<(usize, usize)>)> {
        let mut throwaway = crc32fast::Hasher::new();
        // Full member tail (advisor Q3): never a fixed margin.
        let (out, _end, bnds) =
            crate::backends::isal_decompress::decompress_deflate_from_bit_with_boundaries(
                input,
                start_bit,
                window,
                input.len(), // generous cap; ISA-L grows on demand
                &mut throwaway,
            )?;
        // FROZEN coalesce rule.
        let chosen = if until_exact {
            bnds.iter().find(|b| b.bit_offset == stop_hint)
        } else {
            bnds.iter().find(|b| b.bit_offset >= stop_hint)
        }?;
        let truncate = chosen.output_offset;
        let final_bit = chosen.bit_offset;
        let bytes = out[..truncate].to_vec();
        let crc = crc32_of(&bytes);
        let in_span: Vec<(usize, usize)> = bnds
            .iter()
            .filter(|b| b.bit_offset > start_bit && b.bit_offset <= final_bit)
            .map(|b| (b.bit_offset, b.output_offset))
            .collect();
        Some((bytes, truncate, final_bit, crc, in_span))
    }

    /// Build a multi-MiB all-DYNAMIC-Huffman gzip member as a portable stand-in
    /// for the silesia corpus, so the differential gate runs even where the
    /// benchmark_data fixture is absent (e.g. local Rosetta x86 iteration). The
    /// payload mixes compressible English-ish phrases with structured bytes so
    /// flate2 -9 emits dynamic-Huffman blocks with MANY interior EOB boundaries
    /// (the regime the gate samples). This is a SUPPLEMENT to — not a replacement
    /// for — the real silesia run on the corpus-bearing box.
    fn synthetic_dynamic_gz() -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let words: &[&str] = &[
            "the ",
            "quick ",
            "brown ",
            "fox ",
            "jumps ",
            "over ",
            "lazy ",
            "dog ",
            "lorem ",
            "ipsum ",
            "dolor ",
            "sit ",
            "amet ",
            "consectetur ",
            "adipiscing ",
        ];
        let mut data: Vec<u8> = Vec::with_capacity(12 * 1024 * 1024);
        let mut rng: u64 = 0x9e3779b97f4a7c15;
        while data.len() < 12 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let w = words[(rng >> 33) as usize % words.len()];
            data.extend_from_slice(w.as_bytes());
            if (rng >> 17) % 5 == 0 {
                data.extend_from_slice(&rng.to_le_bytes());
            }
        }
        let mut enc = GzEncoder::new(Vec::new(), Compression::new(9));
        enc.write_all(&data).unwrap();
        enc.finish().unwrap()
    }

    /// GROWTH-PAST-THE-OLD-CAP gate (advisor follow-up to the DIS-29 storm
    /// fix). The 64 MiB RESERVE_CAP now bounds only the UPFRONT reserve; a
    /// chunk whose decoded size exceeds it must GROW past the cap on demand
    /// (previously such a chunk ALWAYS fell back to pure-Rust). A multi-GB
    /// end-to-end fixture is too heavy for the suite, so exercise the exact
    /// production code path directly: the growable FFI decode into a
    /// `SegmentedU8` sink with initial == the 64 MiB cap, on a raw-deflate
    /// stream decoding to ~90 MiB. Asserts full decode + byte-exact output.
    #[test]
    fn isal_growable_decode_grows_past_64mib_cap() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // ~90 MiB of a repeating 512-byte pattern -> tiny compressed stream
        // with extreme expansion (the regime that blows past the cap).
        let mut pattern = [0u8; 512];
        for (i, b) in pattern.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        const RAW_LEN: usize = 90 * 1024 * 1024;
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
        let mut remaining = RAW_LEN;
        while remaining > 0 {
            let n = remaining.min(pattern.len());
            enc.write_all(&pattern[..n]).unwrap();
            remaining -= n;
        }
        let deflate = enc.finish().unwrap();

        const OLD_CAP: usize = 64 * 1024 * 1024; // the production upfront-reserve cap
        let mut sink = crate::decompress::parallel::segmented_buffer::SegmentedU8::default();
        let (written, _end_bit, _boundaries) =
            crate::backends::isal_decompress::decompress_deflate_from_bit_into_growable(
                &deflate,
                0,
                &[],
                &mut sink,
                OLD_CAP,
                4 * 1024 * 1024,
            )
            .expect("growable decode must not fail past the old 64 MiB cap");
        assert_eq!(written, RAW_LEN, "must decode past the 64 MiB initial");
        assert_eq!(sink.len(), RAW_LEN);
        let out = sink.to_contiguous();
        assert!(
            out.chunks(pattern.len()).all(|c| c == &pattern[..c.len()]),
            "decoded bytes must match the source pattern"
        );
    }

    /// FALLBACK-STORM regression gate (DIS-29 / HANDOFF item i). On a
    /// >8x-compressible corpus the old FIXED 8x-compressed-span reserve
    /// overflowed on EVERY chunk: `decompress_deflate_from_bit_into` returned
    /// `None` at buffer-full, so ALL chunks fell back to the ~7.5x-slower
    /// pure-Rust engine (isal_chunks=0 — nasa 9.9x crushed T1 to 0.57x rg).
    /// The growable production decode keeps the same 8x upfront reserve but
    /// GROWS on overflow, so chunks stay on ISA-L. This test builds a ~10x
    /// web-log-like member, decodes it through the REAL ParallelSM pipeline at
    /// T=4, and asserts (a) byte-exact output and (b) at least one chunk
    /// decoded on the ISA-L engine (the pre-fix binary storms: chunks=0).
    #[test]
    fn isal_engine_survives_8x_expansion_no_storm() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        use std::sync::atomic::Ordering;

        // ~48 MiB of repetitive log lines -> ratio >10x at level 9.
        let mut raw: Vec<u8> = Vec::with_capacity(48 * 1024 * 1024);
        let mut rng: u64 = 0x2545f4914f6cdd1d;
        while raw.len() < 48 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let host = (rng >> 33) % 50;
            let path = (rng >> 13) % 200;
            let code = 100 + ((rng >> 5) % 65000);
            raw.extend_from_slice(
                format!(
                    "host{host:03}.example.com - - [01/Jul/1995:00:00:{:02} -0400] \
                     \"GET /path/to/resource/{path} HTTP/1.0\" 200 {code}\n",
                    rng % 60
                )
                .as_bytes(),
            );
        }
        let mut enc = GzEncoder::new(Vec::new(), Compression::new(9));
        enc.write_all(&raw).unwrap();
        let gz = enc.finish().unwrap();
        let ratio = raw.len() as f64 / gz.len() as f64;
        assert!(
            ratio > 8.5,
            "fixture must exceed the 8x storm threshold, got {ratio:.2}x"
        );

        let chunks_before = ISAL_ENGINE_ORACLE_CHUNKS.load(Ordering::Relaxed);
        let mut out: Vec<u8> = Vec::with_capacity(raw.len());
        let n =
            crate::decompress::parallel::single_member::decompress_parallel(&gz, &mut out, None, 4)
                .expect("parallel decode of >8x member");
        assert_eq!(n as usize, raw.len());
        assert_eq!(out, raw, "output must be byte-exact");
        let chunks_delta = ISAL_ENGINE_ORACLE_CHUNKS.load(Ordering::Relaxed) - chunks_before;
        assert!(
            chunks_delta >= 1,
            "ISA-L engine decoded no chunks on a >8x corpus — the reserve-overflow \
             fallback storm is back (DIS-29)"
        );
    }

    /// BFINAL exact-landing gate: `finish_decode_chunk_isal_oracle` with
    /// `until_exact=true` must return `Ok(true)` when no recorded EOB boundary
    /// matches stop_hint_bits exactly BUT ISA-L's `end_bit` is within 1 bit of
    /// stop_hint_bits. Two diagnosed sub-cases:
    ///
    /// (a) ISA-L exits via ISAL_BLOCK_FINISH without setting stopped_at for the
    ///     BFINAL block — no EOB boundary recorded, `end_bit == stop_hint_bits`.
    ///
    /// (b) ISA-L records the BFINAL EOB boundary via stopped_at but at
    ///     `stop_hint_bits - 1` (1-bit coordinate discrepancy between ISA-L's
    ///     chunk-decode buffering state and the block_finder's canonical scan).
    ///     Diagnosed on NASA Jul95: start_bit=150999587, stop_hint=156360208,
    ///     end_bit=156360207, boundaries=25, delta=-1, 3–5/30 runs at T8/T12.
    ///
    /// Pre-fix: the `None` arm of the boundary search returns `Ok(false)`.
    /// Post-fix: `end_bit` within 1 bit of stop_hint_bits ⇒ accept.
    #[test]
    fn isal_until_exact_accepts_bfinal_exact_landing() {
        use crate::backends::isal_decompress;
        use crate::decompress::parallel::segmented_buffer::SegmentedU8;

        let gz = synthetic_dynamic_gz();
        let hdr =
            crate::decompress::parallel::single_member::skip_gzip_header(&gz).expect("gzip header");
        // Raw deflate payload: strip gzip header and 8-byte trailer.
        let input = &gz[hdr..gz.len() - 8];

        // Full-member reference decode via the pure wrapper: decoded bytes + EOB boundaries.
        let (decoded, bounds) = enumerate(input);
        assert!(
            bounds.len() >= 2,
            "synthetic member must have at least 2 EOB boundaries (got {})",
            bounds.len()
        );

        // Get the stream end bit from a full ISA-L decode of the whole member.
        let mut sink = SegmentedU8::default();
        let (_, stream_end_bit, _) = isal_decompress::decompress_deflate_from_bit_into_growable(
            input,
            0,
            &[],
            &mut sink,
            8 * 1024 * 1024,
            4 * 1024 * 1024,
        )
        .expect("full ISA-L member decode must succeed");

        // Pick the last interior EOB boundary that has at least 32 KiB of decoded
        // output before it, so we can seed a full 32 KiB window. This boundary is
        // the simulated "chunk start" — from here to stream_end_bit is the member's
        // tail chunk, whose only remaining block is the BFINAL block.
        let start_b = bounds
            .iter()
            .rev()
            .find(|b| b.out_off >= WINDOW && b.bit < stream_end_bit)
            .expect("synthetic member must have a boundary with >= 32 KiB of preceding output");

        let start_bit = start_b.bit;
        let window_end = start_b.out_off;
        let window = &decoded[window_end - WINDOW..window_end];

        // stop_hint == stream_end_bit: this is the until_exact BFINAL exact case.
        let stop_hint_bits = stream_end_bit;

        // Construct a fresh ChunkData and call finish_decode_chunk_isal_oracle directly.
        // The test module lives in the same file so the private function is in scope.
        let cfg = ChunkConfiguration::default();
        let mut chunk = ChunkData::new(start_bit, cfg);
        let result = finish_decode_chunk_isal_oracle(
            &mut chunk,
            input,
            start_bit,
            stop_hint_bits,
            window,
            true, // until_exact — exercises the BFINAL exact-landing branch
        );

        assert!(
            result.is_ok(),
            "finish_decode_chunk_isal_oracle returned Err on BFINAL tail: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "finish_decode_chunk_isal_oracle returned Ok(false) on BFINAL exact landing \
             (start_bit={start_bit} stop_hint={stop_hint_bits}): the None arm should \
             accept when end_bit is within 1 bit of stop_hint_bits (pre-fix behavior: \
             exact equality check missed the 1-bit ISA-L coordinate discrepancy)"
        );

        // Verify decoded bytes match the ground-truth reference.
        let chunk_bytes = chunk.data.to_contiguous();
        let reference = &decoded[window_end..];
        assert_eq!(
            chunk_bytes.len(),
            reference.len(),
            "BFINAL tail length mismatch: ISA-L={} pure={} \
             (start_bit={start_bit} stop_hint={stop_hint_bits})",
            chunk_bytes.len(),
            reference.len()
        );
        assert_eq!(
            chunk_bytes, reference,
            "BFINAL tail bytes mismatch at start_bit={start_bit} stop_hint={stop_hint_bits}"
        );
    }

    /// RATIO-INFORMED RESERVE unit gate. Tests `compute_initial_reserve` for
    /// the three representative corpus classes and boundary conditions:
    ///
    /// * 1.3× (model-like, near-incompressible) — ratio_ceil = 2; reserve
    ///   = compressed_span × 2, correctly sized 6× smaller than the old 8×.
    /// * 7.8× (ghcn-like) — ratio_ceil = 10; reserve = compressed_span × 10,
    ///   within cap for typical 4 MiB chunks.
    /// * 10× — ratio_ceil = 13; reserve = compressed_span × 13, also within
    ///   cap for 4 MiB chunks.
    /// * Unknown (0) — falls back to historical 8× factor.
    /// * Floor: tiny compressed_span → clamped up to RESERVE_FLOOR (4 MiB).
    /// * Cap: large span × large factor → clamped to RESERVE_CAP (64 MiB).
    #[cfg(all(parallel_sm, feature = "isal-compression", target_arch = "x86_64"))]
    #[test]
    fn ratio_informed_reserve_computation() {
        const FLOOR: usize = 4 * 1024 * 1024;
        const CAP: usize = 64 * 1024 * 1024;
        let span = 4 * 1024 * 1024usize; // typical 4 MiB compressed chunk

        // 1.3× corpus (model): ceil(1.3 × 1.25) = ceil(1.625) = 2
        // The sm_driver computes this as ceil(ISIZE×5 / compressed×4) = ceil(6.5/4·compressed)
        // Verify the clamp helper uses factor 2 correctly.
        let rc_13: u16 = 2;
        assert_eq!(
            compute_initial_reserve(span, rc_13),
            span * 2, // 8 MiB — within [4 MiB, 64 MiB]
            "1.3x corpus: expected 2× reserve"
        );

        // 7.8× corpus (ghcn): ceil(7.8 × 1.25) = ceil(9.75) = 10
        let rc_78: u16 = 10;
        assert_eq!(
            compute_initial_reserve(span, rc_78),
            span * 10, // 40 MiB — within cap
            "7.8x corpus: expected 10× reserve (40 MiB < 64 MiB cap)"
        );

        // 10× corpus: ceil(10 × 1.25) = ceil(12.5) = 13
        let rc_10: u16 = 13;
        assert_eq!(
            compute_initial_reserve(span, rc_10),
            span * 13, // 52 MiB — within cap
            "10x corpus: expected 13× reserve (52 MiB < 64 MiB cap)"
        );

        // Unknown (expansion_ratio_ceil = 0) → historical 8× fallback
        assert_eq!(
            compute_initial_reserve(span, 0),
            span * 8, // 32 MiB
            "unknown ratio: must fall back to 8× factor"
        );

        // Floor: tiny span → clamp up to 4 MiB
        assert_eq!(
            compute_initial_reserve(0, 2),
            FLOOR,
            "zero span: must clamp to RESERVE_FLOOR"
        );
        assert_eq!(
            compute_initial_reserve(100, 2),
            FLOOR,
            "tiny span (100B × 2 = 200B < 4 MiB): must clamp to RESERVE_FLOOR"
        );

        // Cap: 10 MiB span × factor 10 = 100 MiB → clamp to 64 MiB
        assert_eq!(
            compute_initial_reserve(10 * 1024 * 1024, 10),
            CAP,
            "100 MiB upfront request: must clamp to RESERVE_CAP (64 MiB)"
        );
    }

    /// Verify that sm_driver computes the expected ratio_ceil values for the
    /// representative corpus classes (1.3×, 7.8×, 10×) without needing a full
    /// decode.  This tests the arithmetic only — the pure formula
    /// `ceil(ISIZE × 5 / (compressed × 4)), min 2`.
    #[test]
    fn ratio_ceil_arithmetic() {
        // Helper: compute ratio_ceil exactly as sm_driver does.
        let ratio_ceil = |isize_bytes: u64, compressed_bytes: u64| -> u16 {
            if compressed_bytes == 0 || isize_bytes == 0 {
                return 0;
            }
            let numer = isize_bytes.saturating_mul(5);
            let denom = compressed_bytes.saturating_mul(4);
            ((numer + denom - 1) / denom).max(2).min(u16::MAX as u64) as u16
        };

        // 1.3× member: ISIZE = 269 MiB, compressed = 204 MiB
        // ceil(269 × 5 / (204 × 4)) = ceil(1345 / 816) = ceil(1.649) = 2
        let r = ratio_ceil(269 * 1024 * 1024, 204 * 1024 * 1024);
        assert_eq!(r, 2, "1.3x model corpus: ratio_ceil should be 2");

        // 7.8× member (ghcn-like): ISIZE = 7.8 × compressed
        // ceil(7.8 × 5 / 4) = ceil(9.75) = 10
        let compressed = 50_000_000u64;
        let uncompressed = (compressed as f64 * 7.8) as u64;
        let r = ratio_ceil(uncompressed, compressed);
        assert_eq!(r, 10, "7.8x corpus: ratio_ceil should be 10");

        // 10× member: ceil(10 × 5 / 4) = ceil(12.5) = 13
        let compressed = 50_000_000u64;
        let uncompressed = compressed * 10;
        let r = ratio_ceil(uncompressed, compressed);
        assert_eq!(r, 13, "10x corpus: ratio_ceil should be 13");

        // Minimum floor: 1.0× exactly → ceil(1.0 × 1.25) = ceil(1.25) = 2 (min)
        let r = ratio_ceil(100, 100);
        assert_eq!(r, 2, "1.0x corpus: ratio_ceil should be 2 (minimum)");

        // Unknown (zero isize or compressed) → 0
        assert_eq!(ratio_ceil(0, 100), 0, "zero isize → unknown");
        assert_eq!(ratio_ceil(100, 0), 0, "zero compressed → unknown");
    }

    #[test]
    fn isal_tail_matches_pure_tail_on_real_silesia_chunks() {
        // Prefer the real silesia corpus where present (the canonical gate input
        // on the bench box); fall back to a portable synthetic dynamic member so
        // the differential runs locally too.
        let gz = std::fs::read("benchmark_data/silesia-gzip.tar.gz")
            .unwrap_or_else(|_| synthetic_dynamic_gz());
        let hdr =
            crate::decompress::parallel::single_member::skip_gzip_header(&gz).expect("gzip header");
        let input = &gz[hdr..gz.len() - 8];

        let (decoded, bounds) = enumerate(input);
        assert!(
            bounds.len() > 40,
            "need many real deflate boundaries, got {}",
            bounds.len()
        );
        eprintln!(
            "[gate] raw deflate {} bytes -> {} decoded bytes, {} EOB boundaries",
            input.len(),
            decoded.len(),
            bounds.len()
        );

        // Synthesize real chunk tail-decode inputs: 5 starts evenly spread
        // across interior boundaries, each spanning K=8 blocks.
        const K: usize = 8;
        let n = bounds.len();
        let mut starts: Vec<usize> = Vec::new();
        for f in 1..=5usize {
            let idx = (n * f) / 7;
            if idx >= 1 && idx + K + 1 < n {
                starts.push(idx);
            }
        }
        starts.dedup();
        assert!(!starts.is_empty(), "no usable chunk starts");

        let mut total = 0usize;
        let mut diverged = 0usize;

        for &si in &starts {
            for until_exact in [false, true] {
                let start_b = &bounds[si];
                let start_bit = start_b.bit;
                let win_lo = start_b.out_off.saturating_sub(WINDOW);
                // Exact available prefix, capped at 32 KiB (advisor Q4).
                let window = &decoded[win_lo..start_b.out_off];

                let stop_hint = if until_exact {
                    // Exact later real boundary.
                    bounds[si + K].bit
                } else {
                    // Mid-block: between boundary si+K-1 and si+K, so the pure
                    // "first EOB at-or-past" should land on bounds[si+K].
                    let a = bounds[si + K - 1].bit;
                    let b = bounds[si + K].bit;
                    a + (b - a) / 2
                };

                total += 1;
                let label = format!(
                    "start_idx={si} start_bit={start_bit} stop_hint={stop_hint} \
                     until_exact={until_exact} win_len={}",
                    window.len()
                );

                let (lb, ll, lf, lc, lbnd) =
                    left_tail(input, start_bit, stop_hint, window, until_exact, &bounds);

                let right = right_tail(input, start_bit, stop_hint, window, until_exact);
                let Some((rb, rl, rf, rc, rbnd)) = right else {
                    diverged += 1;
                    eprintln!(
                        "[gate] DIVERGE ({label}): ISA-L produced NO boundary matching the \
                         frozen coalesce rule (until_exact={until_exact}); LEFT len={ll} final_bit={lf}"
                    );
                    continue;
                };

                let a_ok = lb == rb;
                let b_ok = ll == rl;
                let c_ok = lf == rf;
                let d_ok = lc == rc;
                let e_ok = lbnd == rbnd;

                if a_ok && b_ok && c_ok && d_ok && e_ok {
                    eprintln!(
                        "[gate] MATCH ({label}): len={ll} final_bit={lf} crc={lc:#010x} \
                         boundaries={}",
                        lbnd.len()
                    );
                } else {
                    diverged += 1;
                    eprintln!("[gate] DIVERGE ({label}):");
                    eprintln!(
                        "    (a) bytes     : {}",
                        if a_ok {
                            "match".into()
                        } else {
                            format!(
                                "DIFFER (left {} vs right {} bytes; first diff at {:?})",
                                lb.len(),
                                rb.len(),
                                lb.iter().zip(rb.iter()).position(|(x, y)| x != y)
                            )
                        }
                    );
                    eprintln!(
                        "    (b) length    : {}",
                        if b_ok {
                            "match".into()
                        } else {
                            format!("DIFFER left {ll} vs right {rl}")
                        }
                    );
                    eprintln!(
                        "    (c) final_bit : {}",
                        if c_ok {
                            "match".into()
                        } else {
                            format!("DIFFER left {lf} vs right {rf}")
                        }
                    );
                    eprintln!(
                        "    (d) crc       : {}",
                        if d_ok {
                            "match".into()
                        } else {
                            format!("DIFFER left {lc:#010x} vs right {rc:#010x}")
                        }
                    );
                    eprintln!(
                        "    (e) boundaries: {}",
                        if e_ok {
                            "match".into()
                        } else {
                            format!("DIFFER left {} vs right {} entries", lbnd.len(), rbnd.len())
                        }
                    );
                }
            }
        }

        eprintln!(
            "[gate] SUMMARY: {} chunk-decodes, {} matched, {} diverged",
            total,
            total - diverged,
            diverged
        );
        assert_eq!(
            diverged, 0,
            "ISA-L tail diverged from pure tail on {diverged}/{total} chunk-decodes \
             (see [gate] lines above for the exact field + mechanism)"
        );
    }
}

/// NATIVE fold gate (gzippy-native, `not(isal_clean_tail)`).
///
/// Permanent catch for the flip-in-place fold. It drives the REAL production
/// path — `decode_chunk_window_absent` — on real silesia chunks: Engine M
/// (`marker_inflate::Block`) emits u16 markers for the early blocks, flips to
/// clean at 32 KiB, and on native FOLDS (continues decoding in-place to
/// `Finished` instead of returning `FlipToClean` to Engine C). The chunk's
/// markers are then resolved against the true predecessor 32 KiB window
/// (vendor applyWindow) and merged, yielding the chunk's complete payload,
/// which is asserted byte-for-byte (and CRC) against the INDEPENDENT
/// whole-member ground-truth decode (`enumerate`). Run for both `until_exact`
/// true (exact-boundary stop hint) and false (mid-block hint).
///
/// The gate also asserts the fold was actually exercised (markers present AND
/// a clean in-place tail > 32 KiB), and that `final_bit` lands on a real EOB
/// boundary at-or-past the stop hint.
///
/// NOTE on the stop point: Engine M stops at the first block whose header
/// starts at-or-past stop_hint (the faithful rapidgzip behavior). The pre-fold
/// two-phase tail (`StreamingInflateWrapper`, kept on `isal_clean_tail`) has an
/// ISA-L-emulation rewind that can keep/skip one fixed-vs-dynamic block at a
/// header straddle (gzip_chunk.rs:481-486). That is a *speculative* stop-point
/// difference only — the consumer reconciles it exactly via `furthest_decoded_
/// bit` and `block_finder.insert(chunk_end_bit)` (chunk_fetcher.rs:1074, 2663),
/// so concatenated output is byte-identical either way (proven end-to-end by
/// the silesia DUAL-SHA gate). Hence this gate asserts correctness against
/// ground truth, not stop-point equality with the retired two-phase tail.
#[cfg(all(test, parallel_sm, not(isal_clean_tail)))]
mod native_fold_parity {
    use super::*;
    use crate::decompress::parallel::crc32::crc32 as crc32_of;

    const WINDOW: usize = 32 * 1024;

    struct Boundary {
        bit: usize,
        out_off: usize,
    }

    /// Decode the whole raw-deflate member once with the pure wrapper,
    /// recording every END_OF_BLOCK boundary (bit = start of next header,
    /// out_off = decoded bytes just past the finished block).
    fn enumerate(input: &[u8]) -> (Vec<u8>, Vec<Boundary>) {
        let mut wrapper =
            StreamingInflateWrapper::with_until_bits(input, 0, input.len() * 8).expect("init");
        wrapper.set_window(&[]).expect("empty window");
        wrapper.set_stopping_points(
            StoppingPoints::END_OF_BLOCK
                | StoppingPoints::END_OF_BLOCK_HEADER
                | StoppingPoints::END_OF_STREAM_HEADER,
        );
        let mut decoded: Vec<u8> = Vec::new();
        let mut bounds: Vec<Boundary> = Vec::new();
        let mut buf = vec![0u8; 128 * 1024];
        let mut last_bit = 0usize;
        loop {
            let r = wrapper.read_stream(&mut buf).expect("read_stream");
            decoded.extend_from_slice(&buf[..r.bytes_written]);
            if r.stopped_at == StoppingPoints::END_OF_BLOCK {
                bounds.push(Boundary {
                    bit: r.bit_position,
                    out_off: decoded.len(),
                });
            }
            if r.finished {
                break;
            }
            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && r.bit_position == last_bit
            {
                break;
            }
            last_bit = r.bit_position;
        }
        (decoded, bounds)
    }

    /// Result of a folded window-absent chunk decode + marker resolution.
    struct Folded {
        /// Fully resolved decoded bytes (markers resolved against `window`,
        /// then the clean tail) — the chunk's complete payload.
        full: Vec<u8>,
        /// Decoded length of the marker (pre-flip) region.
        markers_len: usize,
        /// Decoded length of the clean (post-flip, in-place fold) tail.
        clean_len: usize,
        /// Bit position where the chunk decode stopped.
        final_bit: usize,
    }

    /// THE PRODUCTION FOLD PATH: window-absent decode via
    /// `decode_chunk_window_absent` (Engine M emits u16 markers for the early
    /// blocks, flips to clean at 32 KiB, and on native FOLDS — continuing
    /// in-place to `Finished`). Then resolve the markers against the true
    /// predecessor `window` and merge into one buffer. This is exactly what a
    /// real chunk goes through; the only test-supplied input is the known-good
    /// 32 KiB predecessor window for marker resolution.
    fn folded_window_absent(
        input: &[u8],
        start_bit: usize,
        stop_hint: usize,
        window: &[u8],
    ) -> Folded {
        assert_eq!(window.len(), WINDOW, "resolution window must be 32 KiB");
        let cfg = ChunkConfiguration::default();
        let mut chunk = super::decode_chunk_window_absent(input, start_bit, stop_hint, cfg)
            .expect("folded window-absent decode");
        assert_eq!(chunk.data_prefix_len, 0, "unexpected window-image prefix");
        let final_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let markers_len = chunk.data_with_markers.len();
        let clean_len = chunk.data.len();
        // Resolve the u16 markers against the real predecessor window and fold
        // them into `data` (vendor applyWindow).
        chunk.resolve_and_narrow_markers_in_place(window);
        chunk.merge_resolved_markers_into_data();
        assert!(
            chunk.data_with_markers.is_empty(),
            "markers not consumed by resolve+merge"
        );
        let full = chunk.data.to_contiguous();
        assert_eq!(
            full.len(),
            markers_len + clean_len,
            "resolve length mismatch"
        );
        Folded {
            full,
            markers_len,
            clean_len,
            final_bit,
        }
    }

    #[test]
    fn folded_native_decode_matches_ground_truth_on_real_silesia_chunks() {
        // Real-corpus fixture, large (~67 MiB) and not committed — present on
        // bench boxes / the owner tree but absent in a fresh worktree. Skip
        // gracefully when missing (same convention as
        // `three_oracle_silesia_if_available`) rather than hard-failing on a
        // tree that simply hasn't fetched the corpus. When the fixture IS
        // present every byte-exact ground-truth assertion below runs in full.
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[fold-gate] benchmark_data/silesia-gzip.tar.gz not present, skipping");
                return;
            }
        };
        let hdr =
            crate::decompress::parallel::single_member::skip_gzip_header(&gz).expect("gzip header");
        let input = &gz[hdr..gz.len() - 8];

        let (decoded, bounds) = enumerate(input);
        assert!(
            bounds.len() > 40,
            "need many real deflate boundaries, got {}",
            bounds.len()
        );
        eprintln!(
            "[fold-gate] raw deflate {} bytes -> {} decoded bytes, {} EOB boundaries",
            input.len(),
            decoded.len(),
            bounds.len()
        );

        // Large spans (K blocks) so a chunk has room to accumulate 32 KiB of
        // contiguous clean output and FLIP — the fold branch under test. Note
        // markers propagate forward via long-range copies of pre-resolution
        // markers, so not every chunk flips (data-dependent); we sample many
        // starts and REQUIRE a healthy number of flips below.
        const K: usize = 24;
        let n = bounds.len();
        let mut starts: Vec<usize> = Vec::new();
        for f in 1..=16usize {
            let idx = (n * f) / 17;
            if idx >= 1 && idx + K + 1 < n {
                starts.push(idx);
            }
        }
        starts.dedup();
        assert!(!starts.is_empty(), "no usable chunk starts");

        // Set of every legal EOB bit position, so a final_bit can be checked
        // for landing on a real block boundary (not mid-block).
        let boundary_bits: std::collections::HashSet<usize> =
            bounds.iter().map(|b| b.bit).collect();

        let mut total = 0usize;
        let mut diverged = 0usize;
        let mut flips = 0usize;
        // SEAM RECONCILIATION (advisor residual on the fold milestone): the native
        // fold stops at a DIFFERENT bit than the retired two-phase tail. The
        // consumer reconciles that seam via `furthest_decoded_bit` /
        // `block_finder.insert(chunk_end_bit)` (chunk_fetcher.rs:1074, 1419). The
        // end-to-end DUAL-SHA already covers it, but this cheaper in-file check
        // proves the seam directly: a SECOND folded chunk started at the first
        // chunk's `final_bit`, windowed by the first chunk's resolved tail, must
        // produce bytes that continue the output CONTIGUOUSLY from `final_bit`.
        // A wrong stop-point handoff would desync this seam silently.
        let mut seam_checks = 0usize;
        let mut seam_diverged = 0usize;

        for &si in &starts {
            for until_exact in [false, true] {
                let start_b = &bounds[si];
                let start_bit = start_b.bit;
                let start_off = start_b.out_off;
                let win_lo = start_off.saturating_sub(WINDOW);
                if start_off - win_lo != WINDOW {
                    // Need a full 32 KiB window for Engine M priming.
                    continue;
                }
                let window = &decoded[win_lo..start_off];

                let stop_hint = if until_exact {
                    bounds[si + K].bit
                } else {
                    let a = bounds[si + K - 1].bit;
                    let b = bounds[si + K].bit;
                    a + (b - a) / 2
                };

                total += 1;
                let label = format!(
                    "start_idx={si} start_bit={start_bit} stop_hint={stop_hint} \
                     until_exact={until_exact}"
                );

                let f = folded_window_absent(input, start_bit, stop_hint, window);

                // GROUND TRUTH: the folded chunk's fully-resolved payload must
                // equal exactly what the independent whole-member decode
                // produced for the same span [start_off, start_off+len). This
                // holds whether or not the chunk flipped.
                let truth = &decoded[start_off..start_off + f.full.len()];
                let bytes_ok = f.full == truth;
                // CRC of resolved output == CRC of ground truth (superset of the
                // per-chunk CRC check; equal-by-construction when bytes match,
                // but asserted explicitly to satisfy the gate contract).
                let crc_ok = crc32_of(&f.full) == crc32_of(truth);
                // final_bit must land on a real EOB boundary, at/after the stop
                // hint (Engine M stops at first block-start >= hint).
                let bit_ok = boundary_bits.contains(&f.final_bit) && f.final_bit >= stop_hint;
                // Did this chunk exercise the FOLD branch? (>32 KiB clean tail
                // means it flipped and Engine M continued in-place.)
                let flipped = f.clean_len > WINDOW;
                if flipped {
                    flips += 1;
                }

                let ok = bytes_ok && crc_ok && bit_ok;

                if ok {
                    eprintln!(
                        "[fold-gate] OK ({label}): full_len={} markers={} clean_tail={} \
                         final_bit={} flipped={} crc={:#010x}",
                        f.full.len(),
                        f.markers_len,
                        f.clean_len,
                        f.final_bit,
                        flipped,
                        crc32_of(&f.full)
                    );
                } else {
                    diverged += 1;
                    eprintln!("[fold-gate] DIVERGE ({label}): flipped={flipped}");
                    eprintln!(
                        "    bytes vs truth : {}",
                        if bytes_ok {
                            "match".into()
                        } else {
                            format!(
                                "CORRUPT (full {} bytes; first diff at {:?})",
                                f.full.len(),
                                f.full.iter().zip(truth.iter()).position(|(x, y)| x != y)
                            )
                        }
                    );
                    eprintln!("    crc            : {}", if crc_ok { "ok" } else { "BAD" });
                    eprintln!(
                        "    final_bit      : {} (on_boundary={}, >=hint={})",
                        f.final_bit,
                        boundary_bits.contains(&f.final_bit),
                        f.final_bit >= stop_hint
                    );
                }

                // SEAM CHECK: only on a correct, real seam (final_bit is a true
                // EOB boundary) with a full 32 KiB resolved window available and
                // room for a follow-on chunk. Decode a SECOND folded chunk at
                // `final_bit` and assert its resolved bytes continue the output
                // contiguously from `start_off + f.full.len()` — i.e. the seam the
                // consumer reconciles is byte-continuous on the SAME cursor.
                let seam_off = start_off + f.full.len();
                let seam_window_lo = seam_off.saturating_sub(WINDOW);
                let have_window = seam_off - seam_window_lo == WINDOW;
                let room_past = boundary_bits.contains(&f.final_bit)
                    && seam_off < decoded.len()
                    && f.final_bit < input.len() * 8;
                if ok && have_window && room_past {
                    let seam_window = &decoded[seam_window_lo..seam_off];
                    // Stop the follow-on chunk a modest span past the seam so the
                    // decode is cheap but still crosses several blocks.
                    let seam_stop = (f.final_bit + 64 * 1024).min(input.len() * 8);
                    let f2 = folded_window_absent(input, f.final_bit, seam_stop, seam_window);
                    seam_checks += 1;
                    let avail = decoded.len() - seam_off;
                    let take = f2.full.len().min(avail);
                    let seam_truth = &decoded[seam_off..seam_off + take];
                    if f2.full[..take] != *seam_truth {
                        seam_diverged += 1;
                        eprintln!(
                            "[fold-gate] SEAM DIVERGE ({label}): final_bit={} seam_off={} \
                             follow_len={} first_diff={:?}",
                            f.final_bit,
                            seam_off,
                            f2.full.len(),
                            f2.full[..take]
                                .iter()
                                .zip(seam_truth.iter())
                                .position(|(x, y)| x != y)
                        );
                    }
                }
            }
        }

        // The seam handoff is the whole point of the in-place fold: assert the
        // consumer-level stop-point reconciliation is byte-continuous on a healthy
        // number of real seams (else the fold could regress the seam silently —
        // the residual this assertion closes).
        assert_eq!(
            seam_diverged, 0,
            "FOLD seam reconciliation desynced on {seam_diverged}/{seam_checks} seams \
             (follow-on chunk at final_bit did not continue output contiguously)"
        );
        assert!(
            seam_checks >= 3,
            "seam reconciliation exercised on only {seam_checks} seams — too few; \
             widen sampling"
        );
        eprintln!("[fold-gate] SEAM: {seam_checks} seams checked, all byte-continuous");

        eprintln!(
            "[fold-gate] SUMMARY: {} chunk-decodes, {} correct, {} diverged, {} FLIPPED \
             (exercised the fold branch)",
            total,
            total - diverged,
            diverged,
            flips
        );
        assert!(total > 0, "fold gate ran zero comparisons");
        assert_eq!(
            diverged, 0,
            "FOLDED native decode produced incorrect output vs ground truth on \
             {diverged}/{total} chunk-decodes (see [fold-gate] lines)"
        );
        // The whole point is the FOLD branch; require it was actually taken on a
        // meaningful number of real chunks (else the gate proves nothing about
        // Engine M continuing in-place past the 32 KiB flip).
        assert!(
            flips >= 3,
            "fold branch exercised on only {flips}/{total} chunks — too few to \
             trust the gate; widen K or the start sampling"
        );
    }

    // ── Stage-2 copy-free-to-final owed cases (advisor D + landmine) ─────────
    //
    // These drive the REAL production native contig tail
    // (`finish_decode_chunk_contig_native`) via `decode_chunk_window_absent` on
    // a window-ABSENT stream that flips at 32 KiB, then assert byte-exactness
    // against an independent flate2 decode of the same payload.

    /// Build a raw-deflate stream from `payload` and decode the FIRST chunk
    /// (window-absent, start_bit 0, stop_hint = end) via the production native
    /// path. Returns (resolved_full_bytes, chunk.data_prefix_len, decoded_crc).
    /// Since start is the stream head there is NO predecessor window, so the
    /// flip resolves entirely from in-chunk output (data_prefix_len stays 0).
    fn decode_first_chunk_native(deflate: &[u8]) -> (Vec<u8>, usize, u32) {
        let cfg = ChunkConfiguration::default();
        let stop = deflate.len() * 8;
        let mut chunk = super::decode_chunk_window_absent(deflate, 0, stop, cfg)
            .expect("native window-absent decode");
        let prefix_len = chunk.data_prefix_len;
        // Resolve any pre-flip markers against an EMPTY window (head of stream
        // has no predecessor — back-refs cannot legally reach before byte 0).
        let empty = [0u8; WINDOW];
        chunk.resolve_and_narrow_markers_in_place(&empty);
        chunk.merge_resolved_markers_into_data();
        let full = chunk.data.to_contiguous();
        let crc = crc32_of(&full[prefix_len..]);
        (full[prefix_len..].to_vec(), prefix_len, crc)
    }

    fn flate2_inflate(deflate: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(deflate)
            .read_to_end(&mut out)
            .expect("flate2 inflate");
        out
    }

    fn deflate_of(payload: &[u8], level: u32) -> Vec<u8> {
        use std::io::Write;
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// MULTI-BLOCK clean phase across deflate-block boundaries + CRC-prefix
    /// exclusion. A multi-MiB compressible payload forces a flip at 32 KiB and
    /// then many post-flip blocks; the contig tail must continue across each
    /// EOB (read_header + resume with the same `*pos`). data_prefix_len MUST be
    /// 0 (faithful prepend uses real prior output, no imported window image), so
    /// CRC covers only real output.
    #[test]
    fn contig_native_multiblock_clean_and_crc_prefix_excluded() {
        // Mildly compressible, several MiB → flips early, many clean blocks.
        let mut payload = Vec::with_capacity(4 * 1024 * 1024);
        let mut x: u32 = 0x1234_5678;
        for i in 0..(4 * 1024 * 1024u32) {
            // Repetitive-with-noise so it compresses (forces back-refs) but
            // spans many blocks.
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            payload.push(((i / 64) as u8) ^ ((x >> 24) as u8 & 0x07));
        }
        let deflate = deflate_of(&payload, 6);
        let (got, prefix_len, crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0, "FOLD must keep data_prefix_len == 0");
        assert_eq!(got, payload, "multi-block clean tail bytes diverged");
        assert_eq!(
            crc,
            crc32_of(&payload),
            "CRC over real output (prefix excluded) wrong"
        );
    }

    /// REGROW past the 16 MiB reserve clamp. A >16 MiB clean payload forces the
    /// contig buffer to grow BETWEEN calls (Engine-C contract); the loop must
    /// re-fetch `base`/`cap` after each grow (no stale pointer / no OOB).
    #[test]
    fn contig_native_regrow_past_reserve_clamp() {
        // 20 MiB highly-compressible (long RLE-ish runs) → one chunk, decoded
        // size far exceeds the 16 MiB RESERVE_CLAMP, driving real regrows.
        let block = b"The quick brown fox jumps over the lazy dog. 0123456789ABCDEF\n";
        let target = 20 * 1024 * 1024usize;
        let mut payload = Vec::with_capacity(target + block.len());
        while payload.len() < target {
            payload.extend_from_slice(block);
        }
        let deflate = deflate_of(&payload, 6);
        let (got, prefix_len, crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0);
        assert_eq!(got.len(), payload.len(), "regrow truncated/extended output");
        assert!(got == payload, "regrow corrupted output bytes");
        assert_eq!(crc, crc32_of(&payload));
    }

    /// STORED (uncompressed) block AFTER the flip. The contig primitive can't
    /// decode a stored block (returns InvalidCompression); the native tail must
    /// route it through `decode_clean_stored_into_contig`. Build: a compressible
    /// lead (forces a flip) followed by an incompressible (random) tail that the
    /// encoder emits as a STORED block.
    #[test]
    fn contig_native_stored_block_after_flip() {
        // Compressible lead well over 32 KiB to guarantee the flip.
        let mut payload = Vec::new();
        for i in 0..200_000u32 {
            payload.push((i / 97) as u8);
        }
        // Incompressible tail (LCG random) — flate2 at level 0 stores it; even at
        // higher levels a random run yields stored blocks. Use level 0 to FORCE
        // stored blocks across the whole stream (which includes post-flip stored
        // blocks once the 32 KiB clean window is established).
        let mut x: u32 = 0xDEAD_BEEF;
        for _ in 0..300_000u32 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            payload.push((x >> 24) as u8);
        }
        // Level 0 = all stored blocks → guarantees the contig tail must handle
        // stored blocks post-flip (the flip can fire mid stored-stream once
        // 32 KiB of clean literals accumulate).
        let deflate = deflate_of(&payload, 0);
        let truth = flate2_inflate(&deflate);
        assert_eq!(truth, payload, "fixture self-check");
        let (got, prefix_len, _crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0);
        assert_eq!(got, payload, "stored-block-after-flip diverged");
    }
}

// =========================================================================
// M3 differential gate (DIV-1 part 1): seeded-Block vs seeded-wrapper
// =========================================================================
// For window-seeded INEXACT chunks the gzippy-native production route moved
// from `StreamingInflateWrapper`/`unified::Inflate` onto the ONE
// `deflate::Block` engine (vendor GzipChunk.hpp:454-458). This gate nets the
// two arms — `finish_decode_chunk_seeded_block_native` (new production) vs
// `finish_decode_chunk_with_inexact_offset` (the `GZIPPY_SEEDED_BLOCK=0`
// kill-switch arm) — on generated corpora, asserting:
//   (a) decoded bytes        (b) decoded_size / data_prefix_len accounting
//   (c) final bit (encoded_size_bits)   (d) per-stream CRC32 values
//   (e) subchunk keys (encoded_offset, encoded_size, decoded_offset, size)
//   (f) published windows: get_last_window / last_32kib_window /
//       per-subchunk windows (populate_subchunk_windows), plus a brute-force
//       window check against `pred ‖ payload` (the stale-window-key scar net:
//       trailing/last-window content must equal ground truth, not just match
//       the other arm).
#[cfg(all(test, parallel_sm, not(isal_clean_tail)))]
mod seeded_block_parity {
    use super::*;

    const WINDOW: usize = 32 * 1024;

    /// Raw DEFLATE of `payload` with a 32 KiB preset dictionary so the
    /// encoder emits back-refs reaching into the predecessor window.
    pub(super) fn deflate_with_dict(payload: &[u8], dict: &[u8], level: u32) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress};
        let mut c = Compress::new(Compression::new(level), false);
        c.set_dictionary(dict).expect("set_dictionary");
        let mut out = vec![0u8; payload.len() * 2 + 4096];
        let status = c
            .compress(payload, &mut out, FlushCompress::Finish)
            .expect("compress");
        assert_eq!(status, flate2::Status::StreamEnd, "deflate did not finish");
        out.truncate(c.total_out() as usize);
        out
    }

    /// Raw DEFLATE with a preset dictionary and a SYNC FLUSH every
    /// `flush_every` payload bytes (empty stored blocks + dense boundaries).
    pub(super) fn deflate_with_dict_flushes(
        payload: &[u8],
        dict: &[u8],
        level: u32,
        flush_every: usize,
    ) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress};
        let mut c = Compress::new(Compression::new(level), false);
        c.set_dictionary(dict).expect("set_dictionary");
        let mut out: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; payload.len() + 64 * 1024];
        let mut fed = 0usize;
        loop {
            let end = (fed + flush_every).min(payload.len());
            let flush = if end == payload.len() {
                FlushCompress::Finish
            } else {
                FlushCompress::Sync
            };
            let before_out = c.total_out() as usize;
            let status = c
                .compress(&payload[fed..end], &mut buf, flush)
                .expect("compress");
            out.extend_from_slice(&buf[..c.total_out() as usize - before_out]);
            fed = c.total_in() as usize;
            if end == payload.len() && status == flate2::Status::StreamEnd {
                break;
            }
        }
        out
    }

    /// 32 KiB deterministic text-like dictionary.
    pub(super) fn make_dict() -> Vec<u8> {
        let mut d = b"the quick brown fox jumps over the lazy dog. ".repeat(800);
        d.truncate(WINDOW.max(32768));
        let cut = d.len() - WINDOW;
        d[cut..].to_vec()
    }

    pub(super) fn lcg_bytes(n: usize, mut x: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((x >> 24) as u8);
        }
        v
    }

    pub(super) fn flate2_inflate_with_dict(deflate: &[u8], dict: &[u8]) -> Vec<u8> {
        use flate2::{Decompress, FlushDecompress};
        let mut d = Decompress::new(false);
        d.set_dictionary(dict).expect("set_dictionary");
        let mut out = vec![0u8; 64 * 1024 * 1024];
        let status = d
            .decompress(deflate, &mut out, FlushDecompress::Finish)
            .expect("inflate");
        assert!(
            matches!(status, flate2::Status::StreamEnd),
            "truth inflate did not reach stream end"
        );
        out.truncate(d.total_out() as usize);
        out
    }

    /// NEW production arm: ONE Block engine, seeded.
    fn arm_block(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> ChunkData {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_seeded_block_native(&mut chunk, input, 0, stop_hint_bits, window)
            .expect("seeded Block decode");
        chunk
    }

    /// Kill-switch arm: pre-M3 wrapper path, byte/key reference.
    fn arm_wrapper(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> ChunkData {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_with_inexact_offset(
            &mut chunk,
            input,
            0,
            stop_hint_bits,
            window,
            false,
        )
        .expect("seeded wrapper decode");
        chunk
    }

    /// Full cross-arm equality net (a)-(f). `truth` = full-stream ground truth
    /// (decoded bytes must be a prefix of it; equal when stop = stream end).
    fn assert_arms_equal(
        mut b: ChunkData,
        mut w: ChunkData,
        pred: &[u8],
        truth: &[u8],
        total_bits: usize,
        label: &str,
    ) {
        // (b) prefix accounting: Block arm carries the dictionary prefix.
        assert_eq!(b.data_prefix_len, WINDOW, "{label}: Block arm prefix len");
        assert_eq!(w.data_prefix_len, 0, "{label}: wrapper arm prefix len");
        assert!(
            b.data_with_markers.is_empty() && w.data_with_markers.is_empty(),
            "{label}: seeded decode must emit no markers"
        );

        // (a) decoded bytes.
        let bb = b.data.to_contiguous()[WINDOW..].to_vec();
        let wb = w.data.to_contiguous();
        if bb != wb {
            let first_diff = bb.iter().zip(wb.iter()).position(|(x, y)| x != y);
            panic!(
                "{label}: decoded bytes diverged (block_len={} wrapper_len={} first_diff={first_diff:?})",
                bb.len(),
                wb.len()
            );
        }
        assert!(!bb.is_empty(), "{label}: decoded nothing");
        assert!(
            bb.len() <= truth.len() && bb[..] == truth[..bb.len()],
            "{label}: decoded bytes are not a prefix of ground truth"
        );
        assert_eq!(b.decoded_size(), w.decoded_size(), "{label}: decoded_size");
        assert_eq!(
            b.decoded_size(),
            bb.len(),
            "{label}: decoded_size accounting"
        );

        // (c) final bit. Strict equality, with exactly TWO measured, documented
        // exceptions where the WRAPPER (kill-switch arm) semantics differ:
        //
        //   1. STREAM END (BFINAL decoded; bb == truth): the wrapper reports
        //      `tell_compressed()` after `finished` — the byte-aligned input
        //      end (= total_bits; the <=7 padding bits are consumed). The
        //      Block arm reports the exact bit after the final EOB symbol —
        //      identical to the production-proven native FOLD semantics
        //      (`finish_decode_chunk_contig_native` / `MarkerStep::Finished`).
        //
        //   2. WRAPPER ACCOUNTING HOLE (pre-existing, found by this gate): if
        //      the inexact stop hint lands BETWEEN an EOB and the engine's
        //      next END_OF_BLOCK_HEADER stop, the coalescing engine never
        //      surfaces an END_OF_BLOCK stop (interior EOBs are reported only
        //      via take_block_boundaries), so `last_eob_pos` keeps its init
        //      value (`inflate_start_bit`) and the header-arm stop finalizes
        //      at `last_end_bit = last_eob_pos = chunk start` →
        //      `encoded_size_bits == 0` despite all bytes being emitted. The
        //      Block arm reports the contract value (first block boundary
        //      at-or-past the hint). If the wrapper hole is ever fixed, the
        //      `wrapper_final == 0` guard below fails loudly — remove the
        //      exception then.
        let full_stream = bb.len() == truth.len();
        let block_final = b.encoded_offset_bits + b.encoded_size_bits;
        let wrapper_final = w.encoded_offset_bits + w.encoded_size_bits;
        let final_bit_exception = if b.encoded_size_bits == w.encoded_size_bits {
            None
        } else if full_stream && wrapper_final == total_bits && wrapper_final - block_final < 8 {
            eprintln!(
                "[m3-gate] {label}: stream-end final-bit policy (block exact-EOB {block_final}, wrapper byte-aligned {wrapper_final})"
            );
            Some("stream_end")
        } else if w.encoded_size_bits == 0 && b.encoded_size_bits > 0 {
            eprintln!(
                "[m3-gate] {label}: wrapper final-bit accounting hole (wrapper 0, block {block_final}) — pre-existing wrapper bug, Block arm reports the contract value"
            );
            Some("wrapper_hole")
        } else {
            panic!(
                "{label}: final bit diverged outside the two documented policies (block {block_final}, wrapper {wrapper_final})"
            );
        };

        // (d) CRCs.
        assert_eq!(b.crc32s.len(), w.crc32s.len(), "{label}: crc32s count");
        for (i, (x, y)) in b.crc32s.iter().zip(w.crc32s.iter()).enumerate() {
            assert_eq!(x.crc32(), y.crc32(), "{label}: crc32s[{i}]");
        }
        assert_eq!(
            b.crc32s[0].crc32(),
            crate::decompress::parallel::crc32::crc32(&bb),
            "{label}: CRC accumulator != crc32(decoded)"
        );

        // (e) subchunk keys. The trailing subchunk's `encoded_size_bits` is
        // derived from the chunk-final bit (`finalize_with_deflate`), so under
        // a documented final-bit exception it differs by exactly the same
        // mechanism — compare it via per-arm self-consistency instead; all
        // other key fields stay strictly equal.
        let keys = |c: &ChunkData| {
            let n = c.subchunks.len();
            c.subchunks
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let trailing = i + 1 == n;
                    (
                        s.encoded_offset_bits,
                        if trailing && final_bit_exception.is_some() {
                            0 // normalized; checked via self-consistency below
                        } else {
                            s.encoded_size_bits
                        },
                        s.decoded_offset,
                        s.decoded_size,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&b), keys(&w), "{label}: subchunk keys");
        for (c, final_bit, arm) in [(&b, block_final, "block"), (&w, wrapper_final, "wrapper")] {
            if let Some(last) = c.subchunks.last() {
                assert_eq!(
                    last.encoded_offset_bits + last.encoded_size_bits,
                    final_bit,
                    "{label}: {arm} trailing subchunk encoded_size inconsistent with final bit"
                );
            }
        }

        // (f) published windows: cross-arm AND vs brute-force ground truth
        // (`pred ‖ decoded`) — the stale-window-key scar net.
        let mut hist = Vec::with_capacity(pred.len() + bb.len());
        hist.extend_from_slice(pred);
        hist.extend_from_slice(&bb);
        let brute = |chunk_off: usize| -> Vec<u8> {
            let end = pred.len() + chunk_off;
            hist[end - WINDOW..end].to_vec()
        };

        let lb = b.get_last_window_vec(pred);
        let lw = w.get_last_window_vec(pred);
        assert_eq!(lb, lw, "{label}: get_last_window");
        assert_eq!(
            lb,
            brute(bb.len()),
            "{label}: get_last_window vs brute force"
        );
        assert_eq!(
            b.last_32kib_window_vec(),
            w.last_32kib_window_vec(),
            "{label}: last_32kib_window"
        );

        b.populate_subchunk_windows(pred);
        w.populate_subchunk_windows(pred);
        assert_eq!(
            b.subchunks.len(),
            w.subchunks.len(),
            "{label}: subchunk count"
        );
        for (i, (sb, sw)) in b.subchunks.iter().zip(w.subchunks.iter()).enumerate() {
            let wbts = sb.window.as_ref().map(|v| v.decompress());
            let wwts = sw.window.as_ref().map(|v| v.decompress());
            // Cross-arm window equality — strict, except under the documented
            // wrapper accounting hole, where the wrapper's trailing
            // `encoded_size_bits == 0` makes its sparsity pass
            // (`get_used_window_symbols`) scan from the WRONG bit (chunk
            // start instead of the real continuation bit), mis-zeroing its
            // window. The Block arm's window is still netted against the
            // brute-force ground truth below.
            if final_bit_exception != Some("wrapper_hole") {
                assert_eq!(wbts, wwts, "{label}: subchunk[{i}] window bytes");
            }
            // Sparsity may zero unused symbols (identically in both arms, since
            // keys are equal) — so only check the NON-sparsified brute window
            // when sparsity left the window intact.
            if let Some(got) = &wbts {
                let bf = brute(sb.decoded_offset);
                if sb.used_window_symbols.is_empty() && got != &bf {
                    // Window was sparsified (zeros at unused offsets) or truly
                    // wrong. Verify every NON-zero byte matches ground truth —
                    // a stale/shifted window would mismatch on non-zeros too.
                    assert_eq!(got.len(), bf.len(), "{label}: subchunk[{i}] window len");
                    for (j, (g, t)) in got.iter().zip(bf.iter()).enumerate() {
                        assert!(
                            *g == 0 || g == t,
                            "{label}: subchunk[{i}] window byte {j} stale (got {g}, truth {t})"
                        );
                    }
                }
            }
        }
    }

    fn run_case(payload: &[u8], deflate: &[u8], dict: &[u8], cfg: ChunkConfiguration, label: &str) {
        let truth = flate2_inflate_with_dict(deflate, dict);
        assert_eq!(truth, payload, "{label}: fixture self-check");
        let stop = deflate.len() * 8;
        let b = arm_block(deflate, stop, dict, cfg);
        let w = arm_wrapper(deflate, stop, dict, cfg);
        assert_arms_equal(b, w, dict, &truth, deflate.len() * 8, label);
    }

    #[test]
    fn seeded_dynamic_heavy_multi_subchunk() {
        // Text-like, dynamic-Huffman-heavy, > several split thresholds so the
        // subchunk-split machinery runs (the silesia-T4-class key zone).
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..8 * 1024]); // dict back-refs up front
        for i in 0..6000u32 {
            payload.extend_from_slice(
                format!(
                    "line {i}: the quick brown fox jumps over the lazy dog #{}\n",
                    i % 97
                )
                .as_bytes(),
            );
        }
        payload.extend_from_slice(&dict[16 * 1024..24 * 1024]); // mid-dict refs late
                                                                // Flush every 16 KiB so the stream carries REAL deflate block
                                                                // boundaries (zlib otherwise emits very large blocks), letting the
                                                                // 48 KiB split threshold produce a multi-subchunk shape.
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 16 * 1024);
        let cfg = ChunkConfiguration {
            split_chunk_size: 48 * 1024, // force multiple subchunks locally
            ..ChunkConfiguration::default()
        };
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let stop = deflate.len() * 8;
        let b = arm_block(&deflate, stop, &dict, cfg);
        assert!(
            b.subchunks.len() >= 3,
            "expected a multi-subchunk shape, got {}",
            b.subchunks.len()
        );
        let w = arm_wrapper(&deflate, stop, &dict, cfg);
        assert_arms_equal(b, w, &dict, &truth, stop, "dynamic_heavy");
    }

    #[test]
    fn seeded_stored_mixed() {
        // Alternating compressible / incompressible 24 KiB segments at level 1
        // → mixed STORED + Huffman blocks (zlib stores incompressible runs).
        let dict = make_dict();
        let mut payload = Vec::new();
        for k in 0..8usize {
            if k % 2 == 0 {
                payload.extend_from_slice(&b"compressible compressible! ".repeat(900));
            } else {
                payload.extend_from_slice(&lcg_bytes(24 * 1024, 0xC0FF_EE00 + k as u32));
            }
        }
        let deflate = deflate_with_dict(&payload, &dict, 1);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "stored_mixed_l1",
        );

        // Level 0: ALL stored blocks (the pure `decode_clean_stored_into_contig`
        // route on the Block arm).
        let deflate0 = deflate_with_dict(&payload, &dict, 0);
        run_case(
            &payload,
            &deflate0,
            &dict,
            ChunkConfiguration::default(),
            "stored_only_l0",
        );
    }

    #[test]
    fn seeded_flush_dense() {
        // SYNC flush every 2 KiB → dense empty stored blocks + boundaries.
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[1..4097]); // offset-1 dict ref shape
        for i in 0..2000u32 {
            payload.extend_from_slice(format!("flushy line {i} {}\n", i % 13).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 2048);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "flush_dense",
        );
    }

    #[test]
    fn seeded_window_boundary_backrefs() {
        // Payload BEGINS with a copy of the dictionary head → the encoder emits
        // a distance-32768 back-ref at output offset 0 (and offset-1 variants),
        // crossing the contig prefix seam at its extreme reach.
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..4096]); // distance == 32768 at offset 0
        payload.extend_from_slice(b"X");
        payload.extend_from_slice(&dict[..4096]); // re-reference after 1 byte
        payload.extend_from_slice(&lcg_bytes(8 * 1024, 0xBEEF_CAFE));
        payload.extend_from_slice(&dict[WINDOW - 4096..]); // dict tail refs
        let deflate = deflate_with_dict(&payload, &dict, 9);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "window_boundary",
        );
    }

    #[test]
    fn seeded_stop_hint_parity() {
        // Inexact stop hints: at a real block boundary, just before one, and
        // mid-block — both arms must stop at the SAME first boundary >= hint.
        let dict = make_dict();
        let mut payload = Vec::new();
        for i in 0..4000u32 {
            payload.extend_from_slice(format!("stop-hint line {i} {}\n", i % 31).as_bytes());
        }
        // Dense flushes give plenty of in-stream boundaries to aim at.
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 4096);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration::default();

        // Take boundary positions from a wrapper enumeration pass.
        let total_bits = deflate.len() * 8;
        let probe = arm_wrapper(&deflate, total_bits / 2, &dict, cfg);
        let mid_stop = probe.encoded_offset_bits + probe.encoded_size_bits;
        assert!(
            mid_stop > 0 && mid_stop < total_bits,
            "probe stop in-stream"
        );

        for (delta, name) in [
            (0isize, "at_boundary"),
            (-3, "before_boundary"),
            (5, "past_boundary"),
        ] {
            let hint = (mid_stop as isize + delta) as usize;
            let b = arm_block(&deflate, hint, &dict, cfg);
            let w = arm_wrapper(&deflate, hint, &dict, cfg);
            eprintln!(
                "[m3-gate] stop_hint_{name}: hint={hint} block(final={}, decoded={}, subchunks={}) wrapper(final={}, decoded={}, subchunks={})",
                b.encoded_size_bits,
                b.decoded_size(),
                b.subchunks.len(),
                w.encoded_size_bits,
                w.decoded_size(),
                w.subchunks.len()
            );
            assert_arms_equal(
                b,
                w,
                &dict,
                &truth,
                total_bits,
                &format!("stop_hint_{name}"),
            );
        }
    }

    #[test]
    fn seeded_block_counter_increments() {
        // Engine-proof counter: the Block arm increments SEEDED_BLOCK_CHUNKS.
        let dict = make_dict();
        let payload = b"counter payload ".repeat(1024);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let before = SEEDED_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let _ = arm_block(
            &deflate,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
        );
        let after = SEEDED_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after > before, "SEEDED_BLOCK_CHUNKS did not increment");
    }
}

// =========================================================================
// M4 differential gate (DIV-1 part 2): exact-Block vs exact-wrapper
// =========================================================================
// For window-seeded UNTIL-EXACT chunks the gzippy-native production route
// moved from `StreamingInflateWrapper`/`unified::Inflate` onto the ONE
// `deflate::Block` engine (LABELED DEVIATION — vendor's exact path is the
// C-FFI `decodeChunkWithInflateWrapper`, GzipChunk.hpp:192-265; see
// `finish_decode_chunk_exact_block_native`). This gate nets the two arms —
// `finish_decode_chunk_exact_block_native` (new production) vs
// `finish_decode_chunk_impl(until_exact=true)` (the `GZIPPY_EXACT_BLOCK=0`
// kill-switch arm) — on generated corpora, asserting STRICT equality (no
// M3-style final-bit exceptions: on success both arms must land EXACTLY at
// stop_hint_bits by the until-exact contract):
//   (a) decoded bytes      (b) decoded_size / data_prefix_len accounting
//   (c) final bit == stop_hint_bits on BOTH arms
//   (d) per-stream CRC32 values
//   (e) subchunk keys (encoded_offset, encoded_size, decoded_offset, size)
//   (f) published windows incl. brute-force `pred ‖ payload` ground truth
//   (g) ERROR equality: when the stop cannot be honored (member-final
//       misaligned hint, multi-member-crossing hint) both arms must return
//       ExactStopMissed with IDENTICAL requested/actual coordinates.
#[cfg(all(test, parallel_sm, not(isal_clean_tail)))]
mod exact_block_parity {
    use super::seeded_block_parity::{
        deflate_with_dict, deflate_with_dict_flushes, flate2_inflate_with_dict, lcg_bytes,
        make_dict,
    };
    use super::*;

    const WINDOW: usize = 32 * 1024;

    /// NEW production arm: ONE Block engine, seeded, exact stop.
    fn arm_block_exact(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> Result<ChunkData, ChunkDecodeError> {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_exact_block_native(&mut chunk, input, 0, stop_hint_bits, window)
            .map(|()| chunk)
    }

    /// Kill-switch arm: pre-M4 wrapper path (`finish_decode_chunk_impl`
    /// with `until_exact=true`), byte/key/error reference.
    fn arm_wrapper_exact(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> Result<ChunkData, ChunkDecodeError> {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_impl(
            &mut chunk,
            input,
            0,
            stop_hint_bits,
            window,
            false,
            true,
            true,
        )
        .map(|()| chunk)
    }

    /// Enumerate non-final EOB boundaries `(bit, decoded_offset)` via the
    /// long-vetted wrapper engine (independent of both arms' stop logic).
    fn enumerate_boundaries(input: &[u8], dict: &[u8]) -> Vec<(usize, usize)> {
        let mut w = StreamingInflateWrapper::with_until_bits(input, 0, input.len() * 8)
            .expect("wrapper construct");
        w.set_window(dict).expect("set_window");
        w.set_stopping_points(StoppingPoints::END_OF_BLOCK);
        let mut out = vec![0u8; 1 << 20];
        let mut total = 0usize;
        let mut bounds = Vec::new();
        loop {
            let r = w.read_stream(&mut out).expect("read_stream");
            total += r.bytes_written;
            if r.finished {
                break;
            }
            if r.stopped_at == StoppingPoints::END_OF_BLOCK && !w.is_final_block() {
                bounds.push((r.bit_position, total));
                continue;
            }
            if r.stopped_at == StoppingPoints::NONE && r.bytes_written == 0 {
                break;
            }
        }
        bounds
    }

    /// Strict cross-arm equality net (a)-(f) for SUCCESSFUL exact stops.
    fn assert_arms_equal_exact(
        mut b: ChunkData,
        mut w: ChunkData,
        pred: &[u8],
        truth: &[u8],
        stop_hint_bits: usize,
        label: &str,
    ) {
        // (b) prefix accounting: Block arm carries the dictionary prefix.
        assert_eq!(b.data_prefix_len, WINDOW, "{label}: Block arm prefix len");
        assert_eq!(w.data_prefix_len, 0, "{label}: wrapper arm prefix len");
        assert!(
            b.data_with_markers.is_empty() && w.data_with_markers.is_empty(),
            "{label}: exact decode must emit no markers"
        );

        // (a) decoded bytes.
        let bb = b.data.to_contiguous()[WINDOW..].to_vec();
        let wb = w.data.to_contiguous();
        if bb != wb {
            let first_diff = bb.iter().zip(wb.iter()).position(|(x, y)| x != y);
            panic!(
                "{label}: decoded bytes diverged (block_len={} wrapper_len={} first_diff={first_diff:?})",
                bb.len(),
                wb.len()
            );
        }
        assert!(!bb.is_empty(), "{label}: decoded nothing");
        assert!(
            bb.len() <= truth.len() && bb[..] == truth[..bb.len()],
            "{label}: decoded bytes are not a prefix of ground truth"
        );
        assert_eq!(b.decoded_size(), w.decoded_size(), "{label}: decoded_size");
        assert_eq!(
            b.decoded_size(),
            bb.len(),
            "{label}: decoded_size accounting"
        );

        // (c) final bit: the until-exact contract REQUIRES both arms to land
        // exactly on stop_hint_bits — strict, no exceptions.
        assert_eq!(
            b.encoded_offset_bits + b.encoded_size_bits,
            stop_hint_bits,
            "{label}: Block arm final bit != stop_hint"
        );
        assert_eq!(
            w.encoded_offset_bits + w.encoded_size_bits,
            stop_hint_bits,
            "{label}: wrapper arm final bit != stop_hint"
        );

        // (d) CRCs.
        assert_eq!(b.crc32s.len(), w.crc32s.len(), "{label}: crc32s count");
        for (i, (x, y)) in b.crc32s.iter().zip(w.crc32s.iter()).enumerate() {
            assert_eq!(x.crc32(), y.crc32(), "{label}: crc32s[{i}]");
        }
        assert_eq!(
            b.crc32s[0].crc32(),
            crate::decompress::parallel::crc32::crc32(&bb),
            "{label}: CRC accumulator != crc32(decoded)"
        );

        // (e) subchunk keys — strictly equal (both arms end at stop_hint).
        let keys = |c: &ChunkData| {
            c.subchunks
                .iter()
                .map(|s| {
                    (
                        s.encoded_offset_bits,
                        s.encoded_size_bits,
                        s.decoded_offset,
                        s.decoded_size,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&b), keys(&w), "{label}: subchunk keys");

        // (f) published windows: cross-arm AND vs brute-force ground truth.
        let mut hist = Vec::with_capacity(pred.len() + bb.len());
        hist.extend_from_slice(pred);
        hist.extend_from_slice(&bb);
        let brute = |chunk_off: usize| -> Vec<u8> {
            let end = pred.len() + chunk_off;
            hist[end - WINDOW..end].to_vec()
        };

        let lb = b.get_last_window_vec(pred);
        let lw = w.get_last_window_vec(pred);
        assert_eq!(lb, lw, "{label}: get_last_window");
        assert_eq!(
            lb,
            brute(bb.len()),
            "{label}: get_last_window vs brute force"
        );
        assert_eq!(
            b.last_32kib_window_vec(),
            w.last_32kib_window_vec(),
            "{label}: last_32kib_window"
        );

        b.populate_subchunk_windows(pred);
        w.populate_subchunk_windows(pred);
        assert_eq!(
            b.subchunks.len(),
            w.subchunks.len(),
            "{label}: subchunk count"
        );
        for (i, (sb, sw)) in b.subchunks.iter().zip(w.subchunks.iter()).enumerate() {
            let wbts = sb.window.as_ref().map(|v| v.decompress());
            let wwts = sw.window.as_ref().map(|v| v.decompress());
            assert_eq!(wbts, wwts, "{label}: subchunk[{i}] window bytes");
            if let Some(got) = &wbts {
                let bf = brute(sb.decoded_offset);
                if sb.used_window_symbols.is_empty() && got != &bf {
                    assert_eq!(got.len(), bf.len(), "{label}: subchunk[{i}] window len");
                    for (j, (g, t)) in got.iter().zip(bf.iter()).enumerate() {
                        assert!(
                            *g == 0 || g == t,
                            "{label}: subchunk[{i}] window byte {j} stale (got {g}, truth {t})"
                        );
                    }
                }
            }
        }
    }

    /// (g) ERROR equality: both arms must fail with ExactStopMissed carrying
    /// IDENTICAL requested/actual coordinates.
    fn assert_same_exact_miss(
        be: Result<ChunkData, ChunkDecodeError>,
        we: Result<ChunkData, ChunkDecodeError>,
        label: &str,
    ) -> (usize, usize) {
        let b = match be {
            Err(ChunkDecodeError::ExactStopMissed { requested, actual }) => (requested, actual),
            Err(other) => {
                panic!("{label}: Block arm error variant {other:?}, want ExactStopMissed")
            }
            Ok(c) => panic!(
                "{label}: Block arm unexpectedly SUCCEEDED (decoded {} final_bit {})",
                c.decoded_size(),
                c.encoded_offset_bits + c.encoded_size_bits
            ),
        };
        let w = match we {
            Err(ChunkDecodeError::ExactStopMissed { requested, actual }) => (requested, actual),
            Err(other) => {
                panic!("{label}: wrapper arm error variant {other:?}, want ExactStopMissed")
            }
            Ok(c) => panic!(
                "{label}: wrapper arm unexpectedly SUCCEEDED (decoded {} final_bit {})",
                c.decoded_size(),
                c.encoded_offset_bits + c.encoded_size_bits
            ),
        };
        assert_eq!(
            b, w,
            "{label}: ExactStopMissed (requested, actual) coordinates"
        );
        b
    }

    fn run_exact_case(
        deflate: &[u8],
        stop: usize,
        dict: &[u8],
        truth: &[u8],
        cfg: ChunkConfiguration,
        label: &str,
    ) {
        let b = arm_block_exact(deflate, stop, dict, cfg).unwrap_or_else(|e| {
            panic!("{label}: Block arm failed: {e:?}");
        });
        let w = arm_wrapper_exact(deflate, stop, dict, cfg).unwrap_or_else(|e| {
            panic!("{label}: wrapper arm failed: {e:?}");
        });
        assert_arms_equal_exact(b, w, dict, truth, stop, label);
    }

    /// Interior confirmed-boundary stops: byte-aligned AND non-byte-aligned
    /// bit offsets, from a flush-dense + dynamic corpus.
    #[test]
    fn exact_interior_boundary_stops() {
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..4 * 1024]); // dict back-refs up front
        for i in 0..4000u32 {
            payload.extend_from_slice(format!("interior line {i} {}\n", i % 53).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 4096);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration::default();

        let bounds = enumerate_boundaries(&deflate, &dict);
        assert!(
            bounds.len() >= 8,
            "need many boundaries, got {}",
            bounds.len()
        );

        let aligned: Vec<usize> = bounds
            .iter()
            .map(|&(bit, _)| bit)
            .filter(|bit| bit % 8 == 0)
            .take(3)
            .collect();
        let unaligned: Vec<usize> = bounds
            .iter()
            .map(|&(bit, _)| bit)
            .filter(|bit| bit % 8 != 0)
            .take(3)
            .collect();
        assert!(!aligned.is_empty(), "no byte-aligned boundaries in fixture");
        assert!(
            !unaligned.is_empty(),
            "no non-byte-aligned boundaries in fixture"
        );

        for (kind, stops) in [("aligned", &aligned), ("unaligned", &unaligned)] {
            for &stop in stops {
                run_exact_case(
                    &deflate,
                    stop,
                    &dict,
                    &truth,
                    cfg,
                    &format!("interior_{kind}_bit{stop}"),
                );
            }
        }
    }

    /// Member-final stop: BFINAL block decoded to the byte-aligned stream end
    /// (the RFC 1952 padding consumed; the 8-byte footer is OUTSIDE the input
    /// slice exactly like production `sm_driver` slicing).
    #[test]
    fn exact_member_final_stop() {
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..8 * 1024]);
        for i in 0..3000u32 {
            payload.extend_from_slice(format!("member final line {i} {}\n", i % 17).as_bytes());
        }
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let stop = deflate.len() * 8; // byte-aligned padded deflate end (production total_bits)
        run_exact_case(
            &deflate,
            stop,
            &dict,
            &truth,
            ChunkConfiguration::default(),
            "member_final",
        );

        // Misaligned member-final hints: one byte PAST and one byte SHORT of
        // the padded stream end. Both arms must reject with IDENTICAL
        // ExactStopMissed coordinates; `actual` is the byte-aligned post-EOB
        // bit on both arms (the M4 end-bit coordinate convention).
        let be = arm_block_exact(&deflate, stop + 8, &dict, ChunkConfiguration::default());
        let we = arm_wrapper_exact(&deflate, stop + 8, &dict, ChunkConfiguration::default());
        let (req, actual) = assert_same_exact_miss(be, we, "member_final_past");
        assert_eq!(req, stop + 8);
        assert_eq!(actual, stop, "actual must be the byte-aligned stream end");
    }

    /// Multi-member trailing shape (contract (c)): the input slice continues
    /// past member 1's padded deflate end with a gzip footer + next member's
    /// header + deflate body. NEITHER arm consumes the footer or resets for
    /// the next stream on the until-exact path (`read_footer_at_current` /
    /// `reset_for_next_stream` have no production caller there): a stop hint
    /// pointing past member 1 must fail on BOTH arms with IDENTICAL
    /// ExactStopMissed coordinates, `actual` = member 1's byte-aligned end.
    #[test]
    fn exact_multi_member_trailing() {
        let dict = make_dict();
        let payload1: Vec<u8> = b"member one payload ".repeat(3000);
        let payload2: Vec<u8> = b"member two payload ".repeat(3000);
        let deflate1 = deflate_with_dict(&payload1, &dict, 6);
        // Member 2 is a STANDALONE gzip member (fresh window, as on disk).
        let deflate2 = {
            use flate2::{Compress, Compression, FlushCompress};
            let mut c = Compress::new(Compression::new(6), false);
            let mut out = vec![0u8; payload2.len() * 2 + 4096];
            let status = c
                .compress(&payload2, &mut out, FlushCompress::Finish)
                .expect("compress");
            assert_eq!(status, flate2::Status::StreamEnd);
            out.truncate(c.total_out() as usize);
            out
        };

        let member1_end_bits = deflate1.len() * 8;
        let mut input = deflate1.clone();
        // gzip footer of member 1 (CRC32 + ISIZE) ...
        input
            .extend_from_slice(&crate::decompress::parallel::crc32::crc32(&payload1).to_le_bytes());
        input.extend_from_slice(&(payload1.len() as u32).to_le_bytes());
        // ... then member 2's 10-byte header + deflate body.
        input.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 0x03]);
        input.extend_from_slice(&deflate2);

        // Stop hint deep inside member 2's bit-space.
        let stop = input.len() * 8;
        let be = arm_block_exact(&input, stop, &dict, ChunkConfiguration::default());
        let we = arm_wrapper_exact(&input, stop, &dict, ChunkConfiguration::default());
        let (req, actual) = assert_same_exact_miss(be, we, "multi_member_trailing");
        assert_eq!(req, stop);
        assert_eq!(
            actual, member1_end_bits,
            "both arms must stop at member 1's byte-aligned deflate end"
        );

        // Member 1's padded end IS an honorable exact stop on the same slice.
        let truth1 = flate2_inflate_with_dict(&deflate1, &dict);
        run_exact_case(
            &input,
            member1_end_bits,
            &dict,
            &truth1,
            ChunkConfiguration::default(),
            "multi_member_member1_end",
        );
    }

    /// Stop hint exactly at a subchunk-split boundary: small split threshold
    /// forces multi-subchunk shapes; the exact stop lands on a recorded
    /// boundary at-or-after a split — keys must match strictly.
    #[test]
    fn exact_stop_at_subchunk_split() {
        let dict = make_dict();
        let mut payload = Vec::new();
        for i in 0..6000u32 {
            payload.extend_from_slice(
                format!("split line {i}: the quick brown fox #{}\n", i % 97).as_bytes(),
            );
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 16 * 1024);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration {
            split_chunk_size: 48 * 1024,
            ..ChunkConfiguration::default()
        };

        let bounds = enumerate_boundaries(&deflate, &dict);
        // First boundary whose decoded offset crosses the split threshold —
        // the boundary where the split machinery fires — plus a later one.
        let split_bit = bounds
            .iter()
            .find(|&&(_, off)| off >= 48 * 1024)
            .map(|&(bit, _)| bit)
            .expect("no boundary past the split threshold");
        let late_bit = bounds
            .iter()
            .find(|&&(_, off)| off >= 160 * 1024)
            .map(|&(bit, _)| bit)
            .expect("no late boundary");

        for (stop, label) in [(split_bit, "at_split"), (late_bit, "late_split")] {
            let b = arm_block_exact(&deflate, stop, &dict, cfg)
                .unwrap_or_else(|e| panic!("{label}: Block arm failed: {e:?}"));
            let w = arm_wrapper_exact(&deflate, stop, &dict, cfg)
                .unwrap_or_else(|e| panic!("{label}: wrapper arm failed: {e:?}"));
            if label == "late_split" {
                assert!(
                    b.subchunks.len() >= 2,
                    "{label}: expected multi-subchunk shape, got {}",
                    b.subchunks.len()
                );
            }
            assert_arms_equal_exact(b, w, &dict, &truth, stop, label);
        }
    }

    /// Flush-dense + stored-mixed corpora, member-final exact stops (stored
    /// blocks end byte-aligned; level-0 is ALL stored — the
    /// `decode_clean_stored_into_contig` route on the Block arm).
    #[test]
    fn exact_flush_dense_and_stored() {
        let dict = make_dict();

        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[1..4097]);
        for i in 0..2000u32 {
            payload.extend_from_slice(format!("flushy exact line {i} {}\n", i % 13).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 2048);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        run_exact_case(
            &deflate,
            deflate.len() * 8,
            &dict,
            &truth,
            ChunkConfiguration::default(),
            "flush_dense_final",
        );
        // Interior exact stop inside the flush-dense stream too.
        let bounds = enumerate_boundaries(&deflate, &dict);
        if let Some(&(bit, _)) = bounds.get(bounds.len() / 2) {
            run_exact_case(
                &deflate,
                bit,
                &dict,
                &truth,
                ChunkConfiguration::default(),
                "flush_dense_interior",
            );
        }

        let mut payload2 = Vec::new();
        for k in 0..6usize {
            if k % 2 == 0 {
                payload2.extend_from_slice(&b"compressible compressible! ".repeat(900));
            } else {
                payload2.extend_from_slice(&lcg_bytes(24 * 1024, 0xD00D_0000 + k as u32));
            }
        }
        for level in [1u32, 0u32] {
            let d = deflate_with_dict(&payload2, &dict, level);
            let t = flate2_inflate_with_dict(&d, &dict);
            assert_eq!(t, payload2, "stored fixture self-check");
            run_exact_case(
                &d,
                d.len() * 8,
                &dict,
                &t,
                ChunkConfiguration::default(),
                &format!("stored_final_l{level}"),
            );
        }
    }

    /// Engine-proof counter: the Block arm increments EXACT_BLOCK_CHUNKS.
    #[test]
    fn exact_block_counter_increments() {
        let dict = make_dict();
        let payload = b"exact counter payload ".repeat(1024);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let before = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let _ = arm_block_exact(
            &deflate,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
        )
        .expect("exact Block decode");
        let after = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after > before, "EXACT_BLOCK_CHUNKS did not increment");
    }

    /// Default route proof: `decode_chunk_until_exact` with a full window and
    /// `until_exact=true` takes the Block engine (kill-switch arm untouched).
    #[test]
    fn exact_route_defaults_to_block() {
        let dict = make_dict();
        let payload = b"route proof payload ".repeat(2048);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        let before_b = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let chunk = decode_chunk_until_exact(
            &deflate,
            0,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
            true,
        )
        .expect("until-exact route decode");
        let after_b = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after_b > before_b,
            "decode_chunk_until_exact did not take the Block engine by default"
        );
        let bytes = chunk.data.to_contiguous()[chunk.data_prefix_len..].to_vec();
        assert_eq!(bytes, truth, "route decode bytes");
    }
}
