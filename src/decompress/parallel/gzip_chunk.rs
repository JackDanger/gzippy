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

#[cfg(parallel_sm)]
#[inline]
fn isal_engine_oracle_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_ISAL_ENGINE_ORACLE").is_ok_and(|v| v == "1"))
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
    // Reserve enough for the chunk: target bytes + slack for the straddling block
    // (a boundary AT-OR-PAST stop_hint can be a full block beyond it). 64 MiB of
    // RESERVED (not allocated-and-filled) contiguous capacity is ample; ISA-L
    // stops at a boundary long before exhausting it. Reserve happens ONCE per
    // chunk (amortized into the chunk's own buffer, recycled by the pool) — it is
    // NOT a fresh per-chunk heap alloc+memset the way the old `Vec` was.
    let reserve_len: usize = 64 * 1024 * 1024;
    let decode_start = chunk.data.len();
    let (written, end_bit, boundaries) = {
        let out = chunk.data.writable_tail_reserve(reserve_len);
        match isal_decompress::decompress_deflate_from_bit_into(
            bounded,
            inflate_start_bit,
            initial_window,
            out,
        ) {
            Some(v) => v,
            None => return Ok(false),
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
            None => return Ok(false), // oracle can't honor exact stop ⇒ fall back
        }
    } else {
        let mut chosen = None;
        for b in &boundaries {
            if b.bit_offset >= stop_hint_bits {
                chosen = Some((b.bit_offset, b.output_offset));
                break;
            }
        }
        chosen.unwrap_or((end_bit, written))
    };

    let keep_len = keep_len.min(written);

    // Commit ONLY the kept region into the chunk's u8 data buffer. The bytes are
    // already physically present in the reserved spare (ISA-L wrote them); commit
    // bumps the logical length with zero copies.
    chunk.data.commit(keep_len);
    chunk.note_inner_decoded_bytes(keep_len);

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

    // Replay boundaries up to (and not past) the natural stop, mirroring the
    // pure path's `append_block_boundary_at` calls. Boundary output offsets are
    // relative to this decode's start; the chunk's absolute decoded offset is
    // `decode_start + output_offset` (== production's `decode_base + rel_off`).
    for b in &boundaries {
        if b.bit_offset > final_bit {
            break;
        }
        let decoded_offset = decode_start + b.output_offset.min(keep_len);
        if decoded_offset > 0 {
            chunk.append_block_boundary_at(b.bit_offset, decoded_offset, Some(input));
        }
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
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    finish_decode_chunk_impl(
        &mut chunk,
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        true,
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
) -> Result<(), ChunkDecodeError> {
    FINISH_DECODE_ENTRIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // PHASE-0 WALL ORACLE (GZIPPY_ISAL_ENGINE_ORACLE=1, measurement-only): decode
    // this clean tail with REAL ISA-L instead of the pure-Rust engine, keeping the
    // entire production pipeline. Falls back to the pure path on any uncovered
    // contract (counted in ISAL_ENGINE_ORACLE_FALLBACKS so a contaminated run is
    // detectable). NOT a production path.
    if isal_engine_oracle_enabled() {
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
            finish_decode_chunk_impl(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
                false,
                true,
            )?;
        } else {
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
/// DecodedData buffer (DecodedData.hpp:278-289). NOTE: "copy-free" here means no
/// `u8buf`/`pending_clean` middle-man — the engine ring write + the ring->data
/// drain memcpy remain (the ISA-L `ocl_cf` ceiling pays neither), so the residual
/// to that 0.925× ceiling is an UPPER BOUND on intrinsic symbol rate, not pure rate.
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
        if self.crc32_enabled {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(bytes);
            }
        }
        *self.non_marker_count += bytes.len() as u64;
        self.data.extend_from_slice(bytes);
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

/// The ONE unified decode driver — vendor `deflate::Block` by default (see
/// `marker_decode_step`; `GZIPPY_MARKER_RING=1` selects legacy `MarkerRing`).
/// Once 32 KiB of clean output exist at a block boundary, control hands off to
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
    // copy#2/3/grow but load-inflated). The residual to the engine-removed
    // ceiling (ocl_cf 0.925×) is ~0.188×, an UPPER BOUND on the intrinsic
    // symbol-rate gap that still includes the ring-write + ring->data drain
    // memcpy `ocl_cf` does not pay — not pure symbol rate.
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

/// Counter: chunks that took the window-present clean-from-block-0 fast path
/// (vendor `setInitialWindow`) instead of full marker decode.
#[cfg(parallel_sm)]
pub static WINDOW_SEEDED_CHUNKS: std::sync::atomic::AtomicU64 =
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
    if std::env::var_os("GZIPPY_MARKER_RING").is_some() {
        return marker_decode_step_marker_ring(ctx, data, stop_hint_bits, initial_window, output);
    }
    marker_decode_step_vendor_block(ctx, data, stop_hint_bits, initial_window, output)
}

/// Legacy fast-LUT bootstrap (`lut_bulk_inflate::MarkerRing`). `GZIPPY_MARKER_RING=1` only.
#[cfg(parallel_sm)]
fn marker_decode_step_marker_ring(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    initial_window: &[u8],
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    use crate::decompress::parallel::lut_bulk_inflate::MarkerRing;
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    use std::cell::RefCell;

    thread_local! {
        static BOOTSTRAP_RING: RefCell<MarkerRing> = RefCell::new(MarkerRing::new());
    }

    BOOTSTRAP_RING.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        if !ctx.block_primed {
            block.reset();
            if initial_window.len() == MAX_WINDOW_SIZE {
                block.set_initial_window_u8(initial_window);
            }
            ctx.block_primed = true;
        }
        marker_decode_step_loop(
            ctx,
            data,
            stop_hint_bits,
            output,
            &mut *block,
            |block, bits| block.read_header(bits),
            |block, bits, output| block.read(bits, output),
            |e| {
                use crate::decompress::parallel::lut_bulk_inflate::BulkDecodeError;
                use std::sync::atomic::Ordering;
                match e {
                    BulkDecodeError::InvalidHuffmanCode => {
                        BODY_FAIL_INVALID_HUFFMAN.fetch_add(1, Ordering::Relaxed);
                    }
                    BulkDecodeError::InvalidLookback => {
                        BODY_FAIL_EXCEEDED_WINDOW.fetch_add(1, Ordering::Relaxed);
                    }
                    BulkDecodeError::BlockTypeReserved => {
                        BODY_FAIL_INVALID_COMPRESSION.fetch_add(1, Ordering::Relaxed);
                    }
                    BulkDecodeError::InvalidCodeLengths => {
                        BODY_FAIL_INVALID_CODE_LENGTHS.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        BODY_FAIL_OTHER_VARIANT.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
        )
    })
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
    use crate::decompress::parallel::marker_inflate::{Block, MAX_WINDOW_SIZE};
    use std::cell::RefCell;

    thread_local! {
        static BOOTSTRAP_BLOCK: RefCell<Block> = RefCell::new(Block::new());
    }

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

/// Engine surface shared by vendor `Block` and legacy `MarkerRing`.
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

#[cfg(parallel_sm)]
impl BootstrapEngine for crate::decompress::parallel::lut_bulk_inflate::MarkerRing {
    fn contains_marker_bytes(&self) -> bool {
        self.contains_marker_bytes()
    }
    fn eob(&self) -> bool {
        self.eob()
    }
    fn is_last_block(&self) -> bool {
        self.is_last_block()
    }
}

/// Shared per-iteration body for vendor `Block` and legacy `MarkerRing`.
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
            // gzippy-native: fall through — no handoff, Engine M continues.
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

    #[test]
    fn isal_tail_matches_pure_tail_on_real_silesia_chunks() {
        let gz = std::fs::read("benchmark_data/silesia-gzip.tar.gz")
            .expect("benchmark_data/silesia-gzip.tar.gz must exist");
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
        let gz = std::fs::read("benchmark_data/silesia-gzip.tar.gz")
            .expect("benchmark_data/silesia-gzip.tar.gz must exist");
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
}
