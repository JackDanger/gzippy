#![cfg(parallel_sm)]

//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_with_rapidgzip`] — vendor `decodeChunkWithRapidgzip` +
//!   `finishDecodeChunkWithInexactOffset` on one [`ChunkData`]:
//!   one outer decode iteration (`worker.decode_chunk`) alternates
//!   `deflate_block` blocks (u16 markers) until 32 KiB clean, then streaming
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
    DeflateCompressionType, IsalInflateWrapper, StoppingPoints,
};
#[cfg(parallel_sm)]
use crate::decompress::parallel::rpmalloc_alloc::types;
#[cfg(parallel_sm)]
use crate::decompress::parallel::trace;

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
/// Commit-1 instrumentation for the bootstrap-unification lever: bytes decoded
/// into the u16 marker ring AFTER `contains_marker_bytes` flips false (the flip
/// is permanent within a chunk — once 32 KiB clean exists, no back-ref can
/// reach the unknown predecessor). These are the bytes that could instead be
/// decoded into a u8 linear buffer with the fast copy path (Design B1). The
/// ratio POST_FLIP / BODY_BYTES sizes the prize. No behavior change.
pub static BOOTSTRAP_POST_FLIP_U16_BYTES: std::sync::atomic::AtomicU64 =
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
    decode_chunk_with_rapidgzip_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
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
    decode_chunk_with_rapidgzip(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
    )
}

/// Option A3+A4 dispatch (`plans/unified-decoder.md` §6, commits
/// `9e14bbe`/`757e7a4`/`a6076e1`). Pre-fills `chunk.data[0..32K]` with
/// the predecessor's window image so every back-reference hits the
/// AVX2 fast path; the consumer skips `chunk.data[..data_prefix_len]`
/// when writing.
///
/// Measured +4.2% T=16 silesia on neurotic, byte-perfect across 11
/// corpora (silesia/logs/software in gzip/pigz/bgzf flavors +
/// urandom-100M) at T∈{1,4,16} — 33/33 pass.
///
/// **Default ON** as of the cross-corpus validation gate. To disable
/// for A/B comparison: `GZIPPY_OPTION_A_PREFILL=0`.
#[cfg(pure_inflate_decode)]
fn use_option_a_prefill_path() -> bool {
    use std::sync::OnceLock;
    static USE_A: OnceLock<bool> = OnceLock::new();
    *USE_A.get_or_init(|| match std::env::var("GZIPPY_OPTION_A_PREFILL") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        _ => true,
    })
}

/// Stateless ISA-L-LUT bulk decode for the clean tail (`isal_lut_bulk`).
/// Default ON on pure-rust builds; disable with `GZIPPY_ISAL_PURE_BULK=0`.
#[cfg(pure_inflate_decode)]
fn use_isal_pure_bulk_tail() -> bool {
    use std::sync::OnceLock;
    static USE_BULK: OnceLock<bool> = OnceLock::new();
    *USE_BULK.get_or_init(|| match std::env::var("GZIPPY_ISAL_PURE_BULK") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        _ => true,
    })
}

#[cfg(pure_inflate_decode)]
enum BulkCleanTailResult {
    /// Entire clean tail decoded; caller should `finalize(final_bit)`.
    Complete { final_bit: usize },
    /// First segment filled or declined mid-stream; resume with `IsalInflateWrapper`.
    Handoff {
        start_bit: usize,
        last_eob_pos: usize,
        last_eob_decoded_bytes: usize,
        pending_stop_after_flush: bool,
    },
    /// Use the existing resumable wrapper path for this call.
    Decline,
}

/// Last ≤32 KiB of the logical decode stream before `chunk.data.len()`.
/// With A3 prefill the window image is already in `chunk.data[0..]`;
/// without A3, bytes from `initial_window` precede committed chunk bytes.
#[cfg(pure_inflate_decode)]
fn bulk_predecessor_window(
    chunk: &ChunkData,
    initial_window: &[u8],
    out: &mut [u8; 32 * 1024],
) -> usize {
    const W: usize = 32 * 1024;
    let n = chunk.data.len();
    if n >= W {
        chunk.data.copy_last_into(out);
        return W;
    }
    let need_iw = W.saturating_sub(n).min(initial_window.len());
    let iw_off = initial_window.len().saturating_sub(need_iw);
    if need_iw > 0 {
        out[..need_iw].copy_from_slice(&initial_window[iw_off..iw_off + need_iw]);
    }
    if n > 0 {
        chunk
            .data
            .copy_range_into(0, &mut out[need_iw..need_iw + n]);
    }
    need_iw + n
}

/// Rapidgzip-shaped clean tail via [`crate::decompress::parallel::isal_lut_bulk`]
/// (stateless ISA-L-LUT block loop — no resumable yield tax).
#[cfg(pure_inflate_decode)]
fn finish_decode_chunk_bulk_lut(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> BulkCleanTailResult {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::isal_lut_bulk::{decode_block, DecoderScratch};

    if !use_isal_pure_bulk_tail() {
        return BulkCleanTailResult::Decline;
    }

    // Bulk LUT has no resumable window ring — only run when A3 prefill or at
    // least 32 KiB of decoded output supplies the DEFLATE lookback window.
    let a3_ready = use_option_a_prefill_path()
        && initial_window.len() == 32 * 1024
        && chunk.data.all_in_first_segment();
    let lookback_bytes = initial_window.len().saturating_add(chunk.data.len());
    if !a3_ready && lookback_bytes < 32 * 1024 {
        return BulkCleanTailResult::Decline;
    }

    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.stream_inflate",
        &format!(
            r#""start_bit":{inflate_start_bit},"stop_hint":{stop_hint_bits},"has_window":{},"bulk_lut":true"#,
            !initial_window.is_empty() || chunk.data.len() >= 32 * 1024
        ),
    );

    let bulk_entry_len = chunk.data.len();
    let bulk_entry_subchunks = chunk.subchunks.len();
    let mut bits = Bits::at_bit_offset(input, inflate_start_bit);
    let mut scratch = DecoderScratch::new();
    let mut pred_buf = [0u8; 32 * 1024];

    let mut last_end_bit = inflate_start_bit;
    let mut last_eob_pos = inflate_start_bit;
    let mut last_eob_decoded_bytes = chunk.data.len();
    let mut pending_stop_after_flush = false;
    let mut stopping_point_reached = false;
    let mut reached_stream_end = false;

    let a3_contiguous = use_option_a_prefill_path()
        && initial_window.len() == 32 * 1024
        && chunk.data.all_in_first_segment();

    while !stopping_point_reached {
        let decode_base = chunk.data.len();

        if a3_contiguous && chunk.data.all_in_first_segment() {
            let cap = chunk.data.first_segment_a3_output().len();
            let mut local_pos = decode_base;
            if local_pos >= cap {
                return BulkCleanTailResult::Handoff {
                    start_bit: bits.bit_position(),
                    last_eob_pos,
                    last_eob_decoded_bytes,
                    pending_stop_after_flush,
                };
            }
            while local_pos < cap && !stopping_point_reached {
                let block_result = {
                    let out_buf = chunk.data.first_segment_a3_output();
                    match decode_block(&mut bits, out_buf, &mut local_pos, &[], &mut scratch) {
                        Ok(r) => r,
                        Err(_) => {
                            chunk.data.truncate(bulk_entry_len);
                            chunk.subchunks.truncate(bulk_entry_subchunks);
                            if let Some(last) = chunk.subchunks.last_mut() {
                                last.decoded_size =
                                    bulk_entry_len.saturating_sub(last.decoded_offset);
                            }
                            return BulkCleanTailResult::Decline;
                        }
                    }
                };
                last_end_bit = bits.bit_position();
                if !block_result.is_final_block {
                    chunk.append_block_boundary_at(last_end_bit, local_pos);
                    last_eob_pos = last_end_bit;
                    last_eob_decoded_bytes = local_pos;
                }
                if block_result.is_final_block {
                    stopping_point_reached = true;
                    reached_stream_end = true;
                    break;
                }
                if last_end_bit >= stop_hint_bits {
                    pending_stop_after_flush = true;
                    stopping_point_reached = true;
                    break;
                }
            }
            let append_len = if pending_stop_after_flush {
                last_eob_decoded_bytes.saturating_sub(decode_base)
            } else {
                local_pos.saturating_sub(decode_base)
            };
            if append_len > 0 {
                if chunk.configuration.crc32_enabled {
                    let out_buf = chunk.data.first_segment_a3_output();
                    if let Some(last_crc) = chunk.crc32s.last_mut() {
                        last_crc.update(&out_buf[decode_base..decode_base + append_len]);
                    }
                }
                chunk.data.commit(append_len);
                chunk.statistics.non_marker_count += append_len as u64;
                chunk.note_inner_decoded_bytes(append_len);
            }
            if !stopping_point_reached && local_pos >= cap {
                return BulkCleanTailResult::Handoff {
                    start_bit: bits.bit_position(),
                    last_eob_pos,
                    last_eob_decoded_bytes,
                    pending_stop_after_flush,
                };
            }
            continue;
        }

        // Multi-segment (or non-A3): one segment tail per outer iteration.
        let pred_len = bulk_predecessor_window(chunk, initial_window, &mut pred_buf);
        let pred = &pred_buf[..pred_len];
        let cap = chunk.data.writable_tail().len();
        if cap == 0 {
            return BulkCleanTailResult::Handoff {
                start_bit: bits.bit_position(),
                last_eob_pos,
                last_eob_decoded_bytes,
                pending_stop_after_flush,
            };
        }
        let mut local_pos = 0usize;
        while local_pos < cap && !stopping_point_reached {
            let block_result = {
                let seg_tail = chunk.data.writable_tail();
                match decode_block(&mut bits, seg_tail, &mut local_pos, pred, &mut scratch) {
                    Ok(r) => r,
                    Err(_) => {
                        chunk.data.truncate(bulk_entry_len);
                        chunk.subchunks.truncate(bulk_entry_subchunks);
                        if let Some(last) = chunk.subchunks.last_mut() {
                            last.decoded_size = bulk_entry_len.saturating_sub(last.decoded_offset);
                        }
                        return BulkCleanTailResult::Decline;
                    }
                }
            };
            last_end_bit = bits.bit_position();
            let decoded_total = decode_base + local_pos;
            if !block_result.is_final_block {
                chunk.append_block_boundary_at(last_end_bit, decoded_total);
                last_eob_pos = last_end_bit;
                last_eob_decoded_bytes = decoded_total;
            }
            if block_result.is_final_block {
                stopping_point_reached = true;
                reached_stream_end = true;
                break;
            }
            if last_end_bit >= stop_hint_bits {
                pending_stop_after_flush = true;
                stopping_point_reached = true;
                break;
            }
        }

        let append_len = if pending_stop_after_flush {
            last_eob_decoded_bytes.saturating_sub(decode_base)
        } else {
            local_pos
        };
        if append_len > 0 {
            if chunk.configuration.crc32_enabled {
                let kept = &chunk.data.writable_tail()[..append_len];
                if let Some(last_crc) = chunk.crc32s.last_mut() {
                    last_crc.update(kept);
                }
            }
            chunk.data.commit(append_len);
            chunk.statistics.non_marker_count += append_len as u64;
            chunk.note_inner_decoded_bytes(append_len);
        }

        if !stopping_point_reached && local_pos >= cap {
            return BulkCleanTailResult::Handoff {
                start_bit: bits.bit_position(),
                last_eob_pos,
                last_eob_decoded_bytes,
                pending_stop_after_flush,
            };
        }
    }

    if reached_stream_end {
        return BulkCleanTailResult::Complete {
            final_bit: bits.bit_position(),
        };
    }

    let final_bit = if pending_stop_after_flush {
        last_eob_pos
    } else if last_eob_pos > inflate_start_bit {
        last_eob_pos
    } else {
        last_end_bit
    };
    BulkCleanTailResult::Complete { final_bit }
}

/// P2 unified clean tail (vendor `finishDecodeChunkWithInexactOffset`,
/// GzipChunk.hpp:280-410): ISA-L LUT bulk loop first; ResumableInflate2 only
/// when lookback is unavailable (< 32 KiB predecessor).
#[cfg(parallel_sm)]
fn finish_clean_tail_decode(
    chunk: &mut ChunkData,
    input: &[u8],
    mut inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    record_decode_duration: bool,
) -> Result<(), ChunkDecodeError> {
    use crate::decompress::parallel::deflate_block::MAX_WINDOW_SIZE;

    // `worker.isal_stream_inflate` span — wraps every invocation of
    // gzippy's ISA-L stream-inflate path. Both call sites land here:
    //   (a) the post-bootstrap call at gzip_chunk.rs:549 (after the
    //       pure-Rust phase-1 produces a clean 32 KiB window)
    //   (b) the non-speculative direct path via `decode_chunk`
    //       at gzip_chunk.rs:150 (when the predecessor window is
    //       already known and bootstrap is skipped)
    //
    // Pre-instrumentation the only proxy for ISA-L bulk-inflate time
    // was `decode_chunk dur - bootstrap dur` — a derived value with
    // no per-call visibility. With this span we can answer the
    // post-Fix-#3 question: is the 1.5x per-chunk gap to vendor in
    // bootstrap or in ISA-L bulk inflate?
    //
    // Args: `start_bit` (where decode starts) + `stop_hint` + a
    // `has_window` flag distinguishing the two call paths above.
    let t_decode = std::time::Instant::now();

    // Option A3 (default ON): pre-fill segment 0 with the predecessor's
    // 32 KiB window so back-references hit copy_match_fast via
    // `output[..out_pos]`. Disable with `GZIPPY_OPTION_A_PREFILL=0`.
    // The consumer skips `data_prefix_len` when writing (A4).
    #[cfg(feature = "pure-rust-inflate")]
    let full_window_tail = use_option_a_prefill_path() && initial_window.len() == MAX_WINDOW_SIZE;
    #[cfg(not(feature = "pure-rust-inflate"))]
    let full_window_tail = false;
    // A3: seed lookback into `chunk.data[0..32K]` once, before any tail bytes.
    // Resumable `set_window` must stay empty when prefilled — otherwise the
    // 32 KiB window is applied twice (bulk + resumable ring) and stored blocks
    // decode with wrong lengths.
    if full_window_tail && chunk.data.is_empty() {
        chunk.prefill_window_prefix(initial_window);
    }
    let resumable_window: &[u8] = if chunk.data_prefix_len >= MAX_WINDOW_SIZE {
        &[]
    } else {
        initial_window
    };

    // P2 unified clean tail: keep the same ISA-L LUT engine across 128 KiB
    // segment fills. Only fall back to ResumableInflate2 when bulk declines
    // (no 32 KiB lookback), not on every segment Handoff.
    #[cfg(pure_inflate_decode)]
    loop {
        match finish_decode_chunk_bulk_lut(
            chunk,
            input,
            inflate_start_bit,
            stop_hint_bits,
            initial_window,
        ) {
            BulkCleanTailResult::Complete { final_bit } => {
                if record_decode_duration {
                    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
                }
                chunk.finalize(final_bit);
                return Ok(());
            }
            BulkCleanTailResult::Handoff { start_bit, .. } => {
                BULK_TAIL_SEGMENT_CONTINUES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                inflate_start_bit = start_bit;
                continue;
            }
            BulkCleanTailResult::Decline => break,
        }
    }

    BULK_TAIL_RESUMABLE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.stream_inflate",
        &format!(
            r#""start_bit":{inflate_start_bit},"stop_hint":{stop_hint_bits},"has_window":{}"#,
            !initial_window.is_empty()
        ),
    );
    // `stop_hint_bits` is an inexact stop *hint* (vendor `untilOffset`), not a hard
    // read cap. Capping `refill_buffer` at a partition guess stops mid-block
    // (e.g. silesia gzip-9 at 33554427 vs hint 33554432 → InvalidBlock on resume).
    let read_cap = input.len() * 8;
    let mut wrapper = IsalInflateWrapper::with_until_bits(input, inflate_start_bit, read_cap)?;
    wrapper.set_window(resumable_window)?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER
            | StoppingPoints::END_OF_STREAM,
    );

    let mut stopping_point_reached = false;
    let mut last_end_bit = inflate_start_bit;
    let mut last_eob_pos = inflate_start_bit;
    let mut last_eob_decoded_bytes: usize = chunk.data.len();
    let mut already_decoded: usize = 0;
    // Set once the decoder reaches the genuine end of the input (final
    // gzip member's footer consumed, or ISA-L reports `finished`). When
    // true the chunk finalizes at `last_end_bit` (post-footer / post-
    // final-block), NOT at `last_eob_pos` — the latter is the boundary
    // BEFORE the BFINAL block, so finalizing there drops the final block
    // from the chunk's encoded range and the consumer re-decodes it,
    // duplicating its bytes in the output.
    let mut reached_stream_end = false;
    let mut pending_stop_after_flush = false;
    // §5 step 5 (NEW wrapper / ResumableInflate2): the pure-rust backend no
    // longer has a `session` accumulator, so `session_pending()` is always
    // false. The old "keep looping to drain session" behavior (a6c0a8b) would
    // run past the partition boundary, clobbering `last_end_bit` with the
    // bit_position of the NEXT block's header — successor chunks would then
    // resume at a non-block-boundary and decode garbage Huffman tables. With
    // ResumableInflate2 the inner loop must STOP as soon as
    // `pending_stop_after_flush` fires, same as the isal-only build.
    const STOP_INNER_ON_PENDING_FLUSH: bool = true;

    while !stopping_point_reached || wrapper.session_pending() {
        // Write ISA-L output DIRECTLY into chunk.data's spare capacity.
        // Previous version allocated a fresh `ALLOCATION_CHUNK_SIZE`
        // (128 KiB) buffer per outer iteration, ran ISA-L into it, then
        // called `append_owned_buffer` which either replaced
        // chunk.data (first call — losing the 80 MiB pre-allocated
        // pool buffer) or extend_from_slice'd it (subsequent calls —
        // paying a memcpy + grow page faults). Per the bench-sm
        // perf-dwarf profile this accounted for ~3.4% of total cycles
        // (`append_owned_buffer → Vec::extend_from_slice → page fault`).
        //
        // Vendor pattern: `GzipChunk::readBlock` writes directly into
        // the chunk's vector via `next_out`; no intermediate buffer.
        // gzippy's `ChunkData::note_clean_bytes_written_in_place`
        // helper was already in place (chunk_data.rs:405) — this just
        // wires it up.
        let prev_data_len = chunk.data.len();
        // FOOTPRINT-ALIGN: `chunk.data` is a `SegmentedU8`. Acquire a fresh
        // 128 KiB segment tail to decode this outer iteration into. The
        // returned slice is exactly ALLOCATION_CHUNK_SIZE bytes (a brand-new
        // segment if the prior tail was full) — uninitialized but writable.
        // We write into it across the inner `read_stream` calls, then
        // `commit(append_len)` the bytes we keep. Back-refs reaching before
        // this segment resolve via the wrapper's internal 32 KiB window ring
        // (resumable.rs:653 feeds each call's output into `state.window`), so
        // segment boundaries are transparent to correctness — vendor's
        // `DecodedData::append` segments output the same way
        // (DecodedData.hpp:243-289).
        // `writable_tail` returns the CURRENT tail segment's spare — which is
        // a FULL 128 KiB when the prior outer iteration filled (or there was
        // none), or a partial remainder when the prior commit left the tail
        // non-full. Either way `buffer_cap` is the real writable spare; the
        // inner loop respects it and the outer loop simply runs again to fill
        // the next segment. (No fixed-128KiB assumption — that was wrong for
        // partially-committed tails and tripped the 2 MB decode test.)
        let seg_tail: &mut [u8] = chunk.data.writable_tail();
        let seg_ptr = seg_tail.as_mut_ptr();
        let buffer_cap = seg_tail.len();
        let mut n_bytes_read: usize = 0;
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;
        let mut end_of_stream_hit = false;

        let decode_base = already_decoded;
        while n_bytes_read < buffer_cap
            && !stopping_point_reached
            && !(STOP_INNER_ON_PENDING_FLUSH
                && pending_stop_after_flush
                && !wrapper.session_pending())
        {
            let bit_before_read = wrapper.tell_compressed();
            // A3: decode into segment 0's full backing store so back-refs
            // resolve via `output[..out_pos]` (includes the prefilled window).
            // Non-A3 / multi-segment: spare slice on the tail segment only;
            // cross-chunk refs use the wrapper's window ring.
            #[cfg(feature = "pure-rust-inflate")]
            let r = if full_window_tail && chunk.data.all_in_first_segment() {
                let out_buf = chunk.data.first_segment_a3_output();
                wrapper.read_stream_starting_at(out_buf, prev_data_len + n_bytes_read)?
            } else {
                let spare: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        seg_ptr.add(n_bytes_read),
                        buffer_cap - n_bytes_read,
                    )
                };
                wrapper.read_stream(spare)?
            };
            #[cfg(not(feature = "pure-rust-inflate"))]
            let r = {
                let spare: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        seg_ptr.add(n_bytes_read),
                        buffer_cap - n_bytes_read,
                    )
                };
                wrapper.read_stream(spare)?
            };
            last_per_call = r.bytes_written;
            n_bytes_read += last_per_call;
            chunk.note_inner_decoded_bytes(last_per_call);

            last_stopped_at = r.stopped_at;
            last_finished = r.finished;
            last_end_bit = r.bit_position;

            // Defense-in-depth termination guard. If `read_stream` made
            // no progress whatsoever — no output, no stopping point, not
            // finished, and the compressed cursor did not advance — then
            // ISA-L is exhausted on this input and every further call
            // would stall identically (the input is fixed; `read_stream`
            // already looped internally until `!made_progress`). Finalize
            // the chunk here instead of spinning forever.
            //
            // In production `consumer_loop`'s sub-byte-tail guard means a
            // chunk too small to hold a deflate block is never scheduled;
            // this is the backstop for any future caller that hands
            // `decode_chunk` such a fragment directly.
            //
            // A genuinely truncated stream also stalls here; its
            // already-decoded prefix is kept (appended by earlier
            // iterations) and the short finalize is caught downstream by
            // CRC32/ISIZE verification.
            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && !r.finished
                && r.bit_position == bit_before_read
            {
                stopping_point_reached = true;
                break;
            }

            // END_OF_STREAM must be detected even when ISA-L reports
            // `finished` on the same call — `read_stream` returns both
            // flags together at the final block. Checking `finished`
            // first would skip the footer-reading branch below, leaving
            // the chunk finalized at the pre-BFINAL-block EOB.
            if r.stopped_at == StoppingPoints::END_OF_STREAM {
                end_of_stream_hit = true;
                break;
            }
            if r.finished {
                break;
            }

            match r.stopped_at {
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                    chunk.append_block_boundary_at(r.bit_position, decode_base + n_bytes_read);
                }
                sp if sp == StoppingPoints::END_OF_BLOCK => {
                    if !wrapper.is_final_block() {
                        chunk.append_block_boundary_at(r.bit_position, decode_base + n_bytes_read);
                        if r.bit_position >= stop_hint_bits {
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
                    let not_final = !wrapper.is_final_block();
                    let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
                    if last_eob_pos >= stop_hint_bits && not_final && not_fixed {
                        last_end_bit = last_eob_pos;
                        pending_stop_after_flush = true;
                    }
                }
                sp if sp == StoppingPoints::NONE
                    && last_per_call == 0
                    && last_eob_pos >= stop_hint_bits =>
                {
                    // ISA-L can return 0 bytes between block boundaries.
                    // Do not end the chunk early while still before `stop_hint_bits`.
                    last_end_bit = last_eob_pos;
                    pending_stop_after_flush = true;
                }
                _ => {}
            }
            if end_of_stream_hit || last_finished {
                break;
            }
        }

        let mut append_len = n_bytes_read;
        if stopping_point_reached {
            // Stopped on NONE+0 / HEADER after the prior END_OF_BLOCK was
            // already appended in a previous outer iteration — do not
            // emit bytes from the next block read into this buffer.
            append_len = last_eob_decoded_bytes.saturating_sub(decode_base);
        } else if pending_stop_after_flush {
            // Session-flush outer iterations after the EOB hint fired:
            // all bytes in this buffer are tail output from the same block.
            append_len = n_bytes_read;
        }
        if append_len > 0 {
            // CRC the kept bytes FIRST (read them straight from the segment
            // we just wrote, before `commit`), then commit `append_len` into
            // the segmented buffer. Bytes `[append_len, n_bytes_read)` that
            // the decoder wrote but we're discarding (between EOB and the
            // truncate point) simply never get committed — the next outer
            // iteration overwrites the segment tail from `append_len`.
            //
            // We deliberately do NOT touch `subchunks.last_mut().decoded_size`
            // here because the inner loop already credited the FULL
            // n_bytes_read via `note_inner_decoded_bytes(last_per_call)`.
            //
            // SAFETY: `append_len ≤ n_bytes_read ≤ buffer_cap`; the decoder
            // wrote initialized bytes through `seg_ptr[0..n_bytes_read]`, so
            // `seg_ptr[0..append_len]` is fully initialized.
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

        if end_of_stream_hit {
            let (crc32, isize_field) = wrapper.read_footer_at_current()?;
            let footer_end_bits = wrapper.tell_compressed();
            chunk.append_footer(crc32, isize_field, footer_end_bits);

            let remaining = wrapper.remaining_input();
            if remaining.is_empty() {
                last_end_bit = footer_end_bits;
                reached_stream_end = true;
                break;
            }
            let (_hdr, hdr_size) = crate::decompress::parallel::gzip_format::read_header(remaining)
                .map_err(|e| {
                    ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("multi-stream gzip header at bit {footer_end_bits}: {e}"),
                    ))
                })?;
            wrapper.advance_input(hdr_size);
            wrapper.reset_for_next_stream();
            wrapper.set_stopping_points(
                StoppingPoints::END_OF_BLOCK
                    | StoppingPoints::END_OF_BLOCK_HEADER
                    | StoppingPoints::END_OF_STREAM_HEADER
                    | StoppingPoints::END_OF_STREAM,
            );
            wrapper.set_window(&[])?;
            last_end_bit = wrapper.tell_compressed();
            last_eob_pos = last_end_bit;
            chunk.append_block_boundary_at(last_end_bit, already_decoded);
            continue;
        }

        if last_finished {
            reached_stream_end = true;
            break;
        }
        if last_stopped_at == StoppingPoints::NONE && last_per_call == 0 {
            if last_eob_pos >= stop_hint_bits {
                last_end_bit = last_eob_pos;
                break;
            }
            continue;
        }
    }

    if record_decode_duration {
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    }
    // A4: window-image prefix STAYS in chunk.data through finalize
    // and into the consumer. The consumer write site reads
    // &chunk.data[chunk.data_prefix_len..] so the prefix never
    // reaches the user's output. Downstream chunk methods that
    // need the decoded portion (last_32kib_window, get_last_window,
    // populate_subchunk_windows) skip data_prefix_len bytes;
    // `finish_clean_tail_decode` reads from
    // `tail.data[tail.data_prefix_len..]`. Eliminating the trim
    // memmove was measured at -3.81pp `__memmove_avx_unaligned_erms`
    // CPU share vs A3-with-trim, lifting net throughput to +4.2%
    // vs default at T=16 on neurotic silesia-gzip9.
    let _ = full_window_tail; // consumed at decode-start (A3 prefill + bulk path)
                              // When the reader runs past `stop_hint_bits` without an inexact
                              // stop, `tell_compressed()` can land mid-block. Successor chunks must
                              // resume at the last END_OF_BLOCK position (pre-header), not mid-block.
    let final_bit = if stopping_point_reached || reached_stream_end {
        last_end_bit
    } else if last_eob_pos > inflate_start_bit {
        // Hit the until read cap without an inexact header-stop — finalize at
        // the last pre-header EOB, never at a mid-block bit cursor.
        last_eob_pos
    } else {
        wrapper.tell_compressed()
    };
    chunk.finalize(final_bit);
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
) -> Result<ChunkData, ChunkDecodeError> {
    use crate::decompress::parallel::deflate_block::MAX_WINDOW_SIZE;

    // Envelope span: `chunk_fetcher::run_decode_task` (`worker.decode_chunk`).
    let t_decode = std::time::Instant::now();

    let start_clean_only = initial_window.len() == MAX_WINDOW_SIZE
        || (initial_window.is_empty() && encoded_offset_bits == 0);

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);

    if start_clean_only {
        let t_inflate = std::time::Instant::now();
        finish_clean_tail_decode(
            &mut chunk,
            input,
            encoded_offset_bits,
            stop_hint_bits,
            initial_window,
            false,
        )?;
        let inflate_us = t_inflate.elapsed().as_micros();
        if trace::is_enabled() {
            trace::emit(
                "worker",
                "decode_span",
                &format!(
                    r#""start_bit":{encoded_offset_bits},"bootstrap_us":0,"inflate_us":{inflate_us},"phase":"clean_only","markers":0,"tail_bytes":{}"#,
                    chunk.data.len(),
                ),
            );
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        return Ok(chunk);
    }

    chunk.data_with_markers.reserve(128 * 1024);
    let mut marker_ctx = MarkerDecodeCtx::new(input, encoded_offset_bits)?;

    let mut marker_us: u128 = 0;
    let mut marker_span: Option<crate::decompress::parallel::trace_v2::SpanGuard> = None;
    loop {
        if marker_span.is_none() {
            marker_span = Some(
                crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
                    "worker.bootstrap",
                    &format!(r#""start_bit":{encoded_offset_bits},"stop_hint":{stop_hint_bits}"#,),
                ),
            );
        }
        let t_marker = std::time::Instant::now();
        let (step, flipped_clean) = marker_decode_step(
            &mut marker_ctx,
            input,
            stop_hint_bits,
            &mut chunk.data_with_markers,
        )?;
        marker_us += t_marker.elapsed().as_micros();
        if flipped_clean {
            UNIFIED_MODE_CLEAN_FLIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        match step {
            MarkerStep::Continue => {}
            MarkerStep::Handoff {
                end_bit_offset,
                clean_window,
            } => {
                let markers_len = chunk.data_with_markers.len();
                crate::decompress::parallel::trace_v2::emit_instant(
                    "worker.bootstrap.outcome",
                    &format!(
                        r#""result":"ok","markers_len":{markers_len},"end_bit":{end_bit_offset},"clean_window":true,"bfinal":false,"handoff_reason":"clean_window_armed","bytes_decoded":{markers_len}"#,
                    ),
                    "t",
                );
                crate::decompress::parallel::trace_v2::emit_instant(
                    "causal.decode_handoff",
                    &format!(
                        r#""start_bit":{encoded_offset_bits},"end_bit":{end_bit_offset},"marker_bytes":{markers_len},"inflate_start_bit":{end_bit_offset}"#,
                    ),
                    "t",
                );
                marker_span.take();
                let t_inflate = std::time::Instant::now();
                finish_clean_tail_decode(
                    &mut chunk,
                    input,
                    end_bit_offset,
                    stop_hint_bits,
                    &clean_window,
                    false,
                )?;
                let inflate_us = t_inflate.elapsed().as_micros();
                apply_slow_bootstrap_probe(marker_us);
                if trace::is_enabled() {
                    trace::emit(
                        "worker",
                        "decode_span",
                        &format!(
                            r#""start_bit":{encoded_offset_bits},"bootstrap_us":{marker_us},"inflate_us":{inflate_us},"phase":"bootstrap+inflate","markers":{markers_len},"tail_bytes":{}"#,
                            chunk.data.len(),
                        ),
                    );
                }
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
                return Ok(chunk);
            }
            MarkerStep::Finished {
                end_bit_offset,
                bfinal_hit,
            } => {
                marker_span.take();
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                if trace::is_enabled() {
                    let phase = if bfinal_hit {
                        "bootstrap_terminal"
                    } else {
                        "bootstrap_only"
                    };
                    trace::emit(
                        "worker",
                        "decode_span",
                        &format!(
                            r#""start_bit":{encoded_offset_bits},"bootstrap_us":{marker_us},"inflate_us":0,"phase":"{phase}","markers":{}"#,
                            chunk.data_with_markers.len(),
                        ),
                    );
                }
                chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
                chunk.finalize(end_bit_offset);
                return Ok(chunk);
            }
        }
    }
}

#[cfg(parallel_sm)]
fn apply_slow_bootstrap_probe(bootstrap_dur_us: u128) {
    if let Some(pct) = std::env::var("GZIPPY_SLOW_BOOTSTRAP")
        .ok()
        .and_then(|s| s.parse::<u128>().ok())
    {
        let extra = std::time::Duration::from_micros((bootstrap_dur_us * pct / 100) as u64);
        if std::env::var_os("GZIPPY_SLOW_BOOTSTRAP_SLEEP").is_some() {
            std::thread::sleep(extra);
        } else {
            let until = std::time::Instant::now() + extra;
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
        }
    }
}

/// One iteration of the vendor `decodeChunkWithRapidgzip` block loop.
#[cfg(parallel_sm)]
enum MarkerStep {
    /// Another deflate block was decoded; call again.
    Continue,
    /// `cleanDataCount >= MAX_WINDOW_SIZE` at a block boundary — run streaming inflate next.
    Handoff {
        end_bit_offset: usize,
        clean_window: Vec<u8>,
    },
    /// Chunk ends in the marker path (BFINAL, stop hint, or no clean dict).
    Finished {
        end_bit_offset: usize,
        bfinal_hit: bool,
    },
}

/// Persistent state for [`marker_decode_step`] (one block per call).
#[cfg(parallel_sm)]
struct MarkerDecodeCtx {
    /// `Bits` slice base in `data` (byte index).
    data_base_byte: usize,
    current_bit_offset: usize,
    trailing_clean: usize,
    block_primed: bool,
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
            block_primed: false,
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

/// Result of one bootstrap pass over deflate blocks via
/// [`deflate_block::Block`]. Mirrors the early-exit contract of
/// rapidgzip's `decodeChunkWithRapidgzip` main loop
/// (vendor/.../chunkdecoding/GzipChunk.hpp:468-654).
#[cfg(parallel_sm)]
struct DeflateBootstrap {
    // The u16 marker/literal output is now written DIRECTLY into the caller's
    // `data_with_markers` buffer (passed as `&mut impl MarkerSink`), not
    // returned here — that's the zero-copy merge. This struct carries only the
    // post-decode metadata the caller needs.
    /// Bit position immediately after the last fully-decoded block.
    /// Always at a real deflate block boundary, suitable as a resume
    /// point for ISA-L's `isal_inflate_set_dict` + decode.
    end_bit_offset: usize,
    /// Last 32 KiB of clean (non-marker) output, present only when the
    /// bootstrap saw ≥ MAX_WINDOW_SIZE cumulative clean bytes ending at
    /// a block boundary (rapidgzip's `cleanDataCount >= MAX_WINDOW_SIZE`
    /// at GzipChunk.hpp:521). When `None`, phase 2 cannot run because we
    /// have no clean dict — either BFINAL fired first, `stop_hint_bits` was
    /// reached, or no block produced enough clean data.
    clean_window: Option<Vec<u8>>,
    /// True when the bootstrap exited because it decoded a BFINAL=1
    /// block. Single-stream tail; no further deflate data follows.
    bfinal_hit: bool,
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
#[cfg(parallel_sm)]
fn bootstrap_with_deflate_block(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::deflate_block::MarkerSink,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::parallel::trace_v2;
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "worker.bootstrap",
        &format!(r#""start_bit":{start_bit_offset},"stop_hint":{stop_hint_bits}"#),
    );
    // Inner function so we can capture the Result and emit an outcome
    // instant event before the SpanGuard drops. Per-attempt outcome
    // attribution lets `scripts/timeline_analyze.py` correlate the
    // 29 retries seen on silesia-large with their specific failure
    // variant (header parse vs body invalid-huffman vs
    // exceeded-window-range vs ...).
    let result = bootstrap_with_deflate_block_inner(data, start_bit_offset, stop_hint_bits, output);
    let markers_len = output.sink_len();
    match &result {
        Ok(b) => {
            // handoff_reason: disambiguates the three Ok exit paths in
            // `bootstrap_with_deflate_block_inner`:
            //   - clean_window_armed: trailing_clean reached
            //     MAX_WINDOW_SIZE at a block boundary; ISA-L will
            //     take over (the "good" path).
            //   - bfinal_hit: BFINAL block decoded; chunk is complete
            //     in pure-Rust phase-1 alone.
            //   - stop_hint_reached: the last block header was at-or-past
            //     stop_hint_bits AND non-FixedHuffman AND not BFINAL;
            //     decode stopped on the upcoming block (caller's
            //     successor will re-decode that block).
            // For the 6 heavy-tail chunks per silesia-large run that
            // burn 200+ ms in bootstrap, the suspicion is they all hit
            // either `bfinal_hit` or `stop_hint_reached` after running
            // through the entire chunk without arming a clean window —
            // this arg confirms or falsifies that prior.
            let handoff_reason = if b.clean_window.is_some() {
                "clean_window_armed"
            } else if b.bfinal_hit {
                "bfinal_hit"
            } else {
                "stop_hint_reached"
            };
            // bytes_decoded: the bootstrap-decoded byte count. Each u16
            // in `b.markers` represents ONE decoded byte (high bits
            // distinguish markers from clean bytes). Join with the
            // worker.bootstrap span's duration to compute per-call MB/s
            // for the Step-2 analyzer.
            trace_v2::emit_instant(
                "worker.bootstrap.outcome",
                &format!(
                    r#""result":"ok","markers_len":{},"end_bit":{},"clean_window":{},"bfinal":{},"handoff_reason":"{}","bytes_decoded":{}"#,
                    markers_len,
                    b.end_bit_offset,
                    b.clean_window.is_some(),
                    b.bfinal_hit,
                    handoff_reason,
                    markers_len,
                ),
                "t",
            );
        }
        Err(e) => {
            let kind = match e {
                ChunkDecodeError::BootstrapFailed(io_err) => {
                    let msg = io_err.to_string();
                    // Extract the first identifying token of the inner
                    // BlockError (e.g. "InvalidHuffmanCode"). The error
                    // strings are formatted in bootstrap as
                    // `"deflate body at bit ... : InvalidHuffmanCode"`
                    // or `"deflate header at bit ...: InvalidHuffmanCode"`.
                    if msg.contains("InvalidHuffmanCode") {
                        "InvalidHuffmanCode"
                    } else if msg.contains("ExceededWindowRange") {
                        "ExceededWindowRange"
                    } else if msg.contains("InvalidCompression") {
                        "InvalidCompression"
                    } else if msg.contains("InvalidCodeLengths") {
                        "InvalidCodeLengths"
                    } else if msg.contains("EndOfFile") {
                        "EndOfFile"
                    } else if msg.contains("past end of data") {
                        "past_eof"
                    } else if msg.contains("header") {
                        "header_other"
                    } else if msg.contains("body") {
                        "body_other"
                    } else {
                        "other"
                    }
                }
                _ => "non_bootstrap_err",
            };
            trace_v2::emit_instant(
                "worker.bootstrap.outcome",
                &format!(r#""result":"err","kind":"{kind}""#),
                "t",
            );
        }
    }
    result
}

/// Decode one deflate block into `output` (vendor block loop body).
#[cfg(parallel_sm)]
fn marker_decode_step(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::deflate_block::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    use crate::decompress::parallel::deflate_block::{Block, MAX_WINDOW_SIZE};
    use crate::decompress::parallel::trace_v2;
    use std::cell::RefCell;

    thread_local! {
        static BOOTSTRAP_BLOCK: RefCell<Block> = RefCell::new(Block::new());
    }

    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        if !ctx.block_primed {
            block.reset(None, None);
            ctx.block_primed = true;
        }
        let block = &mut *block;

        // P2: vendor `decodeChunkWithRapidgzip` decodes many blocks per outer
        // iteration; batch here to avoid per-block consumer-loop overhead.
        loop {
            let slice_byte = ctx.current_bit_offset / 8;
            let mut bits = ctx.open_bits(data);

            let next_block_offset = absolute_bit_pos(slice_byte, &bits);

            if ctx.trailing_clean >= MAX_WINDOW_SIZE {
            let end_bit_offset = next_block_offset;
            ctx.current_bit_offset = end_bit_offset;
            let mut window = Vec::with_capacity(MAX_WINDOW_SIZE);
            if output.copy_last_n_clean_u8(MAX_WINDOW_SIZE, &mut window) {
                return Ok((
                    MarkerStep::Handoff {
                        end_bit_offset,
                        clean_window: window,
                    },
                    false,
                ));
            }
            return Ok((
                MarkerStep::Finished {
                    end_bit_offset,
                    bfinal_hit: false,
                },
                false,
            ));
        }

        let header_res = {
            let _tv2 = trace_v2::SpanGuard::begin("worker.block_header");
            let t_header = std::time::Instant::now();
            let r = block.read_header(&mut bits, false);
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
            let res = block.read(&mut bits, &mut *output, usize::MAX);
            if let Err(e) = res {
                let bits_at_fail = absolute_bit_pos(slice_byte, &bits);
                let bytes_wasted = output.sink_len() - before_len;
                let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                BODY_FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BYTES_WASTED
                    .fetch_add(bytes_wasted as u64, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BITS_INTO_BODY
                    .fetch_add(bits_into_body as u64, std::sync::atomic::Ordering::Relaxed);
                use crate::decompress::parallel::deflate_block::BlockError;
                match &e {
                    BlockError::InvalidHuffmanCode => {
                        BODY_FAIL_INVALID_HUFFMAN
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BlockError::ExceededWindowRange => {
                        BODY_FAIL_EXCEEDED_WINDOW
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BlockError::InvalidCompression => {
                        BODY_FAIL_INVALID_COMPRESSION
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BlockError::InvalidCodeLengths => {
                        BODY_FAIL_INVALID_CODE_LENGTHS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    _ => {
                        BODY_FAIL_OTHER_VARIANT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
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
            BOOTSTRAP_POST_FLIP_U16_BYTES.fetch_add(
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
        }
    })
}

#[cfg(parallel_sm)]
fn bootstrap_with_deflate_block_inner(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::deflate_block::MarkerSink,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::parallel::deflate_block::MAX_WINDOW_SIZE;

    let mut ctx = MarkerDecodeCtx::new(data, start_bit_offset)?;
    loop {
        match marker_decode_step(&mut ctx, data, stop_hint_bits, output)?.0 {
            MarkerStep::Continue => {}
            MarkerStep::Handoff {
                end_bit_offset,
                clean_window,
            } => {
                return Ok(DeflateBootstrap {
                    end_bit_offset,
                    clean_window: Some(clean_window),
                    bfinal_hit: false,
                });
            }
            MarkerStep::Finished {
                end_bit_offset,
                bfinal_hit,
            } => {
                let clean_window = if output.sink_len() >= MAX_WINDOW_SIZE {
                    let mut window = Vec::with_capacity(MAX_WINDOW_SIZE);
                    if output.copy_last_n_clean_u8(MAX_WINDOW_SIZE, &mut window) {
                        Some(window)
                    } else {
                        None
                    }
                } else {
                    None
                };
                return Ok(DeflateBootstrap {
                    end_bit_offset,
                    clean_window,
                    bfinal_hit,
                });
            }
        }
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

    /// ISA-L-LUT bulk clean tail must match the resumable wrapper byte-for-byte.
    #[cfg(pure_inflate_decode)]
    #[test]
    fn bulk_clean_tail_matches_resumable_wrapper() {
        let payload = b"the quick brown fox ".repeat(50_000);
        let deflate = make_multi_block_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let stop_hint_bits = deflate.len() * 8;
        let window = [0u8; 32 * 1024];

        let chunk_bulk = {
            std::env::set_var("GZIPPY_ISAL_PURE_BULK", "1");
            std::env::set_var("GZIPPY_OPTION_A_PREFILL", "1");
            decode_chunk(&deflate, 0, stop_hint_bits, &window[..], cfg).unwrap()
        };
        let chunk_wrap = {
            std::env::set_var("GZIPPY_ISAL_PURE_BULK", "0");
            std::env::set_var("GZIPPY_OPTION_A_PREFILL", "1");
            decode_chunk(&deflate, 0, stop_hint_bits, &window[..], cfg).unwrap()
        };
        assert_eq!(
            chunk_bulk.data.to_contiguous(),
            chunk_wrap.data.to_contiguous(),
            "bulk_lut vs resumable wrapper clean tail diverged"
        );
        assert_eq!(flatten(&chunk_bulk), flatten(&chunk_wrap));
    }

    #[test]
    fn decode_chunk_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let stop_hint_bits = deflate.len() * 8;
        let chunk = decode_chunk(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
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
