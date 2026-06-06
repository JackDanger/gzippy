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
    DeflateCompressionType, IsalInflateWrapper, StoppingPoints,
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

/// Option A3+A4: predecessor window image in `chunk.data[0..32K]` so
/// `read_stream_starting_at` hits the copy_match fast path. `=0` disables.
/// Default ON (measured +4.2% T=16 silesia).
#[cfg(parallel_sm)]
fn option_a_prefill_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| match std::env::var("GZIPPY_OPTION_A_PREFILL") {
        Ok(v) => v != "0",
        Err(_) => true,
    })
}

/// Chunks that took the A3 output-prefix fast path in `finish_decode_chunk_impl`.
#[cfg(parallel_sm)]
pub static OPTION_A3_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(pure_inflate_decode)]
#[allow(dead_code)]
fn record_bulk_decline(err: crate::decompress::parallel::isal_lut_bulk::BulkDecodeError) {
    use crate::decompress::parallel::isal_lut_bulk::BulkDecodeError;
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
    let mut wrapper = IsalInflateWrapper::with_until_bits(input, inflate_start_bit, read_cap)?;

    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    #[cfg(feature = "pure-rust-inflate")]
    let use_a3 = option_a_prefill_enabled()
        && initial_window.len() == MAX_WINDOW_SIZE
        && chunk.data_prefix_len == 0
        && chunk.data.is_empty();
    #[cfg(not(feature = "pure-rust-inflate"))]
    let use_a3 = false;

    if use_a3 {
        OPTION_A3_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        chunk.prefill_window_prefix(initial_window);
        wrapper.set_window(&[])?;
    } else {
        wrapper.set_window(initial_window)?;
    }

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
        // A3 only for the first 128 KiB of decoded payload (after the 32 KiB
        // prefix). Beyond that, `read_stream` on `writable_tail` reuses the
        // sliding window built during A3 — never call `set_window` mid-stream.
        let a3_active = use_a3
            && chunk.data.len().saturating_sub(chunk.data_prefix_len) < ALLOCATION_CHUNK_SIZE;
        let (seg_ptr, buffer_cap, out_pos_base) = if a3_active {
            let (ptr, cap, out_pos) = chunk.data.a3_decode_view();
            (ptr, cap.saturating_sub(out_pos), out_pos)
        } else {
            let seg_tail = chunk.data.writable_tail();
            (seg_tail.as_mut_ptr(), seg_tail.len(), 0usize)
        };
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
            let r = if a3_active {
                let out_pos = out_pos_base + n_bytes_read;
                let total_cap = out_pos_base + buffer_cap;
                let buf: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(seg_ptr, total_cap) };
                wrapper.read_stream_starting_at(buf, out_pos)?
            } else {
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
                let kept: &[u8] = if a3_active {
                    unsafe { std::slice::from_raw_parts(seg_ptr.add(out_pos_base), append_len) }
                } else {
                    unsafe { std::slice::from_raw_parts(seg_ptr, append_len) }
                };
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

/// Records block boundaries during marker bootstrap, then applies them to
/// [`ChunkData`] so vendor subchunk split + sparsity can run.
#[cfg(parallel_sm)]
struct RecordingMarkerSink<'a> {
    inner: &'a mut crate::decompress::parallel::segmented_markers::SegmentedU16,
    boundaries: &'a mut Vec<(usize, usize)>,
}

#[cfg(parallel_sm)]
impl crate::decompress::parallel::marker_inflate::MarkerSink for RecordingMarkerSink<'_> {
    fn push_slice(&mut self, values: &[u16]) {
        self.inner.push_slice(values);
    }
    fn sink_len(&self) -> usize {
        self.inner.sink_len()
    }
    fn as_slice(&self) -> &[u16] {
        self.inner.as_slice()
    }
    fn trailing_clean_since(&self, from: usize) -> usize {
        self.inner.trailing_clean_since(from)
    }
    fn copy_last_n_clean_u8(&self, n: usize, out: &mut Vec<u8>) -> bool {
        self.inner.copy_last_n_clean_u8(n, out)
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
    let mut pending_boundaries: Vec<(usize, usize)> = Vec::new();
    loop {
        let mut sink = RecordingMarkerSink {
            inner: &mut chunk.data_with_markers,
            boundaries: &mut pending_boundaries,
        };
        let (step, flipped_clean) =
            marker_decode_step(&mut marker_ctx, input, stop_hint_bits, &[], &mut sink)?;
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

/// Window-present clean-decode (vendor `setInitialWindow`) toggle.
/// `GZIPPY_NO_WINDOW_SEED=1` (or forced-u16) reverts to the old discard-the-
/// window marker-every-chunk behavior for one-binary A/B.
#[cfg(parallel_sm)]
fn window_seed_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var_os("GZIPPY_NO_WINDOW_SEED").is_none()
            && std::env::var_os("GZIPPY_U16_CLEAN_TAIL").is_none()
    })
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
    block_primed: bool,
    /// Set once the unified driver has consumed the single `FlipToClean`
    /// signal and switched the decode sink to the clean u8 tail. Gates the
    /// top-of-loop flip check so it fires EXACTLY once; subsequent steps
    /// (now draining the clean sink) decode normally on the same `MarkerRing`
    /// and the same bit cursor instead of re-returning `FlipToClean`.
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

/// Legacy fast-LUT bootstrap (`isal_lut_bulk::MarkerRing`). `GZIPPY_MARKER_RING=1` only.
#[cfg(parallel_sm)]
fn marker_decode_step_marker_ring(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    initial_window: &[u8],
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    use crate::decompress::parallel::isal_lut_bulk::MarkerRing;
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
                use crate::decompress::parallel::isal_lut_bulk::BulkDecodeError;
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
impl BootstrapEngine for crate::decompress::parallel::isal_lut_bulk::MarkerRing {
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

        if !block.contains_marker_bytes() && !ctx.flipped {
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
    /// (`ResumableInflate2` via `IsalInflateWrapper`) on THIS code state, over a
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
        use crate::decompress::parallel::inflate_wrapper::{IsalInflateWrapper, StoppingPoints};
        use crate::decompress::parallel::isal_lut_bulk::{decode_block, DecoderScratch};
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
                IsalInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8).unwrap();
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
                IsalInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8).unwrap();
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
