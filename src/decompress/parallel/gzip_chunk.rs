#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]

//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_isal`] — on-demand decode when the predecessor
//!   window is known (or chunk 0 with a zero dict). Clean bytes only.
//! - [`decode_chunk_marker_bootstrap_then_isal`] — speculative prefetch when
//!   no window yet: marker bootstrap for cross-chunk refs, then ISA-L bulk.
//!
//! `stop_hint_bits` is an inexact stop hint (vendor `untilOffset`): the
//! decoder runs to the first block boundary at-or-past it, then stops.

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::inflate_wrapper::InflateError;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::inflate_wrapper::{
    DeflateCompressionType, IsalInflateWrapper, StoppingPoints,
};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::rpmalloc_alloc::types;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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
#[allow(dead_code)] // used by x86_64+isal-compression decode_chunk_isal path
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

/// Per-failure structured log. Writes one JSON line to the path in
/// `GZIPPY_BODY_FAIL_LOG` env var (no-op if unset). Use this for
/// distributional analysis: cluster failures by offset to see if they
/// fall in specific corpus regions.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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

#[cfg(not(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
)))]
fn body_fail_log(_: usize, _: usize, _: usize, _: &str) {}

/// ISA-L decode of one chunk when the predecessor window is known.
///
/// Stops at the first block boundary at-or-past `stop_hint_bits` (an
/// inexact hint — the decoder may overshoot), or at end-of-stream.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub fn decode_chunk_isal(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_isal_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
    )
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn decode_chunk_isal_impl(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    // `stop_hint_bits` is an inexact stop *hint* (vendor `untilOffset`), not a hard
    // read cap. Capping `refill_buffer` at a partition guess stops mid-block
    // (e.g. silesia gzip-9 at 33554427 vs hint 33554432 → InvalidBlock on resume).
    let read_cap = input.len() * 8;
    let mut wrapper = IsalInflateWrapper::with_until_bits(input, encoded_offset_bits, read_cap)?;
    wrapper.set_window(initial_window)?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER
            | StoppingPoints::END_OF_STREAM,
    );

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let mut stopping_point_reached = false;
    let mut last_end_bit = encoded_offset_bits;
    let mut last_eob_pos = encoded_offset_bits;
    let mut last_eob_decoded_bytes: usize = 0;
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
        let mut buffer: types::U8 = types::u8_with_capacity(ALLOCATION_CHUNK_SIZE);
        #[allow(clippy::uninit_vec)]
        unsafe {
            buffer.set_len(ALLOCATION_CHUNK_SIZE)
        };
        let mut n_bytes_read: usize = 0;
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;
        let mut end_of_stream_hit = false;

        let decode_base = already_decoded;
        while n_bytes_read < buffer.len()
            && !stopping_point_reached
            && !(STOP_INNER_ON_PENDING_FLUSH
                && pending_stop_after_flush
                && !wrapper.session_pending())
        {
            let bit_before_read = wrapper.tell_compressed();
            let r = wrapper.read_stream(&mut buffer[n_bytes_read..])?;
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
            // `decode_chunk_isal` such a fragment directly.
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
        buffer.truncate(append_len);
        if !buffer.is_empty() {
            chunk.append_owned_buffer(buffer);
        }
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

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    // When the reader runs past `stop_hint_bits` without an inexact
    // stop, `tell_compressed()` can land mid-block. Successor chunks must
    // resume at the last END_OF_BLOCK position (pre-header), not mid-block.
    let final_bit = if stopping_point_reached || reached_stream_end {
        last_end_bit
    } else if last_eob_pos > encoded_offset_bits {
        // Hit the until read cap without an inexact header-stop — finalize at
        // the last pre-header EOB, never at a mid-block bit cursor.
        last_eob_pos
    } else {
        wrapper.tell_compressed()
    };
    chunk.finalize(final_bit);
    Ok(chunk)
}

/// Marker-bootstrap then ISA-L for speculative prefetch (no predecessor window).
/// Requires a real deflate block boundary at `encoded_offset_bits`.
///
/// Returns a `ChunkData` whose `data_with_markers` covers the bootstrap
/// prefix (still containing markers; resolved by `apply_window` in the
/// consumer) and whose `data` covers the ISA-L bulk (already clean).
/// The chunk stops at the next deflate block boundary at-or-past
/// `stop_hint_bits`, or at BFINAL, or when accumulated `decoded_size`
/// crosses `max_decoded_chunk_size`.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub fn decode_chunk_marker_bootstrap_then_isal(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    _initial_window_unused: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();

    // Phase 1 — bootstrap with deflate_block::Block, the rapidgzip-
    // faithful port of vendor/.../gzip/deflate.hpp's deflate::Block<>.
    // Decodes deflate blocks one-by-one until any of:
    //   (a) cumulative clean (non-marker) bytes reach MAX_WINDOW_SIZE
    //       AND the next block header is past `stop_hint_bits` is not yet
    //       required — phase 2 (ISA-L) can take over;
    //   (b) BFINAL block is decoded (single-member tail);
    //   (c) we're at a non-fixed-Huffman block boundary at-or-past
    //       `stop_hint_bits` (chunk's end condition).
    //
    // Mirrors GzipChunk.hpp::decodeChunkWithRapidgzip's main loop
    // (vendor/.../chunkdecoding/GzipChunk.hpp:468-654), in particular
    // the `cleanDataCount >= deflate::MAX_WINDOW_SIZE` handoff at
    // L520-525.
    let t_bootstrap = std::time::Instant::now();
    let mut bootstrap = bootstrap_with_deflate_block(input, encoded_offset_bits, stop_hint_bits)?;
    let bootstrap_dur_us = t_bootstrap.elapsed().as_micros();

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    if !bootstrap.markers.is_empty() {
        chunk.append_markered(&bootstrap.markers);
    }
    // EXPERIMENT: return the markers Vec to the per-thread pool.
    return_bootstrap_output_to_pool(std::mem::take(&mut bootstrap.markers));

    // Whenever the bootstrap reached a block boundary at-or-past
    // stop_hint_bits (rare on chunks > 32 KiB), or BFINAL fired, or no
    // clean window accumulated — we're done. The chunk is marker-only.
    let Some(clean_window) = bootstrap.clean_window else {
        if trace::is_enabled() {
            trace::emit(
                "worker",
                "decode_span",
                &format!(
                    r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":0,"phase":"bootstrap_only","markers":{}"#,
                    bootstrap.markers.len(),
                ),
            );
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    };
    if bootstrap.bfinal_hit || bootstrap.end_bit_offset >= stop_hint_bits {
        if trace::is_enabled() {
            trace::emit(
                "worker",
                "decode_span",
                &format!(
                    r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":0,"phase":"bootstrap_terminal","markers":{}"#,
                    bootstrap.markers.len(),
                ),
            );
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    }

    // Phase 2 — ISA-L bulk decode from the bootstrap's block boundary
    // with the clean 32 KiB tail as dict. Uses the same production
    // `IsalInflateWrapper` path as on-demand decode.
    let bit_offset = bootstrap.end_bit_offset;
    let t_inflate = std::time::Instant::now();
    let tail = decode_chunk_isal_impl(
        input,
        bit_offset,
        stop_hint_bits,
        &clean_window,
        configuration,
    )?;
    let inflate_dur_us = t_inflate.elapsed().as_micros();
    let tail_bytes = tail.data.len();
    absorb_isal_tail(&mut chunk, tail);
    if trace::is_enabled() {
        trace::emit(
            "worker",
            "decode_span",
            &format!(
                r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":{inflate_dur_us},"phase":"bootstrap+inflate","markers":{},"tail_bytes":{tail_bytes}"#,
                bootstrap.markers.len(),
            ),
        );
    }
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    Ok(chunk)
}

/// Merge an ISA-L tail segment (clean bytes + block boundaries) into a
/// chunk that already holds a marker bootstrap prefix.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn absorb_isal_tail(dst: &mut ChunkData, tail: ChunkData) {
    let end_bit = tail.encoded_offset_bits + tail.encoded_size_bits;
    let decoded_base = dst.decoded_size();

    if !tail.data.is_empty() {
        dst.append_clean(&tail.data);
    }

    for f in &tail.footers {
        dst.append_footer(f.crc32, f.uncompressed_size, f.end_bit_offset);
    }

    for sc in tail.subchunks.iter().skip(1) {
        let _ =
            dst.append_block_boundary_at(sc.encoded_offset_bits, decoded_base + sc.decoded_offset);
    }

    dst.stopped_preemptively = tail.stopped_preemptively;
    dst.finalize(end_bit);
}

/// Result of one bootstrap pass over deflate blocks via
/// [`deflate_block::Block`]. Mirrors the early-exit contract of
/// rapidgzip's `decodeChunkWithRapidgzip` main loop
/// (vendor/.../chunkdecoding/GzipChunk.hpp:468-654).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
struct DeflateBootstrap {
    /// u16 output spanning every block decoded in this bootstrap pass.
    /// Values < MARKER_BASE are literal bytes; values ≥ MARKER_BASE are
    /// MapMarkers cross-chunk back-references (consumer resolves via
    /// `apply_window`).
    markers: Vec<u16>,
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

/// Phase 1 bootstrap: drive [`deflate_block::Block`] block-by-block from
/// `start_bit_offset` until any of (a) cumulative clean bytes reach
/// MAX_WINDOW_SIZE at a block boundary, (b) BFINAL fires, or (c) we're
/// at a non-fixed block boundary at-or-past `stop_hint_bits`.
///
/// EXPERIMENT: per-thread pool of bootstrap output buffers.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
thread_local! {
    static BOOTSTRAP_OUTPUT_POOL: std::cell::RefCell<Vec<u16>> = std::cell::RefCell::new(Vec::with_capacity(128 * 1024));
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn take_bootstrap_output_from_pool() -> Vec<u16> {
    BOOTSTRAP_OUTPUT_POOL.with(|cell| {
        let mut v = std::mem::take(&mut *cell.borrow_mut());
        v.clear();
        if v.capacity() < 128 * 1024 {
            v.reserve(128 * 1024);
        }
        v
    })
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn return_bootstrap_output_to_pool(mut v: Vec<u16>) {
    v.clear();
    BOOTSTRAP_OUTPUT_POOL.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Keep the larger of the two.
        if v.capacity() > slot.capacity() {
            *slot = v;
        }
    });
}

/// Mirror of the `while ( true )` loop in
/// `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654), restricted to the
/// single-member case (no multi-stream loop) and with the handoff
/// triggered exclusively by `cleanDataCount` (GzipChunk.hpp:520-525).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn bootstrap_with_deflate_block(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::deflate_block::{Block, MAX_WINDOW_SIZE};
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
    use crate::decompress::parallel::trace_v2;
    use std::cell::RefCell;

    let _tv2 = trace_v2::SpanGuard::begin_with(
        "worker.bootstrap",
        &format!(r#""start_bit":{start_bit_offset},"stop_hint":{stop_hint_bits}"#),
    );

    // Per-thread Block recycling. Block::new() allocates a 128 KiB
    // ring + initializes the marker zone (64 KiB writes). Doing that
    // per chunk was a measured ~4 pp of CPU in `clear_page_erms` on
    // silesia. Per-thread reuse amortizes the cost to once-per-worker.
    // `Block::reset` re-primes marker zone but skips the alloc.
    thread_local! {
        static BOOTSTRAP_BLOCK: RefCell<Block> = RefCell::new(Block::new());
    }

    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        )));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }

    // Bootstrap output: typically ~32 KiB + one trailing block, up to
    // ~128 KiB worst case for a single dense dynamic block.
    // EXPERIMENT: take pooled buffer from thread-local instead of fresh alloc.
    let mut output: Vec<u16> = take_bootstrap_output_from_pool();
    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let block = &mut *block;
        let output = &mut output;

        // Handoff signal: use `Block::contains_marker_bytes()` directly.
        // It flips to false (sticky) when `distance_to_last_marker_byte
        // >= MAX_WINDOW_SIZE`, which guarantees the trailing 32 KiB of
        // output is clean. Mirror of vendor's flip at deflate.hpp:1282-1287
        // inside `Block::read`, which is what triggers `setInitialWindow`
        // and exits marker mode.
        //
        // Previous implementation used a `trailing_clean` counter updated
        // only at block boundaries. On silesia, individual blocks (~85 KiB)
        // frequently contain a marker mid-block which reset the counter to
        // the trailing clean tail of the block — typically < 32 KiB. As a
        // result `trailing_clean` rarely reached MAX_WINDOW_SIZE and
        // bootstrap consumed ~60% of bytes (vs vendor ~0%). Switching to
        // the per-symbol distance-to-last-marker counter that Block
        // already maintains is both vendor-parity and the right signal.
        let mut clean_handoff_armed: bool;
        let mut bfinal_hit = false;
        // `end_bit_offset` always points just past the last completed block.
        // Always assigned before the post-loop read (every `break` is
        // preceded by an assignment), so no initializer is needed.
        let mut end_bit_offset;

        loop {
            // Snapshot the bit position BEFORE reading this block's header.
            // This is what the next chunk's worker resumes from when this
            // chunk hands off to ISA-L (or when this block becomes BFINAL).
            let next_block_offset = absolute_bit_pos(byte_offset, &bits);

            // Handoff check (rapidgzip GzipChunk.hpp:520-525): hand off to
            // ISA-L when Block has flipped to clean mode AND we have
            // enough output to seed a 32 KiB window.
            //
            // `!block.contains_marker_bytes()` is the per-symbol-tracked
            // signal that the last `distance_to_last_marker_byte` ≥
            // MAX_WINDOW_SIZE — i.e., the trailing 32 KiB of output is
            // clean. Cheaper and earlier than the previous block-boundary
            // `trailing_clean` recompute.
            clean_handoff_armed =
                !block.contains_marker_bytes() && output.len() >= MAX_WINDOW_SIZE;
            if clean_handoff_armed {
                end_bit_offset = next_block_offset;
                break;
            }

            // Read this block's header. `treat_last_block_as_error = false`
            // (we WANT to see BFINAL so we can stop).
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
            match header_res {
                Ok(()) => {}
                Err(e) => {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deflate header at bit {next_block_offset}: {e:?}"),
                    )));
                }
            }

            // Preemptive stop condition (rapidgzip GzipChunk.hpp:550-555):
            // if this block's header sits at-or-past stop_hint_bits AND it is
            // non-fixed AND not BFINAL, stop here so the chunk ends on a
            // boundary the successor's BlockFinder can re-find.
            if next_block_offset >= stop_hint_bits && !block.is_last_block() {
                // We've already advanced `bits` past this header. Rewind
                // logically by reporting `end_bit_offset = next_block_offset`
                // — the successor will re-parse the same header.
                end_bit_offset = next_block_offset;
                break;
            }

            // Decode the block's body, one read() call at a time, until
            // EOB. `n_max_to_decode = usize::MAX` matches rapidgzip's
            // `std::numeric_limits<size_t>::max()` at GzipChunk.hpp:568.
            let before_len = output.len();
            let t_body = std::time::Instant::now();
            let _tv2_body = trace_v2::SpanGuard::begin("worker.block_body");
            while !block.eob() {
                let res = block.read(&mut bits, &mut *output, usize::MAX);
                if let Err(e) = res {
                    // Body-failure diagnostics for the speculation-accuracy
                    // investigation (per advisor recommendation). Captures
                    // (a) candidate start_bit, (b) bit position when body
                    // failed (= bits consumed past block start), (c) bytes
                    // emitted before failure (= waste), (d) error variant.
                    let bits_at_fail = absolute_bit_pos(byte_offset, &bits);
                    let bytes_wasted = output.len() - before_len;
                    let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                    BODY_FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    BODY_FAIL_BYTES_WASTED
                        .fetch_add(bytes_wasted as u64, std::sync::atomic::Ordering::Relaxed);
                    BODY_FAIL_BITS_INTO_BODY
                        .fetch_add(bits_into_body as u64, std::sync::atomic::Ordering::Relaxed);
                    // Per-error-variant counter so we can see which
                    // block_error dominates.
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
                    // Detailed per-failure dump (one JSON line) if the
                    // GZIPPY_BODY_FAIL_LOG env var points at a writable
                    // path. Mirror of trace.rs pattern.
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
                (output.len() - before_len) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            // (Previously here: trailing_clean update from block_slice.
            // No longer needed — handoff uses Block::contains_marker_bytes()
            // which is updated per-symbol inside Block::read.)
            end_bit_offset = absolute_bit_pos(byte_offset, &bits);

            if block.is_last_block() {
                bfinal_hit = true;
                break;
            }
        }

        // Build the clean dict if we have one.
        let clean_window = if clean_handoff_armed && output.len() >= MAX_WINDOW_SIZE {
            let start = output.len() - MAX_WINDOW_SIZE;
            // Invariant: the trailing MAX_WINDOW_SIZE values of `output` are
            // < MARKER_BASE (clean). Assert this — corruption here would
            // seed ISA-L with garbage and produce a wrong-CRC chunk.
            let window: Vec<u8> = output[start..]
                .iter()
                .map(|&v| {
                    assert!(
                        v < MARKER_BASE,
                        "bootstrap clean window contained marker at offset {}; \
                     trailing_clean tracker broken",
                        v.saturating_sub(MARKER_BASE)
                    );
                    v as u8
                })
                .collect();
            Some(window)
        } else {
            None
        };

        Ok(DeflateBootstrap {
            markers: std::mem::take(output),
            end_bit_offset,
            clean_window,
            bfinal_hit,
        })
    }) // BOOTSTRAP_BLOCK.with closure
}

/// Compute the absolute bit position within `data` given that `bits`
/// was constructed from `&data[byte_offset..]`. The Bits buffer
/// pre-loads bytes from its slice, so the actual consumed-from-slice
/// count is `bits.pos * 8 - bits.available()`.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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

#[cfg(not(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
)))]
pub fn decode_chunk_marker_bootstrap_then_isal(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
)))]
pub fn decode_chunk_isal(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(test)]
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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
        for v in &chunk.data_with_markers {
            out.push(*v as u8);
        }
        out.extend_from_slice(&chunk.data);
        out
    }

    #[test]
    fn decode_chunk_isal_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let stop_hint_bits = deflate.len() * 8;
        let chunk = decode_chunk_isal(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
    }

    #[test]
    fn decode_chunk_isal_stops_before_eof_when_stop_hint_bits_set() {
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
        let chunk = decode_chunk_isal(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert!(chunk.decoded_size() < payload.len());
        let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        assert!(chunk_end >= stop_hint_bits);
    }

    /// Neurotic profile fixture: gzip(1) -9 on 64 MiB silesia head. Chunk 0
    /// stops at a non-byte-aligned bit; chunk 1 must resume with the published
    /// 32 KiB window. Fails with `InvalidBlock` when handoff is wrong.
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
            std::io::Write::write_all(child.stdin.as_mut().expect("stdin"), head).expect("write");
            let mut gz = Vec::new();
            child
                .stdout
                .as_mut()
                .expect("stdout")
                .read_to_end(&mut gz)
                .expect("read gzip stdout");
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
        let chunk0 = decode_chunk_isal(deflate, 0, spacing_bits, &zero, cfg).expect("chunk0");
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
        let chunk1 = decode_chunk_isal(deflate, resume_at, resume_at + spacing_bits, &tail, cfg)
            .expect("chunk1 at chunk0 end");
        assert!(!chunk1.is_empty());
    }

    /// Regression for the parallel-SM hang. Given a sub-block input
    /// fragment — the gzip end-of-stream byte-alignment padding (0-7
    /// zero bits before the footer) — `read_stream` can make no
    /// progress, and `decode_chunk_isal_impl`'s decode loop used to
    /// call it forever. `decode_chunk_isal` must instead return.
    ///
    /// The decode runs in a child thread joined against a deadline, so
    /// a regression fails this test instead of hanging the whole suite.
    #[test]
    fn decode_chunk_isal_terminates_on_sub_byte_eof_padding() {
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
            let _ = decode_chunk_isal(&[0u8], 4, 8, &[], cfg);
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "decode_chunk_isal did not return on a sub-byte EOF-padding \
                 fragment: isal_inflate is spinning"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        worker.join().expect("decode worker panicked");
    }
}
