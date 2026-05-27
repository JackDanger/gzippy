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
    // `worker.isal_stream_inflate` span — wraps every invocation of
    // gzippy's ISA-L stream-inflate path. Both call sites land here:
    //   (a) the post-bootstrap call at gzip_chunk.rs:549 (after the
    //       pure-Rust phase-1 produces a clean 32 KiB window)
    //   (b) the non-speculative direct path via `decode_chunk_isal`
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
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.isal_stream_inflate",
        &format!(
            r#""start_bit":{encoded_offset_bits},"stop_hint":{stop_hint_bits},"has_window":{}"#,
            !initial_window.is_empty()
        ),
    );
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
        // Reserve at least ALLOCATION_CHUNK_SIZE of spare capacity.
        // chunk.data starts with `max_decoded_chunk_size` (80 MiB)
        // capacity from `take_u8`, so this is a no-op until we exceed
        // that — at which point amortized growth handles it.
        chunk.data.reserve(ALLOCATION_CHUNK_SIZE);
        let buffer_cap = ALLOCATION_CHUNK_SIZE;
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
            // SAFETY: we reserved `ALLOCATION_CHUNK_SIZE` of spare
            // capacity above; `prev_data_len + n_bytes_read` is within
            // the allocation and the slice covers
            // `[prev_data_len + n_bytes_read, prev_data_len + buffer_cap)`
            // bytes which are uninitialized but writable. ISA-L writes
            // monotonically forward; we only `set_len` AFTER the inner
            // loop has decided how many bytes are valid.
            let spare: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    chunk.data.as_mut_ptr().add(prev_data_len + n_bytes_read),
                    buffer_cap - n_bytes_read,
                )
            };
            let r = wrapper.read_stream(spare)?;
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
        if append_len > 0 {
            // Commit the bytes ISA-L wrote into chunk.data's spare
            // capacity. Bytes beyond `prev_data_len + append_len`
            // (between EOB and the truncate point) are left
            // uninitialized in the Vec — the NEXT outer iteration
            // will overwrite them starting at
            // `chunk.data.len() = prev_data_len + append_len`.
            //
            // We deliberately do NOT call
            // `chunk.note_clean_bytes_written_in_place` here because
            // its `subchunks.last_mut().decoded_size += written`
            // would double-count: the inner loop already updated
            // subchunks via `note_inner_decoded_bytes(last_per_call)`
            // for the FULL n_bytes_read (before truncation).
            //
            // SAFETY: `prev_data_len + append_len ≤ prev_data_len +
            // n_bytes_read ≤ prev_data_len + buffer_cap`, which is
            // within the capacity reserved at the start of the
            // outer iteration; ISA-L wrote initialized bytes through
            // `prev_data_len + n_bytes_read`.
            unsafe { chunk.data.set_len(prev_data_len + append_len) };
            if chunk.configuration.crc32_enabled {
                if let Some(last_crc) = chunk.crc32s.last_mut() {
                    last_crc.update(&chunk.data[prev_data_len..prev_data_len + append_len]);
                }
            }
            chunk.statistics.non_marker_count += append_len as u64;
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
    // `worker.absorb_isal_tail` span — wraps the merge of the
    // ISA-L bulk-decode result into the chunk that holds the
    // bootstrap marker prefix. Copies `tail.data` into `dst.data`,
    // appends the tail's CRC into `dst.crc32s` via the constant-time
    // polynomial-append path (vendor `crc32.hpp:214-258`), and
    // merges subchunk metadata.
    //
    // For a typical 7.5 MB silesia chunk where most of the chunk
    // came through ISA-L (bootstrap was small), tail.data can be
    // several MB — large memcpy + statistics merge.
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.absorb_isal_tail",
        &format!(
            r#""tail_bytes":{},"tail_crcs":{},"tail_footers":{}"#,
            tail.data.len(),
            tail.crc32s.len(),
            tail.footers.len()
        ),
    );
    let end_bit = tail.encoded_offset_bits + tail.encoded_size_bits;
    let decoded_base = dst.decoded_size();

    if !tail.data.is_empty() {
        // Per the post-inline-always bench-sm profile, the prior
        // `dst.append_clean(&tail.data)` re-CRC'd `tail.data` from
        // scratch — 2.45% of total cycles in
        // `crc32fast::specialized::pclmulqdq::calculate` traced
        // back here. `decode_chunk_isal_impl` had ALREADY computed
        // those bytes' CRC into `tail.crc32s.last_mut()` during
        // the in-place commit step. Combine that CRC into
        // `dst.crc32s` via the constant-time polynomial `append`
        // (vendor `crc32.hpp:214-258`) and skip the bytewise
        // pclmulqdq pass.
        //
        // We still need the bytes in `dst.data`, the statistics
        // update, and the subchunk size bump — inline those bits
        // of `append_clean` directly.
        if dst.configuration.crc32_enabled {
            // For parallel-SM (single-member at routing time),
            // tail's crc32s has exactly one entry. If `tail` ever
            // crosses a stream boundary mid-decode (multi-member
            // input that slipped past `is_likely_multi_member`'s
            // 16 MiB scan and was misrouted as single-member),
            // tail.crc32s.len() > 1 and the per-stream entries
            // need to be split by the corresponding footers.
            // Handle that explicitly below; the common case is
            // the fast single-`append` path.
            if tail.crc32s.len() == 1 && tail.footers.is_empty() {
                if let (Some(dst_last), Some(tail_only)) =
                    (dst.crc32s.last_mut(), tail.crc32s.first())
                {
                    dst_last.append(tail_only);
                }
            } else {
                // Multi-stream tail: walk the (crc, footer) pairs
                // in order. Each crc[i] covers bytes between
                // footer[i-1] and footer[i] (with footer[0] being
                // the first footer if i == 0, etc). Approximate by
                // appending crc[0] into dst's current trailing
                // hasher, then for each footer push a fresh
                // hasher and append the next tail crc into it.
                for (i, footer) in tail.footers.iter().enumerate() {
                    if let (Some(dst_last), Some(tail_crc)) =
                        (dst.crc32s.last_mut(), tail.crc32s.get(i))
                    {
                        dst_last.append(tail_crc);
                    }
                    dst.append_footer(
                        footer.crc32,
                        footer.uncompressed_size,
                        footer.end_bit_offset,
                    );
                }
                // The crc entry AFTER the last footer (if any) goes
                // into the freshly-pushed hasher from append_footer.
                if let (Some(dst_last), Some(tail_trailing)) =
                    (dst.crc32s.last_mut(), tail.crc32s.get(tail.footers.len()))
                {
                    dst_last.append(tail_trailing);
                }
            }
        }

        // Bytes + statistics + subchunk size — the non-CRC half of
        // `append_clean`. We extend in place rather than via
        // `append_clean` because that path would re-CRC.
        //
        // `allocator_api2::vec::Vec::extend_from_slice` does NOT
        // specialize for `Copy` source types (unlike `std::vec::Vec`
        // which has `SpecExtend`) — it falls back to the generic
        // `extend → Cloned::next → Option::cloned → Clone::clone`
        // iterator chain. Per the bench-sm profile post-absorb-fix,
        // that chain accounted for 2.47% of total cycles for u8
        // bytes (where `Clone::clone` is a no-op load that should
        // have been folded into a memcpy). Replace with an explicit
        // `reserve` + `copy_nonoverlapping` + `set_len` — the same
        // shape `std::vec::Vec`'s specialization would have used.
        let added = tail.data.len();
        let prev_len = dst.data.len();
        dst.data.reserve(added);
        // SAFETY: `reserve(added)` ensures capacity covers
        // `prev_len + added`; `tail.data` and `dst.data` are
        // distinct allocations (separate Vecs); set_len with the
        // post-copy length is legal because all `added` bytes are
        // initialized from `tail.data`'s slice.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tail.data.as_ptr(),
                dst.data.as_mut_ptr().add(prev_len),
                added,
            );
            dst.data.set_len(prev_len + added);
        }
        dst.statistics.non_marker_count += added as u64;
        if let Some(last) = dst.subchunks.last_mut() {
            last.decoded_size += added;
        }
    }

    // Multi-stream case already emitted footers above (interleaved
    // with the per-stream CRC propagation). For the
    // single-stream-or-empty case, footers still need pushing.
    if tail.crc32s.len() == 1 || tail.data.is_empty() {
        for f in &tail.footers {
            dst.append_footer(f.crc32, f.uncompressed_size, f.end_bit_offset);
        }
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
        let cap_before = v.capacity();
        if cap_before < 128 * 1024 {
            BOOTSTRAP_OUTPUT_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v.reserve(128 * 1024);
        } else {
            BOOTSTRAP_OUTPUT_REUSED_BYTES.fetch_add(
                (cap_before * std::mem::size_of::<u16>()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        BOOTSTRAP_OUTPUT_TAKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    })
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn return_bootstrap_output_to_pool(mut v: Vec<u16>) {
    v.clear();
    BOOTSTRAP_OUTPUT_RETURNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BOOTSTRAP_OUTPUT_POOL.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Keep the larger of the two.
        if v.capacity() > slot.capacity() {
            *slot = v;
        } else {
            BOOTSTRAP_OUTPUT_DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// RAII guard that owns the bootstrap output buffer and returns it to
/// the thread-local pool on Drop — UNCONDITIONALLY (success OR failure).
///
/// Background: prior to this guard, the bootstrap function held the
/// buffer as a local `Vec<u16>` and only returned it to the pool on
/// the success path (via the caller's
/// `return_bootstrap_output_to_pool(std::mem::take(&mut bootstrap.markers))`
/// at the call site). On Err, the function early-returned and the
/// local Vec dropped — its underlying allocation (often 7 MB, well
/// above glibc's 128 KiB `MMAP_THRESHOLD`) went through `munmap` and
/// the next `take_*_from_pool` on that thread paid a fresh
/// `mmap + first-touch faults` (≈ 2.7 ms per leaked Vec).
///
/// The May 26 2026 visibility counters made this measurable: across
/// silesia-large 16T, `takes=98 returns=64` — i.e. **34 takes never
/// returned**, matching the 33 bootstrap failures observed via
/// `worker.bootstrap.outcome` instrumentation. With this guard, the
/// failure path also returns the buffer.
///
/// On success the buffer is consumed via `std::mem::take(output)`
/// inside the closure; the guard's `inner` then has capacity = 0
/// (mem::take swaps with `Vec::default()`), and the Drop's
/// `cap == 0` check makes the post-mem::take Drop a no-op so we
/// don't pollute the pool's slot with an empty Vec.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
struct BootstrapBuffer {
    inner: Option<Vec<u16>>,
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
impl BootstrapBuffer {
    fn new() -> Self {
        Self {
            inner: Some(take_bootstrap_output_from_pool()),
        }
    }
    fn get_mut(&mut self) -> &mut Vec<u16> {
        self.inner
            .as_mut()
            .expect("BootstrapBuffer::get_mut after take")
    }
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
impl Drop for BootstrapBuffer {
    fn drop(&mut self) {
        if let Some(v) = self.inner.take() {
            // On the success path, the closure called
            // `std::mem::take(output)` to transfer the buffer into
            // `DeflateBootstrap.markers` — the guard's inner now has
            // cap=0 because `mem::take` swapped in `Vec::default()`.
            // Skip the return; the success-path caller will
            // explicitly return the BIG markers Vec to the pool.
            //
            // On the failure path the buffer is intact (cap >= 128 KiB
            // from `take_*_from_pool`) and we want to put it back so
            // the next take on this thread reuses the same warm pages.
            if v.capacity() > 0 {
                return_bootstrap_output_to_pool(v);
            }
        }
    }
}

/// Counter: fresh `Vec::reserve(128*1024)` events from
/// `take_bootstrap_output_from_pool`. Each is an mmap-eligible
/// allocation (256 KiB u16 buffer > glibc mmap threshold). With
/// pool reuse working, expected ≈ num_worker_threads (one allocation
/// per thread, then reused on every subsequent call). If this counter
/// approaches the bootstrap call count, the pool is broken.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub static BOOTSTRAP_OUTPUT_ALLOCS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: total `take` calls. Pool effectiveness =
/// `1 - (allocs / takes)`. Vendor-equivalent: rpmalloc per-thread arena
/// reuse rate, which approaches 1.0 after warm-up.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub static BOOTSTRAP_OUTPUT_TAKES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: sum of capacity (in BYTES) of Vecs returned from the
/// pool with cap ≥ 128 KiB. Total bytes the pool "saved" from a fresh
/// allocation. A run-end value of `~BOOTSTRAP_OUTPUT_TAKES *
/// 256 KiB` means every take hit the pool path.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub static BOOTSTRAP_OUTPUT_REUSED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: every call to `return_bootstrap_output_to_pool`. Mirrors
/// TAKES on the success path; if RETURNS << TAKES, bootstraps are
/// erroring out and not returning their buffer (it's still moved via
/// the chunk_fetcher.rs path at line 508, so should always match).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub static BOOTSTRAP_OUTPUT_RETURNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: returns where the incoming Vec was DROPPED instead of
/// retained (because the pool's slot already had a larger cap).
/// Non-zero means we threw away a hot allocation — should be 0 with
/// 1-Vec-per-thread pool when caps are stable.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub static BOOTSTRAP_OUTPUT_DROPPED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
    let result = bootstrap_with_deflate_block_inner(data, start_bit_offset, stop_hint_bits);
    match &result {
        Ok(b) => {
            trace_v2::emit_instant(
                "worker.bootstrap.outcome",
                &format!(
                    r#""result":"ok","markers_len":{},"end_bit":{},"clean_window":{},"bfinal":{}"#,
                    b.markers.len(),
                    b.end_bit_offset,
                    b.clean_window.is_some(),
                    b.bfinal_hit
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

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn bootstrap_with_deflate_block_inner(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::deflate_block::{Block, MAX_WINDOW_SIZE};
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
    use crate::decompress::parallel::trace_v2;
    use std::cell::RefCell;

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
    // ~128 KiB worst case for a single dense dynamic block. The
    // `BootstrapBuffer` RAII guard ensures the buffer goes back to
    // the thread-local pool on BOTH the success and failure paths —
    // see the type's doc comment for the leak this plugs.
    let mut buffer_guard = BootstrapBuffer::new();
    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let block = &mut *block;
        let output: &mut Vec<u16> = buffer_guard.get_mut();

        // Tracks trailing CLEAN-byte run length. When it reaches
        // MAX_WINDOW_SIZE AT a block boundary, the next 32 KiB of `output`
        // forms a clean dict for ISA-L. Cumulative count saturates at
        // MAX_WINDOW_SIZE (we only need the latest 32 KiB).
        let mut trailing_clean: usize = 0;
        // True once `trailing_clean >= MAX_WINDOW_SIZE` AND the most recent
        // block has just finished (so the next bit position is a real block
        // header start that ISA-L can resume from).
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

            // Handoff check (rapidgzip GzipChunk.hpp:520-525): if we have a
            // clean 32 KiB tail AND we're at a real block boundary, stop
            // bootstrap and let phase 2 run. We do this BEFORE reading the
            // next header so `end_bit_offset` is at a clean boundary.
            clean_handoff_armed = trailing_clean >= MAX_WINDOW_SIZE;
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

            // Block fully decoded. Update trailing_clean from the bytes
            // just produced. Markers (≥ MARKER_BASE) reset the run; clean
            // bytes extend it (saturating at MAX_WINDOW_SIZE).
            let block_slice = &output[before_len..];
            if !block_slice.is_empty() {
                let trailing_this_block = block_slice
                    .iter()
                    .rev()
                    .take_while(|&&v| v < MARKER_BASE)
                    .count();
                if trailing_this_block == block_slice.len() {
                    // Entire block was clean — extend the prior run.
                    trailing_clean = (trailing_clean + trailing_this_block).min(MAX_WINDOW_SIZE);
                } else {
                    // A marker appeared mid-block; the trailing clean run
                    // restarts from after that last marker.
                    trailing_clean = trailing_this_block.min(MAX_WINDOW_SIZE);
                }
            }

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
