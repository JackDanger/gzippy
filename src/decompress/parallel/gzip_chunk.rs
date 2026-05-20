//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_isal_inexact`] — on-demand decode when the predecessor
//!   window is known (or chunk 0 with a zero dict). Clean bytes only.
//! - [`decode_chunk_marker_bootstrap_then_isal`] — speculative prefetch when
//!   no window yet: marker bootstrap for cross-chunk refs, then ISA-L bulk.
//!
//! `until_bits` caps how far ISA-L may read (`IsalInflateWrapper::with_until_bits`
//! / `refill_buffer`), matching vendor `m_encodedUntilOffset`.

#![allow(dead_code)]

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::inflate_wrapper::InflateError;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::inflate_wrapper::{
    DeflateCompressionType, IsalInflateWrapper, StoppingPoints,
};

#[derive(Debug)]
pub enum ChunkDecodeError {
    InflateFailed(InflateError),
    BootstrapFailed(std::io::Error),
    ExactStopMissed { requested: usize, actual: usize },
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

/// Raw isal-sys decode with stopping points — used when the fast path
/// starts at a non-byte-aligned bit offset (IsalInflateWrapper cross-chunk
/// resume is unreliable on real data despite matching unit tests).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_chunk_isal_inexact_raw(
    input: &[u8],
    bit_offset: usize,
    until_bits: usize,
    dict: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    use isal::isal_sys::igzip_lib as isal_raw;

    let t_decode = std::time::Instant::now();
    let mut chunk = ChunkData::new(bit_offset, configuration);
    let byte_idx = bit_offset / 8;
    let bit_skip = bit_offset % 8;
    if byte_idx >= input.len() {
        return Err(ChunkDecodeError::InflateFailed(
            crate::decompress::parallel::inflate_wrapper::InflateError::StartBitPastEnd,
        ));
    }
    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    state.crc_flag = isal_raw::ISAL_DEFLATE;
    state.points_to_stop_at = isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK
        | isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER
        | isal_raw::ISAL_STOPPING_POINT_END_OF_STREAM;

    let stop_hint = until_bits.min(input.len() * 8);
    let read_cap = input.len() * 8;
    let max_bits = read_cap.saturating_sub(bit_offset);
    if bit_skip > 0 {
        state.read_in = (input[byte_idx] as u64) >> bit_skip;
        state.read_in_length = (8 - bit_skip) as i32;
        state.next_in = unsafe { input.as_ptr().add(byte_idx + 1) as *mut u8 };
        let consumed = (8 - bit_skip) as usize;
        let max_bytes = if max_bits > consumed {
            (max_bits - consumed) / 8
        } else {
            0
        };
        state.avail_in = max_bytes.min(input.len() - byte_idx - 1) as u32;
    } else {
        state.next_in = unsafe { input.as_ptr().add(byte_idx) as *mut u8 };
        let max_bytes = max_bits / 8;
        state.avail_in = max_bytes.min(input.len() - byte_idx) as u32;
    }

    static ZERO_WINDOW: [u8; 32768] = [0u8; 32768];
    let dict = if dict.is_empty() {
        &ZERO_WINDOW[..]
    } else {
        dict
    };
    let ret_dict = unsafe {
        isal_raw::isal_inflate_set_dict(&mut state, dict.as_ptr() as *mut u8, dict.len() as u32)
    };
    if ret_dict != 0 {
        return Err(ChunkDecodeError::InflateFailed(
            crate::decompress::parallel::inflate_wrapper::InflateError::SetDictFailed,
        ));
    }

    let cap = configuration.max_decoded_chunk_size;
    let mut output: Vec<u8> = Vec::with_capacity(ALLOCATION_CHUNK_SIZE.min(cap));
    let mut out_pos: usize = 0;
    let mut last_eob_pos = bit_offset;
    let mut chunk_end_override: Option<usize> = None;

    loop {
        if out_pos >= cap {
            chunk.stopped_preemptively = true;
            break;
        }
        let want = ALLOCATION_CHUNK_SIZE.min(cap - out_pos);
        if output.len() < out_pos + want {
            output.resize(out_pos + want, 0);
        }
        let remaining = want;
        state.avail_out = remaining as u32;
        state.next_out = unsafe { output.as_mut_ptr().add(out_pos) };

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        let written = remaining - state.avail_out as usize;
        if written > 0 {
            chunk.append_clean(&output[out_pos..out_pos + written]);
        }
        out_pos += written;

        match ret {
            0 | 1 => {}
            -1 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidBlock,
                ));
            }
            -2 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidSymbol,
                ));
            }
            -3 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidLookback,
                ));
            }
            other => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::Internal(other),
                ));
            }
        }

        let bit_pos =
            input.len() * 8 - state.avail_in as usize * 8 - state.read_in_length.max(0) as usize;

        if state.stopped_at == isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK {
            chunk.append_block_boundary(bit_pos);
            last_eob_pos = bit_pos;
            state.stopped_at = isal_raw::ISAL_STOPPING_POINT_NONE;
        } else if state.stopped_at == isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER {
            let not_final = state.bfinal == 0;
            let not_fixed = state.btype != 1;
            if last_eob_pos >= stop_hint && not_final && not_fixed {
                chunk_end_override = Some(last_eob_pos);
                break;
            }
            state.stopped_at = isal_raw::ISAL_STOPPING_POINT_NONE;
        } else if state.stopped_at == isal_raw::ISAL_STOPPING_POINT_END_OF_STREAM {
            break;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 && state.stopped_at == 0 {
            break;
        }
    }

    let isal_pos =
        input.len() * 8 - state.avail_in as usize * 8 - state.read_in_length.max(0) as usize;
    let final_bit_pos = chunk_end_override.unwrap_or(isal_pos);
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    chunk.finalize(final_bit_pos);
    Ok(chunk)
}

/// **Inexact** ISA-L decode when the predecessor window is known.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decode_chunk_isal_inexact(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    // `until_bits` is an inexact stop *hint* (vendor `untilOffset`), not a hard
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
    let mut already_decoded: usize = 0;

    while !stopping_point_reached {
        let mut buffer: Vec<u8> = Vec::with_capacity(ALLOCATION_CHUNK_SIZE);
        #[allow(clippy::uninit_vec)]
        unsafe {
            buffer.set_len(ALLOCATION_CHUNK_SIZE)
        };
        let mut n_bytes_read: usize = 0;
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;
        let mut end_of_stream_hit = false;

        while n_bytes_read < buffer.len() && !stopping_point_reached {
            let r = wrapper.read_stream(&mut buffer[n_bytes_read..])?;
            last_per_call = r.bytes_written;
            n_bytes_read += last_per_call;
            chunk.note_inner_decoded_bytes(last_per_call);

            last_stopped_at = r.stopped_at;
            last_finished = r.finished;
            last_end_bit = r.bit_position;

            if r.finished {
                break;
            }

            match r.stopped_at {
                sp if sp == StoppingPoints::END_OF_STREAM => {
                    end_of_stream_hit = true;
                    break;
                }
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                    chunk.append_block_boundary_at(r.bit_position, already_decoded + n_bytes_read);
                }
                sp if sp == StoppingPoints::END_OF_BLOCK => {
                    if !wrapper.is_final_block() {
                        chunk.append_block_boundary_at(
                            r.bit_position,
                            already_decoded + n_bytes_read,
                        );
                        last_eob_pos = r.bit_position;
                    }
                }
                sp if sp == StoppingPoints::END_OF_BLOCK_HEADER => {
                    let not_final = !wrapper.is_final_block();
                    let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
                    if last_eob_pos >= until_bits && not_final && not_fixed {
                        last_end_bit = last_eob_pos;
                        stopping_point_reached = true;
                    }
                }
                sp if sp == StoppingPoints::NONE => {
                    if last_per_call == 0 {
                        stopping_point_reached = true;
                    }
                }
                _ => {}
            }
        }

        already_decoded += n_bytes_read;
        buffer.truncate(n_bytes_read);
        if !buffer.is_empty() {
            chunk.append_owned_buffer(buffer);
        }

        if end_of_stream_hit {
            let (crc32, isize_field) = wrapper.read_footer_at_current()?;
            let footer_end_bits = wrapper.tell_compressed();
            chunk.append_footer(crc32, isize_field, footer_end_bits);

            let remaining = wrapper.remaining_input();
            if remaining.is_empty() {
                last_end_bit = footer_end_bits;
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
            break;
        }
        if last_stopped_at == StoppingPoints::NONE && last_per_call == 0 {
            break;
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    // When the reader hits `until_bits` (refill cap) without an inexact
    // stop, `tell_compressed()` can land mid-block. Successor chunks must
    // resume at the last END_OF_BLOCK position (pre-header), not the cap.
    let final_bit = if stopping_point_reached {
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
/// `until_bits`, or at BFINAL, or when accumulated `decoded_size`
/// crosses `max_decoded_chunk_size`.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decode_chunk_marker_bootstrap_then_isal(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    _initial_window_unused: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();

    // Phase 1 — bootstrap with deflate_block::Block, the rapidgzip-
    // faithful port of vendor/.../gzip/deflate.hpp's deflate::Block<>.
    // Decodes deflate blocks one-by-one until any of:
    //   (a) cumulative clean (non-marker) bytes reach MAX_WINDOW_SIZE
    //       AND the next block header is past `until_bits` is not yet
    //       required — phase 2 (ISA-L) can take over;
    //   (b) BFINAL block is decoded (single-member tail);
    //   (c) we're at a non-fixed-Huffman block boundary at-or-past
    //       `until_bits` (chunk's end condition).
    //
    // Mirrors GzipChunk.hpp::decodeChunkWithRapidgzip's main loop
    // (vendor/.../chunkdecoding/GzipChunk.hpp:468-654), in particular
    // the `cleanDataCount >= deflate::MAX_WINDOW_SIZE` handoff at
    // L520-525.
    let bootstrap = bootstrap_with_deflate_block(input, encoded_offset_bits, until_bits)?;

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    if !bootstrap.markers.is_empty() {
        chunk.append_markered(&bootstrap.markers);
    }

    // Whenever the bootstrap reached a block boundary at-or-past
    // until_bits (rare on chunks > 32 KiB), or BFINAL fired, or no
    // clean window accumulated — we're done. The chunk is marker-only.
    let Some(clean_window) = bootstrap.clean_window else {
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    };
    if bootstrap.bfinal_hit || bootstrap.end_bit_offset >= until_bits {
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    }

    // Phase 2 — ISA-L bulk decode from the bootstrap's block boundary
    // with the clean 32 KiB tail as dict. Direct port of the working
    // v0.6 pattern (backends/isal_decompress.rs::
    // decompress_deflate_from_bit_with_boundaries), which is proven to
    // fire ISA-L END_OF_BLOCK stops correctly. Uses raw isal-sys bindings
    // rather than IsalInflateWrapper to keep the exact known-good shape:
    // points_to_stop_at set BEFORE set_dict, one isal_inflate call per
    // iteration against a pre-allocated buffer big enough that
    // OUT_OVERFLOW won't preempt the stop check.
    use isal::isal_sys::igzip_lib as isal_raw;

    let bit_offset = bootstrap.end_bit_offset;
    let byte_idx = bit_offset / 8;
    let bit_skip = bit_offset % 8;
    if byte_idx >= input.len() {
        return Err(ChunkDecodeError::InflateFailed(
            crate::decompress::parallel::inflate_wrapper::InflateError::StartBitPastEnd,
        ));
    }
    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    state.crc_flag = isal_raw::ISAL_DEFLATE;
    // points_to_stop_at BEFORE set_dict matches v0.6's working order.
    // Request both stop kinds: END_OF_BLOCK fires AFTER a block's
    // payload (position = start of next block's header = pre-header
    // boundary). END_OF_BLOCK_HEADER fires after the next header is
    // parsed (we learn the next block's btype here). We stop only at
    // non-fixed boundaries past until_bits, reporting the pre-header
    // position — matches what BlockFinder candidates accept.
    state.points_to_stop_at = isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK
        | isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER;

    let stop_hint = until_bits.min(input.len() * 8);
    let read_cap = input.len() * 8;
    let max_bits = read_cap.saturating_sub(bit_offset);
    if bit_skip > 0 {
        state.read_in = (input[byte_idx] as u64) >> bit_skip;
        state.read_in_length = (8 - bit_skip) as i32;
        state.next_in = unsafe { input.as_ptr().add(byte_idx + 1) as *mut u8 };
        let consumed = (8 - bit_skip) as usize;
        let max_bytes = if max_bits > consumed {
            (max_bits - consumed) / 8
        } else {
            0
        };
        state.avail_in = max_bytes.min(input.len() - byte_idx - 1) as u32;
    } else {
        state.next_in = unsafe { input.as_ptr().add(byte_idx) as *mut u8 };
        let max_bytes = max_bits / 8;
        state.avail_in = max_bytes.min(input.len() - byte_idx) as u32;
    }

    let ret_dict = unsafe {
        isal_raw::isal_inflate_set_dict(
            &mut state,
            clean_window.as_ptr() as *mut u8,
            clean_window.len() as u32,
        )
    };
    if ret_dict != 0 {
        return Err(ChunkDecodeError::InflateFailed(
            crate::decompress::parallel::inflate_wrapper::InflateError::SetDictFailed,
        ));
    }

    // Output buffer grows dynamically up to max_decoded_chunk_size.
    // Earlier versions pre-allocated + set_len(80 MiB) per worker,
    // which committed 16 × 80 = 1.28 GiB resident under T=16 — a
    // per-worker fixed cost regardless of how much each chunk
    // actually decoded. Now we grow the Vec on demand via ISA-L
    // 128 KiB chunks (ALLOCATION_CHUNK_SIZE), so workers' resident
    // memory matches their actual output size.
    let cap = configuration.max_decoded_chunk_size;
    let mut output: Vec<u8> = Vec::with_capacity(ALLOCATION_CHUNK_SIZE.min(cap));
    let mut out_pos: usize = 0;
    let debug_isal = std::env::var("GZIPPY_DEBUG_ISAL").is_ok();
    let mut iter = 0usize;
    let mut stops_eob = 0usize;
    // Track the position of the last END_OF_BLOCK (= start of next
    // block's header). When END_OF_BLOCK_HEADER fires with a non-fixed
    // btype past until_bits, stop at last_eob_pos (pre-header
    // position) so successor can resume cleanly. See decode_chunk_isal_inexact
    // for the same logic in the fast path.
    let mut last_eob_pos = bit_offset;
    let mut chunk_end_override: Option<usize> = None;
    if debug_isal {
        eprintln!(
            "  [isal] pre-loop: pts={} stopped={} tmpstop={} bstate={} avail_in={} read_in_len={} cap={}",
            state.points_to_stop_at,
            state.stopped_at,
            state.tmp_out_stopped_at,
            state.block_state,
            state.avail_in,
            state.read_in_length,
            cap,
        );
    }

    loop {
        if out_pos >= cap {
            chunk.stopped_preemptively = true;
            break;
        }
        // Grow output buffer by ALLOCATION_CHUNK_SIZE if needed. We
        // never reserve more than `cap` total. Each iteration tells
        // ISA-L it has up to ALLOCATION_CHUNK_SIZE bytes of room.
        let want = ALLOCATION_CHUNK_SIZE.min(cap - out_pos);
        if output.len() < out_pos + want {
            output.resize(out_pos + want, 0);
        }
        let remaining = want;
        state.avail_out = remaining as u32;
        state.next_out = unsafe { output.as_mut_ptr().add(out_pos) };

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        let written = remaining - state.avail_out as usize;
        // Append bytes BEFORE handling any stopping point. append_block_boundary
        // closes the current subchunk and opens a new one; the bytes ISA-L
        // just produced belong to the OLD subchunk (the one that just ended
        // at bit_pos). If we deferred the append until after the loop, every
        // intermediate subchunk would record decoded_size=0 and the final
        // subchunk would absorb the entire chunk — which corrupts the
        // BlockMap with zero-width decoded ranges.
        if written > 0 {
            chunk.append_clean(&output[out_pos..out_pos + written]);
        }
        out_pos += written;
        iter += 1;

        match ret {
            0 | 1 => {} // OK or END_INPUT — continue
            -1 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidBlock,
                ));
            }
            -2 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidSymbol,
                ));
            }
            -3 => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::InvalidLookback,
                ));
            }
            other => {
                return Err(ChunkDecodeError::InflateFailed(
                    crate::decompress::parallel::inflate_wrapper::InflateError::Internal(other),
                ));
            }
        }

        let bit_pos =
            input.len() * 8 - state.avail_in as usize * 8 - state.read_in_length.max(0) as usize;

        if state.stopped_at == isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK {
            stops_eob += 1;
            chunk.append_block_boundary(bit_pos);
            last_eob_pos = bit_pos;
            state.stopped_at = isal_raw::ISAL_STOPPING_POINT_NONE;
        } else if state.stopped_at == isal_raw::ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER {
            // We just parsed the header of the block that starts at
            // last_eob_pos. btype tells us its type.
            let not_final = state.bfinal == 0;
            let not_fixed = state.btype != 1;
            if last_eob_pos >= stop_hint && not_final && not_fixed {
                chunk_end_override = Some(last_eob_pos);
                break;
            }
            state.stopped_at = isal_raw::ISAL_STOPPING_POINT_NONE;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 && state.stopped_at == 0 {
            break;
        }
        if out_pos >= cap {
            chunk.stopped_preemptively = true;
            break;
        }
    }

    let isal_pos =
        input.len() * 8 - state.avail_in as usize * 8 - state.read_in_length.max(0) as usize;
    // When chunk_end_override is set, the next-block-header parse has
    // already consumed bits past the boundary we want to report. The
    // actual data bytes emitted are for blocks up to and including the
    // block ending at last_eob_pos, so chunk.data is correct as-is.
    let final_bit_pos = chunk_end_override.unwrap_or(isal_pos);

    // Bytes have already been appended per loop iteration above so each
    // subchunk records its own slice's decoded_size. No trailing dump.

    if debug_isal {
        eprintln!(
            "  [isal] done iter={} eob={} final_bit={} isal_pos={} decoded={}",
            iter,
            stops_eob,
            final_bit_pos,
            isal_pos,
            chunk.decoded_size()
        );
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    chunk.finalize(final_bit_pos);
    Ok(chunk)
}

/// Result of one bootstrap pass over deflate blocks via
/// [`deflate_block::Block`]. Mirrors the early-exit contract of
/// rapidgzip's `decodeChunkWithRapidgzip` main loop
/// (vendor/.../chunkdecoding/GzipChunk.hpp:468-654).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
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
    /// have no clean dict — either BFINAL fired first, `until_bits` was
    /// reached, or no block produced enough clean data.
    clean_window: Option<Vec<u8>>,
    /// True when the bootstrap exited because it decoded a BFINAL=1
    /// block. Single-stream tail; no further deflate data follows.
    bfinal_hit: bool,
}

/// Phase 1 bootstrap: drive [`deflate_block::Block`] block-by-block from
/// `start_bit_offset` until any of (a) cumulative clean bytes reach
/// MAX_WINDOW_SIZE at a block boundary, (b) BFINAL fires, or (c) we're
/// at a non-fixed block boundary at-or-past `until_bits`.
///
/// Mirror of the `while ( true )` loop in
/// `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654), restricted to the
/// single-member case (no multi-stream loop) and with the handoff
/// triggered exclusively by `cleanDataCount` (GzipChunk.hpp:520-525).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn bootstrap_with_deflate_block(
    data: &[u8],
    start_bit_offset: usize,
    until_bits: usize,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::deflate_block::{Block, CompressionType, MAX_WINDOW_SIZE};
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
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
    // ~128 KiB worst case for a single dense dynamic block. Pre-reserve
    // generously to avoid early growth.
    let mut output: Vec<u16> = Vec::with_capacity(128 * 1024);
    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let block = &mut *block;
        let output = &mut output;

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
        let mut end_bit_offset = absolute_bit_pos(byte_offset, &bits);

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
            match block.read_header(&mut bits, false) {
                Ok(()) => {}
                Err(e) => {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deflate header at bit {next_block_offset}: {e:?}"),
                    )));
                }
            }

            // Preemptive stop condition (rapidgzip GzipChunk.hpp:550-555):
            // if this block's header sits at-or-past until_bits AND it is
            // non-fixed AND not BFINAL, stop here so the chunk ends on a
            // boundary the successor's BlockFinder can re-find.
            let is_fixed = block.compression_type() == CompressionType::FixedHuffman;
            if next_block_offset >= until_bits && !block.is_last_block() && !is_fixed {
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
            while !block.eob() {
                let _ = block
                    .read(&mut bits, &mut *output, usize::MAX)
                    .map_err(|e| {
                        ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("deflate body at bit {next_block_offset}: {e:?}"),
                        ))
                    })?;
            }

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
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
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

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decode_chunk_marker_bootstrap_then_isal(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _until_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decode_chunk_isal_inexact(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _until_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(test)]
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
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
    fn decode_chunk_isal_inexact_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let until_bits = deflate.len() * 8;
        let chunk = decode_chunk_isal_inexact(&deflate, 0, until_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
    }

    #[test]
    fn decode_chunk_isal_inexact_stops_before_eof_when_until_bits_set() {
        let payload = b"x".repeat(500_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 256 * 1024,
            max_decoded_chunk_size: 20 * 256 * 1024,
            crc32_enabled: false,
        };
        let until_bits = deflate.len() * 8 / 2;
        let chunk = decode_chunk_isal_inexact(&deflate, 0, until_bits, &[], cfg).unwrap();
        assert!(chunk.decoded_size() < payload.len());
        let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        assert!(chunk_end >= until_bits);
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
            let (_hdr, hdr_len) =
                crate::decompress::parallel::gzip_format::read_header(&gz).expect("gzip hdr");
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
        let chunk0 =
            decode_chunk_isal_inexact(&deflate, 0, spacing_bits, &zero, cfg).expect("chunk0");
        let resume_at = chunk0.encoded_offset_bits + chunk0.encoded_size_bits;
        assert!(
            resume_at > 0 && resume_at % 8 != 0,
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
        let chunk1 =
            decode_chunk_isal_inexact(deflate, resume_at, resume_at + spacing_bits, &tail, cfg)
                .expect("chunk1 at chunk0 end");
        assert!(!chunk1.is_empty());
    }
}
