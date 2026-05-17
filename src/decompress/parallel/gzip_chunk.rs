//! Port of the chunk-level decoder entry points in
//! `rapidgzip::GzipChunk` (chunkdecoding/GzipChunk.hpp).
//!
//! `finish_decode_chunk_with_inexact_offset` is the workhorse the
//! parallel fetcher (chunk_fetcher.rs) calls per worker. It assumes
//! `encoded_offset_bits` points at a REAL deflate block boundary (the
//! caller — usually [`crate::decompress::parallel::block_finder::BlockFinder`]
//! plus [`crate::decompress::parallel::fast_marker_inflate::validate_boundary`]
//! — is responsible for that). From there it follows rapidgzip's
//! pattern verbatim:
//!
//!   1. **Bootstrap with the marker decoder** — pure-Rust
//!      [`fast_marker_inflate::decode_chunk_bootstrap`] decodes until a
//!      32 KiB clean tail accumulates at a deflate block boundary (or
//!      BFINAL fires, or `until_bits` is reached). The output is
//!      `Vec<u16>` with markers ≥ MARKER_BASE encoding cross-chunk
//!      back-references the consumer will resolve via `apply_window`.
//!
//!   2. **Hand off to patched ISA-L** — when the bootstrap returns a
//!      `clean_window`, construct an [`IsalInflateWrapper`] at the
//!      bootstrap's end-bit-offset, seed its 32 KiB dict with
//!      `clean_window`, and request `END_OF_BLOCK_HEADER` stops so we
//!      know when to terminate the chunk at a real boundary at-or-past
//!      `until_bits`. ISA-L delivers ~163 MB/s/thread vs the marker
//!      decoder's ~50 MB/s.
//!
//!   3. **Subchunk boundaries** are recorded by both phases via
//!      `ChunkData::append_block_boundary`; the chunk emits a new
//!      subchunk whenever accumulated `decoded_size` crosses
//!      `split_chunk_size`.
//!
//!   4. **Finalize** at the ISA-L wrapper's final `tell_compressed()`.
//!      Stream-trailer / multi-stream handling is the caller's
//!      responsibility (single-member only).
//!
//! `decode_chunk_with_inflate_wrapper` is a sibling for chunk 0 / single-
//! chunk decode where the predecessor window is known to be empty and
//! the exact end offset is the stream length — used by the unit tests
//! to oracle-check the bootstrap path against pure ISA-L.

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

/// Fast-path chunk decoder used by workers when the predecessor's 32 KiB
/// window is available in the shared WindowMap. Skips the marker-decoder
/// bootstrap entirely: seeds the patched-ISA-L wrapper with the known
/// dict, sets the rapidgzip stopping-points, decodes to the first
/// non-fixed END_OF_BLOCK_HEADER at-or-past `until_bits`. The returned
/// chunk has only clean bytes (no markers), and `apply_window` is a
/// no-op on it.
///
/// Mirror of `rapidgzip::GzipChunk::finishDecodeChunkWithInexactOffset`
/// (vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp:280-410) when
/// `initialWindow` is known.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decode_chunk_with_window(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    initial_window: &[u8; 32768],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    let mut wrapper = IsalInflateWrapper::new(input, encoded_offset_bits)?;
    wrapper.set_window(&initial_window[..])?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER,
    );

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let mut buf = vec![0u8; ALLOCATION_CHUNK_SIZE];
    let mut stopping_point_reached = false;
    let mut last_end_bit = encoded_offset_bits;
    // Track the position of the last END_OF_BLOCK (= start of the next
    // block's header). When the next block's header is parsed and we
    // get END_OF_BLOCK_HEADER, we KNOW btype. If non-fixed and past
    // until_bits, stop the chunk at `last_eob_pos` (the PRE-header
    // position) rather than `bit_position` (post-header). This makes
    // chunk.end_bit a position the next chunk's worker can resume
    // from (block_state=NEW_HDR will re-parse the same header).
    //
    // Critically: stopping at non-fixed-only boundaries matches what
    // BlockFinder candidates accept, so the speculative-start match
    // rate jumps from ~3% to near 100%, eliminating consumer-thread
    // serialization on authoritative re-dispatches.
    let mut last_eob_pos = encoded_offset_bits;

    while !stopping_point_reached {
        let r = wrapper.read_stream(&mut buf)?;
        if r.bytes_written > 0 {
            chunk.append_clean(&buf[..r.bytes_written]);
        }
        last_end_bit = r.bit_position;
        if r.finished {
            break;
        }
        match r.stopped_at {
            sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                chunk.append_block_boundary(r.bit_position);
            }
            sp if sp == StoppingPoints::END_OF_BLOCK => {
                if !wrapper.is_final_block() {
                    chunk.append_block_boundary(r.bit_position);
                    last_eob_pos = r.bit_position;
                }
            }
            sp if sp == StoppingPoints::END_OF_BLOCK_HEADER => {
                let not_final = !wrapper.is_final_block();
                let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
                if last_eob_pos >= until_bits && not_final && not_fixed {
                    // Stop at the pre-header position so next chunk
                    // can resume cleanly.
                    last_end_bit = last_eob_pos;
                    stopping_point_reached = true;
                }
            }
            sp if sp == StoppingPoints::NONE => {
                if r.bytes_written == 0 {
                    stopping_point_reached = true;
                }
            }
            _ => {}
        }
        if chunk.decoded_size() >= configuration.max_decoded_chunk_size {
            chunk.stopped_preemptively = true;
            stopping_point_reached = true;
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    chunk.finalize(last_end_bit);
    Ok(chunk)
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decode_chunk_with_window(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _until_bits: usize,
    _initial_window: &[u8; 32768],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

/// Exact-stop decode for chunk 0 (empty initial window) or test fixtures
/// where the caller has a verified exact end offset. Mirror of
/// `decodeChunkWithInflateWrapper` (GzipChunk.hpp:190-268).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decode_chunk_with_inflate_wrapper(
    input: &[u8],
    encoded_offset_bits: usize,
    exact_until_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut wrapper = IsalInflateWrapper::new(input, encoded_offset_bits)?;
    wrapper.set_window(initial_window)?;
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let t_decode = std::time::Instant::now();

    let mut buf = vec![0u8; ALLOCATION_CHUNK_SIZE];
    loop {
        let r = wrapper.read_stream(&mut buf)?;
        if r.bytes_written > 0 {
            chunk.append_clean(&buf[..r.bytes_written]);
        }
        if r.finished {
            break;
        }
        if r.bytes_written == 0 {
            break;
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    let actual_end = wrapper.tell_compressed();
    let within_padding = actual_end >= exact_until_bits && actual_end <= exact_until_bits + 8;
    if !within_padding && actual_end != exact_until_bits {
        return Err(ChunkDecodeError::ExactStopMissed {
            requested: exact_until_bits,
            actual: actual_end,
        });
    }
    chunk.finalize(actual_end);
    Ok(chunk)
}

/// Decode a chunk seeded at a REAL deflate block boundary with an
/// initially-unknown (empty) predecessor window. Uses the marker
/// decoder for the bootstrap phase (so cross-chunk back-references are
/// tagged as markers), then hands off to patched ISA-L for the bulk
/// decode once a 32 KiB clean tail is in hand.
///
/// Returns a `ChunkData` whose `data_with_markers` covers the bootstrap
/// prefix (still containing markers; resolved by `apply_window` in the
/// consumer) and whose `data` covers the ISA-L bulk (already clean).
/// The chunk stops at the next deflate block boundary at-or-past
/// `until_bits`, or at BFINAL, or when accumulated `decoded_size`
/// crosses `max_decoded_chunk_size`.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn finish_decode_chunk_with_inexact_offset(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    _initial_window_unused: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();

    // Phase 1 — marker-decoder bootstrap. Runs at ~50 MB/s/thread but
    // only for the chunk's leading ~32 KiB + one block: enough to
    // accumulate a clean 32 KiB tail for ISA-L's dict.
    let bootstrap = crate::decompress::parallel::fast_marker_inflate::decode_chunk_bootstrap(
        input,
        encoded_offset_bits,
        Some(until_bits),
    )?;

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    if !bootstrap.markers.is_empty() {
        chunk.append_markered(&bootstrap.markers);
    }

    // Whenever the marker bootstrap reached a block boundary at-or-past
    // until_bits (rare on chunks > 32 KiB), or BFINAL fired, or no clean
    // window accumulated — we're done. The chunk is marker-only.
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

    if bit_skip > 0 {
        state.read_in = (input[byte_idx] as u64) >> bit_skip;
        state.read_in_length = (8 - bit_skip) as i32;
        state.next_in = unsafe { input.as_ptr().add(byte_idx + 1) as *mut u8 };
        state.avail_in = (input.len() - byte_idx - 1) as u32;
    } else {
        state.next_in = unsafe { input.as_ptr().add(byte_idx) as *mut u8 };
        state.avail_in = (input.len() - byte_idx) as u32;
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

    // Single pre-allocated buffer, capped at max_decoded_chunk_size so
    // a runaway chunk can't OOM. v0.6 uses up to 3 GiB; we use the
    // configured cap (80 MiB at 4 MiB split * 20 multiplier).
    let cap = configuration.max_decoded_chunk_size;
    let mut output: Vec<u8> = Vec::with_capacity(cap);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(cap)
    };
    let mut out_pos: usize = 0;
    let debug_isal = std::env::var("GZIPPY_DEBUG_ISAL").is_ok();
    let mut iter = 0usize;
    let mut stops_eob = 0usize;
    // Track the position of the last END_OF_BLOCK (= start of next
    // block's header). When END_OF_BLOCK_HEADER fires with a non-fixed
    // btype past until_bits, stop at last_eob_pos (pre-header
    // position) so successor can resume cleanly. See decode_chunk_with_window
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
        let remaining = cap - out_pos;
        if remaining == 0 {
            chunk.stopped_preemptively = true;
            break;
        }
        state.avail_out = remaining as u32;
        state.next_out = unsafe { output.as_mut_ptr().add(out_pos) };

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        let written = remaining - state.avail_out as usize;
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
            if last_eob_pos >= until_bits && not_final && not_fixed {
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

    if out_pos > 0 {
        chunk.append_clean(&output[..out_pos]);
    }

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

// Stubs for non-x86_64 / no-feature builds.
#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decode_chunk_with_inflate_wrapper(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _exact_until_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn finish_decode_chunk_with_inexact_offset(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _until_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

// ── Unit tests ───────────────────────────────────────────────────────────

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
    fn finish_decode_inexact_from_bit_0_byte_identical_with_oracle() {
        let payload = b"abcdefghij".repeat(200_000); // ~2 MB
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let until_bits = deflate.len() * 8;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();
        let output = flatten(&chunk);
        assert_eq!(output.len(), payload.len());
        assert_eq!(output, payload);
    }

    #[test]
    fn finish_decode_inexact_emits_subchunks_when_split_threshold_reached() {
        let payload = vec![b'x'; 5_000_000];
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let until_bits = deflate.len() * 8;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();
        assert!(
            chunk.subchunks.len() >= 1,
            "expected ≥1 subchunk, got {}",
            chunk.subchunks.len()
        );
        let total: usize = chunk.subchunks.iter().map(|s| s.decoded_size).sum();
        assert_eq!(total, chunk.decoded_size());
    }

    #[test]
    fn decode_chunk_with_inflate_wrapper_exact_stop_succeeds() {
        let payload = b"hello world".repeat(10_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration::default();
        let exact_until = deflate.len() * 8;
        let chunk = decode_chunk_with_inflate_wrapper(&deflate, 0, exact_until, &[], cfg).unwrap();
        assert_eq!(chunk.decoded_size(), payload.len());
        assert_eq!(chunk.data, payload);
    }

    #[test]
    fn finish_decode_with_until_bits_partway_stops_at_block_boundary() {
        let payload = b"abcdefghij".repeat(500_000); // ~5 MB
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 256 * 1024,
            max_decoded_chunk_size: 20 * 256 * 1024,
            crc32_enabled: true,
        };
        // Stop partway through the stream.
        let until_bits = deflate.len() * 8 / 2;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();
        // Chunk's encoded range should at least reach until_bits (within
        // one block's worth of overshoot is expected and valid).
        let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        assert!(
            chunk_end >= until_bits,
            "chunk should reach at least until_bits ({}), got {}",
            until_bits,
            chunk_end
        );
        // And it should NOT decode the full stream (otherwise the
        // chunk-stop condition is broken).
        assert!(
            chunk.decoded_size() < payload.len(),
            "chunk should stop at a block boundary partway, not at EOF"
        );
    }
}
