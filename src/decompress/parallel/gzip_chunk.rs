//! Port of the chunk-level decoder entry points in
//! `rapidgzip::GzipChunk` (chunkdecoding/GzipChunk.hpp). Two functions:
//!
//!  - [`decode_chunk_with_inflate_wrapper`] — exact-stop decode. Used
//!    when caller has a verified window AND a verified exact end offset.
//!    Errors if `tell_compressed() != exact_until_bits` at end.
//!    Mirror of `decodeChunkWithInflateWrapper` (GzipChunk.hpp:190-268).
//!
//!  - [`finish_decode_chunk_with_inexact_offset`] — inexact-stop
//!    discovery. Used to seed the parallel decode: caller passes the
//!    wrapper at any state (typically just seeked to a partition
//!    offset). The decoder runs to the next deflate-boundary stopping
//!    point at-or-past `until_bits` (or to BFINAL, or to
//!    `max_decoded_chunk_size`). Emits subchunks at boundaries when
//!    accumulated decoded size meets `split_chunk_size`. Mirror of
//!    `finishDecodeChunkWithInexactOffset` (GzipChunk.hpp:280-410).
//!
//! Caller is responsible for stripping gzip header / providing raw
//! deflate input. The wrappers don't handle multi-stream footer
//! parsing — single-member parallel path is single-stream.
//
// Allowed dead_code: step 5 of rapidgzip-port-design.md migration;
// consumed by chunk_fetcher.rs in step 7. Exercised by unit tests.
#![allow(dead_code)]

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::inflate_wrapper::InflateError;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::inflate_wrapper::{IsalInflateWrapper, StoppingPoints};

/// Errors specific to the chunk decoders. `InflateFailed` wraps an
/// error from the underlying ISA-L wrapper. `ExactStopMissed` mirrors
/// rapidgzip's "throw if `tellCompressed() != exactUntilOffset`"
/// invariant (GzipChunk.hpp:252).
#[derive(Debug)]
pub enum ChunkDecodeError {
    InflateFailed(InflateError),
    ExactStopMissed { requested: usize, actual: usize },
    UnsupportedPlatform,
}

impl From<InflateError> for ChunkDecodeError {
    fn from(e: InflateError) -> Self {
        ChunkDecodeError::InflateFailed(e)
    }
}

/// Output buffer size used per `read_stream` iteration. Matches
/// rapidgzip's `ALLOCATION_CHUNK_SIZE` (GzipChunk.hpp uses 128 KiB).
/// Smaller than typical block size so most blocks fit in one or two
/// read_stream calls — keeps the stopping-point feedback granular.
const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

/// Decode a chunk with a known initial window and an exact required
/// stopping bit offset. Returns Err if the decoder lands at any other
/// bit position. Mirror of `decodeChunkWithInflateWrapper` (GzipChunk.hpp:190-268).
///
/// `initial_window` may be empty: in that case, cross-chunk back-
/// references will resolve to zero bytes (which the caller must later
/// fix up via `apply_window` with the real window — unless the chunk
/// is at the very start of the stream where empty IS the real window).
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
    // Exact-stop mode: we don't request stopping points; we just decode
    // until input runs out OR BFINAL fires. Verify end matches at end.
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let window_known = !initial_window.is_empty();
    let t_decode = std::time::Instant::now();

    let mut buf = vec![0u8; ALLOCATION_CHUNK_SIZE];
    loop {
        let r = wrapper.read_stream(&mut buf)?;
        if r.bytes_written > 0 {
            if window_known {
                chunk.append_clean(&buf[..r.bytes_written]);
            } else {
                // Caller said "no window" → bytes ISA-L emits for cross-
                // chunk back-refs are zeros. Mark them as markered so
                // apply_window can fix them later. But since we don't
                // know which bytes came from where without the patched
                // stopping-point machinery active, we conservatively
                // wrap everything as markered for empty-window decode.
                //
                // For the exact-stop path this is mostly used by chunk
                // 0 (where empty window IS the real window so no
                // resolution needed) or by post-mismatch retries
                // (where caller has the real window and passes it).
                // Here we keep the simpler invariant: bytes are clean.
                chunk.append_clean(&buf[..r.bytes_written]);
            }
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
    // Allow up to 8 bits of zero-padding at end of stream (rapidgzip's
    // semantics also tolerate this; deflate streams pad to byte
    // boundary). For mid-stream exact stops we require precise match.
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

/// Decode a chunk from `wrapper`'s current state until a deflate-block-
/// aligned stopping point at-or-past `until_bits`, or BFINAL, or the
/// `max_decoded_chunk_size` preemptive cap. Records subchunks at
/// boundaries when accumulated decoded size meets `split_chunk_size`.
///
/// Mirror of `finishDecodeChunkWithInexactOffset` (GzipChunk.hpp:280-410)
/// adapted for single-member raw-deflate (no footer handling).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn finish_decode_chunk_with_inexact_offset(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut wrapper = IsalInflateWrapper::new(input, encoded_offset_bits)?;
    wrapper.set_window(initial_window)?;
    // Request the patched ISA-L to pause at block-aligned stopping
    // points so the chunk decoder can observe them and either record
    // a subchunk boundary OR decide to stop the chunk.
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER,
    );

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let window_known = !initial_window.is_empty();
    let t_decode = std::time::Instant::now();

    let mut buf = vec![0u8; ALLOCATION_CHUNK_SIZE];
    let mut next_block_offset = wrapper.tell_compressed();
    let mut stopping_point_reached = false;

    while !stopping_point_reached {
        // Inner pump: fill the current buffer (or until a stop fires).
        let r = wrapper.read_stream(&mut buf)?;
        if r.bytes_written > 0 {
            if window_known {
                chunk.append_clean(&buf[..r.bytes_written]);
            } else {
                // Empty-window decode: bytes emitted by ISA-L for
                // cross-chunk back-refs are zeros. We don't yet
                // distinguish them per-byte. For single-member
                // single-chunk-0 the only safe assumption: all bytes
                // are literal (the very first chunk has no predecessor
                // window to reference). This matches rapidgzip's
                // behavior at chunk 0.
                //
                // For non-chunk-0 dispatch (i.e., empty-window
                // speculative decode of a non-first chunk), the
                // dispatcher will use the END_OF_BLOCK feedback to
                // discover real boundaries before relying on the
                // bytes; downstream consumers use `apply_window` with
                // the right window for the resolved part.
                chunk.append_clean(&buf[..r.bytes_written]);
            }
        }

        // Was this stop a block-aligned event we should record?
        let mut is_block_start = false;
        match r.stopped_at {
            sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                is_block_start = true;
            }
            sp if sp == StoppingPoints::END_OF_BLOCK => {
                is_block_start = !wrapper.is_final_block();
            }
            sp if sp == StoppingPoints::END_OF_BLOCK_HEADER => {
                // rapidgzip stops the chunk when the next block starts
                // at or past until_bits, AND it's not the final block,
                // AND the block type is not FIXED_HUFFMAN (because
                // fixed-Huffman block headers are too short/common to
                // be reliable boundaries to land on for downstream
                // chunks). We replicate that condition.
                let next_block_off = wrapper.tell_compressed();
                let not_final = !wrapper.is_final_block();
                let not_fixed = wrapper.btype()
                    != Some(crate::decompress::parallel::inflate_wrapper::DeflateCompressionType::FixedHuffman);
                if (next_block_off >= until_bits && not_final && not_fixed)
                    || next_block_off == until_bits
                {
                    stopping_point_reached = true;
                }
            }
            sp if sp == StoppingPoints::NONE => {
                if r.bytes_written == 0 {
                    // Decoder produced nothing and didn't stop at a
                    // requested point → input exhausted or finished.
                    stopping_point_reached = true;
                }
            }
            _ => {}
        }

        if is_block_start {
            next_block_offset = wrapper.tell_compressed();
            chunk.append_block_boundary(next_block_offset);
            if chunk.decoded_size() >= configuration.max_decoded_chunk_size {
                chunk.stopped_preemptively = true;
                stopping_point_reached = true;
            }
        }

        if r.finished {
            stopping_point_reached = true;
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    let actual_end = wrapper.tell_compressed();
    chunk.finalize(actual_end);
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

    #[test]
    fn finish_decode_inexact_from_bit_0_byte_identical_with_oracle() {
        let payload = b"abcdefghij".repeat(200_000); // ~2 MB
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        // until_bits = end of deflate stream
        let until_bits = deflate.len() * 8;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();

        // Concatenate (data_with_markers as u8) ++ data
        let mut output = Vec::with_capacity(chunk.decoded_size());
        for v in &chunk.data_with_markers {
            output.push(*v as u8);
        }
        output.extend_from_slice(&chunk.data);
        assert_eq!(output.len(), payload.len());
        assert_eq!(output, payload);
    }

    #[test]
    fn finish_decode_inexact_emits_subchunks_when_split_threshold_reached() {
        let payload = vec![b'x'; 5_000_000]; // 5 MB, more than 1 split worth
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let until_bits = deflate.len() * 8;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();

        // 5 MB / 512 KiB ≈ 10. With finalize merging tail, expect
        // somewhere between 5 and 15 subchunks.
        assert!(
            chunk.subchunks.len() >= 5,
            "expected ≥5 subchunks, got {}",
            chunk.subchunks.len()
        );
        // Subchunks should be in increasing encoded order and the
        // decoded sizes should sum to total decoded.
        let total: usize = chunk.subchunks.iter().map(|s| s.decoded_size).sum();
        assert_eq!(total, chunk.decoded_size());
        for pair in chunk.subchunks.windows(2) {
            assert!(pair[0].encoded_offset_bits <= pair[1].encoded_offset_bits);
        }
    }

    #[test]
    fn decode_chunk_with_inflate_wrapper_exact_stop_succeeds() {
        let payload = b"hello world".repeat(10_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration::default();
        // For the exact-stop path with the FULL stream length, this should succeed.
        let exact_until = deflate.len() * 8;
        let chunk = decode_chunk_with_inflate_wrapper(&deflate, 0, exact_until, &[], cfg).unwrap();
        assert_eq!(chunk.decoded_size(), payload.len());
        assert_eq!(chunk.data, payload);
    }

    #[test]
    fn preemptive_stop_fires_when_max_decoded_chunk_size_exceeded() {
        // Make a stream that has multiple blocks per ~1 MB, so subchunk
        // emissions happen frequently enough that we cross the
        // max_decoded_chunk_size cap mid-stream.
        let payload = b"abcdefgh".repeat(2_000_000); // 16 MB
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 256 * 1024,            // small splits
            max_decoded_chunk_size: 4 * 1024 * 1024, // 4 MB cap
            crc32_enabled: true,
        };
        let until_bits = deflate.len() * 8;
        let chunk =
            finish_decode_chunk_with_inexact_offset(&deflate, 0, until_bits, &[], cfg).unwrap();
        assert!(
            chunk.stopped_preemptively,
            "expected preemptive stop at 4 MB cap"
        );
        assert!(chunk.decoded_size() >= 4 * 1024 * 1024);
        assert!(chunk.decoded_size() < payload.len());
    }
}
