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
    // with the clean 32 KiB tail as dict. Stops at:
    //   - any END_OF_BLOCK whose successor is at-or-past until_bits
    //     (chunks naturally bounded by the requested partition end),
    //   - any END_OF_BLOCK_HEADER on a non-FIXED block past until_bits
    //     (records a high-quality subchunk boundary for downstream),
    //   - BFINAL,
    //   - the `max_decoded_chunk_size` per-iteration guard. This last
    //     fires regardless of stop type so a single huge dynamic-
    //     Huffman block (common on Silesia gzip -9, where one block
    //     can decode hundreds of MB) can't run unbounded.
    let mut wrapper = IsalInflateWrapper::new(input, bootstrap.end_bit_offset)?;
    wrapper.set_window(&clean_window)?;
    wrapper.set_stopping_points(StoppingPoints::END_OF_BLOCK | StoppingPoints::END_OF_BLOCK_HEADER);

    let mut buf = vec![0u8; ALLOCATION_CHUNK_SIZE];
    let debug_isal = std::env::var("GZIPPY_DEBUG_ISAL").is_ok();
    let mut iter = 0usize;
    let mut stops_eob = 0usize;
    let mut stops_eobh = 0usize;
    let mut stops_other = 0usize;
    loop {
        let r = wrapper.read_stream(&mut buf)?;
        if r.bytes_written > 0 {
            chunk.append_clean(&buf[..r.bytes_written]);
        }
        iter += 1;
        if debug_isal && iter <= 100 {
            eprintln!(
                "  [isal] iter={} wrote={} stopped_at={:?} bit={} until={} finished={}",
                iter, r.bytes_written, r.stopped_at, r.bit_position, until_bits, r.finished
            );
        }

        // Per-iteration max guard. Single huge blocks can produce
        // hundreds of MB across many read_stream calls before any
        // stopping point fires; without this check the chunk would
        // run unbounded.
        if chunk.decoded_size() >= configuration.max_decoded_chunk_size {
            chunk.stopped_preemptively = true;
            break;
        }
        if r.finished {
            break;
        }
        if r.stopped_at == StoppingPoints::END_OF_BLOCK {
            stops_eob += 1;
            let next_block_off = wrapper.tell_compressed();
            // Position is at the START of the next block (or BFINAL
            // padding). Stop if we've crossed the partition's until.
            if next_block_off >= until_bits {
                break;
            }
            wrapper.clear_stop();
            continue;
        }
        if r.stopped_at == StoppingPoints::END_OF_BLOCK_HEADER {
            stops_eobh += 1;
            let next_block_off = wrapper.tell_compressed();
            let not_final = !wrapper.is_final_block();
            let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
            chunk.append_block_boundary(next_block_off);
            if next_block_off >= until_bits && not_final && not_fixed {
                break;
            }
            wrapper.clear_stop();
            continue;
        }
        if r.stopped_at != StoppingPoints::NONE {
            stops_other += 1;
        }
        if r.bytes_written == 0 {
            // Decoder produced nothing and didn't stop at a requested
            // point → input exhausted or finished mid-block.
            break;
        }
    }
    if debug_isal {
        eprintln!(
            "  [isal] done iter={} eob={} eobh={} other_stops={} bit={} decoded={}",
            iter,
            stops_eob,
            stops_eobh,
            stops_other,
            wrapper.tell_compressed(),
            chunk.decoded_size()
        );
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
