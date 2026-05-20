//! Per-chunk deflate decode for parallel single-member.
//!
//! [`decode_chunk_isal_inexact`] is the only worker decode entry. `until_bits`
//! is an inexact stop hint (first non-fixed block boundary at-or-past it);
//! the ISA-L wrapper does **not** byte-cap input. Empty `initial_window`
//! emits cross-chunk markers; a known 32 KiB dict yields clean bytes.

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

/// **Inexact** ISA-L decode (vendor `finishDecodeChunkWithInexactOffset`
/// when `initialWindow` is set, OR `decodeChunkWithRapidgzip` when it
/// is not). `until_bits` is only a stop hint — the wrapper has **no**
/// byte-level cap. Empty `initial_window` emits markers for unknown
/// back-refs (speculative prefetch); a known dict yields clean bytes.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decode_chunk_isal_inexact(
    input: &[u8],
    encoded_offset_bits: usize,
    until_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    let mut wrapper = IsalInflateWrapper::new(input, encoded_offset_bits)?;
    wrapper.set_window(initial_window)?;
    // END_OF_STREAM fires when the deflate stream's BFINAL block has
    // been fully decoded and the bit reader has byte-aligned to the
    // footer. We listen for it so multi-stream gzip is handled inline:
    // after the footer is consumed (via read_footer_at_current) and the
    // next gzip header parsed (via gzip_format::read_header), the
    // wrapper is reset for the next stream and decoding continues.
    // Mirror of rapidgzip's multi-stream loop in
    // `decodeChunkWithRapidgzip` (chunkdecoding/GzipChunk.hpp:468-654)
    // restricted to chunks driven by the fast path.
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER
            | StoppingPoints::END_OF_STREAM,
    );

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
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

    // OUTER/INNER buffer pattern — literal port of rapidgzip's
    // `finishDecodeChunkWithInexactOffset`
    // (vendor/.../chunkdecoding/GzipChunk.hpp:309-390):
    //
    //   while ( !stoppingPointReached ) {
    //       DecodedVector buffer( ALLOCATION_CHUNK_SIZE );   // L310
    //       size_t nBytesRead = 0;
    //       while ( nBytesRead < buffer.size() && !footer && !stoppingPointReached ) {
    //           tie( perCall, footer ) = readStream(...);    // L318
    //           nBytesRead += perCall;
    //           subchunks.back().decodedSize += perCall;     // L321
    //           // ... stopping-point + isBlockStart logic ...
    //       }
    //       alreadyDecoded += nBytesRead;
    //       buffer.resize( nBytesRead );
    //       result.append( std::move( buffer ) );            // L379
    //       if ( footer ) { result.appendFooter(...); }      // L380-384
    //   }
    //
    // The buffer is fresh per OUTER iter (so we can move it into
    // ChunkData zero-copy on the first outer iter), inner ISA-L calls
    // fill it, subchunk decoded sizes update PER INNER call, and the
    // whole buffer is committed once per outer iter.
    let mut already_decoded: usize = 0;

    while !stopping_point_reached {
        // Vendor parity: `deflate::DecodedVector buffer(ALLOCATION_CHUNK_SIZE)`
        // at vendor/.../chunkdecoding/GzipChunk.hpp:310 allocates a fresh
        // `FasterVector<uint8_t>` of the requested size WITHOUT
        // zero-initializing (vendor/.../core/FasterVector.hpp:172-176 — the
        // `initialValue` arg defaults to `std::nullopt`, so `resize` skips
        // `std::fill`). gzippy's previous `vec![0u8; N]` zero-filled 128 KiB
        // per outer iter, multiplied across ~32 outer iters per 4 MiB chunk
        // ≈ 4 MiB of avoidable memset per chunk per worker. The same
        // `with_capacity + set_len` pattern is used elsewhere in this
        // codebase (see `src/backends/isal_decompress.rs:488-492`). All
        // bytes consumed downstream (the `buffer.truncate(n_bytes_read)` +
        // `chunk.append_owned_buffer(buffer)` calls at lines 254-256) are
        // bytes that ISA-L wrote into via `read_stream`, so no uninitialized
        // memory is ever observed.
        let mut buffer: Vec<u8> = Vec::with_capacity(ALLOCATION_CHUNK_SIZE);
        #[allow(clippy::uninit_vec)]
        unsafe {
            buffer.set_len(ALLOCATION_CHUNK_SIZE)
        };
        let mut n_bytes_read: usize = 0;
        // Cached state of the last inner call so the outer can decide
        // whether to break the OUTER loop (mirrors rapidgzip checking
        // `( stoppedAt() == NONE ) && ( nBytesReadPerCall == 0 ) && !footer`
        // at GzipChunk.hpp:386-389).
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;
        // True when END_OF_STREAM fired mid-outer-iter and we need the
        // outer iter to drive the multi-stream reset after committing
        // the buffer (gzippy-specific because our wrapper doesn't
        // return footers from read_stream).
        let mut end_of_stream_hit = false;

        while n_bytes_read < buffer.len() && !stopping_point_reached {
            // Inner inflate call writes into the still-unfilled tail of
            // `buffer`, mirroring rapidgzip's
            // `inflateWrapper.readStream( buffer.data() + nBytesRead,
            //                             buffer.size() - nBytesRead )`
            // at GzipChunk.hpp:318-319.
            let r = wrapper.read_stream(&mut buffer[n_bytes_read..])?;
            last_per_call = r.bytes_written;
            n_bytes_read += last_per_call;
            chunk.note_inner_decoded_bytes(last_per_call); // GzipChunk.hpp:321

            last_stopped_at = r.stopped_at;
            last_finished = r.finished;
            last_end_bit = r.bit_position;

            if r.finished {
                // BFINAL of the (last) stream in our range — exit both
                // loops via the OUTER check below.
                break;
            }

            match r.stopped_at {
                sp if sp == StoppingPoints::END_OF_STREAM => {
                    // Footer handling needs the buffer's bytes already
                    // committed (CRC32 is per-stream, and append_footer
                    // opens a fresh hasher). Break out of inner; outer
                    // commits the buffer, then drives multi-stream
                    // reset. Matches rapidgzip's outer-loop footer
                    // handling at GzipChunk.hpp:380-384.
                    end_of_stream_hit = true;
                    break;
                }
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                    // isBlockStart=true (GzipChunk.hpp:330-332). Push
                    // new subchunk at (encoded=r.bit_position,
                    // decoded=alreadyDecoded + nBytesRead) — line 365.
                    chunk.append_block_boundary_at(r.bit_position, already_decoded + n_bytes_read);
                }
                sp if sp == StoppingPoints::END_OF_BLOCK => {
                    if !wrapper.is_final_block() {
                        // isBlockStart = !isFinalBlock (GzipChunk.hpp:334-336).
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
                        // Stop at the pre-header position so next chunk
                        // can resume cleanly. Mirrors GzipChunk.hpp:339-345
                        // (`stoppingPointReached = true`).
                        last_end_bit = last_eob_pos;
                        stopping_point_reached = true;
                    }
                }
                sp if sp == StoppingPoints::NONE => {
                    if last_per_call == 0 {
                        // Mirrors GzipChunk.hpp:347-351.
                        stopping_point_reached = true;
                    }
                }
                _ => {}
            }

            // NOTE — vendor's `decodeChunkWithInflateWrapper`
            // (GzipChunk.hpp:192-268) does NOT check
            // maxDecompressedChunkSize. The fast path is bounded only
            // by `exactUntilOffset` (the next chunk's start). The
            // `alreadyDecoded >= max...` preempt vendor has is at
            // GzipChunk.hpp:368-372 inside
            // `finishDecodeChunkWithInexactOffset` (the slow/
            // speculative path), not here. gzippy previously also
            // preempted in the fast path, but that left the ISA-L
            // wrapper mid-block (no real EOB boundary at the stop
            // position), so the successor chunk's worker could not
            // resume from `last_end_bit`. Producing wrong output on
            // any single-member input whose decompressed size
            // exceeded max_decoded_chunk_size (= 20× split_chunk_size
            // = 80 MiB by default). Removing the preempt matches
            // vendor and unblocks large-file decode.
        }

        // OUTER iter end — commit the buffer once. Mirrors GzipChunk.hpp:376-379.
        already_decoded += n_bytes_read;
        buffer.truncate(n_bytes_read);
        if !buffer.is_empty() {
            chunk.append_owned_buffer(buffer);
        }

        if end_of_stream_hit {
            // Multi-stream reset. The deflate body of this gzip stream
            // is done. Read the 8-byte footer at the current cursor
            // (mirror of rapidgzip's IsalInflateWrapper::readFooter
            // call in its multi-stream loop). Record into ChunkData via
            // append_footer; that opens a fresh CRC hasher for any
            // subsequent stream's bytes (ChunkData.hpp:472-489).
            let (crc32, isize_field) = wrapper.read_footer_at_current()?;
            let footer_end_bits = wrapper.tell_compressed();
            chunk.append_footer(crc32, isize_field, footer_end_bits);

            // If more bytes remain, parse the next gzip header and
            // reset the wrapper for the next stream. If no more bytes
            // (this was the last stream in our range), we're done.
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
            // Re-arm the stopping points (reset cleared them) and
            // re-seed the dict — the next stream is independent so an
            // empty window is the correct seed.
            wrapper.set_stopping_points(
                StoppingPoints::END_OF_BLOCK
                    | StoppingPoints::END_OF_BLOCK_HEADER
                    | StoppingPoints::END_OF_STREAM_HEADER
                    | StoppingPoints::END_OF_STREAM,
            );
            wrapper.set_window(&[])?;
            last_end_bit = wrapper.tell_compressed();
            last_eob_pos = last_end_bit;
            // Record the stream-boundary as a subchunk break point.
            chunk.append_block_boundary_at(last_end_bit, already_decoded);
            continue;
        }

        if last_finished {
            break;
        }
        // Mirrors GzipChunk.hpp:386-389: if last call hit no stopping
        // point AND wrote zero bytes AND no footer fired, we're done.
        if last_stopped_at == StoppingPoints::NONE && last_per_call == 0 {
            break;
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    // VENDOR DIVERGENCE (documented 2026-05-19, audit 13): vendor
    // finalizes at `exactUntilOffset` (GzipChunk.hpp:265) — the
    // consumer's requested upper bound, NOT the worker's actual stop.
    // Vendor's IsalInflateWrapper caps `avail_in` at the byte for
    // `m_encodedUntilOffset` (gzip/isal.hpp:231,240,248) so the worker
    // genuinely stops at that bit position; the assertion at
    // GzipChunk.hpp:252-263 throws if `tellCompressed() !=
    // exactUntilOffset`. Vendor's chain invariant therefore holds:
    // chunk_N.encoded_end == until_bits_of_N == partition_seed ==
    // chunk_{N+1}.encoded_offset, so `matchesEncodedOffset` returns
    // true for prefetched chunks and the prefetch is consumed.
    //
    // gzippy's wrapper does NOT cap `avail_in` (inflate_wrapper.rs:116
    // takes input but no until_bit). Worker reads past until_bits to
    // the next EOB ≥ until_bits. last_end_bit > until_bits. Finalizing
    // at last_end_bit means block_finder.insert publishes that value,
    // and the consumer's next_block_offset is > the partition seed
    // under which the prefetch was dispatched. matchesEncodedOffset
    // returns FALSE (max == partition_seed < next_block_offset) →
    // prefetch rejected → on-demand fresh decode.
    //
    // Result (measured via --verbose 2026-05-19): 22 of 37 fetches
    // are on-demand vs vendor's 1 of 21. Pool Efficiency 27.9% vs
    // 64.2%. Decoded CPU 2.19s vs 0.527s (4× more — every rejected
    // prefetch is wasted work).
    //
    // FIX (deferred): port the wrapper's `m_encodedUntilOffset` cap
    // into `IsalInflateWrapper::refill_buffer` (gzippy doesn't have
    // refill today — wrapper sets avail_in once in `new`). Then
    // finalize at until_bits is consistent with the worker's actual
    // stop and the chain invariant holds. Multi-file change requiring
    // careful correctness validation (vendor's assertion at
    // GzipChunk.hpp:252-263 throws when until_bits isn't a real EOB —
    // for gzippy's partition guesses, that would mean reverting to
    // fallback decoding which we don't yet implement). Filed for
    // future autonomous session.
    chunk.finalize(last_end_bit);
    Ok(chunk)
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
}
