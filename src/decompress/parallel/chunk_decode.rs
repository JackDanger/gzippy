#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

//! Per-chunk deflate decode DRIVER for parallel single-member (the decode
//! *logic*, despite the data-flavored name). Port of rapidgzip's
//! `GzipChunkFetcher::decodeChunkWithRapidgzip` / `decodeChunk`
//! (GzipChunkFetcher.hpp). Distinct from: [`super::chunk_data`] (the
//! `ChunkData` *container* it fills) and [`super::chunk_fetcher`] (the
//! `processNextChunk` *orchestration* that calls this).
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
    /// DoS/OOM guard: decoded output exceeded the plausible ceiling derived from
    /// the compressed input length (`input_len × MAX_DEFLATE_EXPANSION`). The
    /// input is malformed — the decoder was fabricating output from zero-padding
    /// past end-of-input. Surfaced as terminal corruption (no allocation runaway).
    OutputCeilingExceeded {
        produced: usize,
        ceiling: usize,
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

/// Whether the clean-tail decode routes through REAL ISA-L FFI.
///
/// The pure-Rust DEFLATE engine is the SOLE production decode path; ISA-L
/// clean-tail decode was a measurement oracle only, controlled by
/// `GZIPPY_ISAL_ENGINE_ORACLE`. The env override was removed 2026-07-07
/// (batch 4f, confirmed no production ISA-L decode exists — the decode graph
/// is pure-Rust; C-FFI is compression-only per CLAUDE.md) — hardcoded OFF
/// (pure-Rust identity). The oracle-only `finish_decode_chunk_isal_oracle`
/// call site this used to gate was a mechanically dead consequence and has
/// been removed (2026-07-07, x86-finish batch), along with its
/// `isal_incremental_growth` sizing knob (`GZIPPY_ISAL_GROW_MIB` /
/// `GZIPPY_ISAL_INITIAL_FACTOR` / `GZIPPY_ISAL_INCREMENTAL_GROWTH`) — both
/// had zero remaining callers once this predicate went hardcoded OFF.
#[cfg(parallel_sm)]
#[inline]
fn isal_engine_oracle_enabled() -> bool {
    false
}

/// M3 (DIV-1 part 1): window-seeded INEXACT chunks decode on the ONE
/// `deflate::Block` engine. Was previously kill-switchable via
/// `GZIPPY_SEEDED_BLOCK=0` (restoring the pre-M3 wrapper path); the env
/// override was removed 2026-07-07 (batch 4f) — hardcoded to the shipped
/// default (ON). Production proof of which engine decoded each seeded chunk:
/// [`SEEDED_BLOCK_CHUNKS`] vs [`SEEDED_WRAPPER_CHUNKS`] (`--verbose` dump).
#[cfg(parallel_sm)]
fn seeded_block_enabled() -> bool {
    true
}

/// Whether the M3 seeded-Block route is taken for a window-seeded inexact
/// chunk: always ON (see [`seeded_block_enabled`] — the
/// `GZIPPY_ISAL_ENGINE_ORACLE` term was dropped 2026-07-07, confirmed no
/// production ISA-L decode graph exists to preserve — single-member decode
/// is pure-Rust ParallelSM at every T, see `decompress/mod.rs`).
#[cfg(parallel_sm)]
fn seeded_block_route_enabled() -> bool {
    seeded_block_enabled()
}

/// M4 (DIV-1 part 2): window-seeded UNTIL-EXACT chunks decode on the ONE
/// `deflate::Block` engine. Was previously kill-switchable via
/// `GZIPPY_EXACT_BLOCK=0` (restoring the pre-M4 wrapper path); the env
/// override was removed 2026-07-07 (batch 4f) — hardcoded to the shipped
/// default (ON). Production proof of which engine decoded each exact chunk:
/// [`EXACT_BLOCK_CHUNKS`] vs [`EXACT_WRAPPER_CHUNKS`] (`--verbose` dump).
#[cfg(parallel_sm)]
fn exact_block_enabled() -> bool {
    true
}

/// Whether the M4 exact-Block route is taken for a window-seeded UNTIL-EXACT
/// chunk: always ON (see [`exact_block_enabled`] — the
/// `GZIPPY_ISAL_ENGINE_ORACLE` term was dropped 2026-07-07, confirmed no
/// production ISA-L decode graph exists to preserve — single-member decode
/// is pure-Rust ParallelSM at every T, see `decompress/mod.rs`).
#[cfg(parallel_sm)]
fn exact_block_route_enabled() -> bool {
    exact_block_enabled()
}

/// Compute the upfront output-reserve byte count for a chunk's clean-tail decode.
///
/// `compressed_span` is the chunk's compressed byte span.  `expansion_ratio_ceil` is
/// the member-level ratio ceiling from `ChunkConfiguration::expansion_ratio_ceil`; a
/// value of 0 means the ratio was unknown at configuration time → falls back to the
/// historical 8× factor.
///
/// Result is clamped to `[RESERVE_FLOOR, RESERVE_CAP]`.  Growth past `RESERVE_CAP`
/// is handled by the GROW_BYTES loop downstream and is always safe — this function
/// only sizes the *upfront* allocation.
///
/// Exposed as `pub(crate)` so the unit-test module can exercise the clamp logic
/// directly.
///
/// Used by the pure-Rust native clean/seeded path
/// (`decode_chunk_unified_marker`, `seed_block_for_contig_native`). Sizing
/// the native per-chunk output reserve from the member ratio (instead of the old
/// `compressed × 8` clamped to 16 MiB) eliminates the grow-realloc storm on
/// expanding corpora (nasa ~10× → the 16 MiB clamp forced 16→32→64 MiB doubling,
/// each grow faulting the new alloc AND memmove-copying accumulated output) while
/// NOT over-reserving on near-incompressible data (ratio ≈ 2 → small reserve),
/// which a blind big-clamp would do.
#[cfg(parallel_sm)]
pub(crate) fn compute_initial_reserve(compressed_span: usize, expansion_ratio_ceil: u16) -> usize {
    const RESERVE_FLOOR: usize = 4 * 1024 * 1024; // never start below 4 MiB
    const RESERVE_CAP: usize = 64 * 1024 * 1024; // upfront ceiling; growth may exceed on demand
                                                 // RESIDENT-OUTPUT-POOL ORACLE (GZIPPY_RESIDENT_OUTPUT_POOL=1, byte-transparent,
                                                 // MEASUREMENT-ONLY). Determination tool for BEAT-IGZIP-T1: pin EVERY chunk's
                                                 // upfront reserve to a single fixed size so all pooled output buffers share an
                                                 // IDENTICAL capacity. Combined with the manual LIFO pool (which retains Vec
                                                 // capacity via `clear()` instead of dropping), the recycled buffer is never
                                                 // realloc'd by-a-hair on reuse → its pages stay RESIDENT, so the next chunk
                                                 // decodes into already-faulted memory (igzip's reused-window fault profile).
                                                 // This tests whether T1 first-touch output faults are recoverable by residency.
    if crate::decompress::parallel::chunk_buffer_pool::resident_output_pool_enabled() {
        // Same value as RESERVE_CAP; the shared constant keeps
        // `SegmentedU8::ensure_buf`'s first-take capacity and this pin in
        // lockstep (one source of truth — lever-2b relies on first-take ==
        // pinned reserve so the buffer's FIRST allocation is already huge).
        return crate::decompress::parallel::chunk_buffer_pool::RESIDENT_PINNED_CAPACITY;
    }
    let factor = if expansion_ratio_ceil == 0 {
        8 // unknown → historical default
    } else {
        expansion_ratio_ceil as usize
    };
    compressed_span
        .saturating_mul(factor)
        .max(RESERVE_FLOOR)
        .min(RESERVE_CAP)
}

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
        // M4 (DIV-1 part 2): window-seeded UNTIL-EXACT chunks decode on the
        // ONE `deflate::Block` engine (always ON — see `exact_block_route_enabled`).
        debug_assert!(exact_block_route_enabled());
        return decode_chunk_exact_block_native(
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
    // DoS/OOM guard: bound decoded output to what this compressed input could
    // plausibly produce (input_len × MAX_DEFLATE_EXPANSION). Malformed input
    // makes the inflate engine fabricate output from zero-padding past EOF;
    // the ceiling turns that runaway into a terminal error instead of an OOM.
    chunk.set_output_ceiling_for_input(input.len());

    // The ISA-L clean-tail measurement-oracle branch that used to sit here
    // (`GZIPPY_ISAL_ENGINE_ORACLE`) was removed 2026-07-07 (batch 4f) —
    // confirmed no production ISA-L decode graph exists (single-member decode
    // is pure-Rust ParallelSM at every T; C-FFI is compression-only per
    // CLAUDE.md). `finish_decode_chunk_isal_oracle` and `isal_incremental_growth`
    // were themselves deleted (x86-finish batch, 2026-07-07) — zero remaining
    // callers once this branch was removed. The ISAL_ENGINE_ORACLE_* counters
    // stay (still read by the `--verbose` dump in `chunk_fetcher.rs`) but
    // are now permanently zero.
    let t_decode = std::time::Instant::now();

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
        // DoS/OOM guard: stop a malformed runaway before it OOMs.
        if let Err(produced) = chunk.ensure_within_output_ceiling() {
            return Err(ChunkDecodeError::OutputCeilingExceeded {
                produced,
                ceiling: chunk.output_ceiling,
            });
        }
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
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER
                    && decode_base + n_bytes_read > 0 =>
                {
                    chunk.append_block_boundary_at(
                        r.bit_position,
                        decode_base + n_bytes_read,
                        Some(input),
                    );
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
        let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
        if until_exact {
            // M4 (DIV-1 part 2): until-exact decodes on the ONE `deflate::Block`
            // engine (always ON — see `exact_block_route_enabled`; the wrapper
            // fallback arm this used to have was removed 2026-07-07, batch 4f).
            // See `finish_decode_chunk_exact_block_native` for the labeled
            // deviation + pre-registered contract.
            debug_assert!(exact_block_route_enabled());
            finish_decode_chunk_exact_block_native(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
            )?;
        } else {
            // M3 (DIV-1 part 1): window-seeded INEXACT chunks decode on the ONE
            // `deflate::Block` engine (vendor GzipChunk.hpp:454-458; always ON —
            // see `seeded_block_route_enabled`; the wrapper fallback arm this
            // used to have was removed 2026-07-07, batch 4f) instead of the
            // second clean engine (`StreamingInflateWrapper`/`unified::Inflate`).
            debug_assert!(seeded_block_route_enabled());
            finish_decode_chunk_seeded_block_native(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
            )?;
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        return Ok(chunk);
    }

    // Vendor tryToDecode: bootstrap failure propagates; caller catches and tries next candidate.
    // STAGE-2d: the whole-file grid sets `configuration.multi_member`, selecting the
    // `MULTI_MEMBER=true` instantiation that walks member boundaries. Single-member
    // decode keeps `::<false>` (byte-identical — the continuation compiles out).
    let mut chunk = if configuration.multi_member {
        decode_chunk_unified_marker::<true>(
            input,
            encoded_offset_bits,
            stop_hint_bits,
            configuration,
        )?
    } else {
        decode_chunk_unified_marker::<false>(
            input,
            encoded_offset_bits,
            stop_hint_bits,
            configuration,
        )?
    };
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
/// DecodedData buffer (DecodedData.hpp:278-289). NOTE: this ContigFoldSink ring
/// path is the ~1% marker-loop dribble on gzippy-native; the BULK clean tail is
/// u8-direct via `decode_clean_into_contig` (no ring, no drain). The
/// A drain/CRC-split measurement measured the remaining drain+CRC second-touch at
/// ~0-1ms (frozen host N=21), so the gap to the ISA-L `ocl_cf` ceiling
/// (matched-comparator 0.945× rg) is ~36ms of essentially PURE symbol rate on the
/// SAME covered chunks — NOT an upper bound padded by ring cost (that earlier
/// caveat is STALE for the contig bulk). See git history (campaign plan, removed).
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

/// Result of a pure-marker cross-member continuation attempt (§3.1).
#[cfg(parallel_sm)]
enum MemberContinuation {
    /// A next member's header was parsed; the marker loop should re-enter and
    /// keep decoding on the same chunk. `marker_ctx` cursor + block are reset.
    Continued,
    /// EOF or trailing garbage after a fully-verified footer — the chunk ends
    /// cleanly at `end_bit` (§3.2 labeled deviation: gzippy + gzip(1) tolerate
    /// trailing garbage; the members decoded so far are `Ok`).
    EndOfData { end_bit: usize },
}

/// STAGE-2b: the pure-marker analogue of stage 1's contig BFINAL continuation
/// (`finish_decode_chunk_contig_native::<true>` — GzipChunk.hpp:602-653). Called
/// from [`decode_chunk_unified_marker`]'s `MarkerStep::Finished{bfinal_hit}` arm
/// when a member-final BFINAL is reached while the chunk is still in marker mode
/// (small members that never accumulate 32 KiB clean). Consumes the footer,
/// verifies ISIZE iff this chunk read the member's header, appends the footer
/// (fresh CRC + size segment), parses the next gzip header, and resets the
/// thread-local marker engine to an EMPTY dictionary so member N+1 decodes with
/// no back-references into member N.
///
/// `flipped` is forced true at the boundary so the 32 KiB fold-flip can NEVER
/// fire on a window that straddles the boundary (member N's clean tail + member
/// N+1's head) — the flip window would otherwise resolve member N+1's back-refs
/// against member N's bytes. Member N+1 keeps decoding via the marker loop's
/// clean-u8 path (empty dict ⇒ no real markers on valid data), which is correct
/// and, for the small members this arm handles, cheap. Large members flip DURING
/// themselves and take the fast contig continuation instead (FlipToContig arm).
#[cfg(parallel_sm)]
fn try_continue_next_member(
    chunk: &mut ChunkData,
    marker_ctx: &mut MarkerDecodeCtx,
    input: &[u8],
    end_bit_offset: usize,
    did_read_header: &mut bool,
    member_start_decoded: &mut usize,
) -> Result<MemberContinuation, ChunkDecodeError> {
    // The deflate stream ends byte-aligned; the 8-byte gzip footer begins at the
    // next byte boundary (the marker loop returns the unaligned post-EOB bit).
    let footer_byte = end_bit_offset.div_ceil(8);
    let footer = crate::decompress::parallel::gzip_format::read_footer(input, footer_byte)
        .map_err(|e| {
            // A truncated/absent footer after a BFINAL is corruption INSIDE the
            // member — terminal (§3.2/§4), never a clean trailing-garbage stop.
            ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("gzip footer at byte {footer_byte}: {e}"),
            ))
        })?;

    // In-chunk ISIZE check iff THIS chunk parsed the member's header
    // (vendor `didReadHeader`, GzipChunk.hpp:629-636). Members whose header
    // preceded this chunk defer to the consumer accumulator (§4).
    if *did_read_header {
        let member_bytes = chunk.decoded_size().saturating_sub(*member_start_decoded);
        if (member_bytes as u32) != footer.uncompressed_size {
            return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "gzip ISIZE mismatch: decoded {} vs footer {}",
                    member_bytes as u32, footer.uncompressed_size
                ),
            )));
        }
    }

    let after_footer_bit = (footer_byte + 8) * 8;
    chunk.append_footer(footer.crc32, footer.uncompressed_size, after_footer_bit);
    *member_start_decoded = chunk.decoded_size();

    // Read the next member's gzip header (vendor reads it at the top of the next
    // iteration, GzipChunk.hpp:470-511). No more bytes ⇒ clean EOF break.
    let header_byte = footer_byte + 8;
    if header_byte >= input.len() {
        return Ok(MemberContinuation::EndOfData {
            end_bit: after_footer_bit,
        });
    }
    match crate::decompress::parallel::gzip_format::read_header(&input[header_byte..]) {
        Ok((_header, header_len)) => {
            // Empty-dictionary reset — install an EMPTY window so member N+1
            // decodes CLEAN (into `chunk.data`, appended AFTER member N), exactly
            // as the contig BFINAL continuation does (`block.reset(None,None)` +
            // `set_initial_window(&[])`, GzipChunk.hpp:508). A bare
            // `reset(None,None)` (the old `block_primed=false` path) leaves the
            // engine in WINDOW-ABSENT/MARKER mode, so member N+1's bytes land in
            // `data_with_markers` — which the output orders BEFORE `chunk.data`,
            // scrambling `[big][s0][s1]` into `[s0][s1][big]` whenever an earlier
            // member decoded clean into `chunk.data`. §3.3 requires markers appear
            // ONLY before the first footer; every member after a boundary is
            // clean by construction (empty dict ⇒ no unresolvable back-refs).
            // `flipped=true` also disables the straddle-prone 32 KiB fold-flip for
            // the rest of this chunk (see fn doc).
            BOOTSTRAP_BLOCK.with(|cell_block| {
                let mut block = cell_block.borrow_mut();
                let mut unused: Vec<u16> = Vec::new();
                block.reset(Some(&mut unused), Some(&[]));
                debug_assert!(
                    unused.is_empty(),
                    "empty-window reset must not drain output"
                );
            });
            marker_ctx.block_primed = true;
            marker_ctx.flipped = true;
            marker_ctx.current_bit_offset = (header_byte + header_len) * 8;
            *did_read_header = true;
            MULTI_MEMBER_CONTINUATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(MemberContinuation::Continued)
        }
        Err(_) => {
            // Trailing bytes after a fully-verified footer are not a parseable
            // gzip header ⇒ clean-stop at this member boundary (§3.2).
            Ok(MemberContinuation::EndOfData {
                end_bit: after_footer_bit,
            })
        }
    }
}

/// STAGE-2b: window-absent (speculative) decode entry that WALKS member
/// boundaries — the `MULTI_MEMBER=true` instantiation of
/// [`decode_chunk_unified_marker`]. This is the entry the whole-file MM parallel
/// pipeline dispatches speculative chunks through (stage 2 driver, item 2); the
/// existing [`decode_chunk_window_absent`] stays byte-identical (`::<false>`).
#[cfg(parallel_sm)]
pub fn decode_chunk_window_absent_multi(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_unified_marker::<true>(input, encoded_offset_bits, stop_hint_bits, configuration)
}

/// The ONE unified decode driver — vendor `deflate::Block` (see
/// `marker_decode_step`). Once 32 KiB of clean output exist at a block boundary, control hands off to
/// `finish_decode_chunk_with_inexact_offset` with that clean window.
///
/// STAGE-2b `MULTI_MEMBER` const-generic (precedent:
/// `finish_decode_chunk_contig_native::<MULTI_MEMBER>`, stage 1): when `true`,
/// a member-final BFINAL reached in the WINDOW-ABSENT/marker path (either the
/// pure-marker `MarkerStep::Finished{bfinal_hit}` arm — small members that never
/// accumulate 32 KiB clean — or the post-flip `FlipToContig` clean tail) walks
/// `… member → footer → header → member …` instead of ending the chunk. This is
/// the gap stage 1 did NOT touch: stage 1 wired the SEEDED/exact/contig arms,
/// but the actual parallel pipeline decodes speculative chunks THROUGH here.
/// When `false` the continuation is compiled out (single-member inertness).
#[cfg(parallel_sm)]
fn decode_chunk_unified_marker<const MULTI_MEMBER: bool>(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let mut marker_ctx = MarkerDecodeCtx::new(input, encoded_offset_bits)?;
    // DoS/OOM guard: bound decoded output to what this compressed input could
    // plausibly produce. A window-absent (markered) chunk on malformed input can
    // fabricate phantom blocks from zero-padding past EOF, growing both the u16
    // marker buffer and the u8 clean buffer without bound; the ceiling turns that
    // into a terminal error. No effect on valid data (max expansion ~1032:1).
    chunk.set_output_ceiling_for_input(input.len());
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
    // copy#2/3/grow but load-inflated). NOTE (2026-06-08, measured): on the
    // gzippy-native build the BULK clean tail does NOT take this ContigFoldSink
    // ring path at all — it takes `finish_decode_chunk_contig_native` ->
    // `decode_clean_into_contig` (u8-DIRECT into chunk.data, no ring, no drain;
    // this sink governs only the ~1% marker-loop dribble). A drain/CRC-split
    // measurement measured the remaining drain+CRC
    // second-touch at ~0-1ms (frozen host, N=21), so the gap to the engine-removed
    // ceiling (ocl_cf, matched-comparator 0.945× rg) is ~36ms of essentially PURE
    // pure-Rust-vs-ISA-L SYMBOL RATE on the SAME covered chunks (coverage symmetry
    // confirmed: native flip_to_clean=12 finished_no_flip=4 window_seeded=2 ==
    // ocl_cf's 14 covered). The earlier "ring-write+drain remain, upper bound only"
    // caveat is STALE for the contig bulk path. See git history (campaign plan, removed).
    {
        // RATIO-INFORMED upfront reserve (shared with the ISA-L oracle path):
        // size from the member's KNOWN ISIZE/compressed ratio
        // (`configuration.expansion_ratio_ceil`) instead of the old fixed
        // `compressed × 8` clamped to 16 MiB. On expanding corpora (nasa ~10×)
        // the 16 MiB clamp forced a 16→32→64 MiB grow-realloc storm (each grow
        // faults the new alloc + memmoves accumulated output); ratio-sizing
        // reserves the realistic decoded size ONCE. An under-reserve still falls
        // back to safe amortized regrow. See `compute_initial_reserve`.
        let compressed_bytes = stop_hint_bits.saturating_sub(encoded_offset_bits) / 8;
        chunk.reserve_clean(compute_initial_reserve(
            compressed_bytes,
            chunk.configuration.expansion_ratio_ceil,
        ));
    }
    let mut pending_boundaries: Vec<(usize, usize)> = Vec::new();
    // STAGE-2b per-member cross-boundary state (MULTI_MEMBER only). A speculative
    // chunk STARTS mid-member, so `did_read_header` is false until this chunk
    // parses a member's gzip header itself; only then does the in-chunk ISIZE
    // check fire at that member's footer (members whose header preceded this
    // chunk defer their ISIZE to the consumer accumulator §4). `member_start_
    // decoded` is the decoded-byte position at the current member's start.
    let mut did_read_header = false;
    let mut member_start_decoded: usize = 0;
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
        let (step, _flipped_clean) =
            marker_decode_step(&mut marker_ctx, input, stop_hint_bits, &[], &mut sink)?;
        marker_ctx.clean_data_count = clean_appended;
        apply_recorded_block_boundaries(&mut chunk, input, &pending_boundaries);
        pending_boundaries.clear();
        // DoS/OOM guard: stop a malformed runaway before it OOMs.
        if let Err(produced) = chunk.ensure_within_output_ceiling() {
            return Err(ChunkDecodeError::OutputCeilingExceeded {
                produced,
                ceiling: chunk.output_ceiling,
            });
        }
        match step {
            MarkerStep::Continue => {}
            MarkerStep::FlipToContig { end_bit_offset } => {
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                // STAGE-2b: the post-flip clean tail carries the cross-member
                // continuation (stage 1's `::<true>` arm). A large member that
                // accumulated 32 KiB clean flips here, then walks its own final
                // BFINAL → footer → next header → next member entirely inside the
                // fast contig driver (buffer-relative back-refs + per-boundary
                // empty-window reset — no straddle hazard). This is the dominant-
                // member path that makes the grid SCALE.
                finish_decode_chunk_contig_native::<MULTI_MEMBER>(
                    &mut chunk,
                    &mut marker_ctx,
                    input,
                    end_bit_offset,
                    stop_hint_bits,
                    false,
                    true,
                )?;
                return Ok(chunk);
            }
            MarkerStep::FlipToClean { end_bit_offset, .. } => {
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
            MarkerStep::Finished {
                end_bit_offset,
                bfinal_hit,
            } => {
                // STAGE-2b pure-marker cross-member continuation: a member-final
                // BFINAL was reached WITHOUT the chunk ever accumulating 32 KiB
                // clean (small members). Walk the footer + next header and keep
                // decoding member N+1 on this same chunk. Non-BFINAL Finished
                // (stop-hint reached) and the single-member instantiation fall
                // through to the finalize below unchanged.
                if MULTI_MEMBER && bfinal_hit {
                    match try_continue_next_member(
                        &mut chunk,
                        &mut marker_ctx,
                        input,
                        end_bit_offset,
                        &mut did_read_header,
                        &mut member_start_decoded,
                    )? {
                        MemberContinuation::Continued => continue,
                        MemberContinuation::EndOfData { end_bit } => {
                            chunk.statistics.non_marker_count +=
                                chunk.data_with_markers.len() as u64;
                            chunk.finalize_with_deflate(end_bit, Some(input));
                            return Ok(chunk);
                        }
                    }
                }
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                chunk.finalize_with_deflate(end_bit_offset, Some(input));
                return Ok(chunk);
            }
        }
    }
}

/// gzippy-native copy-free-to-final clean tail. Resumes the SAME thread-local
/// `Block` (already flipped to clean) and decodes every subsequent deflate block
/// DIRECTLY into `chunk.data`'s reserved contiguous tail — no u8 ring, no
/// ring->chunk.data drain memcpy. Back-refs resolve from `chunk.data[*pos-d]`,
/// the already-committed contiguous clean output (the faithful vendor
/// `setInitialWindow` prepend; `data_prefix_len` stays 0 because the 32 KiB
/// predecessor window is real prior output). Replicates `marker_decode_step_loop`'s
/// per-block bookkeeping (header parse, stop-hint early-out, BFINAL stop, EOB
/// block-boundary recording, CRC + subchunk + non_marker accounting) for the
/// clean phase only. The isal two-phase path (`finish_decode_chunk_with_inexact_offset`)
/// is unchanged.
///
/// `until_exact` (M4, DIV-1 part 2) switches the stop condition from the
/// inexact "first block boundary at-or-past `stop_hint_bits`" to the EXACT
/// contract of the wrapper arm (`finish_decode_chunk_impl` with
/// `until_exact=true`, whose bit reader is hard-capped at `stop_hint_bits`
/// via `with_until_bits`):
///
///   - SUCCESS iff the decode lands EXACTLY at `stop_hint_bits`: either the
///     bit cursor reaches `stop_hint_bits` at a block-header boundary (the
///     wrapper's `try_enter_next_block` cap stop), or the member's BFINAL
///     block ends with its BYTE-ALIGNED post-EOB bit == `stop_hint_bits`.
///   - Otherwise `ChunkDecodeError::ExactStopMissed { requested, actual }`,
///     replicating chunk_decode.rs `finish_decode_chunk_impl`'s
///     `final_bit != stop_hint_bits` assertion with the same coordinates.
///
/// END-BIT COORDINATE CONVENTION (explicit, the BFINAL scar-class lesson):
///   - interior stop: the exact (possibly non-byte-aligned) bit of the
///     confirmed boundary — identical to the wrapper's read-cap cursor.
///   - member-final stop: the post-EOB bit rounded UP to the next byte
///     boundary. The wrapper consumes the RFC 1952 zero padding via
///     `finish_current_block`'s `align_to_byte()` (resumable.rs:823-842),
///     so its `tell_compressed()` at stream end is byte-aligned; the Block
///     arm replicates that by aligning `end_bit_offset` explicitly. The
///     8-byte gzip footer is NOT consumed by either arm (`sm_driver`
///     slices it off the input; production `stop_hint == total_bits` is
///     the padded deflate end, footer excluded). NOTE this differs from
///     the INEXACT Block arm (M3), which reports the UNALIGNED post-EOB
///     bit (documented M3 stream-end exception) — for the exact arm the
///     aligned convention is REQUIRED so the production
///     `stop_hint == total_bits` member-final chunk lands exactly.
#[cfg(parallel_sm)]
// `MULTI_MEMBER` const-generic (precedent:
// `read_internal_compressed_specialized<CONTAINS_MARKERS>`, module doc
// `parallel/mod.rs:76-81`): when `false` the cross-member continuation at BFINAL
// (footer consume + next-header parse + empty-window reset) is compiled OUT, so
// the single-member instantiation is codegen-identical to the pre-port driver
// (verified by the §7 disasm diff). When `true`, the driver walks
// `… deflate block → BFINAL → gzip footer → next gzip header → next member …`
// exactly like rapidgzip's `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654).
fn finish_decode_chunk_contig_native<const MULTI_MEMBER: bool>(
    chunk: &mut ChunkData,
    marker_ctx: &mut MarkerDecodeCtx,
    input: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    until_exact: bool,
    // Per-block EOB boundary recording. The boundary index feeds the block-map /
    // prefetch / subchunk-split (T>1 scaffold). Every remaining caller passes
    // `true` — the former T1-MONOLITH divergence that passed `false` (no
    // boundary recording, pure T>1 scaffold skipped) was removed 2026-07-07
    // (batch 4f, dead opt-in path).
    record_boundaries: bool,
) -> Result<(), ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::{BlockError, CompressionType};

    // Cross-member continuation state (vendor `GzipChunk.hpp:449-452`).
    // `did_read_header` tracks whether the CURRENT member's gzip header was
    // parsed inside this decode call (member 1's is consumed by the caller, so
    // it stays false and its ISIZE defers to the consumer accumulator §4);
    // `stream_bytes_read` is the per-member decoded byte count, reset at every
    // footer. Only ever read/mutated inside `if MULTI_MEMBER` blocks, so both
    // fold away on the single-member (`false`) instantiation.
    let mut did_read_header = false;
    let mut stream_bytes_read: u64 = 0;

    // Per-call contig headroom: one max-length back-ref (258) + the 8-byte
    // word-copy overshoot (matches `decode_clean_into_contig`'s `out_room`
    // reservation `cap - (MAX_RUN_LENGTH + 8)`, MAX_RUN_LENGTH == 258).
    const HEADROOM: usize = 258 + 8;

    marker_ctx.current_bit_offset = start_bit_offset;
    let crc32_enabled = chunk.configuration.crc32_enabled;
    // DoS/OOM guard: bound decoded output to what this compressed input could
    // plausibly produce (input_len × MAX_DEFLATE_EXPANSION). Malformed input
    // makes the decoder fabricate phantom blocks from zero-padding past EOF,
    // growing the output buffer without bound; the ceiling turns that into a
    // terminal error instead of an OOM. No effect on valid data.
    chunk.set_output_ceiling_for_input(input.len());

    BOOTSTRAP_BLOCK.with(|cell_block| -> Result<(), ChunkDecodeError> {
        let mut block = cell_block.borrow_mut();
        debug_assert!(
            !block.contains_marker_bytes(),
            "contig native tail requires a flipped (clean) Block"
        );

        loop {
            let slice_byte = marker_ctx.current_bit_offset / 8;
            let mut bits = marker_ctx.open_bits(input);
            let next_block_offset = absolute_bit_pos(slice_byte, &bits);

            // M4 exact stop at a block-header boundary. Wrapper analog: the
            // bit reader is capped at `stop_hint_bits` (`with_until_bits`),
            // so `try_enter_next_block` returns false the moment the cursor
            // reaches the cap and `final_bit = tell_compressed() ==
            // stop_hint_bits` — success without parsing the next header.
            // A cursor PAST the cap is impossible for the wrapper (the
            // reader refuses those bits) and means a mis-registered stop
            // hint here (never a confirmed boundary) — error with the same
            // ExactStopMissed coordinates the wrapper's final assertion uses.
            if until_exact {
                if next_block_offset == stop_hint_bits {
                    marker_ctx.current_bit_offset = next_block_offset;
                    chunk.finalize_with_deflate(next_block_offset, Some(input));
                    return Ok(());
                }
                if next_block_offset > stop_hint_bits {
                    return Err(ChunkDecodeError::ExactStopMissed {
                        requested: stop_hint_bits,
                        actual: next_block_offset,
                    });
                }
            }

            // Header parse (mirror marker_decode_step_loop:1499-1515).
            {
                if let Err(e) = block.read_header(&mut bits, false) {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deflate header at bit {next_block_offset}: {e:?}"),
                    )));
                }
            }

            // Stop-hint early-out at a block boundary (mirror :1517-1527).
            // INEXACT arm only — the exact arm stops solely on the
            // `next_block_offset == stop_hint_bits` cap check above.
            if !until_exact && next_block_offset >= stop_hint_bits && !block.is_last_block() {
                marker_ctx.current_bit_offset = next_block_offset;
                chunk.finalize_with_deflate(next_block_offset, Some(input));
                return Ok(());
            }

            // Decode the whole block body into chunk.data's contiguous tail.
            // One deflate block may span MULTIPLE contig calls (a block bigger
            // than the per-call out_room); accounting accumulates across them and
            // the block boundary fires only at real EOB.
            let comp_type = block.compression_type();
            while !block.eob() {
                // H4: re-fetch (base, cap, pos) every iteration — a grow inside
                // `contig_decode_window` may have moved the allocation.
                //
                // Request HEADROOM + 1, not HEADROOM: the decoders cap their
                // write budget at `out_room = cap - HEADROOM`, so a returned
                // spare of EXACTLY HEADROOM makes `out_room == pos` → zero
                // budget → `Ok(0)` → a spurious "no progress" error below.
                // (Observed on the M3 seeded route at spare == 266 after the
                // reserve estimate was outgrown; latent on the FOLD path too.)
                // Vec growth is amortized, so the +1 only changes the moment a
                // doubling fires, never the decoded bytes.
                let (base, cap, pos_before) = chunk.data.contig_decode_window(HEADROOM + 1);
                let mut pos = pos_before;
                // H1: release-mode guard — never let the decoder write past the
                // reserved headroom (a contract violation here is a heap OOB,
                // not a CRC-catchable wrong byte, because the contig path has no
                // ring modulo).
                let out_room = cap.saturating_sub(HEADROOM);
                assert!(
                    pos <= out_room && cap >= pos_before + HEADROOM,
                    "contig native tail: insufficient headroom (pos {pos} cap {cap})"
                );

                // SAFETY: `base` is `contig_decode_window`'s pointer, valid for
                // `[0, cap)`; the assert above proves `pos <= out_room` (the
                // headroom contract); `bits.data` is the compressed input, which
                // never aliases the chunk's decode destination.
                let body_res = match comp_type {
                    CompressionType::Uncompressed => unsafe {
                        block.decode_clean_stored_into_contig(
                            &mut bits,
                            base,
                            cap,
                            &mut pos,
                            usize::MAX,
                        )
                    },
                    CompressionType::FixedHuffman | CompressionType::DynamicHuffman => unsafe {
                        block.decode_clean_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
                    },
                    CompressionType::Reserved => Err(BlockError::InvalidCompression),
                };
                let emitted = match body_res {
                    Ok(n) => n,
                    Err(e) => {
                        // Commit whatever was written before the failure so the
                        // logical length stays consistent, then surface the error.
                        chunk.data.commit(pos - pos_before);
                        return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("deflate body at bit {next_block_offset}: {e:?}"),
                        )));
                    }
                };
                debug_assert_eq!(emitted, pos - pos_before);
                // H3: commit BEFORE reading the bytes back for CRC (decoded_range
                // indexes the committed region).
                chunk.data.commit(emitted);
                if emitted > 0 {
                    if crc32_enabled {
                        if let Some(last_crc) = chunk.crc32s.last_mut() {
                            last_crc.update(chunk.data.decoded_range(pos_before, emitted));
                        }
                    }
                    chunk.statistics.non_marker_count += emitted as u64;
                    if let Some(last) = chunk.subchunks.last_mut() {
                        last.decoded_size += emitted;
                    }
                    marker_ctx.clean_data_count += emitted;
                    // Per-member ISIZE accumulator (reset at each footer).
                    // Const-folded to nothing on the single-member instance.
                    if MULTI_MEMBER {
                        stream_bytes_read += emitted as u64;
                    }
                }
                // No forward progress AND not at EOB ⇒ the buffer can't grow
                // enough (degenerate); bail rather than spin.
                if emitted == 0 && !block.eob() {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "contig native tail: no progress",
                    )));
                }
                // DoS/OOM guard: stop a malformed runaway before it OOMs.
                if let Err(produced) = chunk.ensure_within_output_ceiling() {
                    return Err(ChunkDecodeError::OutputCeilingExceeded {
                        produced,
                        ceiling: chunk.output_ceiling,
                    });
                }
            }

            let end_bit_offset = absolute_bit_pos(slice_byte, &bits);
            marker_ctx.current_bit_offset = end_bit_offset;

            if block.is_last_block() {
                if MULTI_MEMBER {
                    // ── rapidgzip cross-member continuation (GzipChunk.hpp:602-653).
                    //    Corrected sequencing [R1-#1]: at BFINAL the footer is
                    //    consumed UNCONDITIONALLY, the next header is read at the
                    //    top of the next iteration, and the stop-hint check runs
                    //    only at deflate-block boundaries (the loop-top checks
                    //    above). So a chunk end is ALWAYS a deflate-block boundary
                    //    (possibly in the next member) or clean EOF — never a
                    //    position inside footer/trailer bytes.

                    // The deflate stream ends byte-aligned; the 8-byte gzip
                    // footer begins at the next byte boundary.
                    let footer_byte = end_bit_offset.div_ceil(8);
                    let footer =
                        crate::decompress::parallel::gzip_format::read_footer(input, footer_byte)
                            .map_err(|e| {
                            // A truncated/absent footer after a BFINAL is corruption
                            // INSIDE the member — terminal (§3.2/§4), never a clean
                            // trailing-garbage stop.
                            ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("gzip footer at byte {footer_byte}: {e}"),
                            ))
                        })?;

                    // Verify ISIZE iff THIS member's gzip header was read in this
                    // chunk (vendor `didReadHeader`, GzipChunk.hpp:629-636).
                    // Member 1's header was consumed by the caller/finder, so its
                    // check defers to the consumer's running accumulator (§4).
                    if did_read_header && (stream_bytes_read as u32) != footer.uncompressed_size {
                        return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "gzip ISIZE mismatch: decoded {} vs footer {}",
                                stream_bytes_read as u32, footer.uncompressed_size
                            ),
                        )));
                    }

                    let after_footer_bit = (footer_byte + 8) * 8;
                    chunk.append_footer(footer.crc32, footer.uncompressed_size, after_footer_bit);

                    // Per-member reset (GzipChunk.hpp:645-647).
                    stream_bytes_read = 0;
                    did_read_header = false;
                    marker_ctx.current_bit_offset = after_footer_bit;

                    // Read the next member's gzip header (vendor reads it at the
                    // top of the next iteration, GzipChunk.hpp:470-511). No more
                    // bytes ⇒ clean EOF break (`bitReader->eof()`,
                    // GzipChunk.hpp:649-652).
                    let header_byte = footer_byte + 8;
                    if header_byte >= input.len() {
                        chunk.finalize_with_deflate(after_footer_bit, Some(input));
                        return Ok(());
                    }
                    match crate::decompress::parallel::gzip_format::read_header(
                        &input[header_byte..],
                    ) {
                        Ok((_header, header_len)) => {
                            // Reset the deflate engine to an EMPTY dictionary — a
                            // new gzip stream carries no back-references into the
                            // previous member (vendor `block->reset(VectorView<
                            // uint8_t>{})`, GzipChunk.hpp:508). The contig back-ref
                            // reader addresses `base[*pos - d]` (buffer-relative),
                            // and resetting `decoded_bytes` to 0 makes the
                            // empty-dict range check bound member N+1's window so
                            // it can never reach member N's bytes on a valid
                            // stream (a corrupt over-reach is caught by the
                            // per-member CRC32, §4).
                            block.reset(None, None);
                            let mut unused: Vec<u16> = Vec::new();
                            block.set_initial_window(&mut unused, &[]).map_err(|e| {
                                ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!("member-boundary empty-window reset: {e:?}"),
                                ))
                            })?;
                            debug_assert!(
                                unused.is_empty(),
                                "empty-window reset must not drain output"
                            );
                            did_read_header = true;
                            marker_ctx.current_bit_offset = (header_byte + header_len) * 8;
                            MULTI_MEMBER_CONTINUATIONS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                        Err(_) => {
                            // Trailing bytes after a fully-verified footer are not
                            // a parseable gzip header ⇒ clean-stop at this member
                            // boundary (§3.2 LABELED DEVIATION from vendor's
                            // chunk-level throw: gzippy + gzip(1) tolerate trailing
                            // garbage; the members decoded so far are `Ok`).
                            chunk.finalize_with_deflate(after_footer_bit, Some(input));
                            return Ok(());
                        }
                    }
                }
                if until_exact {
                    // Member-final exact stop: byte-align the post-EOB bit
                    // (the wrapper consumes the RFC 1952 padding via
                    // `align_to_byte`, resumable.rs:823-842) and require it
                    // to equal the requested stop — the wrapper's
                    // `final_bit != stop_hint_bits` assertion with identical
                    // coordinates. The footer is NOT consumed (input slice
                    // excludes it); a multi-member-crossing stop hint
                    // therefore errors here exactly like the wrapper arm
                    // (no `read_footer_at_current`/`reset_for_next_stream`
                    // call exists on the production until-exact path).
                    let aligned_end = end_bit_offset.div_ceil(8) * 8;
                    marker_ctx.current_bit_offset = aligned_end;
                    if aligned_end != stop_hint_bits {
                        return Err(ChunkDecodeError::ExactStopMissed {
                            requested: stop_hint_bits,
                            actual: aligned_end,
                        });
                    }
                    chunk.finalize_with_deflate(aligned_end, Some(input));
                    return Ok(());
                }
                chunk.finalize_with_deflate(end_bit_offset, Some(input));
                return Ok(());
            }
            // Record the block boundary at the real EOB (decoded_offset = total
            // decoded bytes = markers + clean), mirror :1597. `data_prefix_len`
            // is 0 on the FOLD path (the 32 KiB window is real prior output);
            // on the M3 seeded path the dictionary prefix at `data[0..32768)`
            // is NOT chunk output and must not shift boundary keys.
            if record_boundaries {
                let decoded_offset =
                    chunk.data_with_markers.len() + chunk.data.len() - chunk.data_prefix_len;
                chunk.append_block_boundary_at(end_bit_offset, decoded_offset, Some(input));
            }
        }
    })
}

/// M3 (DIV-1 part 1) — vendor `GzipChunk.hpp:454-458` (non-ISAL build):
///
/// ```c++
/// auto block = std::make_shared<deflate::Block</* CRC32 */ false,
///                                              /* enable analysis */ false>>();
/// if ( initialWindow ) {
///     block->setInitialWindow( *initialWindow );
/// }
/// ```
///
/// A window-KNOWN chunk decodes on the SAME ONE `deflate::Block` engine,
/// seeded clean-from-byte-0 — vendor-native has NO second engine for this
/// path (the ISA-L fork at GzipChunk.hpp:440-444 is compiled out). gzippy
/// mirror: seed the thread-local [`BOOTSTRAP_BLOCK`] (`Block::reset` +
/// `set_initial_window` → `WidthRing::seed_window`, deflate.hpp:1750-1759)
/// and decode every deflate block u8-DIRECT into `chunk.data`'s contiguous
/// tail via `decode_clean_into_contig` — the design's single clean-destination
/// contract (git history (campaign plan, removed) §4.3), the same machinery the FOLD
/// post-flip tail already runs ([`finish_decode_chunk_contig_native`]).
///
/// The 32 KiB seed is installed as a NON-OUTPUT dictionary prefix at
/// `chunk.data[0..32768)` (`prefill_window_prefix`, `data_prefix_len` =
/// 32 KiB — the A3/A4 scaffolding: `decoded_size()`, boundary keys, window
/// extraction and the consumer write all skip it), so back-refs resolve as
/// pure `base[*pos - d]` — byte-equal to vendor's ring-window reads because
/// the prefix bytes ARE the predecessor window (deflate.hpp:1750-1759 prime
/// + DecodedData.hpp:278-289 contiguous clean storage).
///
/// This REPLACES `StreamingInflateWrapper`/`unified::Inflate`
/// (inflate_wrapper.rs:153-161) on the gzippy-native window-seeded INEXACT
/// path — the DIV-1 second clean engine. The until-exact path stays on the
/// wrapper until M4 (its stopping-point/footer contract is pre-registered
/// there); the gzippy-isal clean-tail handoff is untouched (faithful
/// rapidgzip WITH_ISAL, GzipChunk.hpp:440-444/520-526).
///
/// Route: [`seeded_block_route_enabled`] (hardcoded ON; the wrapper arm now
/// runs only on the gzippy-isal build — the `GZIPPY_SEEDED_BLOCK=0` env
/// kill-switch was removed 2026-07-07, batch 4f).
#[cfg(parallel_sm)]
fn finish_decode_chunk_seeded_block_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    SEEDED_BLOCK_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let mut marker_ctx = seed_block_for_contig_native(
        chunk,
        input,
        inflate_start_bit,
        stop_hint_bits,
        initial_window,
    )?;

    // Same per-block driver the FOLD post-flip tail uses: header parse,
    // stop-hint early-out at block boundaries, `decode_clean_into_contig`
    // bodies, EOB boundary recording, BFINAL finalize.
    // STAGE-2d: window-seeded INEXACT chunks that cross a member boundary walk
    // into the next member when the grid sets `multi_member`.
    if chunk.configuration.multi_member {
        finish_decode_chunk_contig_native::<true>(
            chunk,
            &mut marker_ctx,
            input,
            inflate_start_bit,
            stop_hint_bits,
            false,
            true,
        )
    } else {
        finish_decode_chunk_contig_native::<false>(
            chunk,
            &mut marker_ctx,
            input,
            inflate_start_bit,
            stop_hint_bits,
            false,
            true,
        )
    }
}

/// Shared M3/M4 seeding: dictionary prefix + contig reserve + priming the
/// thread-local [`BOOTSTRAP_BLOCK`] clean-from-byte-0 with the predecessor
/// window (vendor GzipChunk.hpp:456-458 → `Block::setInitialWindow`,
/// deflate.hpp:1750-1759).
#[cfg(parallel_sm)]
fn seed_block_for_contig_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<MarkerDecodeCtx, ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    debug_assert_eq!(
        initial_window.len(),
        MAX_WINDOW_SIZE,
        "seeded-Block route requires a full 32 KiB window (caller gate)"
    );

    // Dictionary prefix: the predecessor window at `data[0..32768)`, excluded
    // from output/boundary accounting via `data_prefix_len`.
    chunk.prefill_window_prefix(initial_window);

    // Pre-reserve ONE contiguous clean-data region (mirror of
    // `decode_chunk_unified_marker`'s reserve — same ratio-informed sizing;
    // an under-reserve falls back to safe amortized regrow between calls).
    {
        let compressed_bytes = stop_hint_bits.saturating_sub(inflate_start_bit) / 8;
        chunk.reserve_clean(compute_initial_reserve(
            compressed_bytes,
            chunk.configuration.expansion_ratio_ceil,
        ));
    }

    let mut marker_ctx = MarkerDecodeCtx::new(input, inflate_start_bit)?;

    // Seed the ONE engine: reset the thread-local Block, then prime CLEAN
    // mode from byte 0 with the predecessor window.
    BOOTSTRAP_BLOCK.with(|cell_block| -> Result<(), ChunkDecodeError> {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let mut unused: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut unused, initial_window)
            .map_err(|e| {
                ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("seeded-Block set_initial_window: {e:?}"),
                ))
            })?;
        debug_assert!(unused.is_empty(), "seed must not drain into output");
        Ok(())
    })?;
    marker_ctx.block_primed = true;
    Ok(marker_ctx)
}

/// STAGE-1 multi-member chunked decode entry (`MULTI_MEMBER=true`). Decodes a
/// span of a WHOLE gzip FILE — inter-member footers + headers included in
/// `input` — into ONE `ChunkData` that walks `… member → footer → header →
/// member …` via the cross-member continuation in
/// [`finish_decode_chunk_contig_native`]. Port of rapidgzip's
/// `decodeChunkWithRapidgzip` over a chunk spanning members (GzipChunk.hpp:
/// 468-654), with the single-member driver's window/marker machinery elided by
/// the empty-dictionary start (a member boundary is an empty-window reset, so a
/// clean-from-block-0 decode is exact from member 1).
///
/// STAGE-1 SCOPE: this is the continuation CORE. The whole-file finder span,
/// the `MultiMemberChunked` routing, and the consumer's per-member CRC/ISIZE
/// verification pass are stages 2-3 (§1.2/§4); today this is reached only from
/// the unit + seam tests and instantiates the `true` monomorphization.
///
/// `first_block_bit` is the bit offset of member 1's first deflate block (=
/// `first_header_size * 8`). `stop_hint_bits` is the inexact stop
/// (`input.len() * 8` decodes the whole span to EOF). `reserve_hint` sizes the
/// one output reservation (Σ member ISIZE, or 0 for the amortized-grow
/// fallback).
#[cfg(parallel_sm)]
pub(crate) fn decode_multi_member_native(
    input: &[u8],
    first_block_bit: usize,
    stop_hint_bits: usize,
    reserve_hint: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut chunk = ChunkData::new(first_block_bit, configuration);

    // Empty-dictionary start: member 1 has no prior history, so NO 32 KiB
    // window prefix is installed (`data_prefix_len` stays 0). Back-refs
    // resolve `base[*pos - d]` within the member's own output, bounded by the
    // empty-dict range check — identical to vendor `block->reset(empty)` at
    // GzipChunk.hpp:508 and gzippy's chunk-0 zero-window seed semantics.
    chunk.reserve_clean(reserve_hint.saturating_add(1024));

    let mut marker_ctx = MarkerDecodeCtx::new(input, first_block_bit)?;
    BOOTSTRAP_BLOCK.with(|cell_block| -> Result<(), ChunkDecodeError> {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let mut unused: Vec<u16> = Vec::new();
        block.set_initial_window(&mut unused, &[]).map_err(|e| {
            ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("multi-member seed set_initial_window: {e:?}"),
            ))
        })?;
        debug_assert!(unused.is_empty(), "seed must not drain into output");
        Ok(())
    })?;
    marker_ctx.block_primed = true;

    finish_decode_chunk_contig_native::<true>(
        &mut chunk,
        &mut marker_ctx,
        input,
        first_block_bit,
        stop_hint_bits,
        false,
        true,
    )?;
    Ok(chunk)
}

/// Cross-member continuation deletion-trap: incremented every time the
/// `MULTI_MEMBER=true` contig driver consumes a footer + next header and
/// continues into the following gzip member (`finish_decode_chunk_contig_native`
/// BFINAL arm). Stays 0 on every single-member decode (the continuation is
/// compiled out of the `false` instantiation), so a single-member corpus run
/// asserting this == 0 is the §7 non-execution proof; a multi-member decode
/// asserting it advanced proves the port ran.
pub static MULTI_MEMBER_CONTINUATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// M4 (DIV-1 part 2) — LABELED DEVIATION from the vendor blueprint.
///
/// Vendor's exact-stop path is `decodeChunkWithInflateWrapper<ZlibInflateWrapper/
/// IsalInflateWrapper>` (GzipChunk.hpp:192-265) — a C-FFI inflate wrapper, NOT
/// `deflate::Block`. Putting the until-exact decode on Block-with-exact-stop is
/// justified SOLELY by gzippy-native's no-C-FFI charter (the pure-Rust build has
/// no faithful wrapper engine to hand the chunk to; gzippy-isal keeps the
/// faithful wrapper path untouched, see `exact_block_route_enabled`).
///
/// PRE-REGISTERED CONTRACT (git history (campaign plan, removed) GATE AMENDMENTS §2) —
/// Block must replicate from the `unified::Inflate` wrapper arm
/// (`finish_decode_chunk_impl`, until_exact=true):
///
///  (a) stopping-point reactions: END_OF_BLOCK → every non-final EOB records
///      a block boundary (`append_block_boundary_at`; here at the driver's
///      EOB recording site). END_OF_STREAM_HEADER → on the wrapper arm this
///      stop fires only after `reset_for_next_stream`, which NO production
///      parallel-SM caller invokes (first-hand verified; see
///      inflate_wrapper.rs:1058-1065) — it is unreachable on the until-exact
///      arm, and Block replicates that observable: decode ends at the
///      member's BFINAL EOB with no next-stream continuation.
///  (b) the exact `final_bit != stop_hint_bits => ExactStopMissed` assertion
///      (`finish_decode_chunk_impl`'s final check), same `requested`/`actual`
///      coordinates — enforced in `finish_decode_chunk_contig_native`'s
///      until_exact arm at both stop sites.
///  (c) footer/multi-stream (`read_footer_at_current`/`reset_for_next_stream`):
///      these wrapper APIs are NEVER called by the production until-exact
///      arm — the wrapper stops at the BFINAL EOB (byte-aligned via
///      `align_to_byte`, resumable.rs:823-842) and asserts against
///      `stop_hint_bits` WITHOUT consuming the footer (sm_driver slices the
///      footer off `input`; vendor's wrapper DOES read footers,
///      GzipChunk.hpp:246-251, because its chunks may span members — gzippy's
///      single-member slice cannot). Block replicates the wrapper arm's
///      observable exactly: member-final success iff the byte-aligned
///      post-EOB bit == stop_hint_bits; a member-crossing stop hint errors
///      `ExactStopMissed` identically on both arms (pinned by the
///      `exact_block_parity::exact_multi_member_trailing` net).
///  (d) block-boundary recording (`take_block_boundaries` replay →
///      `append_block_boundary_at`): the driver records every non-final EOB
///      boundary with decoded offsets excluding the dictionary prefix
///      (`data_prefix_len`), key-identical to the wrapper's
///      `decode_base + n_bytes_read` accounting (pinned by the parity nets'
///      subchunk-key equality).
///
/// Route: [`exact_block_route_enabled`] (hardcoded ON; the wrapper arm now
/// runs only on the gzippy-isal build — the `GZIPPY_EXACT_BLOCK=0` env
/// kill-switch was removed 2026-07-07, batch 4f). Engine proof:
/// [`EXACT_BLOCK_CHUNKS`] vs [`EXACT_WRAPPER_CHUNKS`].
#[cfg(parallel_sm)]
fn finish_decode_chunk_exact_block_native(
    chunk: &mut ChunkData,
    input: &[u8],
    inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
) -> Result<(), ChunkDecodeError> {
    EXACT_BLOCK_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let mut marker_ctx = seed_block_for_contig_native(
        chunk,
        input,
        inflate_start_bit,
        stop_hint_bits,
        initial_window,
    )?;

    // STAGE-2d: window-seeded EXACT chunks that cross a member boundary walk
    // into the next member when the grid sets `multi_member`.
    if chunk.configuration.multi_member {
        finish_decode_chunk_contig_native::<true>(
            chunk,
            &mut marker_ctx,
            input,
            inflate_start_bit,
            stop_hint_bits,
            true,
            true,
        )
    } else {
        finish_decode_chunk_contig_native::<false>(
            chunk,
            &mut marker_ctx,
            input,
            inflate_start_bit,
            stop_hint_bits,
            true,
            true,
        )
    }
}

/// Vendor `decodeChunkWithInflateWrapper`-shaped envelope for the M4 Block
/// route: fresh `ChunkData` + window-seeded exact decode + decode-duration
/// accounting (mirror of `decode_chunk_with_inflate_wrapper`'s envelope).
#[cfg(parallel_sm)]
fn decode_chunk_exact_block_native(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    finish_decode_chunk_exact_block_native(
        &mut chunk,
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
    )?;
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    Ok(chunk)
}

/// M3 engine proof: window-seeded INEXACT chunks decoded on the ONE
/// `deflate::Block` engine (`finish_decode_chunk_seeded_block_native`).
#[cfg(parallel_sm)]
pub static SEEDED_BLOCK_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// M3 engine proof (complement): window-seeded INEXACT chunks decoded on the
/// pre-M3 wrapper arm (the gzippy-isal build or the ISA-L measurement oracle;
/// the `GZIPPY_SEEDED_BLOCK=0` env kill-switch was removed 2026-07-07).
#[cfg(parallel_sm)]
pub static SEEDED_WRAPPER_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// M4 engine proof: UNTIL-EXACT chunks decoded on the ONE `deflate::Block`
/// engine (`finish_decode_chunk_exact_block_native`).
#[cfg(parallel_sm)]
pub static EXACT_BLOCK_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// M4 engine proof (complement): UNTIL-EXACT chunks decoded on the wrapper
/// arm (the gzippy-isal build or the ISA-L measurement oracle; the
/// `GZIPPY_EXACT_BLOCK=0` env kill-switch was removed 2026-07-07).
#[cfg(parallel_sm)]
pub static EXACT_WRAPPER_CHUNKS: std::sync::atomic::AtomicU64 =
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
    /// Never constructed on the gzippy-native build (the fold keeps Engine M
    /// decoding in-place); the match arm stays live but unreached.
    #[allow(dead_code)]
    FlipToClean {
        end_bit_offset: usize,
        window_len: usize,
    },
    /// gzippy-native FOLD copy-free-to-final tail. At the ctx-flip point
    /// (`clean_appended_len() >= 32768`) the engine has ALREADY flipped to clean
    /// (`contains_marker_bytes==false`) and ≥32 KiB of contiguous clean output
    /// is in `chunk.data` — the 32 KiB predecessor window is that contiguous
    /// tail. Instead of continuing the ring engine + draining (the
    /// ring->chunk.data memcpy), the driver resumes the SAME thread-local
    /// `Block` and decodes subsequent clean blocks DIRECTLY into `chunk.data`'s
    /// reserved tail via `decode_clean_into_contig` (faithful vendor prepend;
    /// `data_prefix_len` stays 0 because the window is real prior output).
    FlipToContig { end_bit_offset: usize },
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

// Mirror of the `while ( true )` loop in
// `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654), restricted to the
// single-member case (no multi-stream loop) and with the handoff
// triggered exclusively by `cleanDataCount` (GzipChunk.hpp:520-525).

/// Decode one deflate block into `output` (vendor block loop body).
#[cfg(parallel_sm)]
fn marker_decode_step(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    initial_window: &[u8],
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    marker_decode_step_vendor_block(ctx, data, stop_hint_bits, initial_window, output)
}

// The per-thread vendor `deflate::Block` engine, persistent across the
// `marker_decode_step` calls of ONE chunk (primed once via `ctx.block_primed`)
// and reused across chunks (reset on the next chunk's first call). Module-scoped
// so the gzippy-native copy-free-to-final tail (`finish_decode_chunk_contig_native`)
// can re-borrow the SAME engine to continue the post-flip clean decode in-place.
#[cfg(parallel_sm)]
thread_local! {
    static BOOTSTRAP_BLOCK: std::cell::RefCell<crate::decompress::parallel::marker_inflate::Block> =
        std::cell::RefCell::new(crate::decompress::parallel::marker_inflate::Block::new());
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
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

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
            |_e| {},
        )
    })
}

/// Engine surface of the ONE vendor `Block` engine, consumed by
/// `marker_decode_step_loop` (the legacy `MarkerRing` impl was deleted in M5).
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

/// Per-iteration body of the vendor `Block` bootstrap loop.
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

    loop {
        let slice_byte = ctx.current_bit_offset / 8;
        let mut bits = ctx.open_bits(data);
        let next_block_offset = absolute_bit_pos(slice_byte, &bits);

        // Vendor GzipChunk.hpp:520-525 — at 32 KiB of clean u8 the engine FOLDS:
        // Engine M (`marker_inflate::Block`) keeps decoding this and subsequent
        // blocks in-place on the SAME `ctx` cursor. `read()` already drains clean
        // u8 directly (marker_inflate.rs:1011 -> push_clean_u8 once
        // `contains_marker_bytes()==false`). The loop terminates naturally at
        // BFINAL (`Finished`, :1293) or stop_hint (:1222).
        // `ctx.flipped` is set once so this check fires a single time per chunk.
        if output.clean_appended_len() >= MAX_WINDOW_SIZE && !ctx.flipped {
            ctx.flipped = true;
            // gzippy-native: copy-free-to-final. The engine has already flipped
            // (clean_appended only grows post-engine-flip) and ≥32 KiB clean is
            // contiguous in chunk.data; hand the contig tail to the driver, which
            // resumes THIS Block decoding straight into chunk.data (no ring, no
            // drain memcpy). The driver re-borrows the same thread-local Block.
            ctx.current_bit_offset = next_block_offset;
            return Ok((
                MarkerStep::FlipToContig {
                    end_bit_offset: next_block_offset,
                },
                false,
            ));
        }

        let header_res = read_header(block, &mut bits);
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
        while !block.eob() {
            if let Err(e) = read_body(block, &mut bits, output) {
                let bits_at_fail = absolute_bit_pos(slice_byte, &bits);
                let bytes_wasted = output.sink_len() - before_len;
                let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                on_body_fail(&e);
                return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "deflate body at bit {next_block_offset} (failed +{bits_into_body} bits, wasted {bytes_wasted} bytes): {e:?}"
                    ),
                )));
            }
        }

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

    /// P0 REGRESSION (2026-06-12, /tmp/mono-gnu9.tar.gz: deterministic CRC32
    /// mismatch with EXACTLY ONE wrong output byte at 35,335,338 — 'L' for '.').
    ///
    /// Root cause: `emit_backref_ring`'s word-copy rounds the run up to a
    /// multiple of 4 u16; for the FINAL back-ref of a maximally-full `read()`
    /// call (per-call cap `RING_SIZE - MAX_RUN_LENGTH` = 65278, plus up to 257
    /// overshoot ⇒ undrained span up to RING_SIZE-1) the ≤3-slot rounding
    /// overshoot wraps PHYSICALLY onto the OLDEST UNDRAINED slot. The
    /// end-of-call drain then ships the clobbered u16 into
    /// `data_with_markers`. Vendor rapidgzip cannot hit this: its back-ref is
    /// an exact `std::memcpy(..., length * 2)` (deflate.hpp:1376).
    ///
    /// This test reconstructs the EXACT production geometry from the GNU-gzip
    /// monorepo stream, with a hand-built fixed-Huffman deflate stream decoded
    /// window-absent (marker mode) from bit 0:
    ///   * call 1 fills to exactly the cap (65278 single-literal events);
    ///   * periodic marker-propagating back-refs keep `distance_to_last_marker`
    ///     below the 32 Ki flip arming (mirrors the real all-marker chunk);
    ///   * call 2 emits 65277 literals, then a distance-1000 length-258
    ///     back-ref: span + rounded = 65277 + 260 = 65537 > RING_SIZE — the
    ///     overshoot's final slot aliases call 2's FIRST output byte.
    /// Pre-fix: `data_with_markers[65278]` = the u16 at source+259 (wrong).
    /// Post-fix: byte-exact against the reference model.
    #[test]
    fn marker_word_copy_overshoot_does_not_clobber_undrained_output() {
        use crate::decompress::parallel::marker_inflate::RING_SIZE;

        // ── Tiny fixed-Huffman deflate bit-writer ────────────────────────
        struct Bw {
            bytes: Vec<u8>,
            cur: u64,
            n: u32,
        }
        impl Bw {
            fn lsb(&mut self, v: u64, n: u32) {
                self.cur |= v << self.n;
                self.n += n;
                while self.n >= 8 {
                    self.bytes.push((self.cur & 0xFF) as u8);
                    self.cur >>= 8;
                    self.n -= 8;
                }
            }
            // Huffman codes are written MSB-first (RFC 1951 §3.1.1).
            fn code(&mut self, c: u32, n: u32) {
                for i in (0..n).rev() {
                    self.lsb(((c >> i) & 1) as u64, 1);
                }
            }
            fn finish(mut self) -> Vec<u8> {
                if self.n > 0 {
                    self.bytes.push((self.cur & 0xFF) as u8);
                }
                self.bytes
            }
        }
        fn lit(bw: &mut Bw, v: u8) {
            assert!(v < 144, "fixed-Huffman 8-bit literal range");
            bw.code(0x30 + v as u32, 8);
        }
        fn backref(bw: &mut Bw, dist: usize, len: usize) {
            match len {
                3 => bw.code(1, 7),      // sym 257: 7-bit code 0000001
                258 => bw.code(0xC5, 8), // sym 285: 8-bit code 11000101
                _ => panic!("unsupported test length"),
            }
            let (dsym, base, extra) = match dist {
                769..=1024 => (19u32, 769usize, 8u32),
                24577..=32768 => (29u32, 24577usize, 13u32),
                _ => panic!("unsupported test distance"),
            };
            bw.code(dsym, 5);
            if extra > 0 {
                bw.lsb((dist - base) as u64, extra);
            }
        }

        // ── Event plan (output positions) ────────────────────────────────
        #[derive(Clone, Copy)]
        enum Ev {
            Lit(u8),
            Back(usize, usize),
        }
        let p = |i: usize| (i % 140) as u8;
        let mut events: Vec<Ev> = Vec::new();
        let mut out_len = 0usize;
        let lits_to = |events: &mut Vec<Ev>, out_len: &mut usize, upto: usize| {
            while *out_len < upto {
                events.push(Ev::Lit(p(*out_len)));
                *out_len += 1;
            }
        };
        // Markers at 0..3 (distance > position ⇒ MapMarkers values), then
        // re-propagated every 30000 bytes so the clean-run counter never arms
        // the flip (mirrors the production all-marker chunk).
        events.push(Ev::Back(1000, 3));
        out_len += 3;
        for stop in [30_000usize, 60_000, 90_000, 120_000] {
            lits_to(&mut events, &mut out_len, stop);
            events.push(Ev::Back(30_000, 3));
            out_len += 3;
        }
        // Call 1 = [0, 65278) (cap; all single-literal events near the cap).
        // Call 2 = [65278, ...): 65277 literals → final event starts at
        // emitted = 65277 = cap - 1.
        lits_to(&mut events, &mut out_len, 130_555);
        events.push(Ev::Back(1000, 258)); // ← the clobbering back-ref
        out_len += 258;
        let total = out_len;
        assert_eq!(total, 130_813);

        // ── Encode: one fixed-Huffman BFINAL block ───────────────────────
        let mut bw = Bw {
            bytes: Vec::new(),
            cur: 0,
            n: 0,
        };
        bw.lsb(1, 1); // BFINAL
        bw.lsb(1, 2); // BTYPE = 01 (fixed)
        for ev in &events {
            match *ev {
                Ev::Lit(v) => lit(&mut bw, v),
                Ev::Back(d, l) => backref(&mut bw, d, l),
            }
        }
        bw.code(0, 7); // EOB (sym 256)
        let deflate = bw.finish();

        // ── Reference model in marker-u16 space ──────────────────────────
        // distance > position ⇒ pre-init marker zone value RING_SIZE-(d-p)
        // (width_ring.rs `init_marker_zone`); markers propagate via copies.
        let mut expect: Vec<u16> = Vec::with_capacity(total);
        for ev in &events {
            match *ev {
                Ev::Lit(v) => expect.push(v as u16),
                Ev::Back(d, l) => {
                    for _ in 0..l {
                        let pp = expect.len();
                        if d > pp {
                            expect.push((RING_SIZE - (d - pp)) as u16);
                        } else {
                            let v = expect[pp - d];
                            expect.push(v);
                        }
                    }
                }
            }
        }

        // ── Decode window-absent from bit 0 (production speculative path) ─
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 64 * 1024 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let chunk = decode_chunk_window_absent(&deflate, 0, deflate.len() * 8, cfg)
            .expect("window-absent decode");
        // The drain routes output AFTER the last marker (120002) to
        // `chunk.data` as clean u8; everything up to and including it stays
        // in `data_with_markers` (the production split — the real chunk had
        // marker_bytes = decoded - 1 for the same reason).
        let dwm_len = chunk.data_with_markers.len();
        assert_eq!(
            dwm_len, 120_003,
            "marker-mode output must cover through the last marker (no early flip)"
        );
        let mut mismatches = Vec::new();
        for (i, &want) in expect.iter().enumerate().take(dwm_len) {
            let got = chunk.data_with_markers.at(i);
            if got != want {
                mismatches.push((i, got, want));
                if mismatches.len() > 8 {
                    break;
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "u16 output mismatches (idx, got, want): {mismatches:?} — the back-ref \
             word-copy rounding overshoot clobbered undrained output (mono-gnu9 P0)"
        );
        // Clean tail: the post-last-marker bytes, including the final
        // 258-length back-ref's output.
        let tail = chunk.data.to_contiguous();
        assert_eq!(tail.len(), total - dwm_len, "clean-tail length");
        for (k, &b) in tail.iter().enumerate() {
            let want = expect[dwm_len + k];
            assert!(want < 256, "tail must be clean in the reference model");
            assert_eq!(
                b,
                want as u8,
                "clean-tail byte {k} (logical {}) wrong",
                dwm_len + k
            );
        }
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

/// NATIVE fold gate (gzippy-native).
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
/// starts at-or-past stop_hint (the faithful rapidgzip behavior). The retired
/// pre-fold two-phase tail (`StreamingInflateWrapper`) had an
/// ISA-L-emulation rewind that can keep/skip one fixed-vs-dynamic block at a
/// header straddle (chunk_decode.rs:481-486). That is a *speculative* stop-point
/// difference only — the consumer reconciles it exactly via `furthest_decoded_
/// bit` and `block_finder.insert(chunk_end_bit)` (chunk_fetcher.rs:1074, 2663),
/// so concatenated output is byte-identical either way (proven end-to-end by
/// the silesia DUAL-SHA gate). Hence this gate asserts correctness against
/// ground truth, not stop-point equality with the retired two-phase tail.
#[cfg(all(test, parallel_sm))]
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
        // Real-corpus fixture, large (~67 MiB) and not committed — present on
        // bench boxes / the owner tree but absent in a fresh worktree. Skip
        // gracefully when missing (same convention as
        // `three_oracle_silesia_if_available`) rather than hard-failing on a
        // tree that simply hasn't fetched the corpus. When the fixture IS
        // present every byte-exact ground-truth assertion below runs in full.
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[fold-gate] benchmark_data/silesia-gzip.tar.gz not present, skipping");
                return;
            }
        };
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

    // ── Stage-2 copy-free-to-final owed cases (advisor D + landmine) ─────────
    //
    // These drive the REAL production native contig tail
    // (`finish_decode_chunk_contig_native`) via `decode_chunk_window_absent` on
    // a window-ABSENT stream that flips at 32 KiB, then assert byte-exactness
    // against an independent flate2 decode of the same payload.

    /// Build a raw-deflate stream from `payload` and decode the FIRST chunk
    /// (window-absent, start_bit 0, stop_hint = end) via the production native
    /// path. Returns (resolved_full_bytes, chunk.data_prefix_len, decoded_crc).
    /// Since start is the stream head there is NO predecessor window, so the
    /// flip resolves entirely from in-chunk output (data_prefix_len stays 0).
    fn decode_first_chunk_native(deflate: &[u8]) -> (Vec<u8>, usize, u32) {
        let cfg = ChunkConfiguration::default();
        let stop = deflate.len() * 8;
        let mut chunk = super::decode_chunk_window_absent(deflate, 0, stop, cfg)
            .expect("native window-absent decode");
        let prefix_len = chunk.data_prefix_len;
        // Resolve any pre-flip markers against an EMPTY window (head of stream
        // has no predecessor — back-refs cannot legally reach before byte 0).
        let empty = [0u8; WINDOW];
        chunk.resolve_and_narrow_markers_in_place(&empty);
        chunk.merge_resolved_markers_into_data();
        let full = chunk.data.to_contiguous();
        let crc = crc32_of(&full[prefix_len..]);
        (full[prefix_len..].to_vec(), prefix_len, crc)
    }

    fn flate2_inflate(deflate: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(deflate)
            .read_to_end(&mut out)
            .expect("flate2 inflate");
        out
    }

    fn deflate_of(payload: &[u8], level: u32) -> Vec<u8> {
        use std::io::Write;
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// MULTI-BLOCK clean phase across deflate-block boundaries + CRC-prefix
    /// exclusion. A multi-MiB compressible payload forces a flip at 32 KiB and
    /// then many post-flip blocks; the contig tail must continue across each
    /// EOB (read_header + resume with the same `*pos`). data_prefix_len MUST be
    /// 0 (faithful prepend uses real prior output, no imported window image), so
    /// CRC covers only real output.
    #[test]
    fn contig_native_multiblock_clean_and_crc_prefix_excluded() {
        // Mildly compressible, several MiB → flips early, many clean blocks.
        let mut payload = Vec::with_capacity(4 * 1024 * 1024);
        let mut x: u32 = 0x1234_5678;
        for i in 0..(4 * 1024 * 1024u32) {
            // Repetitive-with-noise so it compresses (forces back-refs) but
            // spans many blocks.
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            payload.push(((i / 64) as u8) ^ ((x >> 24) as u8 & 0x07));
        }
        let deflate = deflate_of(&payload, 6);
        let (got, prefix_len, crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0, "FOLD must keep data_prefix_len == 0");
        assert_eq!(got, payload, "multi-block clean tail bytes diverged");
        assert_eq!(
            crc,
            crc32_of(&payload),
            "CRC over real output (prefix excluded) wrong"
        );
    }

    /// REGROW past the 16 MiB reserve clamp. A >16 MiB clean payload forces the
    /// contig buffer to grow BETWEEN calls (Engine-C contract); the loop must
    /// re-fetch `base`/`cap` after each grow (no stale pointer / no OOB).
    #[test]
    fn contig_native_regrow_past_reserve_clamp() {
        // 20 MiB highly-compressible (long RLE-ish runs) → one chunk, decoded
        // size far exceeds the 16 MiB RESERVE_CLAMP, driving real regrows.
        let block = b"The quick brown fox jumps over the lazy dog. 0123456789ABCDEF\n";
        let target = 20 * 1024 * 1024usize;
        let mut payload = Vec::with_capacity(target + block.len());
        while payload.len() < target {
            payload.extend_from_slice(block);
        }
        let deflate = deflate_of(&payload, 6);
        let (got, prefix_len, crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0);
        assert_eq!(got.len(), payload.len(), "regrow truncated/extended output");
        assert!(got == payload, "regrow corrupted output bytes");
        assert_eq!(crc, crc32_of(&payload));
    }

    /// STORED (uncompressed) block AFTER the flip. The contig primitive can't
    /// decode a stored block (returns InvalidCompression); the native tail must
    /// route it through `decode_clean_stored_into_contig`. Build: a compressible
    /// lead (forces a flip) followed by an incompressible (random) tail that the
    /// encoder emits as a STORED block.
    #[test]
    fn contig_native_stored_block_after_flip() {
        // Compressible lead well over 32 KiB to guarantee the flip.
        let mut payload = Vec::new();
        for i in 0..200_000u32 {
            payload.push((i / 97) as u8);
        }
        // Incompressible tail (LCG random) — flate2 at level 0 stores it; even at
        // higher levels a random run yields stored blocks. Use level 0 to FORCE
        // stored blocks across the whole stream (which includes post-flip stored
        // blocks once the 32 KiB clean window is established).
        let mut x: u32 = 0xDEAD_BEEF;
        for _ in 0..300_000u32 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            payload.push((x >> 24) as u8);
        }
        // Level 0 = all stored blocks → guarantees the contig tail must handle
        // stored blocks post-flip (the flip can fire mid stored-stream once
        // 32 KiB of clean literals accumulate).
        let deflate = deflate_of(&payload, 0);
        let truth = flate2_inflate(&deflate);
        assert_eq!(truth, payload, "fixture self-check");
        let (got, prefix_len, _crc) = decode_first_chunk_native(&deflate);
        assert_eq!(prefix_len, 0);
        assert_eq!(got, payload, "stored-block-after-flip diverged");
    }
}

// =========================================================================
// M3 differential gate (DIV-1 part 1): seeded-Block vs seeded-wrapper
// =========================================================================
// For window-seeded INEXACT chunks the gzippy-native production route moved
// from `StreamingInflateWrapper`/`unified::Inflate` onto the ONE
// `deflate::Block` engine (vendor GzipChunk.hpp:454-458). This gate nets the
// two arms — `finish_decode_chunk_seeded_block_native` (new production) vs
// `finish_decode_chunk_with_inexact_offset` (the pre-M3 wrapper arm; its
// `GZIPPY_SEEDED_BLOCK=0` env kill-switch was removed 2026-07-07) — on
// generated corpora, asserting:
//   (a) decoded bytes        (b) decoded_size / data_prefix_len accounting
//   (c) final bit (encoded_size_bits)   (d) per-stream CRC32 values
//   (e) subchunk keys (encoded_offset, encoded_size, decoded_offset, size)
//   (f) published windows: get_last_window / last_32kib_window /
//       per-subchunk windows (populate_subchunk_windows), plus a brute-force
//       window check against `pred ‖ payload` (the stale-window-key scar net:
//       trailing/last-window content must equal ground truth, not just match
//       the other arm).
#[cfg(all(test, parallel_sm))]
mod seeded_block_parity {
    use super::*;

    const WINDOW: usize = 32 * 1024;

    /// Raw DEFLATE of `payload` with a 32 KiB preset dictionary so the
    /// encoder emits back-refs reaching into the predecessor window.
    pub(super) fn deflate_with_dict(payload: &[u8], dict: &[u8], level: u32) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress};
        let mut c = Compress::new(Compression::new(level), false);
        c.set_dictionary(dict).expect("set_dictionary");
        let mut out = vec![0u8; payload.len() * 2 + 4096];
        let status = c
            .compress(payload, &mut out, FlushCompress::Finish)
            .expect("compress");
        assert_eq!(status, flate2::Status::StreamEnd, "deflate did not finish");
        out.truncate(c.total_out() as usize);
        out
    }

    /// Raw DEFLATE with a preset dictionary and a SYNC FLUSH every
    /// `flush_every` payload bytes (empty stored blocks + dense boundaries).
    pub(super) fn deflate_with_dict_flushes(
        payload: &[u8],
        dict: &[u8],
        level: u32,
        flush_every: usize,
    ) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress};
        let mut c = Compress::new(Compression::new(level), false);
        c.set_dictionary(dict).expect("set_dictionary");
        let mut out: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; payload.len() + 64 * 1024];
        let mut fed = 0usize;
        loop {
            let end = (fed + flush_every).min(payload.len());
            let flush = if end == payload.len() {
                FlushCompress::Finish
            } else {
                FlushCompress::Sync
            };
            let before_out = c.total_out() as usize;
            let status = c
                .compress(&payload[fed..end], &mut buf, flush)
                .expect("compress");
            out.extend_from_slice(&buf[..c.total_out() as usize - before_out]);
            fed = c.total_in() as usize;
            if end == payload.len() && status == flate2::Status::StreamEnd {
                break;
            }
        }
        out
    }

    /// 32 KiB deterministic text-like dictionary.
    pub(super) fn make_dict() -> Vec<u8> {
        let mut d = b"the quick brown fox jumps over the lazy dog. ".repeat(800);
        d.truncate(WINDOW.max(32768));
        let cut = d.len() - WINDOW;
        d[cut..].to_vec()
    }

    pub(super) fn lcg_bytes(n: usize, mut x: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((x >> 24) as u8);
        }
        v
    }

    pub(super) fn flate2_inflate_with_dict(deflate: &[u8], dict: &[u8]) -> Vec<u8> {
        use flate2::{Decompress, FlushDecompress};
        let mut d = Decompress::new(false);
        d.set_dictionary(dict).expect("set_dictionary");
        let mut out = vec![0u8; 64 * 1024 * 1024];
        let status = d
            .decompress(deflate, &mut out, FlushDecompress::Finish)
            .expect("inflate");
        assert!(
            matches!(status, flate2::Status::StreamEnd),
            "truth inflate did not reach stream end"
        );
        out.truncate(d.total_out() as usize);
        out
    }

    /// NEW production arm: ONE Block engine, seeded.
    fn arm_block(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> ChunkData {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_seeded_block_native(&mut chunk, input, 0, stop_hint_bits, window)
            .expect("seeded Block decode");
        chunk
    }

    /// Kill-switch arm: pre-M3 wrapper path, byte/key reference.
    fn arm_wrapper(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> ChunkData {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_with_inexact_offset(
            &mut chunk,
            input,
            0,
            stop_hint_bits,
            window,
            false,
        )
        .expect("seeded wrapper decode");
        chunk
    }

    /// Full cross-arm equality net (a)-(f). `truth` = full-stream ground truth
    /// (decoded bytes must be a prefix of it; equal when stop = stream end).
    fn assert_arms_equal(
        mut b: ChunkData,
        mut w: ChunkData,
        pred: &[u8],
        truth: &[u8],
        total_bits: usize,
        label: &str,
    ) {
        // (b) prefix accounting: Block arm carries the dictionary prefix.
        assert_eq!(b.data_prefix_len, WINDOW, "{label}: Block arm prefix len");
        assert_eq!(w.data_prefix_len, 0, "{label}: wrapper arm prefix len");
        assert!(
            b.data_with_markers.is_empty() && w.data_with_markers.is_empty(),
            "{label}: seeded decode must emit no markers"
        );

        // (a) decoded bytes.
        let bb = b.data.to_contiguous()[WINDOW..].to_vec();
        let wb = w.data.to_contiguous();
        if bb != wb {
            let first_diff = bb.iter().zip(wb.iter()).position(|(x, y)| x != y);
            panic!(
                "{label}: decoded bytes diverged (block_len={} wrapper_len={} first_diff={first_diff:?})",
                bb.len(),
                wb.len()
            );
        }
        assert!(!bb.is_empty(), "{label}: decoded nothing");
        assert!(
            bb.len() <= truth.len() && bb[..] == truth[..bb.len()],
            "{label}: decoded bytes are not a prefix of ground truth"
        );
        assert_eq!(b.decoded_size(), w.decoded_size(), "{label}: decoded_size");
        assert_eq!(
            b.decoded_size(),
            bb.len(),
            "{label}: decoded_size accounting"
        );

        // (c) final bit. Strict equality, with exactly TWO measured, documented
        // exceptions where the WRAPPER (kill-switch arm) semantics differ:
        //
        //   1. STREAM END (BFINAL decoded; bb == truth): the wrapper reports
        //      `tell_compressed()` after `finished` — the byte-aligned input
        //      end (= total_bits; the <=7 padding bits are consumed). The
        //      Block arm reports the exact bit after the final EOB symbol —
        //      identical to the production-proven native FOLD semantics
        //      (`finish_decode_chunk_contig_native` / `MarkerStep::Finished`).
        //
        //   2. WRAPPER ACCOUNTING HOLE (pre-existing, found by this gate): if
        //      the inexact stop hint lands BETWEEN an EOB and the engine's
        //      next END_OF_BLOCK_HEADER stop, the coalescing engine never
        //      surfaces an END_OF_BLOCK stop (interior EOBs are reported only
        //      via take_block_boundaries), so `last_eob_pos` keeps its init
        //      value (`inflate_start_bit`) and the header-arm stop finalizes
        //      at `last_end_bit = last_eob_pos = chunk start` →
        //      `encoded_size_bits == 0` despite all bytes being emitted. The
        //      Block arm reports the contract value (first block boundary
        //      at-or-past the hint). If the wrapper hole is ever fixed, the
        //      `wrapper_final == 0` guard below fails loudly — remove the
        //      exception then.
        let full_stream = bb.len() == truth.len();
        let block_final = b.encoded_offset_bits + b.encoded_size_bits;
        let wrapper_final = w.encoded_offset_bits + w.encoded_size_bits;
        let final_bit_exception = if b.encoded_size_bits == w.encoded_size_bits {
            None
        } else if full_stream && wrapper_final == total_bits && wrapper_final - block_final < 8 {
            eprintln!(
                "[m3-gate] {label}: stream-end final-bit policy (block exact-EOB {block_final}, wrapper byte-aligned {wrapper_final})"
            );
            Some("stream_end")
        } else if w.encoded_size_bits == 0 && b.encoded_size_bits > 0 {
            eprintln!(
                "[m3-gate] {label}: wrapper final-bit accounting hole (wrapper 0, block {block_final}) — pre-existing wrapper bug, Block arm reports the contract value"
            );
            Some("wrapper_hole")
        } else {
            panic!(
                "{label}: final bit diverged outside the two documented policies (block {block_final}, wrapper {wrapper_final})"
            );
        };

        // (d) CRCs.
        assert_eq!(b.crc32s.len(), w.crc32s.len(), "{label}: crc32s count");
        for (i, (x, y)) in b.crc32s.iter().zip(w.crc32s.iter()).enumerate() {
            assert_eq!(x.crc32(), y.crc32(), "{label}: crc32s[{i}]");
        }
        assert_eq!(
            b.crc32s[0].crc32(),
            crate::decompress::parallel::crc32::crc32(&bb),
            "{label}: CRC accumulator != crc32(decoded)"
        );

        // (e) subchunk keys. The trailing subchunk's `encoded_size_bits` is
        // derived from the chunk-final bit (`finalize_with_deflate`), so under
        // a documented final-bit exception it differs by exactly the same
        // mechanism — compare it via per-arm self-consistency instead; all
        // other key fields stay strictly equal.
        let keys = |c: &ChunkData| {
            let n = c.subchunks.len();
            c.subchunks
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let trailing = i + 1 == n;
                    (
                        s.encoded_offset_bits,
                        if trailing && final_bit_exception.is_some() {
                            0 // normalized; checked via self-consistency below
                        } else {
                            s.encoded_size_bits
                        },
                        s.decoded_offset,
                        s.decoded_size,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&b), keys(&w), "{label}: subchunk keys");
        for (c, final_bit, arm) in [(&b, block_final, "block"), (&w, wrapper_final, "wrapper")] {
            if let Some(last) = c.subchunks.last() {
                assert_eq!(
                    last.encoded_offset_bits + last.encoded_size_bits,
                    final_bit,
                    "{label}: {arm} trailing subchunk encoded_size inconsistent with final bit"
                );
            }
        }

        // (f) published windows: cross-arm AND vs brute-force ground truth
        // (`pred ‖ decoded`) — the stale-window-key scar net.
        let mut hist = Vec::with_capacity(pred.len() + bb.len());
        hist.extend_from_slice(pred);
        hist.extend_from_slice(&bb);
        let brute = |chunk_off: usize| -> Vec<u8> {
            let end = pred.len() + chunk_off;
            hist[end - WINDOW..end].to_vec()
        };

        let lb = b.get_last_window_vec(pred);
        let lw = w.get_last_window_vec(pred);
        assert_eq!(lb, lw, "{label}: get_last_window");
        assert_eq!(
            lb,
            brute(bb.len()),
            "{label}: get_last_window vs brute force"
        );
        assert_eq!(
            b.last_32kib_window_vec(),
            w.last_32kib_window_vec(),
            "{label}: last_32kib_window"
        );

        b.populate_subchunk_windows(pred);
        w.populate_subchunk_windows(pred);
        assert_eq!(
            b.subchunks.len(),
            w.subchunks.len(),
            "{label}: subchunk count"
        );
        for (i, (sb, sw)) in b.subchunks.iter().zip(w.subchunks.iter()).enumerate() {
            let wbts = sb.window.as_ref().map(|v| v.decompress());
            let wwts = sw.window.as_ref().map(|v| v.decompress());
            // Cross-arm window equality — strict, except under the documented
            // wrapper accounting hole, where the wrapper's trailing
            // `encoded_size_bits == 0` makes its sparsity pass
            // (`get_used_window_symbols`) scan from the WRONG bit (chunk
            // start instead of the real continuation bit), mis-zeroing its
            // window. The Block arm's window is still netted against the
            // brute-force ground truth below.
            if final_bit_exception != Some("wrapper_hole") {
                assert_eq!(wbts, wwts, "{label}: subchunk[{i}] window bytes");
            }
            // Sparsity may zero unused symbols (identically in both arms, since
            // keys are equal) — so only check the NON-sparsified brute window
            // when sparsity left the window intact.
            if let Some(got) = &wbts {
                let bf = brute(sb.decoded_offset);
                if sb.used_window_symbols.is_empty() && got != &bf {
                    // Window was sparsified (zeros at unused offsets) or truly
                    // wrong. Verify every NON-zero byte matches ground truth —
                    // a stale/shifted window would mismatch on non-zeros too.
                    assert_eq!(got.len(), bf.len(), "{label}: subchunk[{i}] window len");
                    for (j, (g, t)) in got.iter().zip(bf.iter()).enumerate() {
                        assert!(
                            *g == 0 || g == t,
                            "{label}: subchunk[{i}] window byte {j} stale (got {g}, truth {t})"
                        );
                    }
                }
            }
        }
    }

    fn run_case(payload: &[u8], deflate: &[u8], dict: &[u8], cfg: ChunkConfiguration, label: &str) {
        let truth = flate2_inflate_with_dict(deflate, dict);
        assert_eq!(truth, payload, "{label}: fixture self-check");
        let stop = deflate.len() * 8;
        let b = arm_block(deflate, stop, dict, cfg);
        let w = arm_wrapper(deflate, stop, dict, cfg);
        assert_arms_equal(b, w, dict, &truth, deflate.len() * 8, label);
    }

    #[test]
    fn seeded_dynamic_heavy_multi_subchunk() {
        // Text-like, dynamic-Huffman-heavy, > several split thresholds so the
        // subchunk-split machinery runs (the silesia-T4-class key zone).
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..8 * 1024]); // dict back-refs up front
        for i in 0..6000u32 {
            payload.extend_from_slice(
                format!(
                    "line {i}: the quick brown fox jumps over the lazy dog #{}\n",
                    i % 97
                )
                .as_bytes(),
            );
        }
        payload.extend_from_slice(&dict[16 * 1024..24 * 1024]); // mid-dict refs late
                                                                // Flush every 16 KiB so the stream carries REAL deflate block
                                                                // boundaries (zlib otherwise emits very large blocks), letting the
                                                                // 48 KiB split threshold produce a multi-subchunk shape.
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 16 * 1024);
        let cfg = ChunkConfiguration {
            split_chunk_size: 48 * 1024, // force multiple subchunks locally
            ..ChunkConfiguration::default()
        };
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let stop = deflate.len() * 8;
        let b = arm_block(&deflate, stop, &dict, cfg);
        assert!(
            b.subchunks.len() >= 3,
            "expected a multi-subchunk shape, got {}",
            b.subchunks.len()
        );
        let w = arm_wrapper(&deflate, stop, &dict, cfg);
        assert_arms_equal(b, w, &dict, &truth, stop, "dynamic_heavy");
    }

    #[test]
    fn seeded_stored_mixed() {
        // Alternating compressible / incompressible 24 KiB segments at level 1
        // → mixed STORED + Huffman blocks (zlib stores incompressible runs).
        let dict = make_dict();
        let mut payload = Vec::new();
        for k in 0..8usize {
            if k % 2 == 0 {
                payload.extend_from_slice(&b"compressible compressible! ".repeat(900));
            } else {
                payload.extend_from_slice(&lcg_bytes(24 * 1024, 0xC0FF_EE00 + k as u32));
            }
        }
        let deflate = deflate_with_dict(&payload, &dict, 1);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "stored_mixed_l1",
        );

        // Level 0: ALL stored blocks (the pure `decode_clean_stored_into_contig`
        // route on the Block arm).
        let deflate0 = deflate_with_dict(&payload, &dict, 0);
        run_case(
            &payload,
            &deflate0,
            &dict,
            ChunkConfiguration::default(),
            "stored_only_l0",
        );
    }

    #[test]
    fn seeded_flush_dense() {
        // SYNC flush every 2 KiB → dense empty stored blocks + boundaries.
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[1..4097]); // offset-1 dict ref shape
        for i in 0..2000u32 {
            payload.extend_from_slice(format!("flushy line {i} {}\n", i % 13).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 2048);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "flush_dense",
        );
    }

    #[test]
    fn seeded_window_boundary_backrefs() {
        // Payload BEGINS with a copy of the dictionary head → the encoder emits
        // a distance-32768 back-ref at output offset 0 (and offset-1 variants),
        // crossing the contig prefix seam at its extreme reach.
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..4096]); // distance == 32768 at offset 0
        payload.extend_from_slice(b"X");
        payload.extend_from_slice(&dict[..4096]); // re-reference after 1 byte
        payload.extend_from_slice(&lcg_bytes(8 * 1024, 0xBEEF_CAFE));
        payload.extend_from_slice(&dict[WINDOW - 4096..]); // dict tail refs
        let deflate = deflate_with_dict(&payload, &dict, 9);
        run_case(
            &payload,
            &deflate,
            &dict,
            ChunkConfiguration::default(),
            "window_boundary",
        );
    }

    #[test]
    fn seeded_stop_hint_parity() {
        // Inexact stop hints: at a real block boundary, just before one, and
        // mid-block — both arms must stop at the SAME first boundary >= hint.
        let dict = make_dict();
        let mut payload = Vec::new();
        for i in 0..4000u32 {
            payload.extend_from_slice(format!("stop-hint line {i} {}\n", i % 31).as_bytes());
        }
        // Dense flushes give plenty of in-stream boundaries to aim at.
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 4096);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration::default();

        // Take boundary positions from a wrapper enumeration pass.
        let total_bits = deflate.len() * 8;
        let probe = arm_wrapper(&deflate, total_bits / 2, &dict, cfg);
        let mid_stop = probe.encoded_offset_bits + probe.encoded_size_bits;
        assert!(
            mid_stop > 0 && mid_stop < total_bits,
            "probe stop in-stream"
        );

        for (delta, name) in [
            (0isize, "at_boundary"),
            (-3, "before_boundary"),
            (5, "past_boundary"),
        ] {
            let hint = (mid_stop as isize + delta) as usize;
            let b = arm_block(&deflate, hint, &dict, cfg);
            let w = arm_wrapper(&deflate, hint, &dict, cfg);
            eprintln!(
                "[m3-gate] stop_hint_{name}: hint={hint} block(final={}, decoded={}, subchunks={}) wrapper(final={}, decoded={}, subchunks={})",
                b.encoded_size_bits,
                b.decoded_size(),
                b.subchunks.len(),
                w.encoded_size_bits,
                w.decoded_size(),
                w.subchunks.len()
            );
            assert_arms_equal(
                b,
                w,
                &dict,
                &truth,
                total_bits,
                &format!("stop_hint_{name}"),
            );
        }
    }

    #[test]
    fn seeded_block_counter_increments() {
        // Engine-proof counter: the Block arm increments SEEDED_BLOCK_CHUNKS.
        let dict = make_dict();
        let payload = b"counter payload ".repeat(1024);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let before = SEEDED_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let _ = arm_block(
            &deflate,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
        );
        let after = SEEDED_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after > before, "SEEDED_BLOCK_CHUNKS did not increment");
    }
}

// =========================================================================
// M4 differential gate (DIV-1 part 2): exact-Block vs exact-wrapper
// =========================================================================
// For window-seeded UNTIL-EXACT chunks the gzippy-native production route
// moved from `StreamingInflateWrapper`/`unified::Inflate` onto the ONE
// `deflate::Block` engine (LABELED DEVIATION — vendor's exact path is the
// C-FFI `decodeChunkWithInflateWrapper`, GzipChunk.hpp:192-265; see
// `finish_decode_chunk_exact_block_native`). This gate nets the two arms —
// `finish_decode_chunk_exact_block_native` (new production) vs
// `finish_decode_chunk_impl(until_exact=true)` (the pre-M4 wrapper arm; its
// `GZIPPY_EXACT_BLOCK=0` env kill-switch was removed 2026-07-07) — on
// generated corpora, asserting STRICT equality (no
// M3-style final-bit exceptions: on success both arms must land EXACTLY at
// stop_hint_bits by the until-exact contract):
//   (a) decoded bytes      (b) decoded_size / data_prefix_len accounting
//   (c) final bit == stop_hint_bits on BOTH arms
//   (d) per-stream CRC32 values
//   (e) subchunk keys (encoded_offset, encoded_size, decoded_offset, size)
//   (f) published windows incl. brute-force `pred ‖ payload` ground truth
//   (g) ERROR equality: when the stop cannot be honored (member-final
//       misaligned hint, multi-member-crossing hint) both arms must return
//       ExactStopMissed with IDENTICAL requested/actual coordinates.
#[cfg(all(test, parallel_sm))]
mod exact_block_parity {
    use super::seeded_block_parity::{
        deflate_with_dict, deflate_with_dict_flushes, flate2_inflate_with_dict, lcg_bytes,
        make_dict,
    };
    use super::*;

    const WINDOW: usize = 32 * 1024;

    /// NEW production arm: ONE Block engine, seeded, exact stop.
    fn arm_block_exact(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> Result<ChunkData, ChunkDecodeError> {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_exact_block_native(&mut chunk, input, 0, stop_hint_bits, window)
            .map(|()| chunk)
    }

    /// Kill-switch arm: pre-M4 wrapper path (`finish_decode_chunk_impl`
    /// with `until_exact=true`), byte/key/error reference.
    fn arm_wrapper_exact(
        input: &[u8],
        stop_hint_bits: usize,
        window: &[u8],
        cfg: ChunkConfiguration,
    ) -> Result<ChunkData, ChunkDecodeError> {
        let mut chunk = ChunkData::new(0, cfg);
        finish_decode_chunk_impl(&mut chunk, input, 0, stop_hint_bits, window, false, true)
            .map(|()| chunk)
    }

    /// Enumerate non-final EOB boundaries `(bit, decoded_offset)` via the
    /// long-vetted wrapper engine (independent of both arms' stop logic).
    fn enumerate_boundaries(input: &[u8], dict: &[u8]) -> Vec<(usize, usize)> {
        let mut w = StreamingInflateWrapper::with_until_bits(input, 0, input.len() * 8)
            .expect("wrapper construct");
        w.set_window(dict).expect("set_window");
        w.set_stopping_points(StoppingPoints::END_OF_BLOCK);
        let mut out = vec![0u8; 1 << 20];
        let mut total = 0usize;
        let mut bounds = Vec::new();
        loop {
            let r = w.read_stream(&mut out).expect("read_stream");
            total += r.bytes_written;
            if r.finished {
                break;
            }
            if r.stopped_at == StoppingPoints::END_OF_BLOCK && !w.is_final_block() {
                bounds.push((r.bit_position, total));
                continue;
            }
            if r.stopped_at == StoppingPoints::NONE && r.bytes_written == 0 {
                break;
            }
        }
        bounds
    }

    /// Strict cross-arm equality net (a)-(f) for SUCCESSFUL exact stops.
    fn assert_arms_equal_exact(
        mut b: ChunkData,
        mut w: ChunkData,
        pred: &[u8],
        truth: &[u8],
        stop_hint_bits: usize,
        label: &str,
    ) {
        // (b) prefix accounting: Block arm carries the dictionary prefix.
        assert_eq!(b.data_prefix_len, WINDOW, "{label}: Block arm prefix len");
        assert_eq!(w.data_prefix_len, 0, "{label}: wrapper arm prefix len");
        assert!(
            b.data_with_markers.is_empty() && w.data_with_markers.is_empty(),
            "{label}: exact decode must emit no markers"
        );

        // (a) decoded bytes.
        let bb = b.data.to_contiguous()[WINDOW..].to_vec();
        let wb = w.data.to_contiguous();
        if bb != wb {
            let first_diff = bb.iter().zip(wb.iter()).position(|(x, y)| x != y);
            panic!(
                "{label}: decoded bytes diverged (block_len={} wrapper_len={} first_diff={first_diff:?})",
                bb.len(),
                wb.len()
            );
        }
        assert!(!bb.is_empty(), "{label}: decoded nothing");
        assert!(
            bb.len() <= truth.len() && bb[..] == truth[..bb.len()],
            "{label}: decoded bytes are not a prefix of ground truth"
        );
        assert_eq!(b.decoded_size(), w.decoded_size(), "{label}: decoded_size");
        assert_eq!(
            b.decoded_size(),
            bb.len(),
            "{label}: decoded_size accounting"
        );

        // (c) final bit: the until-exact contract REQUIRES both arms to land
        // exactly on stop_hint_bits — strict, no exceptions.
        assert_eq!(
            b.encoded_offset_bits + b.encoded_size_bits,
            stop_hint_bits,
            "{label}: Block arm final bit != stop_hint"
        );
        assert_eq!(
            w.encoded_offset_bits + w.encoded_size_bits,
            stop_hint_bits,
            "{label}: wrapper arm final bit != stop_hint"
        );

        // (d) CRCs.
        assert_eq!(b.crc32s.len(), w.crc32s.len(), "{label}: crc32s count");
        for (i, (x, y)) in b.crc32s.iter().zip(w.crc32s.iter()).enumerate() {
            assert_eq!(x.crc32(), y.crc32(), "{label}: crc32s[{i}]");
        }
        assert_eq!(
            b.crc32s[0].crc32(),
            crate::decompress::parallel::crc32::crc32(&bb),
            "{label}: CRC accumulator != crc32(decoded)"
        );

        // (e) subchunk keys — strictly equal (both arms end at stop_hint).
        let keys = |c: &ChunkData| {
            c.subchunks
                .iter()
                .map(|s| {
                    (
                        s.encoded_offset_bits,
                        s.encoded_size_bits,
                        s.decoded_offset,
                        s.decoded_size,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&b), keys(&w), "{label}: subchunk keys");

        // (f) published windows: cross-arm AND vs brute-force ground truth.
        let mut hist = Vec::with_capacity(pred.len() + bb.len());
        hist.extend_from_slice(pred);
        hist.extend_from_slice(&bb);
        let brute = |chunk_off: usize| -> Vec<u8> {
            let end = pred.len() + chunk_off;
            hist[end - WINDOW..end].to_vec()
        };

        let lb = b.get_last_window_vec(pred);
        let lw = w.get_last_window_vec(pred);
        assert_eq!(lb, lw, "{label}: get_last_window");
        assert_eq!(
            lb,
            brute(bb.len()),
            "{label}: get_last_window vs brute force"
        );
        assert_eq!(
            b.last_32kib_window_vec(),
            w.last_32kib_window_vec(),
            "{label}: last_32kib_window"
        );

        b.populate_subchunk_windows(pred);
        w.populate_subchunk_windows(pred);
        assert_eq!(
            b.subchunks.len(),
            w.subchunks.len(),
            "{label}: subchunk count"
        );
        for (i, (sb, sw)) in b.subchunks.iter().zip(w.subchunks.iter()).enumerate() {
            let wbts = sb.window.as_ref().map(|v| v.decompress());
            let wwts = sw.window.as_ref().map(|v| v.decompress());
            assert_eq!(wbts, wwts, "{label}: subchunk[{i}] window bytes");
            if let Some(got) = &wbts {
                let bf = brute(sb.decoded_offset);
                if sb.used_window_symbols.is_empty() && got != &bf {
                    assert_eq!(got.len(), bf.len(), "{label}: subchunk[{i}] window len");
                    for (j, (g, t)) in got.iter().zip(bf.iter()).enumerate() {
                        assert!(
                            *g == 0 || g == t,
                            "{label}: subchunk[{i}] window byte {j} stale (got {g}, truth {t})"
                        );
                    }
                }
            }
        }
    }

    /// (g) ERROR equality: both arms must fail with ExactStopMissed carrying
    /// IDENTICAL requested/actual coordinates.
    fn assert_same_exact_miss(
        be: Result<ChunkData, ChunkDecodeError>,
        we: Result<ChunkData, ChunkDecodeError>,
        label: &str,
    ) -> (usize, usize) {
        let b = match be {
            Err(ChunkDecodeError::ExactStopMissed { requested, actual }) => (requested, actual),
            Err(other) => {
                panic!("{label}: Block arm error variant {other:?}, want ExactStopMissed")
            }
            Ok(c) => panic!(
                "{label}: Block arm unexpectedly SUCCEEDED (decoded {} final_bit {})",
                c.decoded_size(),
                c.encoded_offset_bits + c.encoded_size_bits
            ),
        };
        let w = match we {
            Err(ChunkDecodeError::ExactStopMissed { requested, actual }) => (requested, actual),
            Err(other) => {
                panic!("{label}: wrapper arm error variant {other:?}, want ExactStopMissed")
            }
            Ok(c) => panic!(
                "{label}: wrapper arm unexpectedly SUCCEEDED (decoded {} final_bit {})",
                c.decoded_size(),
                c.encoded_offset_bits + c.encoded_size_bits
            ),
        };
        assert_eq!(
            b, w,
            "{label}: ExactStopMissed (requested, actual) coordinates"
        );
        b
    }

    fn run_exact_case(
        deflate: &[u8],
        stop: usize,
        dict: &[u8],
        truth: &[u8],
        cfg: ChunkConfiguration,
        label: &str,
    ) {
        let b = arm_block_exact(deflate, stop, dict, cfg).unwrap_or_else(|e| {
            panic!("{label}: Block arm failed: {e:?}");
        });
        let w = arm_wrapper_exact(deflate, stop, dict, cfg).unwrap_or_else(|e| {
            panic!("{label}: wrapper arm failed: {e:?}");
        });
        assert_arms_equal_exact(b, w, dict, truth, stop, label);
    }

    /// Interior confirmed-boundary stops: byte-aligned AND non-byte-aligned
    /// bit offsets, from a flush-dense + dynamic corpus.
    #[test]
    fn exact_interior_boundary_stops() {
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..4 * 1024]); // dict back-refs up front
        for i in 0..4000u32 {
            payload.extend_from_slice(format!("interior line {i} {}\n", i % 53).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 4096);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration::default();

        let bounds = enumerate_boundaries(&deflate, &dict);
        assert!(
            bounds.len() >= 8,
            "need many boundaries, got {}",
            bounds.len()
        );

        let aligned: Vec<usize> = bounds
            .iter()
            .map(|&(bit, _)| bit)
            .filter(|bit| bit % 8 == 0)
            .take(3)
            .collect();
        let unaligned: Vec<usize> = bounds
            .iter()
            .map(|&(bit, _)| bit)
            .filter(|bit| bit % 8 != 0)
            .take(3)
            .collect();
        assert!(!aligned.is_empty(), "no byte-aligned boundaries in fixture");
        assert!(
            !unaligned.is_empty(),
            "no non-byte-aligned boundaries in fixture"
        );

        for (kind, stops) in [("aligned", &aligned), ("unaligned", &unaligned)] {
            for &stop in stops {
                run_exact_case(
                    &deflate,
                    stop,
                    &dict,
                    &truth,
                    cfg,
                    &format!("interior_{kind}_bit{stop}"),
                );
            }
        }
    }

    /// Member-final stop: BFINAL block decoded to the byte-aligned stream end
    /// (the RFC 1952 padding consumed; the 8-byte footer is OUTSIDE the input
    /// slice exactly like production `sm_driver` slicing).
    #[test]
    fn exact_member_final_stop() {
        let dict = make_dict();
        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[..8 * 1024]);
        for i in 0..3000u32 {
            payload.extend_from_slice(format!("member final line {i} {}\n", i % 17).as_bytes());
        }
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let stop = deflate.len() * 8; // byte-aligned padded deflate end (production total_bits)
        run_exact_case(
            &deflate,
            stop,
            &dict,
            &truth,
            ChunkConfiguration::default(),
            "member_final",
        );

        // Misaligned member-final hints: one byte PAST and one byte SHORT of
        // the padded stream end. Both arms must reject with IDENTICAL
        // ExactStopMissed coordinates; `actual` is the byte-aligned post-EOB
        // bit on both arms (the M4 end-bit coordinate convention).
        let be = arm_block_exact(&deflate, stop + 8, &dict, ChunkConfiguration::default());
        let we = arm_wrapper_exact(&deflate, stop + 8, &dict, ChunkConfiguration::default());
        let (req, actual) = assert_same_exact_miss(be, we, "member_final_past");
        assert_eq!(req, stop + 8);
        assert_eq!(actual, stop, "actual must be the byte-aligned stream end");
    }

    /// Multi-member trailing shape (contract (c)): the input slice continues
    /// past member 1's padded deflate end with a gzip footer + next member's
    /// header + deflate body. NEITHER arm consumes the footer or resets for
    /// the next stream on the until-exact path (`read_footer_at_current` /
    /// `reset_for_next_stream` have no production caller there): a stop hint
    /// pointing past member 1 must fail on BOTH arms with IDENTICAL
    /// ExactStopMissed coordinates, `actual` = member 1's byte-aligned end.
    #[test]
    fn exact_multi_member_trailing() {
        let dict = make_dict();
        let payload1: Vec<u8> = b"member one payload ".repeat(3000);
        let payload2: Vec<u8> = b"member two payload ".repeat(3000);
        let deflate1 = deflate_with_dict(&payload1, &dict, 6);
        // Member 2 is a STANDALONE gzip member (fresh window, as on disk).
        let deflate2 = {
            use flate2::{Compress, Compression, FlushCompress};
            let mut c = Compress::new(Compression::new(6), false);
            let mut out = vec![0u8; payload2.len() * 2 + 4096];
            let status = c
                .compress(&payload2, &mut out, FlushCompress::Finish)
                .expect("compress");
            assert_eq!(status, flate2::Status::StreamEnd);
            out.truncate(c.total_out() as usize);
            out
        };

        let member1_end_bits = deflate1.len() * 8;
        let mut input = deflate1.clone();
        // gzip footer of member 1 (CRC32 + ISIZE) ...
        input
            .extend_from_slice(&crate::decompress::parallel::crc32::crc32(&payload1).to_le_bytes());
        input.extend_from_slice(&(payload1.len() as u32).to_le_bytes());
        // ... then member 2's 10-byte header + deflate body.
        input.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 0x03]);
        input.extend_from_slice(&deflate2);

        // Stop hint deep inside member 2's bit-space.
        let stop = input.len() * 8;
        let be = arm_block_exact(&input, stop, &dict, ChunkConfiguration::default());
        let we = arm_wrapper_exact(&input, stop, &dict, ChunkConfiguration::default());
        let (req, actual) = assert_same_exact_miss(be, we, "multi_member_trailing");
        assert_eq!(req, stop);
        assert_eq!(
            actual, member1_end_bits,
            "both arms must stop at member 1's byte-aligned deflate end"
        );

        // Member 1's padded end IS an honorable exact stop on the same slice.
        let truth1 = flate2_inflate_with_dict(&deflate1, &dict);
        run_exact_case(
            &input,
            member1_end_bits,
            &dict,
            &truth1,
            ChunkConfiguration::default(),
            "multi_member_member1_end",
        );
    }

    /// Stop hint exactly at a subchunk-split boundary: small split threshold
    /// forces multi-subchunk shapes; the exact stop lands on a recorded
    /// boundary at-or-after a split — keys must match strictly.
    #[test]
    fn exact_stop_at_subchunk_split() {
        let dict = make_dict();
        let mut payload = Vec::new();
        for i in 0..6000u32 {
            payload.extend_from_slice(
                format!("split line {i}: the quick brown fox #{}\n", i % 97).as_bytes(),
            );
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 16 * 1024);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        let cfg = ChunkConfiguration {
            split_chunk_size: 48 * 1024,
            ..ChunkConfiguration::default()
        };

        let bounds = enumerate_boundaries(&deflate, &dict);
        // First boundary whose decoded offset crosses the split threshold —
        // the boundary where the split machinery fires — plus a later one.
        let split_bit = bounds
            .iter()
            .find(|&&(_, off)| off >= 48 * 1024)
            .map(|&(bit, _)| bit)
            .expect("no boundary past the split threshold");
        let late_bit = bounds
            .iter()
            .find(|&&(_, off)| off >= 160 * 1024)
            .map(|&(bit, _)| bit)
            .expect("no late boundary");

        for (stop, label) in [(split_bit, "at_split"), (late_bit, "late_split")] {
            let b = arm_block_exact(&deflate, stop, &dict, cfg)
                .unwrap_or_else(|e| panic!("{label}: Block arm failed: {e:?}"));
            let w = arm_wrapper_exact(&deflate, stop, &dict, cfg)
                .unwrap_or_else(|e| panic!("{label}: wrapper arm failed: {e:?}"));
            if label == "late_split" {
                assert!(
                    b.subchunks.len() >= 2,
                    "{label}: expected multi-subchunk shape, got {}",
                    b.subchunks.len()
                );
            }
            assert_arms_equal_exact(b, w, &dict, &truth, stop, label);
        }
    }

    /// Flush-dense + stored-mixed corpora, member-final exact stops (stored
    /// blocks end byte-aligned; level-0 is ALL stored — the
    /// `decode_clean_stored_into_contig` route on the Block arm).
    #[test]
    fn exact_flush_dense_and_stored() {
        let dict = make_dict();

        let mut payload = Vec::new();
        payload.extend_from_slice(&dict[1..4097]);
        for i in 0..2000u32 {
            payload.extend_from_slice(format!("flushy exact line {i} {}\n", i % 13).as_bytes());
        }
        let deflate = deflate_with_dict_flushes(&payload, &dict, 6, 2048);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        assert_eq!(truth, payload, "fixture self-check");
        run_exact_case(
            &deflate,
            deflate.len() * 8,
            &dict,
            &truth,
            ChunkConfiguration::default(),
            "flush_dense_final",
        );
        // Interior exact stop inside the flush-dense stream too.
        let bounds = enumerate_boundaries(&deflate, &dict);
        if let Some(&(bit, _)) = bounds.get(bounds.len() / 2) {
            run_exact_case(
                &deflate,
                bit,
                &dict,
                &truth,
                ChunkConfiguration::default(),
                "flush_dense_interior",
            );
        }

        let mut payload2 = Vec::new();
        for k in 0..6usize {
            if k % 2 == 0 {
                payload2.extend_from_slice(&b"compressible compressible! ".repeat(900));
            } else {
                payload2.extend_from_slice(&lcg_bytes(24 * 1024, 0xD00D_0000 + k as u32));
            }
        }
        for level in [1u32, 0u32] {
            let d = deflate_with_dict(&payload2, &dict, level);
            let t = flate2_inflate_with_dict(&d, &dict);
            assert_eq!(t, payload2, "stored fixture self-check");
            run_exact_case(
                &d,
                d.len() * 8,
                &dict,
                &t,
                ChunkConfiguration::default(),
                &format!("stored_final_l{level}"),
            );
        }
    }

    /// Engine-proof counter: the Block arm increments EXACT_BLOCK_CHUNKS.
    #[test]
    fn exact_block_counter_increments() {
        let dict = make_dict();
        let payload = b"exact counter payload ".repeat(1024);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let before = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let _ = arm_block_exact(
            &deflate,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
        )
        .expect("exact Block decode");
        let after = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after > before, "EXACT_BLOCK_CHUNKS did not increment");
    }

    /// Default route proof: `decode_chunk_until_exact` with a full window and
    /// `until_exact=true` takes the Block engine (kill-switch arm untouched).
    #[test]
    fn exact_route_defaults_to_block() {
        let dict = make_dict();
        let payload = b"route proof payload ".repeat(2048);
        let deflate = deflate_with_dict(&payload, &dict, 6);
        let truth = flate2_inflate_with_dict(&deflate, &dict);
        let before_b = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        let chunk = decode_chunk_until_exact(
            &deflate,
            0,
            deflate.len() * 8,
            &dict,
            ChunkConfiguration::default(),
            true,
        )
        .expect("until-exact route decode");
        let after_b = EXACT_BLOCK_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after_b > before_b,
            "decode_chunk_until_exact did not take the Block engine by default"
        );
        let bytes = chunk.data.to_contiguous()[chunk.data_prefix_len..].to_vec();
        assert_eq!(bytes, truth, "route decode bytes");
    }

    // ════════════════════════════════════════════════════════════════════════
    // STAGE-1 MULTI-MEMBER cross-member continuation tests (§6 of the port
    // design). Drive `decode_multi_member_native` (the `MULTI_MEMBER=true`
    // instantiation of `finish_decode_chunk_contig_native`) directly, since the
    // whole-file finder span + `MultiMemberChunked` routing are stages 2-3.
    // ════════════════════════════════════════════════════════════════════════
    #[cfg(parallel_sm)]
    mod multi_member_stage1 {
        use super::*;
        use crate::decompress::parallel::gzip_format;
        use std::io::Write;

        fn make_gzip_level(payload: &[u8], level: u32) -> Vec<u8> {
            let mut enc =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
            enc.write_all(payload).unwrap();
            enc.finish().unwrap()
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

        fn mm_cfg() -> ChunkConfiguration {
            ChunkConfiguration {
                split_chunk_size: 4 * 1024 * 1024,
                max_decoded_chunk_size: 64 * 1024 * 1024,
                crc32_enabled: true,
                ..Default::default()
            }
        }

        /// Concatenate `payloads` as independent gzip members and return
        /// `(bytes, first_header_len, Σ payload len)`.
        fn build_multi_member(payloads: &[&[u8]], level: u32) -> (Vec<u8>, usize, usize) {
            let mut input = Vec::new();
            let mut total = 0usize;
            for p in payloads {
                input.extend_from_slice(&make_gzip_level(p, level));
                total += p.len();
            }
            let first_header_len = gzip_format::read_header(&input).unwrap().1;
            (input, first_header_len, total)
        }

        fn decode_mm(input: &[u8], first_header_len: usize, reserve: usize) -> ChunkData {
            decode_multi_member_native(
                input,
                first_header_len * 8,
                input.len() * 8,
                reserve,
                mm_cfg(),
            )
            .expect("multi-member decode")
        }

        /// Two members decode byte-exact across the boundary, with correctly
        /// segmented footers + per-member sizes, and the continuation counter
        /// advances (deletion-trap).
        #[test]
        fn two_member_byte_exact_and_segmented() {
            let a = b"The quick brown fox jumps over the lazy dog. ".repeat(40);
            let b = b"Pack my box with five dozen liquor jugs. ".repeat(53);
            let (input, hlen, total) = build_multi_member(&[&a, &b], 6);

            let before = MULTI_MEMBER_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed);
            let chunk = decode_mm(&input, hlen, total);
            let after = MULTI_MEMBER_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed);

            let mut expected = a.clone();
            expected.extend_from_slice(&b);
            assert_eq!(flatten(&chunk), expected, "cross-member bytes");
            assert_eq!(chunk.footers.len(), 2, "one footer per member");
            // `segment_sizes.len() == footers.len() + 1`: the trailing empty
            // segment (0 bytes after the final footer) mirrors vendor's trailing
            // empty `CRC32Calculator` (ChunkData.hpp:559-561); the consumer
            // carries it forward into the next chunk (a no-op on the last chunk).
            assert_eq!(
                chunk.segment_sizes_final(),
                vec![a.len() as u64, b.len() as u64, 0],
                "per-member decoded sizes + trailing empty segment"
            );
            assert!(after > before, "continuation counter must advance");

            // Per-member CRC/ISIZE segments match each member's own trailer.
            assert_eq!(chunk.footers[0].uncompressed_size, a.len() as u32);
            assert_eq!(chunk.footers[1].uncompressed_size, b.len() as u32);
            assert_eq!(chunk.crc32s[0].crc32(), chunk.footers[0].crc32);
            assert_eq!(chunk.crc32s[1].crc32(), chunk.footers[1].crc32);
        }

        /// Three uneven members (different sizes AND compression levels) —
        /// dominant middle member, mixed dynamic/stored blocks.
        #[test]
        fn three_member_uneven_byte_exact() {
            let m0 = b"small".to_vec();
            let m1 = {
                let mut v = Vec::new();
                for i in 0..(64u32 * 1024) {
                    v.push((i.wrapping_mul(2654435761) >> 24) as u8);
                }
                v
            };
            let m2 = b"trailing member with some repetition ".repeat(20);
            // Member 1 stored (level 0), others dynamic — exercises the stored
            // arm crossing a member boundary.
            let mut input = make_gzip_level(&m0, 6);
            input.extend_from_slice(&make_gzip_level(&m1, 0));
            input.extend_from_slice(&make_gzip_level(&m2, 9));
            let hlen = gzip_format::read_header(&input).unwrap().1;

            let chunk = decode_mm(&input, hlen, m0.len() + m1.len() + m2.len());
            let mut expected = m0.clone();
            expected.extend_from_slice(&m1);
            expected.extend_from_slice(&m2);
            assert_eq!(flatten(&chunk), expected);
            assert_eq!(chunk.footers.len(), 3);
            assert_eq!(
                chunk.segment_sizes_final(),
                vec![m0.len() as u64, m1.len() as u64, m2.len() as u64, 0]
            );
        }

        /// A single member driven through the multi-member path: BFINAL →
        /// consume footer → header offset == EOF → clean break. One footer, no
        /// continuation into a next member.
        #[test]
        fn single_member_through_mm_path_breaks_clean_at_eof() {
            let a = b"lonely member".repeat(30);
            let (input, hlen, total) = build_multi_member(&[&a], 6);
            let chunk = decode_mm(&input, hlen, total);
            assert_eq!(flatten(&chunk), a);
            assert_eq!(chunk.footers.len(), 1);
            assert_eq!(chunk.segment_sizes_final(), vec![a.len() as u64, 0]);
        }

        /// SEAM: sweep the inexact stop hint across EVERY byte offset of a
        /// 2-member stream (including the footer + next-header bytes). Every
        /// position must yield a clean decode whose output is a prefix of the
        /// full two-member output — never a panic or wrong bytes.
        #[test]
        fn stop_hint_every_byte_offset_yields_clean_prefix() {
            let a = b"member one payload data ".repeat(12);
            let b = b"member two different data ".repeat(15);
            let (input, hlen, total) = build_multi_member(&[&a, &b], 6);
            let mut full = a.clone();
            full.extend_from_slice(&b);

            for stop_byte in hlen..=input.len() {
                let res =
                    decode_multi_member_native(&input, hlen * 8, stop_byte * 8, total, mm_cfg());
                let chunk = res.unwrap_or_else(|e| {
                    panic!("stop_hint at byte {stop_byte} errored on valid input: {e:?}")
                });
                let out = flatten(&chunk);
                assert!(
                    full.starts_with(&out),
                    "stop_hint at byte {stop_byte}: output ({} B) is not a prefix of full ({} B)",
                    out.len(),
                    full.len()
                );
            }
        }

        /// §3.2 trailing garbage AT a member boundary ⇒ clean-stop: the members
        /// so far decode `Ok`, the garbage is ignored.
        #[test]
        fn trailing_garbage_at_boundary_clean_stops() {
            let a = b"first".repeat(20);
            let b = b"second".repeat(25);
            let (mut input, hlen, total) = build_multi_member(&[&a, &b], 6);
            // Non-gzip trailing bytes (no valid magic).
            input.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44]);

            let chunk = decode_mm(&input, hlen, total);
            let mut expected = a.clone();
            expected.extend_from_slice(&b);
            assert_eq!(flatten(&chunk), expected, "garbage ignored, members Ok");
            assert_eq!(chunk.footers.len(), 2);
        }

        /// §4 corruption INSIDE a member's deflate body must never silently
        /// corrupt: stage 1 verifies ISIZE (length) at the footer, so a
        /// length-changing corruption errors here; a length-preserving one
        /// surfaces as WRONG BYTES that the consumer's per-member CRC (stage 3)
        /// rejects. Either way the corruption is observable, never silent.
        #[test]
        fn corruption_inside_member_is_never_silent() {
            let a = b"clean first member ".repeat(30);
            let b = b"second member body ".repeat(30);
            let gz_a = make_gzip_level(&a, 6);
            let mut gz_b = make_gzip_level(&b, 6);
            let hlen_b = gzip_format::read_header(&gz_b).unwrap().1;
            // Flip a byte well inside member 2's deflate body.
            gz_b[hlen_b + 3] ^= 0xFF;
            let mut input = gz_a.clone();
            input.extend_from_slice(&gz_b);
            let hlen = gzip_format::read_header(&input).unwrap().1;

            let mut expected = a.clone();
            expected.extend_from_slice(&b);
            match decode_multi_member_native(
                &input,
                hlen * 8,
                input.len() * 8,
                a.len() + b.len(),
                mm_cfg(),
            ) {
                Err(_) => {} // structural / ISIZE error — corruption caught.
                Ok(chunk) => assert_ne!(
                    flatten(&chunk),
                    expected,
                    "length-preserving corruption must be visible (CRC rejects it in stage 3)"
                ),
            }
        }

        /// §4 member-2 ISIZE mismatch: the header was read in-chunk
        /// (`did_read_header`), so the in-decode ISIZE check fires ⇒ terminal Err.
        #[test]
        fn member2_isize_mismatch_is_terminal_err() {
            let a = b"aaaaa".repeat(20);
            let b = b"bbbbb".repeat(20);
            let gz_a = make_gzip_level(&a, 6);
            let mut gz_b = make_gzip_level(&b, 6);
            // Corrupt member 2's ISIZE (last 4 bytes of its footer).
            let n = gz_b.len();
            gz_b[n - 1] ^= 0xFF;
            let mut input = gz_a.clone();
            input.extend_from_slice(&gz_b);
            let hlen = gzip_format::read_header(&input).unwrap().1;

            let res = decode_multi_member_native(
                &input,
                hlen * 8,
                input.len() * 8,
                a.len() + b.len(),
                mm_cfg(),
            );
            assert!(res.is_err(), "member-2 ISIZE mismatch must error");
        }

        /// A truncated FINAL footer (fewer than 8 bytes after the last BFINAL)
        /// is corruption inside the member ⇒ terminal Err (not a clean stop).
        #[test]
        fn truncated_final_footer_is_terminal_err() {
            let a = b"first".repeat(20);
            let b = b"second".repeat(20);
            let (input, hlen, total) = build_multi_member(&[&a, &b], 6);
            // Drop 3 bytes of the last footer.
            let truncated = &input[..input.len() - 3];
            let res = decode_multi_member_native(
                truncated,
                hlen * 8,
                truncated.len() * 8,
                total,
                mm_cfg(),
            );
            assert!(res.is_err(), "truncated final footer must error");
        }

        // ════════════════════════════════════════════════════════════════════
        // STAGE-2b: the WINDOW-ABSENT / marker path continuation
        // (`decode_chunk_window_absent_multi`). Stage 1 wired the contig/exact
        // arms; the actual parallel pipeline decodes speculative chunks THROUGH
        // this marker path. A chunk that STARTS at member 1's first block has an
        // empty predecessor window, so it decodes all-clean (no residual
        // markers) and `flatten` yields real bytes — exercising both the
        // pure-marker small-member continuation and the post-flip contig
        // continuation depending on member sizes.
        // ════════════════════════════════════════════════════════════════════

        fn decode_mm_wa(input: &[u8], first_header_len: usize, reserve: usize) -> ChunkData {
            // Reserve is only a sizing hint on this path; window_absent sizes
            // itself from the compressed span. `reserve` kept for call symmetry.
            let _ = reserve;
            decode_chunk_window_absent_multi(input, first_header_len * 8, input.len() * 8, mm_cfg())
                .expect("window-absent multi-member decode")
        }

        /// SMALL members (each < 32 KiB) never accumulate the 32 KiB clean needed
        /// to flip, so every member-boundary is crossed by the PURE-MARKER
        /// continuation (`try_continue_next_member`). Byte-exact across 4 members,
        /// correct footer/segment counts, continuation counter advances.
        #[test]
        fn window_absent_small_members_pure_marker_continuation() {
            let m0 = b"The quick brown fox. ".repeat(30); // ~630 B
            let m1 = b"Pack my box with five dozen liquor jugs. ".repeat(40);
            let m2 = b"lorem ipsum dolor sit amet ".repeat(50);
            let m3 = b"tail member payload here ".repeat(35);
            let (input, hlen, _total) = build_multi_member(&[&m0, &m1, &m2, &m3], 6);

            let before = MULTI_MEMBER_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed);
            let chunk = decode_mm_wa(&input, hlen, m0.len() + m1.len() + m2.len() + m3.len());
            let after = MULTI_MEMBER_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed);

            let mut expected = m0.clone();
            expected.extend_from_slice(&m1);
            expected.extend_from_slice(&m2);
            expected.extend_from_slice(&m3);
            assert_eq!(
                flatten(&chunk),
                expected,
                "cross-member bytes (marker path)"
            );
            assert_eq!(chunk.footers.len(), 4, "one footer per member");
            assert_eq!(
                chunk.segment_sizes_final(),
                vec![
                    m0.len() as u64,
                    m1.len() as u64,
                    m2.len() as u64,
                    m3.len() as u64,
                    0
                ],
                "per-member sizes + trailing empty segment"
            );
            assert!(
                after >= before + 3,
                "≥3 cross-member continuations must fire (4 members)"
            );
            // NOTE: per-chunk `crc32s[i]` is NOT asserted here. On the window-
            // absent/marker path, small members that never flip live entirely in
            // the u16 marker region, whose CRC is credited by the CONSUMER's
            // apply_window pass (§4), not during decode. Byte-correctness (above)
            // is the item-1 guarantee; per-member CRC verification is item 3.
        }

        /// A LARGE first member (> 32 KiB, decoded CLEAN into `chunk.data`)
        /// followed by SMALL members (pure-marker continuation) — exercises the
        /// mixed geometry that was the stage-2c fold-geometry bug.
        ///
        /// ROOT CAUSE (fixed stage-2c): `try_continue_next_member` left the marker
        /// engine in WINDOW-ABSENT/marker mode across the boundary
        /// (`block_primed=false` ⇒ `reset(None,None)`), so member N+1's bytes
        /// landed in `data_with_markers`. Since the output is
        /// `data_with_markers ++ data` and member1 (large) was already CLEAN in
        /// `chunk.data`, the order came out `[s0][s1][big]` (total size + footer
        /// count correct; only byte ORDER wrong). Fix: install an EMPTY window at
        /// the boundary (`reset(Some,&[])`) so every member after the first footer
        /// decodes CLEAN into `chunk.data` — the §3.3 "markers only before the
        /// first footer" invariant. See `window_absent_small_members_pure_marker_
        /// continuation` (all-small) and the two-large case below.
        #[test]
        fn window_absent_large_then_small_members_both_paths() {
            // Large, dynamic, well-compressible → > 32 KiB clean output.
            let big: Vec<u8> = {
                let mut v = Vec::new();
                let line = b"the mixed large member payload with repetition and structure ";
                while v.len() < 200 * 1024 {
                    v.extend_from_slice(line);
                }
                v
            };
            let s0 = b"small tail A ".repeat(10);
            let s1 = b"small tail B ".repeat(12);
            let (input, hlen, _total) = build_multi_member(&[&big, &s0, &s1], 6);

            let chunk = decode_mm_wa(&input, hlen, big.len() + s0.len() + s1.len());
            let got = flatten(&chunk);
            let mut expected = big.clone();
            expected.extend_from_slice(&s0);
            expected.extend_from_slice(&s1);
            assert_eq!(got, expected, "large+small cross-member bytes");
            assert_eq!(chunk.footers.len(), 3);
            assert_eq!(
                chunk.segment_sizes_final(),
                vec![big.len() as u64, s0.len() as u64, s1.len() as u64, 0]
            );
        }

        /// TWO large (> 32 KiB) members back to back: member 1 decodes CLEAN into
        /// `chunk.data`, then member 2 (also large) must ALSO decode CLEAN into
        /// `chunk.data` (appended after member 1) via the empty-window boundary
        /// reset — NOT into `data_with_markers`. Byte-exact + ordered.
        #[test]
        fn window_absent_two_large_members_both_clean() {
            let big0: Vec<u8> = {
                let mut v = Vec::new();
                let line = b"first large member with structure and repetition ";
                while v.len() < 150 * 1024 {
                    v.extend_from_slice(line);
                }
                v
            };
            let big1: Vec<u8> = {
                let mut v = Vec::new();
                let line = b"second large member entirely distinct payload text ";
                while v.len() < 120 * 1024 {
                    v.extend_from_slice(line);
                }
                v
            };
            let (input, hlen, _total) = build_multi_member(&[&big0, &big1], 6);
            let chunk = decode_mm_wa(&input, hlen, big0.len() + big1.len());
            let mut expected = big0.clone();
            expected.extend_from_slice(&big1);
            assert_eq!(flatten(&chunk), expected, "two-large cross-member bytes");
            assert_eq!(chunk.footers.len(), 2);
            assert_eq!(
                chunk.segment_sizes_final(),
                vec![big0.len() as u64, big1.len() as u64, 0]
            );
        }

        /// SEAM: sweep the inexact stop hint across EVERY byte offset of a
        /// small-member 3-member stream (marker path). Every position yields a
        /// clean decode whose output is a prefix of the full output — never a
        /// panic or wrong bytes.
        #[test]
        fn window_absent_stop_hint_every_byte_yields_clean_prefix() {
            let m0 = b"alpha member ".repeat(9);
            let m1 = b"beta member data ".repeat(11);
            let m2 = b"gamma member payload ".repeat(8);
            let (input, hlen, _total) = build_multi_member(&[&m0, &m1, &m2], 6);
            let mut full = m0.clone();
            full.extend_from_slice(&m1);
            full.extend_from_slice(&m2);

            for stop_byte in hlen..=input.len() {
                let chunk =
                    decode_chunk_window_absent_multi(&input, hlen * 8, stop_byte * 8, mm_cfg())
                        .unwrap_or_else(|e| panic!("stop_hint at byte {stop_byte} errored: {e:?}"));
                let out = flatten(&chunk);
                assert!(
                    full.starts_with(&out),
                    "stop_hint {stop_byte}: output ({} B) not a prefix of full ({} B)",
                    out.len(),
                    full.len()
                );
            }
        }

        /// §3.2 trailing garbage at a member boundary on the marker path ⇒
        /// clean-stop; the members so far decode `Ok`.
        #[test]
        fn window_absent_trailing_garbage_clean_stops() {
            let m0 = b"one ".repeat(20);
            let m1 = b"two ".repeat(25);
            let (mut input, hlen, _total) = build_multi_member(&[&m0, &m1], 6);
            input.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
            let chunk = decode_mm_wa(&input, hlen, m0.len() + m1.len());
            let mut expected = m0.clone();
            expected.extend_from_slice(&m1);
            assert_eq!(flatten(&chunk), expected, "garbage ignored, members Ok");
            assert_eq!(chunk.footers.len(), 2);
        }

        /// A corrupt interior member's ISIZE on the marker path (header read
        /// in-chunk ⇒ in-decode check fires) ⇒ terminal Err.
        #[test]
        fn window_absent_member2_isize_mismatch_is_terminal_err() {
            let a = b"aaaa ".repeat(15);
            let b = b"bbbb ".repeat(15);
            let gz_a = make_gzip_level(&a, 6);
            let mut gz_b = make_gzip_level(&b, 6);
            let n = gz_b.len();
            gz_b[n - 1] ^= 0xFF; // corrupt member 2 ISIZE
            let mut input = gz_a.clone();
            input.extend_from_slice(&gz_b);
            let hlen = gzip_format::read_header(&input).unwrap().1;
            let res = decode_chunk_window_absent_multi(&input, hlen * 8, input.len() * 8, mm_cfg());
            assert!(
                res.is_err(),
                "member-2 ISIZE mismatch (marker path) must error"
            );
        }
    }
}
