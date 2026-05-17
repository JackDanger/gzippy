//! Port of `rapidgzip::GzipChunkFetcher` (GzipChunkFetcher.hpp): the
//! parallel-decode orchestrator. Partitions a raw-deflate stream into
//! fixed-spacing speculative chunks, dispatches one worker per
//! partition that:
//!
//!   1. Finds a real deflate block boundary at-or-past its partition
//!      seed (via [`BlockFinder`] + [`validate_boundary`]). For
//!      partition 0 the boundary is bit 0 (start of stream).
//!
//!   2. Decodes the chunk from that boundary via
//!      [`finish_decode_chunk_with_inexact_offset`] — marker bootstrap,
//!      then patched-ISA-L bulk decode — with an empty initial window.
//!      Cross-chunk back-references in the bootstrap segment are tagged
//!      as markers (Vec<u16> with values ≥ `MARKER_BASE`).
//!
//! The consumer thread (`get_next_chunk`) then walks chunks in stream
//! order: looks up the predecessor's last 32 KiB window via
//! [`WindowMap`], calls [`apply_window`] to resolve markers in place,
//! extracts this chunk's tail as the successor's window, and returns
//! the resolved chunk for writer-side consumption.
//!
//! Overlap reconciliation: workers can overshoot their partition
//! boundary by up to one deflate block. The consumer drops chunks whose
//! entire range is shadowed by the predecessor's overshoot and skips
//! chunks with partial overlap. Mirror of rapidgzip's BlockMap
//! insertion-sort behavior.
//!
//! Out of scope for this commit: rapidgzip's LRU prefetch cache
//! (`BlockFetcher::get`), on-demand re-dispatch on partition mismatch,
//! and subchunk-level partial-overlap trimming. The dispatcher is
//! synchronous ("all workers up front") via `std::thread::scope`.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
use crate::decompress::parallel::window_map::WindowMap;

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::block_finder::BlockFinder;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::fast_marker_inflate::validate_boundary;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::gzip_chunk::finish_decode_chunk_with_inexact_offset;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub enum FetchError {
    Decode(ChunkDecodeError),
    UnsupportedPlatform,
    DispatchExhausted,
}

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// How far past a partition seed we'll search for a real deflate block
/// boundary before giving up on that partition. 8 MiB matches the v0.6
/// `find_real_boundary_for_fetcher` fallback radius — large enough for
/// gzip -9's worst-case sparse-boundary regions, small enough to bound
/// per-worker setup cost.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const BOUNDARY_SEARCH_RADIUS_BYTES: usize = 8 * 1024 * 1024;

/// Per-partition validation: how many bytes the marker decoder must
/// successfully decode from a candidate boundary before we trust it.
/// 32 KiB matches v0.6's `decode_chunk_for_fetcher` setting.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const BOUNDARY_VALIDATE_MIN_BYTES: usize = 32 * 1024;

type ChunkSlot = Mutex<Option<Result<ChunkData, ChunkDecodeError>>>;

pub struct GzipChunkFetcher<'a> {
    input: &'a [u8],
    parallelization: usize,
    configuration: ChunkConfiguration,
    /// Partition seed offsets in compressed-bit coordinates, snapped to
    /// byte boundaries. `partition_offsets[i] = i * chunk_size_bits`.
    /// Workers search forward from each seed for a real block boundary.
    partition_offsets: Vec<usize>,
    /// Per-partition result slots. Workers populate; consumer drains
    /// in partition order.
    chunk_slots: Arc<Vec<ChunkSlot>>,
    /// `window_map[encoded_offset_bits]` = the 32 KiB window seeding
    /// decode at that offset. Seeded with empty at 0; extended after
    /// each consumed chunk.
    window_map: WindowMap,
    next_consumer_index: usize,
    /// Confirmed end (bit offset) of the most recently consumed chunk.
    /// Drives overlap-reconciliation: slot ranges at-or-before this
    /// are dropped as shadowed.
    last_consumed_end_bits: usize,
    dispatched: bool,
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl<'a> GzipChunkFetcher<'a> {
    pub fn new(input: &'a [u8], parallelization: usize, configuration: ChunkConfiguration) -> Self {
        let chunk_size_bits = configuration.split_chunk_size * 8;
        let total_bits = input.len() * 8;
        let n = total_bits.div_ceil(chunk_size_bits).max(1);
        let partition_offsets: Vec<usize> = (0..n).map(|i| i * chunk_size_bits).collect();
        let chunk_slots: Vec<ChunkSlot> = (0..n).map(|_| Mutex::new(None)).collect();
        let mut window_map = WindowMap::new();
        // Chunk 0's input window is empty (start of stream).
        window_map.insert(0, Arc::new([0u8; 32768]));
        Self {
            input,
            parallelization,
            configuration,
            partition_offsets,
            chunk_slots: Arc::new(chunk_slots),
            window_map,
            next_consumer_index: 0,
            last_consumed_end_bits: 0,
            dispatched: false,
        }
    }

    pub fn has_more(&self) -> bool {
        self.next_consumer_index < self.partition_offsets.len()
    }

    /// Find the first deflate block boundary at-or-past `from_bit` that
    /// the marker-decoder validator accepts. Returns None if no valid
    /// boundary is found within `BOUNDARY_SEARCH_RADIUS_BYTES`.
    /// Partition 0 (from_bit = 0) bypasses validation — bit 0 of the
    /// stripped deflate stream is a real boundary by construction.
    fn locate_boundary(input: &[u8], from_bit: usize) -> Option<usize> {
        if from_bit == 0 {
            return Some(0);
        }
        let finder = BlockFinder::new(input);
        let mut cursor = from_bit;
        let absolute_limit = input.len() * 8;
        let window_bits = BOUNDARY_SEARCH_RADIUS_BYTES * 8;
        while cursor < absolute_limit {
            let window_end = (cursor + window_bits).min(absolute_limit);
            for candidate in finder.find_blocks(cursor, window_end) {
                if candidate.bit_offset < cursor {
                    continue;
                }
                // min_blocks=1 + 32 KiB output: catches false positives
                // without over-rejecting gzip -9 boundaries where blocks
                // are larger than a 1-block validation window.
                if validate_boundary(
                    input,
                    candidate.bit_offset,
                    1,
                    BOUNDARY_VALIDATE_MIN_BYTES,
                    false,
                )
                .is_ok()
                {
                    return Some(candidate.bit_offset);
                }
            }
            cursor = window_end;
        }
        None
    }

    fn dispatch_all_blocking(&mut self) {
        if self.dispatched {
            return;
        }
        self.dispatched = true;

        let n = self.partition_offsets.len();
        let total_bits = self.input.len() * 8;
        let input = self.input;
        let configuration = self.configuration;
        let partition_offsets = &self.partition_offsets;
        let chunk_slots = Arc::clone(&self.chunk_slots);
        let next_task = AtomicUsize::new(0);

        std::thread::scope(|s| {
            let workers = self.parallelization.max(1).min(n);
            for _ in 0..workers {
                let chunk_slots = Arc::clone(&chunk_slots);
                let next_task = &next_task;
                s.spawn(move || loop {
                    let idx = next_task.fetch_add(1, Ordering::Relaxed);
                    if idx >= n {
                        break;
                    }
                    let seed = partition_offsets[idx];
                    if seed >= total_bits {
                        break;
                    }
                    let Some(boundary) = Self::locate_boundary(input, seed) else {
                        // No validatable boundary in the search radius.
                        // Slot stays None; consumer skips it; the
                        // predecessor chunk's overshoot covers the range.
                        if std::env::var("GZIPPY_DEBUG").is_ok() {
                            eprintln!(
                                "[parallel_sm] chunk[{}] no boundary in {} MiB past seed bit {}",
                                idx,
                                BOUNDARY_SEARCH_RADIUS_BYTES / (1024 * 1024),
                                seed
                            );
                        }
                        continue;
                    };
                    // Until limit: the next partition's seed, or EOF.
                    let until = partition_offsets
                        .get(idx + 1)
                        .copied()
                        .unwrap_or(total_bits);
                    let result = finish_decode_chunk_with_inexact_offset(
                        input,
                        boundary,
                        until,
                        &[],
                        configuration,
                    );
                    if std::env::var("GZIPPY_DEBUG").is_ok() {
                        match &result {
                            Ok(c) => eprintln!(
                                "[parallel_sm] chunk[{}] OK: seed={} boundary={} until={} end={} decoded={} markers={} clean={}",
                                idx,
                                seed,
                                boundary,
                                until,
                                c.encoded_offset_bits + c.encoded_size_bits,
                                c.decoded_size(),
                                c.data_with_markers.len(),
                                c.data.len(),
                            ),
                            Err(e) => eprintln!(
                                "[parallel_sm] chunk[{}] ERR: seed={} boundary={} until={} err={:?}",
                                idx, seed, boundary, until, e
                            ),
                        }
                    }
                    *chunk_slots[idx].lock().unwrap() = Some(result);
                });
            }
        });
    }

    pub fn get_next_chunk(&mut self) -> Result<ChunkData, FetchError> {
        if !self.dispatched {
            self.dispatch_all_blocking();
        }
        loop {
            if self.next_consumer_index >= self.partition_offsets.len() {
                return Err(FetchError::DispatchExhausted);
            }
            let idx = self.next_consumer_index;
            let slot = self.chunk_slots[idx].lock().unwrap().take();
            self.next_consumer_index += 1;

            let chunk = match slot {
                Some(Ok(c)) => c,
                Some(Err(e)) => return Err(FetchError::Decode(e)),
                None => continue, // boundary search failed; benign skip
            };

            let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
            if chunk_end <= self.last_consumed_end_bits {
                // Entirely shadowed by predecessor's overshoot.
                continue;
            }
            if chunk.encoded_offset_bits < self.last_consumed_end_bits {
                // Partial overlap: predecessor already supplied the
                // overlapped bytes. Subchunk-level trimming is a
                // follow-up — for now we drop the whole chunk. (Rare in
                // practice: the marker bootstrap exits at the first
                // block boundary past 32 KiB clean tail, and ISA-L
                // stops at the first non-fixed boundary past until_bits.)
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!(
                        "[parallel_sm] chunk[{}] partial overlap: start {} < last_end {} — dropping",
                        idx, chunk.encoded_offset_bits, self.last_consumed_end_bits
                    );
                }
                continue;
            }

            // Look up this chunk's seed window. Chunk 0 was pre-seeded
            // with the empty window in `new`; subsequent chunks were
            // seeded by the previous successful consume below.
            let chunk_start = chunk.encoded_offset_bits;
            let Some(window) = self.window_map.get(chunk_start) else {
                // We dropped a predecessor (boundary-search failure or
                // shadow). No window available; resolving markers here
                // would corrupt the output, so return an error to make
                // the routing layer fall back to libdeflate.
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: chunk_start,
                    actual: self.last_consumed_end_bits,
                }));
            };

            let mut chunk = chunk;
            apply_window(&mut chunk, &window[..]);

            // Extract this chunk's last 32 KiB as the successor's window.
            let mut next_window = [0u8; 32768];
            let total_out = chunk.decoded_size();
            if total_out >= 32768 {
                if chunk.data.len() >= 32768 {
                    let start = chunk.data.len() - 32768;
                    next_window.copy_from_slice(&chunk.data[start..]);
                } else {
                    let from_data = chunk.data.len();
                    let from_markers = 32768 - from_data;
                    let m_start = chunk.data_with_markers.len() - from_markers;
                    for (i, v) in chunk.data_with_markers[m_start..].iter().enumerate() {
                        next_window[i] = *v as u8;
                    }
                    next_window[from_markers..].copy_from_slice(&chunk.data);
                }
            } else if total_out > 0 {
                let leading = 32768 - total_out;
                next_window[..leading].copy_from_slice(&window[window.len() - leading..]);
                let mut pos = leading;
                for v in &chunk.data_with_markers {
                    next_window[pos] = *v as u8;
                    pos += 1;
                }
                next_window[pos..].copy_from_slice(&chunk.data);
            } else {
                next_window.copy_from_slice(&window[..]);
            }
            self.window_map.insert(chunk_end, Arc::new(next_window));
            self.last_consumed_end_bits = chunk_end;
            return Ok(chunk);
        }
    }
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
impl<'a> GzipChunkFetcher<'a> {
    pub fn new(
        _input: &'a [u8],
        _parallelization: usize,
        _configuration: ChunkConfiguration,
    ) -> Self {
        unimplemented!("GzipChunkFetcher requires x86_64 + isal-compression")
    }
    pub fn has_more(&self) -> bool {
        false
    }
    pub fn get_next_chunk(&mut self) -> Result<ChunkData, FetchError> {
        Err(FetchError::UnsupportedPlatform)
    }
}

// ── Integration tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    fn drive(fetcher: &mut GzipChunkFetcher<'_>) -> Result<Vec<u8>, FetchError> {
        let mut out = Vec::new();
        while fetcher.has_more() {
            let chunk = fetcher.get_next_chunk()?;
            for v in &chunk.data_with_markers {
                out.push(*v as u8);
            }
            out.extend_from_slice(&chunk.data);
        }
        Ok(out)
    }

    #[test]
    fn fetcher_round_trips_2mb_level6() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut fetcher = GzipChunkFetcher::new(&deflate, 4, cfg);
        let out = drive(&mut fetcher).expect("fetcher");
        assert_eq!(out, payload);
    }

    #[test]
    fn fetcher_round_trips_8mb_level9() {
        let payload = b"the quick brown fox jumps over the lazy dog ".repeat(200_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(9));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut fetcher = GzipChunkFetcher::new(&deflate, 8, cfg);
        let out = drive(&mut fetcher).expect("fetcher");
        assert_eq!(out, payload);
    }
}
