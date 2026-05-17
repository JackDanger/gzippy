//! Port of `rapidgzip::GzipChunkFetcher` (GzipChunkFetcher.hpp). The
//! parallel-decode orchestrator: takes a raw deflate stream, partitions
//! it into fixed-spacing seed offsets, dispatches workers to decode
//! each partition via `finish_decode_chunk_with_inexact_offset`,
//! propagates windows between chunks in order, and applies the windows
//! to resolve cross-chunk markers in parallel post-processing.
//!
//! Consumer-facing API: `get_next_chunk()` returns the next chunk in
//! the stream's natural order, with `data_with_markers` already
//! resolved against the predecessor's window. The consumer writes the
//! chunk's bytes to the output and combines the chunk's CRC.
//!
//! This commit implements the synchronous "all workers up front" form
//! of the dispatcher — the simplest faithful port of rapidgzip's
//! parallel-decode shape. Prefetching, on-demand fetch on mismatch,
//! and dynamic work redistribution (the full GzipChunkFetcher's
//! `get_block` path) are not in this commit; they're "highly
//! performant" improvements per the design doc and will land in
//! follow-up commits.
//
// Allowed dead_code: step 7 of rapidgzip-port-design.md migration;
// consumed by single_member.rs in step 8.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
use crate::decompress::parallel::window_map::WindowMap;

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::gzip_chunk::finish_decode_chunk_with_inexact_offset;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub enum FetchError {
    Decode(ChunkDecodeError),
    UnsupportedPlatform,
    MissingWindow { encoded_offset_bits: usize },
}

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// Parallel-decode orchestrator. Owns the input slice + thread pool +
/// the window map that propagates between chunks. Consumer drives
/// progress by repeated `get_next_chunk()` calls until `has_more()`
/// returns false.
pub struct GzipChunkFetcher<'a> {
    input: &'a [u8],
    parallelization: usize,
    configuration: ChunkConfiguration,
    /// Partition seed offsets in compressed-bit coordinates. Length =
    /// number of speculative chunks the fetcher will dispatch.
    /// `partition_offsets[i] = i * chunk_size_bits`.
    partition_offsets: Vec<usize>,
    /// Per-partition result slots. Populated by worker threads via
    /// `decode_partition`. Read by the consumer in partition order via
    /// `get_next_chunk`.
    chunk_slots: Arc<Vec<Mutex<Option<ChunkData>>>>,
    /// Window propagated forward: window_map[encoded_offset_bits] =
    /// last 32 KiB of the chunk whose output ends at that offset. Used
    /// to apply_window on the next chunk's data_with_markers.
    window_map: WindowMap,
    /// Index of the next chunk the consumer will see.
    next_consumer_index: usize,
    /// True once the thread scope has been entered + workers
    /// dispatched. We can't easily re-enter `std::thread::scope`, so
    /// for this first port we dispatch all workers eagerly at first
    /// `get_next_chunk()` call. Note: this commit uses a stop-the-world
    /// dispatch via `dispatch_all_blocking`, not rapidgzip's gradual
    /// prefetcher.
    dispatched: bool,
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl<'a> GzipChunkFetcher<'a> {
    pub fn new(input: &'a [u8], parallelization: usize, configuration: ChunkConfiguration) -> Self {
        let chunk_size_bits = configuration.split_chunk_size * 8;
        let total_bits = input.len() * 8;
        let n = ((total_bits + chunk_size_bits - 1) / chunk_size_bits).max(1);
        let partition_offsets: Vec<usize> = (0..n).map(|i| i * chunk_size_bits).collect();
        let chunk_slots: Vec<Mutex<Option<ChunkData>>> = (0..n).map(|_| Mutex::new(None)).collect();
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
            dispatched: false,
        }
    }

    pub fn has_more(&self) -> bool {
        self.next_consumer_index < self.partition_offsets.len()
    }

    /// Dispatch one worker per partition. Each worker runs
    /// `finish_decode_chunk_with_inexact_offset(partition_offset,
    /// next_partition_offset_or_end, empty_window)`. Empty window =
    /// speculative: cross-chunk back-refs decode to zeros, which the
    /// consumer's later `apply_window` call (in get_next_chunk) fixes
    /// using the real predecessor window. This is rapidgzip's
    /// "decode-with-empty-window, fix-with-applyWindow" pattern.
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
                    let partition_start = partition_offsets[idx];
                    let end = partition_offsets
                        .get(idx + 1)
                        .copied()
                        .unwrap_or(total_bits);
                    if partition_start >= total_bits {
                        break;
                    }
                    // Workers seek to a real deflate block boundary
                    // at-or-after the partition offset before decoding.
                    // Rapidgzip uses its own BlockFinder for this; we
                    // re-use gzippy's existing block-search primitive
                    // until a faithful BlockFinder.hpp port lands.
                    // Chunk 0 always starts at bit 0.
                    let real_start = if idx == 0 {
                        0
                    } else {
                        match crate::decompress::parallel::single_member::find_real_boundary_for_fetcher(
                            input,
                            partition_start,
                        ) {
                            Some(b) => b,
                            None => continue, // no boundary; slot stays None
                        }
                    };
                    let result = finish_decode_chunk_with_inexact_offset(
                        input,
                        real_start,
                        end,
                        &[], // empty window — speculative
                        configuration,
                    );
                    match result {
                        Ok(chunk) => {
                            *chunk_slots[idx].lock().unwrap() = Some(chunk);
                        }
                        Err(e) => {
                            if std::env::var("GZIPPY_DEBUG").is_ok() {
                                eprintln!(
                                    "[parallel_sm:rapidgzip] chunk[{}] worker failed: real_start={} until={} err={:?}",
                                    idx, real_start, end, e
                                );
                            }
                        }
                    }
                });
            }
        });
    }

    /// Return the next chunk in stream order, with markers resolved via
    /// `apply_window` using the predecessor's last 32 KiB. The
    /// returned ChunkData's `data_with_markers` is now u8-castable
    /// (every value < 256); concat it with `data` for the final byte
    /// stream.
    pub fn get_next_chunk(&mut self) -> Result<ChunkData, FetchError> {
        if !self.dispatched {
            self.dispatch_all_blocking();
        }
        if self.next_consumer_index >= self.partition_offsets.len() {
            return Err(FetchError::MissingWindow {
                encoded_offset_bits: usize::MAX,
            });
        }
        let idx = self.next_consumer_index;
        let mut chunk = self.chunk_slots[idx].lock().unwrap().take().ok_or_else(|| {
            if std::env::var("GZIPPY_DEBUG").is_ok() {
                eprintln!(
                    "[parallel_sm:rapidgzip] chunk[{}] slot was None at consumer (partition_offset={}); \
                     worker either failed to find boundary or decode errored",
                    idx, self.partition_offsets[idx]
                );
            }
            FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                requested: idx,
                actual: 0,
            })
        })?;

        // Look up the window for this chunk's start. Chunk 0 gets the
        // pre-seeded empty window; chunks 1..N get whatever the
        // predecessor's window-extraction put into the map.
        let chunk_start = chunk.encoded_offset_bits;
        let window = self
            .window_map
            .get(chunk_start)
            .ok_or(FetchError::MissingWindow {
                encoded_offset_bits: chunk_start,
            })?;

        // Resolve cross-chunk markers in place.
        apply_window(&mut chunk, &window[..]);

        // Extract this chunk's last 32 KiB as the next chunk's input
        // window. Concatenate (data_with_markers as u8) ++ data and
        // take the tail.
        let end_offset = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let mut next_window = [0u8; 32768];
        let total_out = chunk.decoded_size();
        if total_out >= 32768 {
            // Last 32 KiB lies in `data` if data.len() >= 32768; else
            // it spans the tail of data_with_markers + all of data.
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
            // Smaller than 32 KiB: pad the leading bytes with the
            // PREVIOUS window's tail. (Rare in practice; happens for
            // very tiny final chunks.)
            let leading = 32768 - total_out;
            next_window[..leading].copy_from_slice(&window[window.len() - leading..]);
            // Tail = this chunk's full output.
            let mut tail_pos = leading;
            for v in &chunk.data_with_markers {
                next_window[tail_pos] = *v as u8;
                tail_pos += 1;
            }
            next_window[tail_pos..].copy_from_slice(&chunk.data);
        } else {
            // Empty chunk: window unchanged.
            next_window.copy_from_slice(&window[..]);
        }
        self.window_map.insert(end_offset, Arc::new(next_window));

        self.next_consumer_index += 1;
        Ok(chunk)
    }
}

// Stub for non-x86_64 / no-feature builds.
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

    fn drive_to_completion(fetcher: &mut GzipChunkFetcher<'_>) -> Result<Vec<u8>, FetchError> {
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
    fn fetcher_decodes_2mb_round_trip() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut fetcher = GzipChunkFetcher::new(&deflate, 2, cfg);
        let out = drive_to_completion(&mut fetcher).expect("fetcher should drive");
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn fetcher_decodes_8mb_with_high_parallelism() {
        let payload = b"the quick brown fox jumps over the lazy dog ".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut fetcher = GzipChunkFetcher::new(&deflate, 8, cfg);
        let out = drive_to_completion(&mut fetcher).expect("fetcher should drive");
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }
}
