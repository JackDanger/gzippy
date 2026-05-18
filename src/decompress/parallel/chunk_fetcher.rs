//! Port of `rapidgzip::GzipChunkFetcher::processNextChunk`
//! (vendor/rapidgzip/.../GzipChunkFetcher.hpp:311-362) layered on a
//! `BlockFetcher`-driven dispatch
//! (vendor/.../core/BlockFetcher.hpp:245-329).
//!
//! Cutover (2026-05-17): the previous 1661-line implementation
//! reimplemented the prefetch ring, the dispatch primitive, and the
//! cache-hit / take-from-prefetch flow inline in `consumer_loop` —
//! `BlockFetcher::get` was a thin synchronous wrapper invoked from
//! inside a closure that did everything. This rewrite reverses that
//! relationship: `BlockFetcher::get` IS the dispatch primitive (it owns
//! `m_prefetching` per BlockFetcher.hpp:131 and the cache lookup +
//! prefetch-take + on-demand-submit + insert-into-cache flow per
//! BlockFetcher.hpp:245-329) and the consumer is ~80 lines of
//! processNextChunk-shaped orchestration.
//!
//! Mapping (vendor → gzippy):
//!
//! - `m_blockMap` → `BlockMap` (block_map.rs).
//! - `m_blockFinder` → `GzipBlockFinder` (gzip_block_finder.rs).
//! - `m_windowMap` → `WindowMap` (window_map.rs).
//! - `m_threadPool` → `std::thread::scope` + `mpsc::Sender<Job>` shared
//!   between consumer and workers. Workers pull `Job` from the queue;
//!   the consumer submits `DecodeJob`s via `submit_decode_job`. Each
//!   job carries its own reply `Sender<Result<ChunkData, _>>`, so the
//!   `Receiver` returned to `BlockFetcher::get` plays the role of
//!   `std::future<BlockData>` at BlockFetcher.hpp:131.
//! - `m_prefetching: std::map<size_t, std::future<BlockData>>` → lives
//!   inside `BlockFetcher` now (`block_fetcher.rs`'s
//!   `prefetching: HashMap<Key, Receiver<Result<Value, Err>>>`).
//! - `m_cache` → `BlockFetcher::cache`.
//! - `submitOnDemandTask` / `decodeAndMeasureBlock` → the `submit`
//!   closure passed to `BlockFetcher::get`, which sends a `DecodeJob`
//!   and returns the reply `Receiver`. Mirror of BlockFetcher.hpp:554-558.
//! - `decodeBlock` / `decodeChunk` / `decodeChunkWithRapidgzip` →
//!   `gzip_chunk::decode_chunk_with_window` (fast path) +
//!   `decode_or_iterate` (slow path: tryToDecode + BlockFinder
//!   iteration per GzipChunk.hpp:712-846).
//! - `postProcessChunk` / `applyWindow` → `apply_window` +
//!   `populate_subchunk_windows`, dispatched on the same pool via a
//!   `PostProcessJob`. Vendor parity at GzipChunkFetcher.hpp:553-583.

#![allow(dead_code)]

use crate::decompress::parallel::chunk_data::ChunkConfiguration;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::Arc;

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::block_fetcher::BlockFetcher;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::block_finder::BlockFinder;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::block_map::{append_subchunks_to_block_map, BlockMap};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::crc32::CRC32Calculator;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::gzip_block_finder::{GetReturnCode, GzipBlockFinder};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::gzip_chunk::{
    decode_chunk_with_window, finish_decode_chunk_with_inexact_offset,
};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::prefetcher::FetchNextAdaptive;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::trace;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::window_map::WindowMap;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::mpsc;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::time::Duration;

#[derive(Debug)]
pub enum FetchError {
    Decode(ChunkDecodeError),
    UnsupportedPlatform,
}

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// How long a worker will block waiting for the predecessor's window
/// before falling back to the slow path. Short enough that chunk-0
/// (which never has a predecessor) doesn't waste time; long enough
/// that the predecessor's decode plus apply_window can land for the
/// common case.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const WINDOW_WAIT_TIMEOUT: Duration = Duration::from_millis(50);

/// Default `FetchNextAdaptive` memory size (vendor Prefetcher.hpp uses
/// per-instance values from ParallelGzipReader; 32 biases toward longer
/// sequential runs).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const FETCH_STRATEGY_MEMORY: usize = 32;

// ── Worker pool job types ────────────────────────────────────────────────

/// One unit of decode work submitted to the pool. Mirror of vendor's
/// `decodeAndMeasureBlock` task lambda body
/// (vendor/.../core/BlockFetcher.hpp:555-558).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
struct DecodeJob {
    /// Cache key under which the worker stores the result. Mirror of
    /// vendor's `blockOffset` arg threaded through `submitOnDemandTask`.
    cache_key: usize,
    /// Bit offset where this chunk's decode starts.
    start_bit: usize,
    /// Approximate bit offset where the chunk should stop (the next
    /// chunk's `nextBlockOffset` from `m_blockFinder->get(idx+1)`,
    /// vendor BlockFetcher.hpp:268).
    until_bit: usize,
    /// True iff the caller has guaranteed the predecessor's 32 KiB
    /// window is published in the WindowMap by the time the worker
    /// runs. Used to pick fast vs slow path at the worker.
    authoritative: bool,
    /// Reply channel — playing the role of vendor's
    /// `std::future<BlockData>` (BlockFetcher.hpp:131). Wraps the
    /// decoded chunk in an `Arc` to mirror vendor's
    /// `std::shared_ptr<BlockData>` aliasing model
    /// (BlockFetcher.hpp:46).
    reply: mpsc::Sender<Result<Arc<ChunkData>, ChunkDecodeError>>,
    /// For trace labelling only.
    partition_idx: usize,
}

/// One unit of post-processing work submitted to the pool. Mirror of
/// the lambda enqueued at `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:579-582), which calls
/// `chunkData->applyWindow( *window, chunkData->windowCompressionType() )`.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
struct PostProcessJob {
    partition_idx: usize,
    chunk: ChunkData,
    predecessor_window: Arc<[u8; 32768]>,
    window_map: WindowMap,
    reply: mpsc::Sender<ChunkData>,
}

/// Job variant on the unified work channel. Workers receive `Job` and
/// dispatch on the variant. Vendor's `submitTask` and
/// `submitTaskWithHighPriority` both target `BS::thread_pool`; we keep
/// that 1:1.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
enum Job {
    Decode(DecodeJob),
    PostProcess(PostProcessJob),
}

// ── Public driver entry point ─────────────────────────────────────────────

/// Decompress an entire raw deflate stream, writing output bytes to
/// `writer`, returning the per-chunk-combined CRC32 + total decoded
/// size. Mirror of `ParallelGzipReader::read` flowing through
/// `GzipChunkFetcher::processNextChunk` until EOF
/// (vendor/.../ParallelGzipReader.hpp:702-810 + GzipChunkFetcher.hpp:311-362).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn drive<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    let total_bits = input.len() * 8;
    let pool_size = parallelization.max(1);

    // ── m_blockFinder (vendor GzipChunkFetcher.hpp:283) ─────────────
    let block_finder = Arc::new(GzipBlockFinder::new(
        /* first_block_offset_in_bits = */ 0,
        /* spacing_in_bytes = */ configuration.split_chunk_size,
        /* file_size_in_bits = */ Some(total_bits),
    ));

    // ── m_windowMap (vendor GzipChunkFetcher.hpp:285) ───────────────
    let window_map = WindowMap::new();
    // Chunk 0's input window is empty by definition (start of stream).
    let empty_window: Arc<[u8; 32768]> = Arc::new([0u8; 32768]);
    window_map.insert(0, empty_window);

    // ── m_blockMap (vendor GzipChunkFetcher.hpp:284) ────────────────
    let block_map = Arc::new(BlockMap::new());

    // ── m_chunkFetcher = BlockFetcher (vendor BlockFetcher.hpp:38) ──
    // The BlockFetcher owns the cache, prefetch cache, prefetch queue
    // (`m_prefetching`), fetching strategy, and statistics. Sized to
    // hold the active working set plus prefetched chunks.
    let cache_capacity = pool_size * 2;
    let prefetch_capacity = pool_size * 2;
    let block_fetcher: Arc<
        BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive, ChunkDecodeError>,
    > = Arc::new(BlockFetcher::new(
        cache_capacity,
        prefetch_capacity,
        FetchNextAdaptive::new(FETCH_STRATEGY_MEMORY),
        pool_size,
    ));

    // Running CRC + size accumulators. Mirror of vendor's
    // `m_crc32Calculator` + `m_totalDecompressedSize` updates inside
    // ParallelGzipReader::read.
    let mut total_crc = CRC32Calculator::new();
    let mut total_size: usize = 0;

    // Unified worker pool. Job channel is the only synchronization
    // primitive between consumer and workers.
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

    let result = std::thread::scope(|s| -> Result<(), FetchError> {
        for _ in 0..pool_size {
            let job_rx = Arc::clone(&job_rx);
            let window_map = window_map.clone();
            let configuration = configuration;
            s.spawn(move || {
                worker_loop(input, job_rx, window_map, configuration);
            });
        }

        consumer_loop(
            input,
            writer,
            total_bits,
            &block_finder,
            &block_fetcher,
            &block_map,
            &window_map,
            &job_tx,
            pool_size,
            configuration,
            &mut total_crc,
            &mut total_size,
        )?;

        // Drop our job_tx clone so workers exit when their queue empties.
        drop(job_tx);
        Ok(())
    });
    result?;

    // Vendor parity: `m_blockMap->finalize()` + `m_blockFinder->finalize()`
    // at the end of `processNextChunk`'s EOF branch
    // (GzipChunkFetcher.hpp:324-326 + 352-354). Called once after
    // `consumer_loop` exits.
    block_map.finalize();
    block_finder.finalize();
    block_fetcher
        .statistics
        .base
        .set_block_count(block_map.data_block_count(), true);

    Ok((total_crc.crc32(), total_size))
}

// ── Consumer: processNextChunk port ──────────────────────────────────────

/// Vendor-faithful port of `GzipChunkFetcher::processNextChunk`
/// (vendor/.../GzipChunkFetcher.hpp:311-362), wrapped in a `loop` that
/// plays the role of `ParallelGzipReader::read`'s outer loop
/// (ParallelGzipReader.hpp:702-810). Every per-iteration step is a
/// vendor primitive — no inline ring, no inline cache logic, no
/// inline take-from-prefetch.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn consumer_loop<W: std::io::Write>(
    _input: &[u8],
    writer: &mut W,
    total_bits: usize,
    block_finder: &GzipBlockFinder,
    block_fetcher: &Arc<BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive, ChunkDecodeError>>,
    block_map: &BlockMap,
    window_map: &WindowMap,
    job_tx: &mpsc::Sender<Job>,
    pool_size: usize,
    _configuration: ChunkConfiguration,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    // Vendor's `m_nextUnprocessedBlockIndex` (GzipChunkFetcher.hpp:318 + 410).
    let mut next_unprocessed_block_index: usize = 0;
    // Pending writes (post-process jobs in flight, output order).
    let mut pending: std::collections::VecDeque<PendingWrite> =
        std::collections::VecDeque::with_capacity(pool_size * 2);
    let post_process_inflight_cap = pool_size;

    // The vendor's `processNextChunk` returns one chunk per call; the
    // caller loops in `ParallelGzipReader::read`. We inline that loop
    // here so the local-state mutation (post-process queue + writer +
    // CRC) stays simple.
    loop {
        // Vendor GzipChunkFetcher.hpp:318 — `m_blockFinder->get(m_nextUnprocessedBlockIndex)`.
        let next_block_offset = match block_finder.get(next_unprocessed_block_index) {
            (Some(offset), GetReturnCode::Success) => offset,
            // Vendor GzipChunkFetcher.hpp:320-327 — EOF when no offset
            // or offset past end of file.
            _ => break,
        };
        if next_block_offset >= total_bits {
            break;
        }

        // Vendor GzipChunkFetcher.hpp:329 —
        //   `chunkData = getBlock(*nextBlockOffset, m_nextUnprocessedBlockIndex)`
        // which calls `BaseType::get(*nextBlockOffset, blockIndex,
        // getPartitionOffsetFromOffset)`. That's our `BlockFetcher::get`.
        //
        // The submit closure plays the role of vendor's
        // `submitOnDemandTask` (BlockFetcher.hpp:600). It sends a
        // `DecodeJob` and returns the reply receiver — `BlockFetcher::get`
        // does the wait + cache-insert internally.
        block_fetcher.record_fetch(next_unprocessed_block_index);
        let until_bit = until_bit_for(
            block_finder,
            next_unprocessed_block_index,
            total_bits,
            next_block_offset,
        );
        let next_block_offset_for_submit = next_block_offset;
        let until_bit_for_submit = until_bit;
        let partition_idx_for_trace = next_unprocessed_block_index;
        let chunk_arc_result: Result<Arc<ChunkData>, ChunkDecodeError> = block_fetcher.get(
            next_block_offset,
            |key: usize| -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
                submit_decode_job(
                    job_tx,
                    DecodeJob {
                        cache_key: key,
                        start_bit: next_block_offset_for_submit,
                        until_bit: until_bit_for_submit,
                        authoritative: true,
                        reply: mpsc::channel().0, // overwritten below
                        partition_idx: partition_idx_for_trace,
                    },
                )
            },
        );
        let chunk_arc = chunk_arc_result.map_err(FetchError::Decode)?;
        // Take ownership when we hold the only Arc; otherwise clone the
        // inner ChunkData. Mirror of rapidgzip's shared_ptr aliasing at
        // GzipChunkFetcher.hpp:329.
        let mut chunk: ChunkData = Arc::try_unwrap(chunk_arc).unwrap_or_else(|a| (*a).clone());

        // Vendor GzipChunkFetcher.hpp:349 —
        //   `chunkData->setEncodedOffset(*nextBlockOffset);`
        // Adjusts encoded_offset_bits to the consumer-requested seed
        // (matters when a speculative worker decoded at an earlier
        // candidate inside [next_block_offset, max_encoded_offset_bits]).
        if chunk.encoded_offset_bits != next_block_offset {
            chunk.set_encoded_offset(next_block_offset);
        }

        // Vendor GzipChunkFetcher.hpp:350-355 — EOF mid-decode
        // (`encodedSizeInBits == 0`).
        if chunk.encoded_size_bits == 0 {
            break;
        }
        let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;

        // Vendor GzipChunkFetcher.hpp:343 — `postProcessChunk(chunkData, lastWindow)`.
        // The window emplacement at lines 558-575 (tail-window publish on
        // consumer thread BEFORE handing off to the worker pool) is what
        // unblocks the next chunk's worker without serializing on this
        // chunk's apply_window. We mirror exactly:
        let predecessor_window_for_postprocess: Option<Arc<[u8; 32768]>>;
        if chunk.data_with_markers.is_empty() {
            // No markers → apply_window is a no-op. Worker has already
            // published the tail window via `last_32kib_window()`.
            predecessor_window_for_postprocess = None;
        } else {
            // Vendor lines 334 + 558-574: pull the predecessor's window
            // and compute the tail window on the consumer thread.
            let window_key = chunk.max_encoded_offset_bits;
            let window = window_map
                .get_or_wait(window_key, Duration::from_secs(60))
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: window_key,
                    actual: next_block_offset,
                }))?;
            let tail = chunk.get_last_window(&window[..]);
            window_map.insert(chunk_end_bit, Arc::new(tail));
            predecessor_window_for_postprocess = Some(window);
        }

        // Vendor GzipChunkFetcher.hpp:357 —
        //   `appendSubchunksToIndexes(chunkData, chunkData->subchunks(), *lastWindow);`
        // Pushes subchunks into BlockMap (line 373) and BlockFinder
        // (line 374).
        append_subchunks_to_block_map(block_map, &chunk);
        if chunk.subchunks.is_empty() {
            block_finder.insert(chunk_end_bit);
        } else {
            for sc in &chunk.subchunks {
                block_finder.insert(sc.encoded_offset_bits + sc.encoded_size_bits);
            }
        }
        // Vendor GzipChunkFetcher.hpp:359 — `m_statistics.merge(*chunkData)`.
        block_fetcher.statistics.base.record_get();

        // Vendor GzipChunkFetcher.hpp:410 — `m_nextUnprocessedBlockIndex += subchunks.size()`.
        let inserted = chunk.subchunks.len().max(1);
        next_unprocessed_block_index += inserted;

        // Hand off to post-process worker if there are markers; else
        // queue as immediately-ready for ordered write.
        match predecessor_window_for_postprocess {
            Some(window) => {
                let rx = submit_post_process_job(
                    job_tx,
                    PostProcessJob {
                        partition_idx: partition_idx_for_trace,
                        chunk,
                        predecessor_window: window,
                        window_map: window_map.clone(),
                        reply: mpsc::channel().0, // overwritten below
                    },
                );
                pending.push_back(PendingWrite::Async {
                    idx: partition_idx_for_trace,
                    rx,
                    cache_key: next_block_offset,
                });
            }
            None => {
                pending.push_back(PendingWrite::Ready {
                    idx: partition_idx_for_trace,
                    chunk,
                    cache_key: next_block_offset,
                });
            }
        }

        // Vendor parity: drain post-processes that are FIFO-ready to keep
        // the in-flight cap bounded. Vendor's
        // `waitForReplacedMarkers` (GzipChunkFetcher.hpp:478-518) does
        // this implicitly by blocking on the per-chunk future.
        while pending.len() > post_process_inflight_cap {
            drain_one_pending(&mut pending, writer, total_crc, total_size, block_fetcher)?;
        }
    }

    // Final drain — flush remaining post-processes in encoded order.
    while !pending.is_empty() {
        drain_one_pending(&mut pending, writer, total_crc, total_size, block_fetcher)?;
    }

    Ok(())
}

/// Compute the `until_bit` hint for the worker. Mirror of vendor's
/// `nextBlockOffset = m_blockFinder->get(validDataBlockIndex + 1)` at
/// BlockFetcher.hpp:268. Falls back to `total_bits` if the block
/// finder doesn't have the next offset (end of stream).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn until_bit_for(
    block_finder: &GzipBlockFinder,
    block_index: usize,
    total_bits: usize,
    floor: usize,
) -> usize {
    match block_finder.get(block_index + 1) {
        (Some(offset), GetReturnCode::Success) | (Some(offset), GetReturnCode::Failure) => {
            offset.max(floor)
        }
        _ => total_bits,
    }
}

/// Send a `DecodeJob` to the worker pool, returning the reply receiver.
/// Mirror of vendor's `submitOnDemandTask(blockOffset, nextBlockOffset)`
/// at BlockFetcher.hpp:600 — both arrange for an asynchronous decode
/// whose `std::future<BlockData>` the caller waits on.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn submit_decode_job(
    job_tx: &mpsc::Sender<Job>,
    template: DecodeJob,
) -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
    let (tx, rx) = mpsc::channel();
    let job = DecodeJob {
        reply: tx,
        ..template
    };
    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "submit_decode",
            &format!(
                r#""partition_idx":{},"cache_key":{},"start_bit":{},"until_bit":{},"authoritative":{}"#,
                job.partition_idx, job.cache_key, job.start_bit, job.until_bit, job.authoritative,
            ),
        );
    }
    job_tx.send(Job::Decode(job)).expect("worker pool dropped");
    rx
}

/// Send a `PostProcessJob` to the worker pool, returning the reply
/// receiver. Mirror of vendor's `submitTaskWithHighPriority(applyWindow)`
/// at GzipChunkFetcher.hpp:579.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn submit_post_process_job(
    job_tx: &mpsc::Sender<Job>,
    template: PostProcessJob,
) -> mpsc::Receiver<ChunkData> {
    let (tx, rx) = mpsc::channel();
    let job = PostProcessJob {
        reply: tx,
        ..template
    };
    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "submit_post_process",
            &format!(r#""partition_idx":{}"#, job.partition_idx),
        );
    }
    job_tx
        .send(Job::PostProcess(job))
        .expect("worker pool dropped");
    rx
}

// ── Worker loop ──────────────────────────────────────────────────────────

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn worker_loop(
    input: &[u8],
    job_rx: Arc<std::sync::Mutex<mpsc::Receiver<Job>>>,
    window_map: WindowMap,
    configuration: ChunkConfiguration,
) {
    loop {
        let job = {
            let rx = job_rx.lock().unwrap();
            match rx.recv() {
                Ok(j) => j,
                Err(_) => return, // sender dropped → shutdown
            }
        };

        match job {
            Job::Decode(d) => run_decode_job(input, d, &window_map, configuration),
            Job::PostProcess(p) => run_post_process_job(p),
        }
    }
}

/// Worker-side execution of a `DecodeJob`. Mirror of vendor's
/// `decodeBlock(blockOffset, nextBlockOffset)`
/// (GzipChunkFetcher.hpp:692-729) — picks the fast path
/// (`decodeChunkWithIsal` / `decode_chunk_with_window`) when the
/// initial window is known, else the slow path
/// (`decodeChunkWithRapidgzip` / `decode_or_iterate`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn run_decode_job(
    input: &[u8],
    job: DecodeJob,
    window_map: &WindowMap,
    configuration: ChunkConfiguration,
) {
    let label = trace::worker_label(job.partition_idx);
    let t0 = std::time::Instant::now();

    // Try to acquire the predecessor's window. The consumer guarantees
    // it for authoritative dispatches (vendor: post_process always
    // emplaces the tail BEFORE the next worker can be unblocked); for
    // speculative dispatches we wait a short timeout.
    let window = if job.start_bit == 0 {
        Some(Arc::new([0u8; 32768]))
    } else if job.authoritative {
        window_map.get_or_wait(job.start_bit, Duration::from_secs(60))
    } else {
        window_map.get_or_wait(job.start_bit, WINDOW_WAIT_TIMEOUT)
    };

    let chunk_result = match window {
        Some(w) => decode_chunk_with_window(input, job.start_bit, job.until_bit, &w, configuration),
        None => decode_or_iterate(input, job.start_bit, job.until_bit, configuration),
    };

    // Publish tail window if cleanly available — workers can usually
    // do this without waiting for apply_window. Mirror of vendor's
    // optimisation at GzipChunkFetcher.hpp:573-575 (`getLastWindow`
    // shortcut when the chunk has no markers).
    if let Ok(ref c) = chunk_result {
        if let Some(tail) = c.last_32kib_window() {
            let end_bit = c.encoded_offset_bits + c.encoded_size_bits;
            window_map.insert(end_bit, Arc::new(tail));
        }
    }

    // Wrap in Arc to match BlockFetcher's `Value = Arc<ChunkData>`
    // (vendor's `std::shared_ptr<BlockData>` at BlockFetcher.hpp:46).
    let result = chunk_result.map(Arc::new);

    if trace::is_enabled() {
        let dur_us = t0.elapsed().as_micros();
        let path = if window_map.get(job.start_bit).is_some() {
            "fast"
        } else {
            "slow"
        };
        match &result {
            Ok(c) => trace::emit(
                &label,
                "decode_ok",
                &format!(
                    r#""partition_idx":{},"path":"{}","start_bit":{},"end_bit":{},"decoded":{},"duration_us":{dur_us}"#,
                    job.partition_idx,
                    path,
                    job.start_bit,
                    c.encoded_offset_bits + c.encoded_size_bits,
                    c.decoded_size(),
                ),
            ),
            Err(e) => trace::emit(
                &label,
                "decode_err",
                &format!(
                    r#""partition_idx":{},"path":"{}","start_bit":{},"until_bit":{},"err":"{}","duration_us":{dur_us}"#,
                    job.partition_idx,
                    path,
                    job.start_bit,
                    job.until_bit,
                    trace::esc(&format!("{e:?}")),
                ),
            ),
        }
    }

    let _ = job.reply.send(result);
}

/// Worker-side execution of a `PostProcessJob`. Mirror of the lambda
/// body at `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:579-582).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn run_post_process_job(mut job: PostProcessJob) {
    apply_window(&mut job.chunk, &job.predecessor_window[..]);
    job.chunk
        .populate_subchunk_windows(&job.predecessor_window[..]);

    // Per-subchunk window publication (vendor GzipChunkFetcher.hpp:430-458).
    for sc in &job.chunk.subchunks {
        if let Some(ref w) = sc.window {
            let sc_end_bit = sc.encoded_offset_bits + sc.encoded_size_bits;
            if sc_end_bit > sc.encoded_offset_bits {
                job.window_map.insert(sc_end_bit, w.clone());
            }
        }
    }

    let _ = job.reply.send(job.chunk);
}

// ── Slow-path decoder (tryToDecode + BlockFinder iteration) ──────────────

/// Try to decode at `start_bit` directly; if that fails, iterate
/// BlockFinder candidates in [start_bit, start_bit + 512 KiB] until
/// one succeeds. Mirror of rapidgzip's `tryToDecode` + candidate
/// iteration at vendor/.../GzipChunk.hpp:712-841.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_or_iterate(
    input: &[u8],
    start_bit: usize,
    until_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let finder = BlockFinder::new(input);
    let scan_end = (start_bit + 512 * 1024 * 8).min(input.len() * 8);
    let candidates = finder.find_blocks(start_bit, scan_end);
    let mut last_err: Option<ChunkDecodeError> = None;
    for candidate in candidates {
        if candidate.bit_offset < start_bit {
            continue;
        }
        match finish_decode_chunk_with_inexact_offset(
            input,
            candidate.bit_offset,
            until_bit,
            &[],
            configuration,
        ) {
            Ok(mut c) => {
                let actual = c.encoded_offset_bits;
                c.encoded_offset_bits = start_bit;
                c.max_encoded_offset_bits = actual;
                c.encoded_size_bits += actual - start_bit;
                return Ok(c);
            }
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }
    Err(last_err.unwrap_or(ChunkDecodeError::ExactStopMissed {
        requested: start_bit,
        actual: scan_end,
    }))
}

// ── Pending-write queue (order-preserved output) ─────────────────────────

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
enum PendingWrite {
    Ready {
        idx: usize,
        chunk: ChunkData,
        cache_key: usize,
    },
    Async {
        idx: usize,
        rx: mpsc::Receiver<ChunkData>,
        cache_key: usize,
    },
}

/// Pull the head of the pending FIFO, wait on its post-process if
/// needed, then write its bytes + advance the stream CRC. Mirror of
/// the tail of `GzipChunkFetcher::waitForReplacedMarkers` + write loop
/// at vendor lines 516 + 333-342.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn drain_one_pending<W: std::io::Write>(
    pending: &mut std::collections::VecDeque<PendingWrite>,
    writer: &mut W,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive, ChunkDecodeError>,
) -> Result<(), FetchError> {
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    let (idx, chunk, cache_key) = match head {
        PendingWrite::Ready {
            idx,
            chunk,
            cache_key,
        } => (idx, chunk, cache_key),
        PendingWrite::Async { idx, rx, cache_key } => {
            let chunk = rx.recv().map_err(|_| {
                FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: idx,
                    actual: 0,
                })
            })?;
            (idx, chunk, cache_key)
        }
    };

    // Mirror of vendor's per-chunk write loop (GzipChunkFetcher.hpp:333-342).
    let mut written_crc = CRC32Calculator::new();
    if !chunk.data_with_markers.is_empty() {
        let dwm_len = chunk.data_with_markers.len();
        let mut narrowed: Vec<u8> = Vec::with_capacity(dwm_len);
        for v in &chunk.data_with_markers {
            narrowed.push(*v as u8);
        }
        written_crc.update(&narrowed);
        writer
            .write_all(&narrowed)
            .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        *total_size += narrowed.len();
    }
    if !chunk.data.is_empty() {
        written_crc.update(&chunk.data);
        writer
            .write_all(&chunk.data)
            .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        *total_size += chunk.data.len();
    }
    total_crc.append(&written_crc);

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"decoded":{}"#,
                chunk.decoded_size()
            ),
        );
    }
    block_fetcher.insert(cache_key, Arc::new(chunk));
    Ok(())
}

// Non-x86_64 / non-isal stub.
#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn drive<W: std::io::Write>(
    _input: &[u8],
    _writer: &mut W,
    _parallelization: usize,
    _configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    Err(FetchError::UnsupportedPlatform)
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

    #[test]
    fn drive_round_trips_2mb_level6() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 4, cfg).expect("drive");
        assert_eq!(size, payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn drive_round_trips_8mb_level9() {
        let payload = b"the quick brown fox jumps over the lazy dog ".repeat(200_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(9));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 8, cfg).expect("drive");
        assert_eq!(size, payload.len());
        assert_eq!(out, payload);
    }
}
