//! Port of `rapidgzip::GzipChunkFetcher::processNextChunk`
//! (vendor/rapidgzip/.../GzipChunkFetcher.hpp:311-362) on top of the
//! [`BlockFetcher`](crate::decompress::parallel::block_fetcher::BlockFetcher) +
//! [`BlockMap`](crate::decompress::parallel::block_map::BlockMap) +
//! [`WindowMap`](crate::decompress::parallel::window_map::WindowMap) +
//! [`FetchNextAdaptive`](crate::decompress::parallel::prefetcher::FetchNextAdaptive) +
//! [`ChunkFetcherStatistics`](crate::decompress::parallel::statistics::ChunkFetcherStatistics)
//! pipeline.
//!
//! Architecture
//! ------------
//! Rapidgzip composes a generic `BlockFetcher` (LRU cache + prefetch
//! cache + FetchingStrategy + Statistics) with a domain-specific
//! `GzipChunkFetcher` that drives `processNextChunk` over a `BlockMap`
//! ordered by encoded bit offset. This module mirrors that composition.
//!
//! The vendor uses `BS::thread_pool` + `std::future` for dispatch. Our
//! port keeps the existing std::thread worker pool but distributes
//! jobs via a lock-free `crossbeam_channel::unbounded` MPMC queue
//! (mirroring rapidgzip's ThreadPool which lets each worker pull from
//! a shared queue without serializing — vendor/.../core/ThreadPool.hpp).
//! The earlier `Arc<Mutex<mpsc::Receiver>>` design forced all workers
//! to serialize on a single lock per `recv()`, leaving N-1 of N workers
//! stalled on `.lock()` whenever the queue was non-empty. Each pool worker is the per-block
//! "decodeBlock" job; its result is both sent on a per-job reply channel
//! AND inserted into `BlockFetcher` (prefetch cache for speculative
//! jobs, main cache for authoritative). The consumer's
//! `processNextChunk` body:
//!
//!   1. Compute the next chunk's expected encoded bit offset (the
//!      predecessor's actual end, or `partition_offsets[0] = 0` for
//!      chunk 0). Record the access through
//!      [`BlockFetcher::record_fetch`] so [`FetchNextAdaptive`] tracks
//!      the sequential pattern.
//!   2. Consult [`BlockFetcher::get_if_available`]: cache hit (the
//!      common path on hot partitions) returns the prefetched chunk
//!      with `record_prefetch_cache_hit` already counted.
//!   3. Miss → wait on the per-partition reply channel (speculative
//!      job's result is en route). If the speculative chunk's start
//!      doesn't match expected (`matches_encoded_offset` +
//!      `decoded_offset_for`), submit an authoritative re-decode at
//!      `expected_start`, counted via
//!      [`BlockFetcher::record_on_demand_fetch`].
//!   4. Resolve markers via [`apply_window`]; push subchunks into
//!      [`block_map::append_subchunks_to_block_map`]; populate the
//!      [`WindowMap`] per the rapidgzip
//!      `appendSubchunksToIndexes` cascade (GzipChunkFetcher.hpp:430-458).
//!   5. Write decoded bytes, combine CRC32, move on.
//!
//! Multi-stream gzip is handled inside the per-chunk decode loop via
//! [`IsalInflateWrapper::read_footer_at_current`] +
//! [`IsalInflateWrapper::reset_for_next_stream`] when a chunk's decode
//! crosses a gzip footer — see [`gzip_chunk`].
//!
//! The rapidgzip threading-pool difference is intentional. See the
//! "Threading model" note in `block_fetcher.rs` and `docs/rapidgzip-port-reference.md`
//! §I "Consumer rewrite last".

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
use crossbeam_channel as cbc;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::mpsc;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::time::Duration;

#[derive(Debug)]
pub enum FetchError {
    Decode(ChunkDecodeError),
    UnsupportedPlatform,
}

/// Test-only observability: number of times the production `drive`
/// path called `BlockFetcher::record_get`. Used by
/// `test_block_fetcher_in_drive` to lock in that the cutover's
/// BlockFetcher wiring is actually exercised (a deletion-trap killer
/// in the same spirit as `MARKER_PIPELINE_RUNS`).
#[cfg(any(test, debug_assertions))]
pub static BLOCK_FETCHER_GETS_OBSERVED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// How far past a partition seed we'll search for a deflate block
/// boundary candidate. Matches rapidgzip's tryToDecode scan budget
/// (vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp ~ L800).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const BOUNDARY_SEARCH_RADIUS_BYTES: usize = 512 * 1024;

/// How long a worker will block waiting for the predecessor's window
/// before falling back to the slow path. Short enough that chunk-0
/// (which never has a predecessor) doesn't waste time; long enough
/// that the predecessor's decode plus apply_window can land for the
/// common case.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const WINDOW_WAIT_TIMEOUT: Duration = Duration::from_millis(50);

/// Default `FetchNextAdaptive` memory size. Matches the rapidgzip
/// default (Prefetcher.hpp uses 32; the production
/// `ParallelGzipReader` constructs it with the parallelization count
/// — we use a fixed 32 to bias toward longer sequential runs).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const FETCH_STRATEGY_MEMORY: usize = 32;

/// One unit of decode work submitted to the pool. The reply channel
/// carries the worker's result. Workers exit when the work-queue
/// sender is dropped (the scope's main thread, which holds the
/// sender, exits at end of `drive`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
struct DecodeJob {
    partition_idx: usize,
    start_bit: usize,
    until_bit: usize,
    /// True for the consumer's authoritative re-dispatch: we KNOW the
    /// predecessor's tail is available at `start_bit`, so workers must
    /// take the fast path (no fallback). For false (speculative
    /// prefetch) the worker waits a short timeout and falls back to
    /// the slow path on miss.
    authoritative: bool,
    /// Cache key under which the worker stores the result in
    /// `BlockFetcher`. For speculative jobs this is the partition seed
    /// (`partition_offsets[idx]`); for authoritative jobs the real
    /// expected start bit. Mirror of rapidgzip's `blockOffset` arg to
    /// `BlockFetcher::get` / `insert`.
    cache_key: usize,
    reply: mpsc::Sender<Result<ChunkData, ChunkDecodeError>>,
}

/// Public driver function. Replaces the old `GzipChunkFetcher` struct
/// API with a single entry point: decompress the entire raw deflate
/// stream, writing output bytes to `writer`, and returning the per-
/// chunk-combined CRC32 + total decoded size.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn drive<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    let chunk_size_bits = configuration.split_chunk_size * 8;
    let total_bits = input.len() * 8;
    let n_partitions = total_bits.div_ceil(chunk_size_bits).max(1);
    let partition_offsets: Vec<usize> = (0..n_partitions).map(|i| i * chunk_size_bits).collect();

    let window_map = WindowMap::new();
    // Chunk 0's input window is empty by definition (start of stream).
    let empty_window: Arc<[u8; 32768]> = Arc::new([0u8; 32768]);
    window_map.insert(0, empty_window);

    let pool_size = parallelization.max(1);

    // Cache sizing (mirror of rapidgzip BlockFetcher.hpp:160-170):
    // main cache holds the consumer's recent chunks for replay; the
    // prefetch cache holds in-flight prefetch results. Both grow with
    // parallelization. We bound to `2 * pool_size` for each, matching
    // rapidgzip's default `(parallelization + 1) * 8` heuristic
    // halved for our smaller chunk sizes (4 MiB vs 16 MiB).
    let cache_capacity = pool_size * 2;
    let prefetch_capacity = pool_size * 2;

    let block_fetcher: Arc<BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>> =
        Arc::new(BlockFetcher::new(
            cache_capacity,
            prefetch_capacity,
            FetchNextAdaptive::new(FETCH_STRATEGY_MEMORY),
            pool_size,
        ));
    let block_map = Arc::new(BlockMap::new());

    // Lock-free MPMC job queue. Each worker holds its own clone of the
    // Receiver and calls `recv()` directly — no `Mutex` serialization
    // around the channel. Mirror of rapidgzip's ThreadPool which lets
    // each worker pull from a shared deque without a global lock
    // (vendor/.../core/ThreadPool.hpp).
    let (job_tx, job_rx) = cbc::unbounded::<DecodeJob>();

    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size: usize = 0;

    let result = std::thread::scope(|s| -> Result<(), FetchError> {
        // Spawn worker pool.
        for _ in 0..pool_size {
            let job_rx = job_rx.clone();
            let window_map = window_map.clone();
            let block_fetcher = Arc::clone(&block_fetcher);
            let configuration = configuration;
            s.spawn(move || {
                // Arc::as_ref → &BlockFetcher matches worker_loop's
                // borrow signature; the Arc itself is moved into the
                // closure so the refcount keeps the fetcher alive for
                // the worker's lifetime.
                let bf: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive> = &block_fetcher;
                worker_loop(input, job_rx, window_map, bf, configuration)
            });
        }
        // Drop the parent thread's receiver clone so it doesn't keep
        // the queue alive past the consumer. crossbeam
        // `Receiver::recv()` returns `Err` only when ALL senders are
        // dropped AND the queue is empty — the consumer dropping
        // `job_tx` at scope exit cleanly signals worker shutdown after
        // every queued job is drained.
        drop(job_rx);

        consumer_loop(
            input,
            writer,
            n_partitions,
            &partition_offsets,
            total_bits,
            &job_tx,
            &window_map,
            &block_fetcher,
            &block_map,
            configuration,
            pool_size,
            &mut total_crc,
            &mut total_size,
        )?;

        // Dropping job_tx signals workers to exit.
        drop(job_tx);
        Ok(())
    });
    result?;

    // Finalize the BlockMap and stats (rapidgzip GzipChunkFetcher.hpp:324:
    // `m_blockMap->finalize();` after EOF).
    block_map.finalize();
    block_fetcher
        .statistics
        .base
        .set_block_count(block_map.data_block_count(), true);

    let crc = total_crc.finalize();
    Ok((crc, total_size))
}

/// Try to decode at `start_bit` directly; if that fails, iterate
/// BlockFinder candidates in [start_bit, start_bit + 512 KiB] until
/// one succeeds. Returns a chunk whose `encoded_offset_bits` is the
/// requested `start_bit` (matches the cache key) but whose
/// `max_encoded_offset_bits` is the actual candidate used.
/// Mirror of rapidgzip's tryToDecode + candidate iteration at
/// vendor/rapidgzip/.../GzipChunk.hpp:712-841.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_or_iterate(
    input: &[u8],
    start_bit: usize,
    until_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    // Direct port of rapidgzip's tryToDecode + alternating BlockFinder
    // search (GzipChunk.hpp:712-846). We DON'T try direct decode at
    // start_bit blindly — our finish_decode_chunk_with_inexact_offset's
    // marker bootstrap is too lenient with malformed headers (false
    // "success" on random bits produces a tiny chunk that misses
    // expected_start downstream). Instead, run BlockFinder to get
    // validated candidates (find_blocks does Kraft-precode +
    // Huffman-code validation matching rapidgzip's
    // seekToNonFinalDynamicDeflateBlock), iterate them — including a
    // candidate AT start_bit if find_blocks emits one — and accept the
    // first that try-decodes cleanly.
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
                // Adopt rapidgzip's semantic: encoded_offset = requested,
                // max_encoded_offset = actual candidate used.
                let actual = c.encoded_offset_bits;
                c.encoded_offset_bits = start_bit;
                c.max_encoded_offset_bits = actual;
                // encoded_size_bits is relative to encoded_offset_bits.
                // Decoder set it relative to `actual`; adjust so the
                // chunk reports its total span from the requested start.
                c.encoded_size_bits += actual - start_bit;
                // Subchunks were recorded with absolute encoded_offset_bits
                // — those are still valid (consumer's decoded_offset_for
                // looks up by absolute bit position).
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

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn worker_loop(
    input: &[u8],
    job_rx: cbc::Receiver<DecodeJob>,
    window_map: WindowMap,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>,
    configuration: ChunkConfiguration,
) {
    loop {
        // Lock-free recv on a cloned MPMC Receiver. Multiple workers
        // dequeue concurrently without serialization — replacing the
        // old `Arc<Mutex<mpsc::Receiver>>` design where exactly one
        // worker could be waiting on `recv()` at a time while N-1
        // others stalled on `.lock()`. Returns Err only when ALL
        // senders are dropped AND the queue is empty (clean shutdown).
        let job = match job_rx.recv() {
            Ok(j) => j,
            Err(_) => return,
        };

        let label = trace::worker_label(job.partition_idx);
        let t0 = std::time::Instant::now();
        let t_clock = trace::now_secs();
        block_fetcher
            .statistics
            .base
            .note_decode_block_start(t_clock);

        // For chunk 0 (start_bit==0) the empty window is the right
        // initial dict; insert is a no-op since we pre-seeded it.
        let window = if job.start_bit == 0 {
            window_map.get(0)
        } else if job.authoritative {
            // Consumer guarantees predecessor's window is in the map.
            window_map.get_or_wait(job.start_bit, Duration::from_secs(60))
        } else {
            // Speculative job: short wait. If miss, take slow path
            // with the empty window. Slow path is bounded (the
            // deflate_block::Block bootstrap stops at the first non-
            // fixed block boundary at-or-past until_bits) so a phantom
            // boundary returns Err quickly rather than hanging.
            window_map.get_or_wait(job.start_bit, WINDOW_WAIT_TIMEOUT)
        };

        let (result, path) = match window {
            Some(w) => {
                trace::emit(
                    &label,
                    "fast_path_start",
                    &format!(
                        r#""partition_idx":{},"start_bit":{},"until_bit":{},"authoritative":{},"rss_kib":{}"#,
                        job.partition_idx,
                        job.start_bit,
                        job.until_bit,
                        job.authoritative,
                        trace::rss_kib(),
                    ),
                );
                let r = decode_chunk_with_window(
                    input,
                    job.start_bit,
                    job.until_bit,
                    &w,
                    configuration,
                );
                (r, "fast")
            }
            None => {
                trace::emit(
                    &label,
                    "slow_path_start",
                    &format!(
                        r#""partition_idx":{},"start_bit":{},"until_bit":{},"rss_kib":{}"#,
                        job.partition_idx,
                        job.start_bit,
                        job.until_bit,
                        trace::rss_kib(),
                    ),
                );
                // Try direct at start_bit first (rapidgzip's tryToDecode
                // at GzipChunk.hpp:739). If it fails, iterate BlockFinder
                // candidates inside the worker, up to a 512 KiB scan.
                // The returned chunk reports encoded_offset_bits =
                // job.start_bit (the requested position) and
                // max_encoded_offset_bits = the actual candidate used.
                let r = decode_or_iterate(input, job.start_bit, job.until_bit, configuration);
                (r, "slow")
            }
        };

        let dur_us = t0.elapsed().as_micros();
        block_fetcher
            .statistics
            .base
            .add_decode_block_time(t0.elapsed().as_secs_f64());
        block_fetcher
            .statistics
            .base
            .note_decode_block_end(trace::now_secs());

        match &result {
            Ok(c) => {
                trace::emit(
                    &label,
                    "decode_ok",
                    &format!(
                        r#""partition_idx":{},"path":"{}","start_bit":{},"end_bit":{},"decoded":{},"markers":{},"clean":{},"preemptive":{},"duration_us":{dur_us}"#,
                        job.partition_idx,
                        path,
                        job.start_bit,
                        c.encoded_offset_bits + c.encoded_size_bits,
                        c.decoded_size(),
                        c.data_with_markers.len(),
                        c.data.len(),
                        c.stopped_preemptively,
                    ),
                );
                // Publish tail window if we can do it without waiting
                // for apply_window. Fast-path chunks always can. Slow-
                // path chunks whose trailing 32 KiB is in the clean
                // segment also can; otherwise the consumer publishes
                // after apply_window.
                if let Some(tail) = c.last_32kib_window() {
                    let end_bit = c.encoded_offset_bits + c.encoded_size_bits;
                    window_map.insert(end_bit, Arc::new(tail));
                }
                // Record the prefetch in BlockFetcher stats. We don't
                // insert the full ChunkData into the prefetch cache —
                // ownership must move via the reply channel so the
                // consumer can mutate it (apply_window, append_footer)
                // without an expensive Arc<Mutex<>> dance. Mirror of
                // rapidgzip's BlockFetcher.hpp:432-460 stats path; the
                // cache itself is populated by the consumer on
                // completion (see `block_fetcher.insert` below).
                block_fetcher.statistics.base.record_prefetch();
            }
            Err(e) => {
                trace::emit(
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
                );
                // Mirror rapidgzip BlockFetcher.hpp:600-620: failed
                // prefetches are remembered so the consumer doesn't
                // re-issue at the same key.
                if !job.authoritative {
                    block_fetcher.mark_failed_prefetch(job.cache_key);
                }
            }
        }
        let _ = job.reply.send(result);
    }
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn consumer_loop<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    n_partitions: usize,
    partition_offsets: &[usize],
    total_bits: usize,
    job_tx: &cbc::Sender<DecodeJob>,
    window_map: &WindowMap,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>,
    block_map: &BlockMap,
    configuration: ChunkConfiguration,
    pool_size: usize,
    total_crc: &mut crc32fast::Hasher,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    let _ = (input, configuration);

    // Port of `rapidgzip::GzipChunkFetcher::processNextChunk`
    // (GzipChunkFetcher.hpp:311-362). Iterates the BlockMap in
    // partition order, dispatching speculative prefetches via
    // BlockFetcher's stats path and falling back to authoritative
    // on-demand fetches when speculation misses.
    //
    // Each partition idx has a speculative job in flight, dispatched at
    // partition_offsets[idx]. The worker decodes at the seed and
    // iterates BlockFinder candidates within 512 KiB until one succeeds.
    // The returned chunk carries:
    //   encoded_offset_bits = partition_offsets[idx]   (requested seed)
    //   max_encoded_offset_bits = actual candidate bit-position used.
    //
    // Consumer at expected_start checks the speculative result:
    //   matches_encoded_offset(expected_start)
    //     ⇔ seed ≤ expected_start ≤ actual_candidate
    //     AND decoded_offset_for(expected_start).is_some()
    //     ⇔ there's a subchunk boundary at expected_start.
    // On hit, we trim trim_bytes and consume. On miss, we re-dispatch
    // an AUTHORITATIVE job at expected_start (worker fast-paths because
    // the predecessor's window is in the WindowMap), and discard the
    // speculative.

    let prefetch_count = std::env::var("GZIPPY_PREFETCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(pool_size * 2);

    let mut spec: Vec<Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>> =
        (0..n_partitions).map(|_| None).collect();

    let until_for = |idx: usize| -> usize {
        partition_offsets
            .get(idx + 1)
            .copied()
            .unwrap_or(total_bits)
    };

    let submit_job = |idx: usize,
                      start: usize,
                      until: usize,
                      authoritative: bool,
                      cache_key: usize|
     -> mpsc::Receiver<Result<ChunkData, ChunkDecodeError>> {
        let (tx, rx) = mpsc::channel();
        let event = if authoritative {
            "authoritative_prefetch"
        } else {
            "speculative_prefetch"
        };
        trace::emit(
            "consumer",
            event,
            &format!(r#""partition_idx":{idx},"start_bit":{start},"until_bit":{until}"#),
        );
        if authoritative {
            block_fetcher.statistics.base.record_on_demand_fetch();
        } else {
            block_fetcher.note_prefetch_started(cache_key);
        }
        job_tx
            .send(DecodeJob {
                partition_idx: idx,
                start_bit: start,
                until_bit: until,
                authoritative,
                cache_key,
                reply: tx,
            })
            .expect("worker pool dropped");
        rx
    };

    // Submit chunk 0 authoritative (start is known: bit 0). Also pre-fill
    // up to prefetch_count speculatives starting at partition 1.
    spec[0] = Some(submit_job(0, 0, until_for(0), true, 0));
    for i in 1..(1 + prefetch_count).min(n_partitions) {
        spec[i] = Some(submit_job(
            i,
            partition_offsets[i],
            until_for(i),
            false,
            partition_offsets[i],
        ));
    }

    let mut expected_start: usize = 0;
    let mut next_spec_to_dispatch: usize = (1 + prefetch_count).min(n_partitions);

    for idx in 0..n_partitions {
        // Update FetchingStrategy (rapidgzip BlockFetcher.hpp:280-282).
        block_fetcher.record_fetch(idx);

        // First, see if the chunk is already resolved in either cache.
        // For speculative jobs the worker inserts into the prefetch
        // cache and stats are recorded there. We track the cache hit
        // for completeness — currently the streaming consumer always
        // waits on the per-job channel (clone-cost of ChunkData is too
        // high to redundantly stash a full copy in the cache).
        let cache_key_for_partition = if idx == 0 { 0 } else { partition_offsets[idx] };
        let _ = block_fetcher.get_if_available(&cache_key_for_partition);

        // If speculation is disabled (prefetch_count = 0) or this slot
        // wasn't pre-filled, dispatch authoritative on demand.
        let rx = match spec[idx].take() {
            Some(rx) => rx,
            None => {
                let until = partition_offsets
                    .iter()
                    .skip(idx + 1)
                    .find(|&&s| s > expected_start)
                    .copied()
                    .unwrap_or(total_bits);
                submit_job(idx, expected_start, until, true, expected_start)
            }
        };
        trace::emit(
            "consumer",
            "speculative_wait",
            &format!(r#""partition_idx":{idx},"expected_start":{expected_start}"#),
        );
        let spec_result = match rx.recv() {
            Ok(r) => r,
            Err(_) => {
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: expected_start,
                    actual: 0,
                }));
            }
        };
        // The corresponding speculative prefetch (if any) is now done.
        // Mark it complete in the in-flight tracker.
        if idx != 0 {
            block_fetcher.note_prefetch_completed(&cache_key_for_partition);
        }

        // Hit test: speculative chunk must (a) decode successfully,
        // (b) cover expected_start, (c) have a subchunk boundary at
        // expected_start (so decoded_offset_for resolves). Chunk 0 is
        // authoritative (submitted with start = 0); for it, a hit is
        // guaranteed if Ok.
        let hit_chunk: Option<ChunkData> = match spec_result {
            Ok(mut c) => {
                if c.matches_encoded_offset(expected_start)
                    && c.decoded_offset_for(expected_start).is_some()
                {
                    trace::emit(
                        "consumer",
                        "speculative_hit",
                        &format!(
                            r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                            c.encoded_offset_bits, c.max_encoded_offset_bits,
                        ),
                    );
                    block_fetcher
                        .statistics
                        .base
                        .record_prefetch_cache_hit(true);
                    // Literal port: rapidgzip's processNextChunk calls
                    // setEncodedOffset(actual_offset) after the chunk
                    // resolves to collapse the [encoded, max] range to
                    // the exact start. No-op for chunks where encoded
                    // == max == expected_start, but preserves the
                    // rapidgzip semantic for future range-matching
                    // candidates (stored-block range offsets per
                    // Uncompressed.hpp's pair return).
                    if c.encoded_offset_bits != expected_start {
                        c.set_encoded_offset(expected_start);
                    }
                    Some(c)
                } else {
                    trace::emit(
                        "consumer",
                        "speculative_miss",
                        &format!(
                            r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                            c.encoded_offset_bits, c.max_encoded_offset_bits,
                        ),
                    );
                    block_fetcher.statistics.base.record_prefetch_cache_miss();
                    None
                }
            }
            Err(e) => {
                // Chunk 0 was dispatched authoritative; if it fails the
                // input is invalid. Speculative chunks may fail on
                // phantom boundaries — fall through to authoritative.
                if idx == 0 {
                    return Err(FetchError::Decode(e));
                }
                trace::emit(
                    "consumer",
                    "speculative_err",
                    &format!(
                        r#""partition_idx":{idx},"expected_start":{expected_start},"err":"{}""#,
                        trace::esc(&format!("{e:?}")),
                    ),
                );
                block_fetcher.statistics.base.record_prefetch_cache_miss();
                None
            }
        };

        let chunk = match hit_chunk {
            Some(c) => c,
            None => {
                // Re-dispatch authoritative at expected_start. Worker
                // will fast-path because predecessor's window is in
                // the WindowMap (just inserted after consume of N-1).
                let until = partition_offsets
                    .iter()
                    .skip(idx + 1)
                    .find(|&&s| s > expected_start)
                    .copied()
                    .unwrap_or(total_bits);
                let auth_rx = submit_job(idx, expected_start, until, true, expected_start);
                match auth_rx.recv() {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => return Err(FetchError::Decode(e)),
                    Err(_) => {
                        return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                            requested: expected_start,
                            actual: 0,
                        }));
                    }
                }
            }
        };

        let trim_bytes = chunk.decoded_offset_for(expected_start).unwrap_or(0);

        // Process the chunk: apply_window if markers; write bytes;
        // combine CRC; publish tail window.
        //
        // Window lookup is keyed by the ACTUAL decode start
        // (max_encoded_offset_bits), not the requested seed
        // (encoded_offset_bits). For speculative chunks these differ:
        // data corresponds to compressed range [actual, end] and the
        // markers in the leading region are back-refs into bytes ending
        // at compressed position `actual`. Predecessor publishes its
        // tail at its actual_end, which on a hit equals our `actual`.
        let mut chunk = chunk;
        if !chunk.data_with_markers.is_empty() {
            let window_key = chunk.max_encoded_offset_bits;
            let window = window_map
                .get_or_wait(window_key, Duration::from_secs(60))
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: window_key,
                    actual: expected_start,
                }))?;
            let aw_t0 = std::time::Instant::now();
            let marker_count = chunk.data_with_markers.len();
            apply_window(&mut chunk, &window[..]);
            // Literal port of `appendSubchunksToIndexes` window
            // emplacement (vendor/.../GzipChunkFetcher.hpp:430-458):
            // each subchunk gets its 32 KiB resume-window populated.
            chunk.populate_subchunk_windows(&window[..]);
            trace::emit(
                "consumer",
                "apply_window_done",
                &format!(
                    r#""partition_idx":{idx},"marker_bytes":{marker_count},"duration_us":{}"#,
                    aw_t0.elapsed().as_micros()
                ),
            );
            if let Some(tail) = chunk.last_32kib_window() {
                let end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                window_map.insert(end_bit, Arc::new(tail));
            }
            // Publish per-subchunk windows so future workers waiting on
            // an intermediate subchunk's bit position can fast-path.
            // Mirrors rapidgzip's per-subchunk WindowMap emplacement
            // (GzipChunkFetcher.hpp:430-458).
            for sc in &chunk.subchunks {
                if let Some(ref w) = sc.window {
                    let sc_end_bit = sc.encoded_offset_bits + sc.encoded_size_bits;
                    if sc_end_bit > sc.encoded_offset_bits {
                        window_map.insert(sc_end_bit, w.clone());
                    }
                }
            }
        }

        // Push subchunks into the BlockMap — literal port of
        // `appendSubchunksToIndexes` at GzipChunkFetcher.hpp:371-375.
        // The map is the random-access index keyed by encoded bit
        // offset; sequential streaming doesn't consult it but
        // maintaining it preserves rapidgzip's semantics.
        append_subchunks_to_block_map(block_map, &chunk);

        // Write bytes, skipping `trim_bytes` leading bytes that belong
        // to the predecessor's range.
        let mut written_crc = crc32fast::Hasher::new();
        let mut remaining_skip = trim_bytes;
        if !chunk.data_with_markers.is_empty() {
            let dwm_len = chunk.data_with_markers.len();
            let skip_in_dwm = remaining_skip.min(dwm_len);
            remaining_skip -= skip_in_dwm;
            if skip_in_dwm < dwm_len {
                let mut narrowed: Vec<u8> = Vec::with_capacity(dwm_len - skip_in_dwm);
                for v in &chunk.data_with_markers[skip_in_dwm..] {
                    narrowed.push(*v as u8);
                }
                written_crc.update(&narrowed);
                writer
                    .write_all(&narrowed)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                *total_size += narrowed.len();
            }
        }
        if !chunk.data.is_empty() {
            let skip_in_data = remaining_skip.min(chunk.data.len());
            if skip_in_data < chunk.data.len() {
                let slice = &chunk.data[skip_in_data..];
                written_crc.update(slice);
                writer
                    .write_all(slice)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                *total_size += slice.len();
            }
        }
        total_crc.combine(&written_crc);
        expected_start = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        // record_get covers the "consumer pulled chunk idx out of the
        // pipeline" event; mirrors rapidgzip's stats.recordBlockIndexGet
        // at BlockFetcher.hpp:270-272.
        block_fetcher.statistics.base.record_get();
        #[cfg(any(test, debug_assertions))]
        BLOCK_FETCHER_GETS_OBSERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let trace_decoded = chunk.decoded_size();
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"end_bit":{expected_start},"decoded":{trace_decoded},"trim_bytes":{trim_bytes},"rss_kib":{}"#,
                trace::rss_kib(),
            ),
        );
        // Insert the fully-processed chunk into BlockFetcher's main
        // cache for potential future random-access replay. Mirror of
        // rapidgzip's `insertIntoCache` after consumer completion at
        // BlockFetcher.hpp:320-327. The LRU cache will evict older
        // entries on overflow; the streaming consumer never replays so
        // the cache is structurally maintained but practically unused
        // by this code path. Wrapped in Arc to match BlockFetcher's
        // Value-type constraint and to keep insertion O(1).
        block_fetcher.insert(cache_key_for_partition, Arc::new(chunk));

        // Refill the speculative pipeline: keep prefetch_count chunks
        // outstanding ahead of the consumer.
        if next_spec_to_dispatch < n_partitions {
            spec[next_spec_to_dispatch] = Some(submit_job(
                next_spec_to_dispatch,
                partition_offsets[next_spec_to_dispatch],
                until_for(next_spec_to_dispatch),
                false,
                partition_offsets[next_spec_to_dispatch],
            ));
            next_spec_to_dispatch += 1;
        }
    }

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

    /// Spec §I "Tests required (new)" #4: assert that the production
    /// `drive` path constructs and uses a BlockFetcher. The
    /// BLOCK_FETCHER_GETS_OBSERVED counter increments each time the
    /// consumer calls `block_fetcher.statistics.base.record_get()`. If
    /// a future refactor "forgets" to thread BlockFetcher through, the
    /// counter stays flat and this test fails — the same deletion-trap
    /// pattern `MARKER_PIPELINE_RUNS` uses.
    #[test]
    fn test_block_fetcher_in_drive() {
        use std::sync::atomic::Ordering;
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let before = BLOCK_FETCHER_GETS_OBSERVED.load(Ordering::Relaxed);
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 4, cfg).expect("drive");
        assert_eq!(size, payload.len());
        let after = BLOCK_FETCHER_GETS_OBSERVED.load(Ordering::Relaxed);
        assert!(
            after > before,
            "BlockFetcher::record_get was not invoked during drive (before={before}, \
             after={after}). The BlockFetcher pipeline was bypassed — \
             chunk_fetcher::drive's BlockFetcher wiring has regressed."
        );
    }
}
