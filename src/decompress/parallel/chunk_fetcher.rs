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
//! port keeps the existing std::thread + mpsc worker pool — equivalent
//! semantics, simpler dependencies. The consumer's `processNextChunk`
//! body (literal port of GzipChunkFetcher.hpp:311-362):
//!
//!   1. Query the next encoded bit offset from
//!      [`GzipBlockFinder::get`] (literal port of
//!      GzipChunkFetcher.hpp:318 `m_blockFinder->get(...)`). Record
//!      the access through [`BlockFetcher::record_fetch`] so
//!      [`FetchNextAdaptive`] tracks the sequential pattern.
//!   2. Call [`BlockFetcher::get`] (literal port of
//!      BlockFetcher.hpp:245-329). The dispatch closure encapsulates:
//!      (a) drain the speculatively-prefetched `SpecSlot` if present,
//!      (b) validate the hit (matches `expected_start` + has a
//!      subchunk boundary at it), (c) on miss/failure, submit
//!      authoritative at `expected_start` to the mpsc worker pool and
//!      wait on the reply. On cache hit `get` returns the cached
//!      `Arc<ChunkData>` directly (literal port of
//!      BlockFetcher.hpp:302-309); on miss it caches the dispatch
//!      result under the offset (literal port of
//!      BlockFetcher.hpp:320 `insertIntoCache`).
//!   3. `chunkData->setEncodedOffset(*nextBlockOffset)` (mirror of
//!      GzipChunkFetcher.hpp:349) — done inside the dispatch closure
//!      when the speculative seed differs from `expected_start`.
//!   4. Resolve markers via [`apply_window`]; push subchunks into
//!      [`block_map::append_subchunks_to_block_map`]; populate the
//!      [`WindowMap`] per the rapidgzip
//!      `appendSubchunksToIndexes` cascade (GzipChunkFetcher.hpp:430-458).
//!   5. `m_blockFinder->insert(subchunk_end)` per subchunk (literal
//!      port of GzipChunkFetcher.hpp:374) — closes the partitioner-
//!      feedback loop so subsequent `get(idx)` queries return
//!      confirmed offsets.
//!   6. Write decoded bytes, combine CRC32, move on.
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

/// Test-only observability: number of times the production `drive`
/// path called `BlockFetcher::record_get`. Used by
/// `test_block_fetcher_in_drive` to lock in that the cutover's
/// BlockFetcher wiring is actually exercised (a deletion-trap killer
/// in the same spirit as `MARKER_PIPELINE_RUNS`).
#[cfg(any(test, debug_assertions))]
pub static BLOCK_FETCHER_GETS_OBSERVED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Test-only observability: number of times the production `drive` path
/// called `GzipBlockFinder::insert(actual_end)` after a chunk
/// completed. Deletion-trap for the partitioner-feedback wiring; if a
/// refactor "forgets" to call insert, this counter stays flat and
/// `test_gzip_block_finder_in_drive` fails. Mirror of rapidgzip's
/// `m_blockFinder->insert(subchunk.encodedOffset + subchunk.encodedSize)`
/// at vendor/.../GzipChunkFetcher.hpp:374.
#[cfg(any(test, debug_assertions))]
pub static GZIP_BLOCK_FINDER_INSERTS_OBSERVED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Test-only observability: number of times the production `drive` path
/// invoked the `BlockFetcher::get` synchronous dispatch primitive.
/// Deletion-trap for the rapidgzip `BlockFetcher::get(blockOffset)` port
/// — see BlockFetcher.hpp:245-329. If a refactor reverts the consumer
/// back to the spec-ring + reply-channel dispatch without going through
/// `BlockFetcher::get`, this counter stays flat.
#[cfg(any(test, debug_assertions))]
pub static BLOCK_FETCHER_GET_CALLS_OBSERVED: std::sync::atomic::AtomicU64 =
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

/// One unit of post-processing work submitted to the pool. Mirror of
/// rapidgzip's lambda enqueued via `submitTaskWithHighPriority` at
/// `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:577-583), which calls
/// `chunkData->applyWindow( *window, chunkData->windowCompressionType() )`.
/// Our task additionally calls `populate_subchunk_windows` (the gzippy
/// equivalent of vendor's per-subchunk window emplacement at
/// GzipChunkFetcher.hpp:430-458, which vendor also produces inside the
/// applyWindow path via `chunkData->getWindowAt(...)`).
///
/// The tail window has ALREADY been computed via
/// `ChunkData::get_last_window` and inserted into the `WindowMap` on
/// the consumer thread BEFORE this job is enqueued — vendor parity at
/// GzipChunkFetcher.hpp:558-575 (the `m_windowMap->emplace(...)` calls
/// at lines 570 + 572-573 run before the `submitTaskWithHighPriority`
/// at line 579).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
struct PostProcessJob {
    partition_idx: usize,
    chunk: ChunkData,
    predecessor_window: Arc<[u8; 32768]>,
    /// Window map the job will use to publish per-subchunk windows
    /// AFTER `populate_subchunk_windows` runs. Mirror of vendor's
    /// per-subchunk emplacement at
    /// `appendSubchunksToIndexes` (GzipChunkFetcher.hpp:430-458). The
    /// tail window has already been published on the consumer thread
    /// via `get_last_window`; this only emplaces interior subchunk
    /// resume-windows.
    window_map: WindowMap,
    reply: mpsc::Sender<ChunkData>,
}

/// Job variant on the unified work channel. Workers receive `Job`,
/// dispatch on the variant. One pool, two work classes — vendor's
/// `submitTask` / `submitTaskWithHighPriority` collapse to the same
/// `BS::thread_pool` instance, so we keep that 1:1.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
enum Job {
    Decode(DecodeJob),
    PostProcess(PostProcessJob),
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

    // ── Partitioner: `GzipBlockFinder` ──────────────────────────────
    // Literal port of rapidgzip's partitioning model. The block finder
    // is constructed with the first confirmed offset (`0` for a
    // single-member stream — the deflate body starts at bit 0 of the
    // raw deflate stream we receive) and the spacing in bytes equal to
    // the chunk size. Workers do not consult it; the consumer queries
    // `get(idx)` for the seed of partition `idx` and `get(idx+1)` for
    // its `until_bit`. After each chunk completes the consumer calls
    // `insert(actual_end)` to promote that boundary from a guess to a
    // confirmed offset — mirror of
    // vendor/rapidgzip/.../GzipChunkFetcher.hpp:374
    // (`m_blockFinder->insert(subchunk.encodedOffset + subchunk.encodedSize)`).
    let block_finder_par = Arc::new(GzipBlockFinder::new(
        /* first_block_offset_in_bits = */ 0,
        /* spacing_in_bytes = */ configuration.split_chunk_size,
        /* file_size_in_bits = */ Some(total_bits),
    ));

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

    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

    // Mirror of `m_crc32Calculator` (GzipChunkFetcher.hpp:309) — the
    // running stream CRC, accumulated across chunks via append() so the
    // gzip footer verification at the end of stream goes through the
    // ported `CRC32Calculator::verify` surface
    // (vendor/.../gzip/crc32.hpp:308-319). The polynomial combine in
    // append() / prepend() goes through `combine_crc32`
    // (crc32.hpp:214-258) rather than `crc32fast::Hasher::combine` so the
    // new ports drive production behavior.
    let mut total_crc = CRC32Calculator::new();
    let mut total_size: usize = 0;

    let result = std::thread::scope(|s| -> Result<(), FetchError> {
        // Spawn worker pool.
        for _ in 0..pool_size {
            let job_rx = Arc::clone(&job_rx);
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

        consumer_loop(
            input,
            writer,
            n_partitions,
            &block_finder_par,
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
    // `m_blockMap->finalize();` after EOF). Mirror of
    // GzipChunkFetcher.hpp:325 + 407: finalize both the block map AND the
    // block finder once the input has been fully consumed.
    block_map.finalize();
    block_finder_par.finalize();
    block_fetcher
        .statistics
        .base
        .set_block_count(block_map.data_block_count(), true);

    let crc = total_crc.crc32();
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

/// Worker-side execution of a `PostProcessJob`. Mirror of the lambda
/// body at `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:579-582):
///
///     this->submitTaskWithHighPriority(
///         [chunkData, window = std::move( previousWindow )] () {
///             chunkData->applyWindow( *window, chunkData->windowCompressionType() );
///         } )
///
/// Runs `apply_window` (resolves cross-chunk markers + folds resolved
/// bytes' CRC into the chunk's per-stream calculator), then
/// `populate_subchunk_windows` (per-subchunk 32 KiB resume windows),
/// then ships the processed chunk back to the consumer via `reply`.
/// All output-write + stream-CRC combine work stays on the consumer
/// thread because both are order-dependent.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn run_post_process_job(mut job: PostProcessJob) {
    let trace_on = trace::is_enabled();
    let t0 = if trace_on {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let marker_count = job.chunk.data_with_markers.len();

    // Vendor parity: `chunkData->applyWindow(*window, ...)` at
    // vendor/.../GzipChunkFetcher.hpp:581.
    apply_window(&mut job.chunk, &job.predecessor_window[..]);
    // Per-subchunk windows are derived from the chunk's own resolved
    // output prefixed by the predecessor's tail. Vendor produces these
    // either inside `applyWindow` (vendor ChunkData.hpp:333-366) via
    // `getWindowAt(window, decodedOffsetInBlock)` or, for skipped
    // subchunks, in `appendSubchunksToIndexes`
    // (GzipChunkFetcher.hpp:443-446 `getWindowAt(lastWindow, ...)`).
    // gzippy collapses both into one explicit call here so the consumer
    // never touches markers.
    job.chunk
        .populate_subchunk_windows(&job.predecessor_window[..]);

    // Per-subchunk windows: publish so future workers waiting on an
    // intermediate subchunk's bit position can fast-path. Mirror of
    // rapidgzip's per-subchunk WindowMap emplacement
    // (GzipChunkFetcher.hpp:430-458). The TAIL window has already been
    // published by the consumer thread via `get_last_window` BEFORE
    // this job was enqueued (vendor lines 558-575).
    for sc in &job.chunk.subchunks {
        if let Some(ref w) = sc.window {
            let sc_end_bit = sc.encoded_offset_bits + sc.encoded_size_bits;
            if sc_end_bit > sc.encoded_offset_bits {
                job.window_map.insert(sc_end_bit, w.clone());
            }
        }
    }

    if let Some(t) = t0 {
        trace::emit(
            "postprocess",
            "apply_window_done",
            &format!(
                r#""partition_idx":{},"marker_bytes":{marker_count},"duration_us":{}"#,
                job.partition_idx,
                t.elapsed().as_micros()
            ),
        );
    }
    let _ = job.reply.send(job.chunk);
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn worker_loop(
    input: &[u8],
    job_rx: Arc<std::sync::Mutex<mpsc::Receiver<Job>>>,
    window_map: WindowMap,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>,
    configuration: ChunkConfiguration,
) {
    loop {
        // Pull one job. Mutex serializes only the recv() — the actual
        // decode/post-process runs in parallel across workers.
        let job = {
            let rx = job_rx.lock().unwrap();
            match rx.recv() {
                Ok(j) => j,
                Err(_) => return, // sender dropped → shutdown
            }
        };

        let job = match job {
            Job::Decode(d) => d,
            Job::PostProcess(p) => {
                run_post_process_job(p);
                continue;
            }
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
                // Gate format!/rss_kib() behind is_enabled() — the vendor
                // has no equivalent per-worker syscall/string-format on the
                // hot path. `rss_kib()` reads and parses
                // `/proc/self/status` on every call, which we evaluate
                // eagerly before `emit` short-circuits. Skipping the
                // string-formatting and the syscall when tracing is OFF
                // makes the worker entry as cheap as the vendor's.
                if trace::is_enabled() {
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
                }
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
                if trace::is_enabled() {
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
                }
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
                if trace::is_enabled() {
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
                }
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
                if trace::is_enabled() {
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
                }
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
    block_finder: &GzipBlockFinder,
    total_bits: usize,
    job_tx: &mpsc::Sender<DecodeJob>,
    window_map: &WindowMap,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>,
    block_map: &BlockMap,
    configuration: ChunkConfiguration,
    pool_size: usize,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    let _ = (input, configuration);

    // Helper: query the GzipBlockFinder for partition `idx`'s seed.
    // Mirror of rapidgzip's `m_blockFinder->get(blockIndex)` pattern at
    // vendor/.../GzipChunkFetcher.hpp:318. The block finder returns the
    // confirmed offset when `idx` is known (after a predecessor's
    // `insert(actual_end)`), or a spacing-aligned guess when not.
    //
    // IMPORTANT: this helper takes the BLOCK INDEX (vendor's
    // `m_nextUnprocessedBlockIndex`), which after subchunk inserts is
    // NOT the same as the partition counter `idx` in the consumer loop.
    // Use `seed_for(next_block_index)` at the call site; see
    // `next_block_index` initialization in `consumer_loop`.
    let seed_for = |block_index: usize| -> usize {
        match block_finder.get(block_index) {
            (Some(offset), GetReturnCode::Success) => offset,
            // Past-EOF / failure: clamp to file size so `until_for` is
            // monotonic. The consumer will hit this on the final
            // partition's `until_for(n - 1) = get(n)` query.
            (Some(offset), GetReturnCode::Failure) => offset,
            (None, _) => total_bits,
        }
    };
    let until_for = |block_index: usize| -> usize { seed_for(block_index + 1).min(total_bits) };
    // Compute the spacing-aligned guess for partition `idx`. Used for
    // the INITIAL speculative dispatch round (before any subchunk inserts)
    // and for the speculative-refill ring during consume. Equivalent to
    // `block_finder.get(idx)` when block_finder has only the initial seed
    // (mirror of GzipBlockFinder.hpp::get's guessed-offset branch at
    // L134-157, which returns `partition_index * spacing_in_bits`).
    let partition_seed_for = |partition_idx: usize| -> usize {
        let chunk_size_bits = configuration.split_chunk_size * 8;
        (partition_idx * chunk_size_bits).min(total_bits)
    };
    let partition_until_for =
        |partition_idx: usize| -> usize { partition_seed_for(partition_idx + 1).min(total_bits) };

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

    /// Tracks one outstanding speculative job. We carry the seed that
    /// was active at dispatch time so the cache key remains stable
    /// across subsequent `block_finder.insert(...)` calls that would
    /// otherwise shift `seed_for(idx)` (see
    /// vendor/.../GzipBlockFinder.hpp:30-32: "block confirmation
    /// effectively invalidates previous indexes"). Mirror of
    /// rapidgzip's keeping the original `blockOffset` (not blockIndex)
    /// as the BlockFetcher cache key across the get/insert cycle —
    /// vendor/.../GzipChunkFetcher.hpp:267 + 596.
    struct SpecSlot {
        rx: mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>,
        dispatched_seed: usize,
    }
    let mut spec: Vec<Option<SpecSlot>> = (0..n_partitions).map(|_| None).collect();

    let submit_job = |idx: usize,
                      start: usize,
                      until: usize,
                      authoritative: bool,
                      cache_key: usize|
     -> mpsc::Receiver<Result<ChunkData, ChunkDecodeError>> {
        let (tx, rx) = mpsc::channel();
        if trace::is_enabled() {
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
        }
        if authoritative {
            block_fetcher.statistics.base.record_on_demand_fetch();
        } else {
            block_fetcher.note_prefetch_started(cache_key);
        }
        job_tx
            .send(Job::Decode(DecodeJob {
                partition_idx: idx,
                start_bit: start,
                until_bit: until,
                authoritative,
                cache_key,
                reply: tx,
            }))
            .expect("worker pool dropped");
        rx
    };

    // ── Post-process job dispatch (vendor parity) ──────────────────────
    // Mirror of `queueChunkForPostProcessing`
    // (vendor/.../GzipChunkFetcher.hpp:553-583). The caller MUST have
    // already inserted the chunk's TAIL window into the WindowMap via
    // `ChunkData::get_last_window` (vendor lines 558-575) BEFORE
    // calling this helper — that is the critical-path step that lets
    // downstream chunk workers unblock while this chunk's apply_window
    // is still queued.
    let submit_post_process = |idx: usize,
                               chunk: ChunkData,
                               predecessor_window: Arc<[u8; 32768]>|
     -> mpsc::Receiver<ChunkData> {
        let (tx, rx) = mpsc::channel();
        if trace::is_enabled() {
            trace::emit(
                "consumer",
                "post_process_dispatch",
                &format!(
                    r#""partition_idx":{idx},"markers":{}"#,
                    chunk.data_with_markers.len(),
                ),
            );
        }
        job_tx
            .send(Job::PostProcess(PostProcessJob {
                partition_idx: idx,
                chunk,
                predecessor_window,
                window_map: window_map.clone(),
                reply: tx,
            }))
            .expect("worker pool dropped");
        rx
    };

    // Submit chunk 0 authoritative (start is known: bit 0). Also pre-fill
    // up to prefetch_count speculatives starting at partition 1. The
    // seed for each partition comes from `GzipBlockFinder::get(idx)` —
    // a confirmed offset when one of idx's predecessors has finished
    // and called `insert(actual_end)`, or a spacing-aligned guess
    // otherwise. Mirror of rapidgzip's
    // GzipChunkFetcher.hpp:318 (`m_blockFinder->get(...)`) +
    // BlockFetcher.hpp:479 (prefetch-time `get(blockIndexToPrefetch, 0)`).
    // Initial dispatch uses partition-aligned seeds (block_finder has
    // only the initial seed `[0]` at this point, so `block_finder.get(i)`
    // returns the spacing-aligned guess `i * spacing`). After the loop
    // starts processing chunks and inserting subchunk boundaries, the
    // block_finder's index-to-offset mapping shifts; we use
    // `partition_seed_for` to keep the spec ring aligned to partition
    // boundaries regardless of insert progress.
    spec[0] = Some(SpecSlot {
        rx: submit_job(0, 0, partition_until_for(0), true, 0),
        dispatched_seed: 0,
    });
    for i in 1..(1 + prefetch_count).min(n_partitions) {
        let seed = partition_seed_for(i);
        spec[i] = Some(SpecSlot {
            rx: submit_job(i, seed, partition_until_for(i), false, seed),
            dispatched_seed: seed,
        });
    }

    let mut expected_start: usize = 0;
    let mut next_spec_to_dispatch: usize = (1 + prefetch_count).min(n_partitions);
    // Mirror of vendor `m_nextUnprocessedBlockIndex`
    // (vendor/.../GzipChunkFetcher.hpp:318 + 410). Tracks the running
    // count of subchunks INSERTED into `block_finder`. Each chunk's
    // subchunks bump this by `chunk.subchunks.len()` after processing
    // (line 410: `m_nextUnprocessedBlockIndex += subchunks.size();`).
    // We query `block_finder.get(next_block_index)` to find the next
    // chunk's seed — equivalent to vendor's line 318 query.
    //
    // CRITICAL: the SPEC-RING `idx` is still a 0..n_partitions counter
    // (used to address `spec[idx]`), but the seed/until queries MUST
    // use `next_block_index` instead. After idx=0 inserts N subchunk
    // boundaries, `block_finder.get(1)` returns the end of subchunk 0
    // (an INTERNAL position in chunk 0's range), not the next chunk's
    // start. Vendor avoids this by advancing the index by subchunks.size(),
    // which is what we do here.
    let mut next_block_index: usize = 0;

    // ── Post-process pending queue (vendor parity) ────────────────────
    //
    // `pending` holds chunks awaiting their write-to-output step. Order
    // is encoded order (push_back on dispatch, pop_front on drain).
    // `post_process_inflight_cap` bounds how many post-process jobs are
    // allowed to be in flight simultaneously — anything beyond that
    // forces a drain on the oldest. We size it to `pool_size` so the
    // entire worker pool can be saturated with apply_window tasks while
    // the consumer is still acquiring downstream decoded chunks. Mirror
    // of vendor's `m_markersBeingReplaced` map +
    // `waitForReplacedMarkers` driving behavior
    // (vendor/.../GzipChunkFetcher.hpp:267-518), where the same pool
    // accepts both decode tasks and `applyWindow` tasks and the
    // consumer's wait is implicit through the per-chunk future.
    let mut pending: std::collections::VecDeque<PendingWrite> =
        std::collections::VecDeque::with_capacity(pool_size * 2);
    let post_process_inflight_cap = pool_size;

    for idx in 0..n_partitions {
        // Update FetchingStrategy (rapidgzip BlockFetcher.hpp:280-282).
        block_fetcher.record_fetch(idx);

        // Take the in-flight slot (if any) so we can use its
        // dispatched_seed as the stable cache key — see SpecSlot's
        // comment about block-confirmation invalidating indexes
        // (vendor/.../GzipBlockFinder.hpp:30-32).
        let slot = spec[idx].take();
        let cache_key_for_partition = match &slot {
            Some(s) => s.dispatched_seed,
            None if idx == 0 => 0,
            None => partition_seed_for(idx),
        };

        // ALL chunk acquisition flows through `block_fetcher.get`
        // (rapidgzip BlockFetcher::get at
        // vendor/rapidgzip/.../core/BlockFetcher.hpp:245-329). The
        // dispatch closure encapsulates: (a) drain the speculative
        // slot if pre-filled, (b) validate the hit (matches expected
        // start + has a subchunk boundary), (c) on miss/failure
        // re-dispatch authoritative at `expected_start`. This makes
        // the consumer's call shape a literal port of
        // `processNextChunk`'s single-call `getBlock`/`BaseType::get`
        // pattern at GzipChunkFetcher.hpp:293 + 654.
        //
        // The `until` hint for the worker MUST come from the SUBCHUNK-
        // indexed BlockFinder via `next_block_index`, NOT from the
        // partition counter `idx`. Mirror of vendor's
        // `BlockFetcher::get` which computes
        // `nextBlockOffset = m_blockFinder->get(validDataBlockIndex + 1)`
        // (vendor/.../core/BlockFetcher.hpp:268). After predecessors
        // insert subchunk boundaries, `block_finder.get(idx + 1)` would
        // return an INTERIOR position from a prior chunk's subchunks
        // (because `idx` indexes into block_offsets, not partitions).
        // Using `next_block_index + 1` skips past the predecessor's
        // subchunks correctly and yields the NEXT chunk's seed (or a
        // spacing-aligned guess if not yet confirmed).
        let until = until_for(next_block_index).max(expected_start);
        let auth_result: Result<Arc<ChunkData>, ChunkDecodeError> = block_fetcher.get(
            expected_start,
            || -> Result<Arc<ChunkData>, ChunkDecodeError> {
                #[cfg(any(test, debug_assertions))]
                BLOCK_FETCHER_GET_CALLS_OBSERVED
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Step 1: drain the pre-filled speculative slot (if
                // any) — equivalent to rapidgzip's
                // `takeFromPrefetchQueue` at BlockFetcher.hpp:385-410.
                let speculative_chunk: Option<ChunkData> = match slot {
                    Some(s) => {
                        if trace::is_enabled() {
                            trace::emit(
                                "consumer",
                                "speculative_wait",
                                &format!(
                                    r#""partition_idx":{idx},"expected_start":{expected_start}"#
                                ),
                            );
                        }
                        let spec_result = match s.rx.recv() {
                            Ok(r) => r,
                            Err(_) => {
                                return Err(ChunkDecodeError::ExactStopMissed {
                                    requested: expected_start,
                                    actual: 0,
                                });
                            }
                        };
                        if idx != 0 {
                            block_fetcher.note_prefetch_completed(&cache_key_for_partition);
                        }
                        match spec_result {
                            Ok(mut c) => {
                                // Validate the speculative hit: must
                                // cover expected_start AND have a
                                // subchunk boundary there.
                                if c.matches_encoded_offset(expected_start)
                                    && c.decoded_offset_for(expected_start).is_some()
                                {
                                    if trace::is_enabled() {
                                        trace::emit(
                                            "consumer",
                                            "speculative_hit",
                                            &format!(
                                                r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                                                c.encoded_offset_bits,
                                                c.max_encoded_offset_bits,
                                            ),
                                        );
                                    }
                                    block_fetcher
                                        .statistics
                                        .base
                                        .record_prefetch_cache_hit(true);
                                    // Mirror of rapidgzip
                                    // GzipChunkFetcher.hpp:349
                                    // `chunkData->setEncodedOffset(*nextBlockOffset)`.
                                    if c.encoded_offset_bits != expected_start {
                                        c.set_encoded_offset(expected_start);
                                    }
                                    Some(c)
                                } else {
                                    if trace::is_enabled() {
                                        trace::emit(
                                            "consumer",
                                            "speculative_miss",
                                            &format!(
                                                r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                                                c.encoded_offset_bits,
                                                c.max_encoded_offset_bits,
                                            ),
                                        );
                                    }
                                    block_fetcher.statistics.base.record_prefetch_cache_miss();
                                    None
                                }
                            }
                            Err(e) => {
                                // Chunk 0 is authoritative: failure is
                                // a hard error per
                                // GzipChunkFetcher.hpp:656-661.
                                if idx == 0 {
                                    return Err(e);
                                }
                                if trace::is_enabled() {
                                    trace::emit(
                                        "consumer",
                                        "speculative_err",
                                        &format!(
                                            r#""partition_idx":{idx},"expected_start":{expected_start},"err":"{}""#,
                                            trace::esc(&format!("{e:?}")),
                                        ),
                                    );
                                }
                                block_fetcher.statistics.base.record_prefetch_cache_miss();
                                None
                            }
                        }
                    }
                    None => None,
                };

                // Step 2: on speculative hit, return the validated
                // chunk. On miss (or no slot pre-filled), dispatch
                // authoritative — mirror of
                // GzipChunkFetcher.hpp:654 (`BaseType::get(blockOffset,
                // blockIndex, getPartitionOffsetFromOffset)`).
                if let Some(c) = speculative_chunk {
                    return Ok(Arc::new(c));
                }
                let rx = submit_job(idx, expected_start, until, true, expected_start);
                match rx.recv() {
                    Ok(Ok(c)) => Ok(Arc::new(c)),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(ChunkDecodeError::ExactStopMissed {
                        requested: expected_start,
                        actual: 0,
                    }),
                }
            },
        );
        let auth_arc = match auth_result {
            Ok(a) => a,
            Err(e) => return Err(FetchError::Decode(e)),
        };
        // Take ownership via Arc::try_unwrap when we hold the only
        // ref; otherwise clone the inner ChunkData. This mirrors
        // rapidgzip's pattern of mutating through the shared_ptr —
        // in Rust the safe equivalent is copy-on-write when the cache
        // still aliases it. ChunkData is Clone (chunk_data.rs:106), so
        // clone-on-shared is structurally cheap and faithful to the
        // rapidgzip aliasing semantics at GzipChunkFetcher.hpp:329.
        let chunk: ChunkData = Arc::try_unwrap(auth_arc).unwrap_or_else(|arc| (*arc).clone());

        // Mirror of vendor `GzipChunkFetcher::processNextChunk`
        // (vendor/.../GzipChunkFetcher.hpp:350-355):
        //
        //   /* Should only happen when encountering EOF during decodeBlock call. */
        //   if ( chunkData->encodedSizeInBits == 0 ) {
        //       m_blockMap->finalize();
        //       m_blockFinder->finalize();
        //       return {};
        //   }
        //
        // Reaching this with `encoded_size_bits == 0` means the worker
        // decoded zero compressed bits at `expected_start` — the
        // deflate stream has ended.
        if chunk.encoded_size_bits == 0 {
            if trace::is_enabled() {
                trace::emit(
                    "consumer",
                    "eof_terminate",
                    &format!(
                        r#""partition_idx":{idx},"expected_start":{expected_start},"total_size":{}"#,
                        *total_size,
                    ),
                );
            }
            break;
        }

        let trim_bytes = chunk.decoded_offset_for(expected_start).unwrap_or(0);

        // ── Vendor-parity tail-window publication (critical path) ─────
        //
        // Mirror of rapidgzip's `getLastWindow` + `WindowMap::emplace`
        // pair at `GzipChunkFetcher::queueChunkForPostProcessing`
        // (vendor/.../GzipChunkFetcher.hpp:553-575). The tail window is
        // computed synchronously on the consumer thread (O(W) work) so
        // that the successor chunk's worker — which may already be
        // running speculatively — can unblock IMMEDIATELY, BEFORE this
        // chunk's full `applyWindow` runs on a worker thread (vendor
        // line 579: `submitTaskWithHighPriority(... applyWindow ...)`).
        //
        // Without this split, `applyWindow` ran synchronously on the
        // consumer thread and serialised the entire pipeline behind the
        // O(N) marker resolution + per-subchunk window population. The
        // bench-sm gap from rapidgzip (442 MB/s vs 1349 MB/s) was
        // dominated by this serialisation.
        //
        // Window lookup is keyed by the ACTUAL decode start
        // (max_encoded_offset_bits), not the requested seed
        // (encoded_offset_bits). For speculative chunks these differ:
        // data corresponds to compressed range [actual, end] and the
        // markers in the leading region are back-refs into bytes ending
        // at compressed position `actual`. Predecessor publishes its
        // tail at its actual_end, which on a hit equals our `actual`.
        let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let predecessor_window: Option<Arc<[u8; 32768]>>;
        if !chunk.data_with_markers.is_empty() {
            let window_key = chunk.max_encoded_offset_bits;
            let window = window_map
                .get_or_wait(window_key, Duration::from_secs(60))
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: window_key,
                    actual: expected_start,
                }))?;
            // Vendor parity: lines 558-574 — if the chunk's tail position
            // (`windowOffset`) is not already in the map, compute it via
            // `getLastWindow(*previousWindow)` and emplace it (with
            // `CompressionType::NONE` — vendor doesn't compress the
            // critical-path window). We always emplace because the
            // worker's `last_32kib_window()` shortcut only fires for the
            // all-clean case (which doesn't enter this branch).
            let tail = chunk.get_last_window(&window[..]);
            window_map.insert(chunk_end_bit, Arc::new(tail));
            predecessor_window = Some(window);
        } else {
            // No markers → tail already published by the decode worker
            // via `last_32kib_window()` (workers do this before sending
            // their reply, see `worker_loop`'s `Ok(c)` branch).
            predecessor_window = None;
        }

        // Bookkeeping that does NOT depend on apply_window. Push
        // subchunks into the BlockMap (literal port of
        // `appendSubchunksToIndexes` at GzipChunkFetcher.hpp:371-375)
        // and promote subchunk-end positions to confirmed offsets in
        // the partitioner (literal port of vendor line 374,
        // `m_blockFinder->insert(...)`).
        append_subchunks_to_block_map(block_map, &chunk);
        if chunk.subchunks.is_empty() {
            block_finder.insert(chunk_end_bit);
            #[cfg(any(test, debug_assertions))]
            GZIP_BLOCK_FINDER_INSERTS_OBSERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            for sc in &chunk.subchunks {
                let actual_end = sc.encoded_offset_bits + sc.encoded_size_bits;
                block_finder.insert(actual_end);
                #[cfg(any(test, debug_assertions))]
                GZIP_BLOCK_FINDER_INSERTS_OBSERVED
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // record_get covers the "consumer pulled chunk idx out of the
        // pipeline" event; mirrors rapidgzip's stats.recordBlockIndexGet
        // at BlockFetcher.hpp:270-272.
        block_fetcher.statistics.base.record_get();
        #[cfg(any(test, debug_assertions))]
        BLOCK_FETCHER_GETS_OBSERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Capture subchunk count + decoded size BEFORE moving the chunk
        // into the post-process job (or directly into the cache for the
        // no-marker branch).
        let inserted = if chunk.subchunks.is_empty() {
            1
        } else {
            chunk.subchunks.len()
        };

        expected_start = chunk_end_bit;
        next_block_index += inserted;

        // ── Pending-write queue: post-process dispatch + ordered drain ─
        //
        // Mirror of vendor's `m_markersBeingReplaced` map +
        // `waitForReplacedMarkers` (vendor/.../GzipChunkFetcher.hpp:478-518).
        // The consumer submits an `applyWindow` task per chunk and
        // stashes its future; output writes (which are order-dependent)
        // pull futures in encoded order. While the head of the queue
        // is being processed, downstream chunks' post-process jobs run
        // in parallel on the same pool.
        if let Some(predecessor_window) = predecessor_window {
            // Marker case: submit post-process job. Job will run
            // apply_window + populate_subchunk_windows + emplace
            // per-subchunk windows.
            let rx = submit_post_process(idx, chunk, predecessor_window);
            pending.push_back(PendingWrite::Async {
                idx,
                rx,
                trim_bytes,
                cache_key: cache_key_for_partition,
            });
        } else {
            // Clean case: chunk has no markers, no apply_window needed,
            // no per-subchunk windows to populate (markers are already
            // resolved by the decode worker). Queue it as immediately
            // ready so writes stay in encoded order vs any in-flight
            // marker post-processes ahead of it.
            pending.push_back(PendingWrite::Ready {
                idx,
                chunk,
                trim_bytes,
                cache_key: cache_key_for_partition,
            });
        }

        // Refill the speculative pipeline: keep prefetch_count chunks
        // outstanding ahead of the consumer. The seed uses the
        // PARTITION-aligned guess (`partition_seed_for`) because the
        // spec ring is indexed by partition counter, not subchunk
        // count — using `block_finder.get(next_spec_to_dispatch)`
        // would return an interior subchunk position from a prior
        // chunk's inserts.
        if next_spec_to_dispatch < n_partitions {
            let seed = partition_seed_for(next_spec_to_dispatch);
            spec[next_spec_to_dispatch] = Some(SpecSlot {
                rx: submit_job(
                    next_spec_to_dispatch,
                    seed,
                    partition_until_for(next_spec_to_dispatch),
                    false,
                    seed,
                ),
                dispatched_seed: seed,
            });
            next_spec_to_dispatch += 1;
        }

        // Drain the pending queue down to the in-flight cap. Vendor
        // does this implicitly via `waitForReplacedMarkers` which
        // blocks on the current chunk's future; we make it explicit so
        // up to `post_process_inflight_cap` apply_windows can overlap
        // with each other AND with the decode pipeline. When the cap is
        // exceeded, drain the oldest entries (in encoded order — that's
        // how `pending` is ordered by construction).
        while pending.len() > post_process_inflight_cap {
            drain_one_pending(&mut pending, writer, total_crc, total_size, block_fetcher)?;
        }
    }

    // Final drain: flush remaining post-processes in order.
    while !pending.is_empty() {
        drain_one_pending(&mut pending, writer, total_crc, total_size, block_fetcher)?;
    }

    Ok(())
}

/// Items waiting to be written. Order is encoded order (partition
/// idx ascending). `Ready` is a no-op chunk (no markers, no post-
/// process needed); `Async` carries a receiver for the result of a
/// post-process job currently in-flight on a worker.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
enum PendingWrite {
    Ready {
        idx: usize,
        chunk: ChunkData,
        trim_bytes: usize,
        cache_key: usize,
    },
    Async {
        idx: usize,
        rx: mpsc::Receiver<ChunkData>,
        trim_bytes: usize,
        cache_key: usize,
    },
}

/// Pull the head of the pending FIFO, wait on its post-process if
/// needed, then write its bytes + advance the stream CRC. Mirror of
/// the tail of `GzipChunkFetcher::waitForReplacedMarkers` + write loop
/// at vendor lines 516 + 333-342 (the consumer's per-chunk write step
/// after `markerReplaceFuture->second.get()`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn drain_one_pending<W: std::io::Write>(
    pending: &mut std::collections::VecDeque<PendingWrite>,
    writer: &mut W,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive>,
) -> Result<(), FetchError> {
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    let (idx, chunk, trim_bytes, cache_key) = match head {
        PendingWrite::Ready {
            idx,
            chunk,
            trim_bytes,
            cache_key,
        } => (idx, chunk, trim_bytes, cache_key),
        PendingWrite::Async {
            idx,
            rx,
            trim_bytes,
            cache_key,
        } => {
            // Mirror of vendor line 516:
            // `markerReplaceFuture->second.get();` — block on the
            // queued apply_window task. By the time we get here, this
            // chunk's post-process job has typically completed in
            // parallel with later chunks' jobs.
            let chunk = rx.recv().map_err(|_| {
                FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: idx,
                    actual: 0,
                })
            })?;
            (idx, chunk, trim_bytes, cache_key)
        }
    };

    // Write bytes, skipping `trim_bytes` leading bytes that belong to
    // the predecessor's range. Per-chunk CRC accumulates via the
    // ported `CRC32Calculator::update` (crc32.hpp:296-303); the total-
    // stream CRC is then advanced by `append`-ing it (mirror of
    // `m_crc32Calculator.append( chunkCalculator )` at
    // vendor/.../GzipChunkFetcher.hpp:340).
    let mut written_crc = CRC32Calculator::new();
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
    total_crc.append(&written_crc);

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"decoded":{},"trim_bytes":{trim_bytes},"rss_kib":{}"#,
                chunk.decoded_size(),
                trace::rss_kib(),
            ),
        );
    }
    // Insert the fully-processed chunk into BlockFetcher's main cache
    // for potential future random-access replay. Mirror of rapidgzip's
    // `insertIntoCache` after consumer completion at
    // BlockFetcher.hpp:320-327.
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

    /// Deletion-trap killer for the `GzipBlockFinder` partitioner-feedback
    /// wiring (rapidgzip GzipChunkFetcher.hpp:374). Asserts the
    /// production `drive` path called `block_finder.insert(actual_end)`
    /// at least once during a successful run. Pre-port, the static
    /// `partition_offsets: Vec<usize>` vec never fed back; this counter
    /// would have stayed at zero. Post-port, every processed chunk's
    /// subchunks bump it.
    #[test]
    fn test_gzip_block_finder_in_drive() {
        use std::sync::atomic::Ordering;
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let before = GZIP_BLOCK_FINDER_INSERTS_OBSERVED.load(Ordering::Relaxed);
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 4, cfg).expect("drive");
        assert_eq!(size, payload.len());
        let after = GZIP_BLOCK_FINDER_INSERTS_OBSERVED.load(Ordering::Relaxed);
        assert!(
            after > before,
            "GzipBlockFinder::insert was not invoked during drive \
             (before={before}, after={after}). The partitioner-feedback \
             wiring regressed — consumer is no longer informing the \
             block finder of actual chunk ends. Mirror of rapidgzip \
             GzipChunkFetcher.hpp:374."
        );
    }

    /// Process-global lock to serialize tests that mutate the
    /// `GZIPPY_PREFETCH` env var (which `consumer_loop` reads to
    /// control speculation depth). Without this guard, parallel
    /// `cargo test` execution would race other tests calling `drive`.
    static PREFETCH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Deletion-trap killer for the `BlockFetcher::get` synchronous
    /// dispatch primitive (rapidgzip BlockFetcher.hpp:245-329).
    /// Speculative-miss chunks re-dispatch via `block_fetcher.get`,
    /// which bumps `BLOCK_FETCHER_GET_CALLS_OBSERVED`. We force at
    /// least one miss by setting `prefetch_count = 0` so chunk 1+
    /// always re-dispatches authoritative — and that authoritative
    /// path now flows through `block_fetcher.get`.
    #[test]
    fn test_block_fetcher_get_in_drive() {
        use std::sync::atomic::Ordering;
        // Disable speculation so the consumer always falls through to
        // the authoritative `block_fetcher.get` re-dispatch path.
        let _env_guard = PREFETCH_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("GZIPPY_PREFETCH", "0");

        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };

        let before = BLOCK_FETCHER_GET_CALLS_OBSERVED.load(Ordering::Relaxed);
        let mut out = Vec::new();
        let drive_result = drive(&deflate, &mut out, 4, cfg);
        std::env::remove_var("GZIPPY_PREFETCH");
        let (_crc, size) = drive_result.expect("drive");
        assert_eq!(size, payload.len());
        let after = BLOCK_FETCHER_GET_CALLS_OBSERVED.load(Ordering::Relaxed);
        assert!(
            after > before,
            "BlockFetcher::get was not invoked during drive \
             (before={before}, after={after}). The synchronous-dispatch \
             primitive port regressed — authoritative re-dispatch is no \
             longer flowing through `block_fetcher.get`. Mirror of \
             rapidgzip BlockFetcher.hpp:245-329."
        );
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
