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
//! Step 3 (2026-05-17): the `std::thread::scope + Mutex<mpsc::Receiver<Job>>`
//! worker pool is replaced by the literal port of `rapidgzip::ThreadPool`
//! (`thread_pool.rs`, vendor/.../core/ThreadPool.hpp:33-248). Decode
//! tasks AND post-process tasks both submit through the same
//! `ThreadPool::submit(closure, priority)` returning a `Future<R>`
//! (mirror of `std::future<BlockData>` at BlockFetcher.hpp:686). The
//! submit closure inside `BlockFetcher::get` unwraps the `Future` into
//! its underlying `mpsc::Receiver` (via `Future::into_receiver`) so the
//! `BlockFetcher`'s in-flight prefetch map can stash it.
//!
//! Mapping (vendor → gzippy):
//!
//! - `m_blockMap` → `BlockMap` (block_map.rs).
//! - `m_blockFinder` → `GzipBlockFinder` (gzip_block_finder.rs).
//! - `m_windowMap` → `WindowMap` (window_map.rs).
//! - `m_threadPool` → `ThreadPool` (thread_pool.rs, literal port of
//!   vendor/.../core/ThreadPool.hpp:33-248). One pool, shared across
//!   decode and post-process submissions, mirroring vendor's
//!   `m_threadPool` at BlockFetcher.hpp:686.
//! - `m_prefetching: std::map<size_t, std::future<BlockData>>` → lives
//!   inside `BlockFetcher` now (`block_fetcher.rs`'s
//!   `prefetching: HashMap<Key, Receiver<Result<Value, Err>>>`).
//! - `m_cache` → `BlockFetcher::cache`.
//! - `submitOnDemandTask` / `decodeAndMeasureBlock` → the `submit`
//!   closure passed to `BlockFetcher::get`, which calls
//!   `thread_pool.submit(run_decode_job, /* priority */ 0)` and returns
//!   the future's receiver. Mirror of BlockFetcher.hpp:554-558.
//! - `decodeBlock` / `decodeChunk` / `decodeChunkWithRapidgzip` →
//!   `gzip_chunk::decode_chunk_with_window` (fast path) +
//!   `decode_or_iterate` (slow path: tryToDecode + BlockFinder
//!   iteration per GzipChunk.hpp:712-846).
//! - `postProcessChunk` / `applyWindow` → `apply_window` +
//!   `populate_subchunk_windows`, dispatched on the SAME pool via
//!   `thread_pool.submit(run_post_process_job, /* priority */ -1)`
//!   (vendor's `submitTaskWithHighPriority` at BlockFetcher.hpp:606-611).

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
use crate::decompress::parallel::thread_pool::{ThreadPinning, ThreadPool};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::trace;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::window_map::WindowMap;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::mpsc;

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

/// Default `FetchNextAdaptive` memory size (vendor Prefetcher.hpp uses
/// per-instance values from ParallelGzipReader; 32 biases toward longer
/// sequential runs).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const FETCH_STRATEGY_MEMORY: usize = 32;

// ── Worker pool job parameters ───────────────────────────────────────────

/// Static descriptor for a single decode task — replaces the prior
/// `DecodeJob` struct now that dispatch goes through
/// `ThreadPool::submit(closure, priority)` directly (no enum-tagged
/// work channel). Mirror of the arguments captured by vendor's
/// `decodeAndMeasureBlock` task lambda body at
/// vendor/.../core/BlockFetcher.hpp:555-558.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct DecodeParams {
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
    /// For trace labelling only.
    partition_idx: usize,
}

/// `Send`-able wrapper around a raw pointer + length to the
/// deflate-stream input buffer. The buffer is owned by the caller of
/// `chunk_fetcher::drive` and the `ThreadPool` is stopped (joining all
/// workers) before `drive` returns, so no `InputSlice` ever outlives
/// the buffer it points into.
///
/// Why raw pointers: `ThreadPool::submit` requires `F: Send + 'static`
/// (mirror of `std::future`'s value semantics — captured state must be
/// owned or `'static`-borrowed). Vendor's C++ ignores lifetimes;
/// rapidgzip simply captures `this` and assumes the `BlockFetcher`
/// outlives the futures. We encode the same invariant explicitly.
///
/// Mirror of the `BlockFetcher`'s implicit "captured by reference"
/// pattern in `submitOnDemandTask` and the lambda at BlockFetcher.hpp:554
/// (`[this, ...] () { return decodeAndMeasureBlock(...); }`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct InputSlice {
    ptr: *const u8,
    len: usize,
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
unsafe impl Send for InputSlice {}
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
unsafe impl Sync for InputSlice {}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl InputSlice {
    /// SAFETY: caller must guarantee the slice outlives every
    /// `InputSlice` derived from it. `drive` enforces this by holding
    /// the `ThreadPool` (and dropping it, joining all workers) before
    /// returning.
    unsafe fn from_slice(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// SAFETY: caller must hold the `drive` lifetime invariant — i.e.
    /// the original slice is still valid for reads.
    unsafe fn as_slice(self) -> &'static [u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
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
    // CompressionType::None: vendor defaults to Zlib for the seekable-
    // reader where windows accumulate (memory pressure matters). For
    // single-pass single-member decode each window is published and
    // consumed once; compress/decompress overhead is pure waste.
    let window_map = WindowMap::with_compression(
        crate::decompress::parallel::compressed_vector::CompressionType::None,
    );
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

    // ── m_threadPool (vendor BlockFetcher.hpp:686) ──────────────────
    // The single thread pool that backs both decode and post-process
    // submissions, mirroring vendor's single `BS::thread_pool` shared
    // through `submit` + `submitTaskWithHighPriority`. `Arc` so the
    // submit closure inside `BlockFetcher::get` (cloned per call) can
    // hold a reference for `ThreadPool::submit`.
    let thread_pool = Arc::new(ThreadPool::new(pool_size, ThreadPinning::new()));

    // Running CRC + size accumulators. Mirror of vendor's
    // `m_crc32Calculator` + `m_totalDecompressedSize` updates inside
    // ParallelGzipReader::read.
    let mut total_crc = CRC32Calculator::new();
    let mut total_size: usize = 0;

    // SAFETY: `input_view` is consumed only by closures submitted to
    // `thread_pool`. `consumer_loop` returns only after every
    // submitted decode/post-process future has been awaited (via
    // `BlockFetcher::get`'s `rx.recv()` and `drain_one_pending`). We
    // additionally `thread_pool.stop()` below to join any prefetch
    // tasks still spawned. Therefore no closure holding `input_view`
    // can outlive the borrowed `input` slice.
    let input_view = unsafe { InputSlice::from_slice(input) };

    let consumer_result = consumer_loop(
        input_view,
        writer,
        total_bits,
        &block_finder,
        &block_fetcher,
        &block_map,
        &window_map,
        &thread_pool,
        pool_size,
        configuration,
        &mut total_crc,
        &mut total_size,
    );

    // Stop the pool BEFORE returning so any straggler prefetch tasks
    // are joined and `InputSlice` is no longer reachable. Mirror of
    // vendor's `stopThreadPool()` at BlockFetcher.hpp:600-604, called
    // from `~GzipChunkFetcher` before member destruction (line 686).
    thread_pool.stop();

    consumer_result?;

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
    input: InputSlice,
    writer: &mut W,
    total_bits: usize,
    block_finder: &GzipBlockFinder,
    block_fetcher: &Arc<BlockFetcher<usize, Arc<ChunkData>, FetchNextAdaptive, ChunkDecodeError>>,
    block_map: &BlockMap,
    window_map: &WindowMap,
    thread_pool: &Arc<ThreadPool>,
    pool_size: usize,
    configuration: ChunkConfiguration,
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
        // Vendor BlockFetcher.hpp:279-280 — `lastFetchedIndex` snapshot
        // + `m_fetchingStrategy.fetch(idx)`. Captured BEFORE the call so
        // the prefetch trigger predicate at line 297 ("index changed")
        // is well-defined.
        let last_fetched_before = block_fetcher.last_fetched();
        block_fetcher.record_fetch(next_unprocessed_block_index);
        let should_drive_prefetch = last_fetched_before
            .map(|li| li != next_unprocessed_block_index)
            .unwrap_or(true);

        let until_bit = until_bit_for(
            block_finder,
            next_unprocessed_block_index,
            total_bits,
            next_block_offset,
        );
        let partition_idx_for_trace = next_unprocessed_block_index;
        let params = DecodeParams {
            start_bit: next_block_offset,
            until_bit,
            authoritative: true,
            partition_idx: partition_idx_for_trace,
        };

        // The submit_for_prefetch closure mirrors vendor's
        // `m_threadPool.submit([this, offset, nextOffset] () {
        //     return decodeAndMeasureBlock(offset, nextOffset); }, 0)`
        // at BlockFetcher.hpp:554-557, with `nextOffset` derived from
        // the prefetched block's index+1.
        // Mirror of vendor's `decodeAndMeasureBlock(offset, nextOffset)`
        // capture at BlockFetcher.hpp:555-557 — the prefetched
        // `next_offset` is computed by `prefetch_new_blocks` from the
        // prefetched index + 1 (BlockFetcher.hpp:519-520) and passed in.
        // Critical: `next_offset` is the worker's stop hint; using
        // anything derived from `next_unprocessed_block_index` here
        // (the consumer's index, NOT the prefetched index) produced an
        // until_bit equal to `start_bit` for every prefetch past the
        // first one, causing the worker to immediately exit with an
        // empty chunk.
        let prefetch_submit = |offset: usize,
                               next_offset: usize|
         -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
            let prefetch_until_bit = next_offset.max(offset);
            let prefetch_params = DecodeParams {
                start_bit: offset,
                until_bit: prefetch_until_bit,
                // Prefetch tasks may race ahead of the consumer; the
                // worker waits a short window for the predecessor's
                // tail and falls back to the slow path on miss.
                authoritative: false,
                partition_idx: usize::MAX, // trace marker for "prefetch"
            };
            submit_decode_to_pool(
                thread_pool,
                input,
                prefetch_params,
                window_map,
                configuration,
            )
        };

        // `lookup_block_offset` mirrors vendor's `m_blockFinder->get(idx, 0)`
        // at BlockFetcher.hpp:479 — non-blocking lookup, returns the
        // confirmed block offset for `idx` if known.
        let lookup_block_offset = |idx: usize| -> Option<usize> {
            match block_finder.get(idx) {
                (Some(offset), GetReturnCode::Success) => Some(offset),
                _ => None,
            }
        };
        // `is_finalized_and_index_too_high` mirrors vendor's
        // BlockFetcher.hpp:504 `if ( m_blockFinder->finalized() &&
        // ( blockIndexToPrefetch >= m_blockFinder->size() ) )`.
        let is_finalized_too_high =
            |idx: usize| -> bool { block_finder.finalized() && idx >= block_finder.size() };
        // `partition_offset_for` mirrors vendor's
        // `getPartitionOffsetFromOffset` lambda at
        // GzipChunkFetcher.hpp:595-596:
        //   `[this] ( auto offset ) {
        //        return m_blockFinder->partitionOffsetContainingOffset(offset); }`.
        // Flows into BlockFetcher::prefetchNewBlocks at
        // BlockFetcher.hpp:486-489 / 537-538 for the double-key check.
        let partition_offset_for =
            |offset: &usize| -> usize { block_finder.partition_offset_containing_offset(*offset) };

        // Vendor `GzipChunkFetcher::getBlock` at
        // vendor/.../rapidgzip/GzipChunkFetcher.hpp:591-687 — FIRST try
        // the cache/prefetch lookup keyed by partition offset (where the
        // prefetch was submitted, since `m_blockFinder->get(idx)` returns
        // the partition-aligned guess for not-yet-confirmed indexes —
        // see GzipBlockFinder.hpp:134-157). If that succeeds AND the
        // returned chunk's accepted range contains the real offset
        // (`matchesEncodedOffset` at ChunkData.hpp:397, the same check
        // vendor uses at GzipChunkFetcher.hpp:647), the prefetched chunk
        // is reused. Otherwise we fall through to a full `get` at the
        // real offset, dispatching an on-demand task (matching vendor's
        // `BaseType::get(blockOffset, blockIndex, ...)` at
        // GzipChunkFetcher.hpp:654).
        let partition_offset = partition_offset_for(&next_block_offset);
        let mut chunk_arc_from_partition: Option<Arc<ChunkData>> = None;
        if partition_offset != next_block_offset {
            if let Some(Ok(arc)) = block_fetcher.try_take_prefetched(&partition_offset) {
                // Vendor GzipChunkFetcher.hpp:646-648 — accept the chunk
                // only if `matchesEncodedOffset(blockOffset)`. If not,
                // discard and dispatch on-demand at the real offset.
                if arc.matches_encoded_offset(next_block_offset) {
                    chunk_arc_from_partition = Some(arc);
                }
                // If !matches, the Arc is dropped here (vendor "throws
                // away" the partition-keyed result and re-issues at real
                // offset below — line 654). Same for Some(Err(...)) —
                // vendor's `catch ( const NoBlockInRange& )` at
                // GzipChunkFetcher.hpp:604-609 silently discards
                // prefetch failures.
            }
        }

        let chunk_arc = match chunk_arc_from_partition {
            Some(arc) => arc,
            None => {
                let (chunk_arc_result, _prefetched) = block_fetcher.get_with_prefetch(
                    next_block_offset,
                    |_key: usize| -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
                        submit_decode_to_pool(thread_pool, input, params, window_map, configuration)
                    },
                    lookup_block_offset,
                    prefetch_submit,
                    is_finalized_too_high,
                    partition_offset_for,
                    should_drive_prefetch,
                );
                chunk_arc_result.map_err(FetchError::Decode)?
            }
        };
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
            // Vendor `waitForReplacedMarkers` (GzipChunkFetcher.hpp:478-518)
            // blocks until the PREDECESSOR'S marker-replace future
            // completes — JUST the predecessor's, not all in-flight
            // chunks. Vendor lookup is `m_windowMap->get(*nextBlockOffset)`,
            // a blocking call that waits on the predecessor's
            // future-via-WindowMap-condvar; once it returns, downstream
            // (later) chunks' post-processes can keep running in the
            // worker pool without the consumer waiting on them.
            //
            // Our equivalent is to drain `pending` UNTIL the predecessor's
            // window appears at `next_block_offset`. The post-process queue
            // is FIFO, so draining from the front advances earliest-first;
            // a drain stops as soon as the consumer can move forward.
            // Pre-fix used `while !pending.is_empty()` — that drained ALL
            // queued post-processes per consumer iter, serializing the
            // whole pipeline (slow-path chunks effectively single-threaded
            // through the consumer). Now we drain only what's needed.
            while window_map.get(next_block_offset).is_none() {
                if pending.is_empty() {
                    // No post-process can produce the missing window —
                    // bubble the same error vendor would (a logic_error
                    // about a missing predecessor window).
                    break;
                }
                drain_one_pending(&mut pending, writer, total_crc, total_size, block_fetcher)?;
            }
            // Vendor `GzipChunkFetcher.hpp:334` —
            //   `auto sharedLastWindow = m_windowMap->get( *nextBlockOffset );`
            // Always look up the predecessor's window at the consumer's
            // expected start (`next_block_offset`), NOT at the chunk's
            // post-finalize `max_encoded_offset_bits`. The two coincide
            // ONLY after `set_encoded_offset(next_block_offset)` re-
            // anchors max → next_block_offset; in the "chunk's encoded
            // offset already matches next_block_offset" branch above
            // (line 487-489) we SKIP set_encoded_offset, and max stays
            // at the post-finalize chunk-end value. Looking it up there
            // would land on the position THIS chunk's own apply_window
            // will publish, not the predecessor's — guaranteed miss.
            // Pre-fix: this branch was unreachable because bootstrap
            // always failed; once the marker decoder works, every
            // partition-aligned slow-path chunk took the wrong key.
            let window_key = next_block_offset;
            let window = window_map.get(window_key).ok_or(FetchError::Decode(
                ChunkDecodeError::ExactStopMissed {
                    requested: window_key,
                    actual: next_block_offset,
                },
            ))?;
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
                let rx = submit_post_process_to_pool(
                    thread_pool,
                    chunk,
                    window,
                    window_map.clone(),
                    partition_idx_for_trace,
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

/// Submit a decode task to the `ThreadPool`, returning the
/// `mpsc::Receiver` that `BlockFetcher::get` will wait on. Mirror of
/// vendor's `submitOnDemandTask(blockOffset, nextBlockOffset)` at
/// BlockFetcher.hpp:573-589 — both arrange for an asynchronous decode
/// whose `std::future<BlockData>` the caller waits on. Vendor calls
/// `m_threadPool.submit([this, ...] () { return decodeAndMeasureBlock(...); }, 0)`;
/// we mirror that with `thread_pool.submit(run_decode_task, /* priority */ 0)`.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn submit_decode_to_pool(
    thread_pool: &Arc<ThreadPool>,
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    configuration: ChunkConfiguration,
) -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "submit_decode",
            &format!(
                r#""partition_idx":{},"start_bit":{},"until_bit":{},"authoritative":{}"#,
                params.partition_idx, params.start_bit, params.until_bit, params.authoritative,
            ),
        );
    }
    let window_map = window_map.clone();
    let future = thread_pool.submit(
        move || run_decode_task(input, params, &window_map, configuration),
        /* priority */ 0,
    );
    future.into_receiver()
}

/// Submit a post-process task to the `ThreadPool`, returning the
/// `mpsc::Receiver` that the consumer's pending-write queue will wait
/// on. Mirror of vendor's `submitTaskWithHighPriority(applyWindow)` at
/// GzipChunkFetcher.hpp:579 (which forwards to
/// `m_threadPool.submit(task, /* priority */ -1)` at
/// BlockFetcher.hpp:606-611).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn submit_post_process_to_pool(
    thread_pool: &Arc<ThreadPool>,
    chunk: ChunkData,
    predecessor_window: Arc<[u8; 32768]>,
    window_map: WindowMap,
    partition_idx: usize,
) -> mpsc::Receiver<ChunkData> {
    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "submit_post_process",
            &format!(r#""partition_idx":{}"#, partition_idx),
        );
    }
    let future = thread_pool.submit(
        move || run_post_process_task(chunk, predecessor_window, window_map),
        /* priority */ -1,
    );
    future.into_receiver()
}

/// Pool-side execution of a decode task. Mirror of vendor's
/// `decodeBlock(blockOffset, nextBlockOffset)`
/// (GzipChunkFetcher.hpp:692-729) — picks the fast path
/// (`decodeChunkWithIsal` / `decode_chunk_with_window`) when the
/// initial window is known, else the slow path
/// (`decodeChunkWithRapidgzip` / `decode_or_iterate`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn run_decode_task(
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    configuration: ChunkConfiguration,
) -> Result<Arc<ChunkData>, ChunkDecodeError> {
    // SAFETY: `drive`'s contract — input outlives the thread pool.
    let input_bytes: &[u8] = unsafe { input.as_slice() };

    let label = trace::worker_label(params.partition_idx);
    let t0 = std::time::Instant::now();

    // Audit step 5: WindowMap is now Condvar-free (vendor
    // WindowMap.hpp:19-186 has no condition_variable). Workers do
    // NON-blocking `get`: if the predecessor's tail has been
    // published, take the fast path; otherwise fall through to the
    // slow path (decode with markers, consumer's post-process
    // resolves them). This is the vendor model — workers never block
    // on the WindowMap; the dispatch order from `BlockFetcher::get`
    // and the post-process queue do the synchronization.
    let window = if params.start_bit == 0 {
        Some(Arc::new([0u8; 32768]))
    } else {
        window_map.get(params.start_bit)
    };

    let chunk_result = match window {
        Some(w) => decode_chunk_with_window(
            input_bytes,
            params.start_bit,
            params.until_bit,
            &w,
            configuration,
        ),
        None => decode_or_iterate(
            input_bytes,
            params.start_bit,
            params.until_bit,
            configuration,
        ),
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
        let path = if window_map.get(params.start_bit).is_some() {
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
                    params.partition_idx,
                    path,
                    params.start_bit,
                    c.encoded_offset_bits + c.encoded_size_bits,
                    c.decoded_size(),
                ),
            ),
            Err(e) => trace::emit(
                &label,
                "decode_err",
                &format!(
                    r#""partition_idx":{},"path":"{}","start_bit":{},"until_bit":{},"err":"{}","duration_us":{dur_us}"#,
                    params.partition_idx,
                    path,
                    params.start_bit,
                    params.until_bit,
                    trace::esc(&format!("{e:?}")),
                ),
            ),
        }
    }

    result
}

/// Pool-side execution of a post-process task. Mirror of the lambda
/// body at `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:579-582).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn run_post_process_task(
    mut chunk: ChunkData,
    predecessor_window: Arc<[u8; 32768]>,
    window_map: WindowMap,
) -> ChunkData {
    apply_window(&mut chunk, &predecessor_window[..]);
    chunk.populate_subchunk_windows(&predecessor_window[..]);

    // Per-subchunk window publication (vendor GzipChunkFetcher.hpp:430-458).
    for sc in &chunk.subchunks {
        if let Some(ref w) = sc.window {
            let sc_end_bit = sc.encoded_offset_bits + sc.encoded_size_bits;
            if sc_end_bit > sc.encoded_offset_bits {
                window_map.insert(sc_end_bit, w.clone());
            }
        }
    }

    chunk
}

// ── Slow-path decoder (tryToDecode + BlockFinder iteration) ──────────────

/// Try to decode at `start_bit` directly; if that fails, iterate
/// BlockFinder candidates in [start_bit, start_bit + 512 KiB] until
/// one succeeds. Mirror of rapidgzip's `tryToDecode` + candidate
/// iteration at vendor/.../GzipChunk.hpp:712-841.
///
/// The returned chunk's `encoded_offset_bits` is the REAL candidate
/// position the decode ran from — not a fabricated `start_bit`. Vendor
/// invariant (GzipChunk.hpp:716-722): `bitReader.seekTo(offset.second)`
/// then `result.encodedOffsetInBits = offset.first`, with `first ==
/// second` for compressed blocks. The two-key consumer wrapper at
/// `GzipChunkFetcher::getBlock` (GzipChunkFetcher.hpp:646-654) uses
/// `matchesEncodedOffset` to detect when a speculatively-prefetched
/// chunk landed on a different valid boundary than the real one, and
/// falls back to an on-demand decode at the real offset.
///
/// Previously this function fabricated `encoded_offset_bits = start_bit`
/// and stored `max_encoded_offset_bits = actual`, claiming the chunk
/// spanned `[start_bit, actual + size]`. But the decoded bytes ONLY
/// covered `[actual, actual + size]`, so consumers that took the chunk
/// at face value (after the partition-keyed cache lookup landed) would
/// emit wrong output — bytes `[start_bit, actual)` were missing from the
/// payload but counted in the metadata. The lie was harmless in the
/// pre-port era because the prefetch queue's key mismatch caused every
/// prefetched chunk to be silently discarded. With the two-key lookup
/// in place the lie surfaces; drop it to match vendor exactly.
/// Slow-path decoder for prefetched chunks whose predecessor window is
/// not yet published. Mirror of vendor's `tryToDecode` cascade at
/// vendor/.../chunkdecoding/GzipChunk.hpp:712-846.
///
/// **Vendor parity** — the cascade at GzipChunk.hpp:803-846 walks 8 KiB
/// sub-chunks (`CHUNK_SIZE = 8_Ki * BYTE_SIZE`, line 804), calls
/// `findNextDynamic`/`findNextUncompressed` ONCE per sub-chunk to get
/// the next candidate, calls `tryToDecode(offset)`, and on success
/// returns immediately (line 837-841). The pre-2026-05-17 gzippy
/// equivalent enumerated ALL candidates in a 512 KiB window via
/// `find_blocks`, then trial-decoded EVERY candidate even after the
/// first failure — a layering on top of vendor's cascade, not a port of
/// it.
///
/// **Investigation finding (trace silesia-large.gz @ gzip -9, 16T)**:
/// `find_blocks` returns 55-137 candidates per 512 KiB window. The
/// first candidate IS the real next-block boundary (typical Δ from
/// partition_offset = 12-150 KiB). EVERY candidate, including the real
/// one, fails trial-decode with `InvalidHuffmanCode` — a marker-decoder
/// bug we haven't isolated. With 67 attempts × 200 µs per false-positive
/// trial decode, each failed prefetch burns ~14 ms of worker time. With
/// 15-38 such prefetches per silesia decode, that's 200-500 ms of pure
/// waste, half the total wall time.
///
/// Until the bootstrap marker decoder is fixed, restrict iteration to
/// the first candidate — vendor's `tryToDecode` returns on first success
/// at GzipChunk.hpp:837-841, and we have no evidence that retry past
/// the first candidate ever salvages a real-world prefetch (the
/// production tracer showed zero `iterate_ok` events across 15 failed
/// prefetches). Restricting to one trial cuts per-failed-prefetch cost
/// from ~14 ms to ~200 µs, freeing workers to take on-demand work the
/// consumer dispatches. Effective parallelism climbs as soon as the
/// consumer is no longer waiting on a worker that's grinding through 67
/// doomed trial decodes.
///
/// Vendor parity (GzipChunk.hpp:803-846): scan in 8 KiB CHUNK_SIZE
/// increments but keep iterating until 512 KiB total or until a valid
/// boundary is found. Real gzip -9 boundaries on silesia typically sit
/// 8-150 KiB past the partition seed, so a SINGLE 8 KiB pass misses
/// the boundary on ~80% of partitions (observed via decode_err trace);
/// the outer 512 KiB cap matches vendor's `chunkBegin - blockOffset >=
/// 512_Ki * BYTE_SIZE` break at GzipChunk.hpp:811.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_or_iterate(
    input: &[u8],
    start_bit: usize,
    until_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    const CHUNK_SIZE_BITS: usize = 8 * 1024 * 8;
    const MAX_SCAN_BITS: usize = 512 * 1024 * 8;
    let finder = BlockFinder::new(input);
    let input_bits = input.len() * 8;
    let max_end = (start_bit + MAX_SCAN_BITS).min(input_bits);
    let mut chunk_begin = start_bit;
    while chunk_begin < max_end {
        let chunk_end = (chunk_begin + CHUNK_SIZE_BITS).min(max_end);
        let candidates = finder.find_blocks(chunk_begin, chunk_end);
        if let Some(first) = candidates.into_iter().find(|c| c.bit_offset >= chunk_begin) {
            return finish_decode_chunk_with_inexact_offset(
                input,
                first.bit_offset,
                until_bit,
                &[],
                configuration,
            );
        }
        chunk_begin = chunk_end;
    }
    Err(ChunkDecodeError::ExactStopMissed {
        requested: start_bit,
        actual: max_end,
    })
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

    /// Regression for the silesia-large.gz failure caught by `make
    /// bench-sm`: 4 MiB-compressed chunks crossing partition boundaries
    /// surface the two-key cache lookup bug. Uses production
    /// split_chunk_size and an input long enough to span multiple
    /// chunks.
    #[test]
    fn drive_round_trips_60mb_level9_prod_split() {
        // ~60 MB payload at -9 compresses to ~40 MB → ~10 chunks at the
        // production 4 MiB spacing. Enough to exercise multiple iter +
        // partition crossings without exploding test runtime.
        let payload = b"the quick brown fox jumps over the lazy dog ".repeat(1_500_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(9));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 20 * 4 * 1024 * 1024,
            crc32_enabled: true,
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 8, cfg).expect("drive");
        assert_eq!(size, payload.len(), "size mismatch (suggests early break)");
        assert_eq!(out, payload, "byte mismatch");
    }
}
