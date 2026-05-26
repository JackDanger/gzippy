#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]

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
//! - Worker decode: `decode_chunk_isal` (window known or chunk 0) or
//!   `speculative_decode_find_boundary` → marker bootstrap (prefetch, no window).
//! - `postProcessChunk` / `applyWindow` → `apply_window` on the pool
//!   (priority −1). **WindowMap publishes stay on the consumer** (vendor
//!   orchestrator thread): tail before post-process, subchunk windows
//!   after `apply_window` completes (`appendSubchunksToIndexes`).

use crate::decompress::parallel::chunk_data::ChunkConfiguration;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use std::sync::Arc;

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::block_fetcher::BlockFetcher;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::block_map::{append_subchunks_to_block_map, BlockMap};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::compressed_vector::CompressionType;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::crc32::CRC32Calculator;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::gzip_block_finder::{GetReturnCode, GzipBlockFinder};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::gzip_chunk::{
    decode_chunk_isal, decode_chunk_marker_bootstrap_then_isal,
};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::prefetcher::FetchMultiStream;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::raw_block_finder::RawBlockFinderCoordinator;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::streamed_results::StreamedGetReturnCode;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::thread_pool::ThreadPool;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::trace;
use crate::decompress::parallel::trace_v2;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use crate::decompress::parallel::window_map::{Window, WindowMap};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use std::borrow::Cow;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use std::sync::mpsc;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
use std::time::Duration;

/// Materialize a `Window`'s raw bytes. Mirror of vendor's
/// `sharedLastWindow->decompress()` call at
/// GzipChunkFetcher.hpp:341. For `CompressionType::None` (the
/// single-pass single-member production case) this is a zero-alloc
/// slice borrow into the existing `CompressedVector`. For Zlib
/// (seekable-reader path) it allocates a fresh `Vec<u8>`.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn materialize_window(w: &Window) -> Cow<'_, [u8]> {
    if w.compression_type() == CompressionType::None {
        Cow::Borrowed(w.raw_bytes())
    } else {
        Cow::Owned(w.decompress())
    }
}

/// Reverse-lookup map: subchunk encoded-bit offset → cache key of the
/// parent chunk that produced it. Mirror of vendor's
/// `m_unsplitBlocks: std::unordered_map<size_t, size_t>` declared at
/// `GzipChunkFetcher.hpp:781` and populated inside
/// `appendSubchunksToIndexes` (`:380-396`). When a chunk decodes into
/// multiple subchunks, each subchunk's offset is recorded so that a
/// random-access read for an internal subchunk offset can resolve to
/// the parent chunk in cache (vendor's `getIndexedChunk` query at
/// `:264-289`). Single-pass single-member streaming never queries
/// this; it is scaffolding for the seekable-reader path
/// (`sm_driver.rs`).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub type UnsplitBlocks = Arc<std::sync::Mutex<std::collections::HashMap<usize, usize>>>;

/// Construct a fresh, empty `UnsplitBlocks`. Mirror of the default
/// construction of `m_unsplitBlocks` as a `GzipChunkFetcher` member.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn new_unsplit_blocks() -> UnsplitBlocks {
    Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Deletion-trap counter incremented every time an entry is emplaced
/// into the `unsplit_blocks` map. Process-global; tests snapshot
/// before/after a multi-subchunk decode to prove the vendor
/// `appendSubchunksToIndexes` emplace site (`GzipChunkFetcher.hpp:393`)
/// has a live caller. Without this, the emplace branch could rot
/// silently (no production read site exists yet — it's scaffolding
/// for the seekable-reader path).
///
/// Not cfg-gated: the static must be addressable from tests on every
/// build configuration. The actual increment site IS cfg-gated to the
/// production path (x86_64 + isal-compression); on other targets the
/// counter stays at 0 and tests skip the assertion (mirror of the
/// MARKER_PIPELINE_RUNS pattern at single_member.rs:44).
pub static UNSPLIT_BLOCKS_EMPLACED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter incremented every time `lookup_next_block_offset` accepts a
/// `GetReturnCode::Failure` with offset == `file_size_in_bits` as a
/// valid worker stop hint (mirror of vendor's BlockFetcher.hpp:533-535
/// asymmetry). Without this asymmetry, the LAST prefetch in any file
/// was always skipped because the `idx+1` lookup returned Failure even
/// though the offset value was usable. On the 3-partition 221 MB
/// fixture this expands prefetch dispatch from 1→3 chunks in parallel.
///
/// Not cfg-gated for the same reason as UNSPLIT_BLOCKS_EMPLACED.
pub static PREFETCH_NEXT_FILESIZE_ACCEPT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[derive(Debug)]
#[allow(dead_code)] // error payloads surfaced via Debug in production
pub enum FetchError {
    Decode(ChunkDecodeError),
    #[allow(dead_code)] // non-x86 cfg stub; constructed only off the SM hot path
    UnsupportedPlatform,
}

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// Vendor production defaults for `FetchMultiStream`
/// (ParallelGzipReader.hpp:85, Prefetcher.hpp:234-336).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
const FETCH_MEMORY_PER_STREAM: usize = 3;
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
const FETCH_MAX_STREAM_COUNT: usize = 16;

// ── Worker pool job parameters ───────────────────────────────────────────

/// Static descriptor for a single decode task — replaces the prior
/// `DecodeJob` struct now that dispatch goes through
/// `ThreadPool::submit(closure, priority)` directly (no enum-tagged
/// work channel). Mirror of the arguments captured by vendor's
/// `decodeAndMeasureBlock` task lambda body at
/// vendor/.../core/BlockFetcher.hpp:555-558.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#[derive(Clone, Copy)]
struct DecodeParams {
    /// Bit offset where this chunk's decode starts.
    start_bit: usize,
    /// Inexact stop hint: first deflate block boundary at-or-past this
    /// bit (vendor `untilOffset` when `untilOffsetIsExact == false`).
    /// NOT a hard byte cap on the ISA-L reader — see `decode_chunk_isal`.
    stop_hint_bit: usize,
    /// True for partition-aligned prefetches that may run before the
    /// predecessor window is published (speculative path). False for
    /// on-demand decodes at a confirmed `block_finder` offset.
    is_speculative_prefetch: bool,
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
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#[derive(Clone, Copy)]
struct InputSlice {
    ptr: *const u8,
    len: usize,
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
unsafe impl Send for InputSlice {}
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
unsafe impl Sync for InputSlice {}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
pub fn drive<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "drive",
        &format!(
            r#""input_bytes":{},"parallelization":{}"#,
            input.len(),
            parallelization
        ),
    );
    let total_bits = input.len() * 8;
    // Vendor `availableCores()` (AffinityHelpers.hpp:18-21): avoid pinning
    // more workers than physical cores — SMT siblings collide on cache.
    let pool_size = parallelization.max(1).min(num_cpus::get_physical().max(1));

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
    // Vendor pattern: insert an empty / zero window so subsequent
    // get(0) lookups return a valid SharedWindow rather than nullptr.
    let zero_window = [0u8; 32768];
    window_map.insert_bytes(0, &zero_window);

    // ── m_blockMap (vendor GzipChunkFetcher.hpp:284) ────────────────
    let block_map = Arc::new(BlockMap::new());

    // ── m_chunkFetcher = BlockFetcher (vendor BlockFetcher.hpp:38) ──
    // The BlockFetcher owns the cache, prefetch cache, prefetch queue
    // (`m_prefetching`), fetching strategy, and statistics. Sized to
    // hold the active working set plus prefetched chunks.
    let cache_capacity = pool_size * 2;
    // Tried 4x — regressed (wall 486→510ms; misses dropped 4→3 but
    // overhead elsewhere ate it). Keep 2x.
    let prefetch_capacity = pool_size * 2;
    // Step 0 of plans/cache-miss-fix.md: falsifiability test for the
    // "cache misses are the lever" hypothesis. Per adversarial advisor:
    // the existing `prefetching_len() + 1 >= parallelization` gate at
    // block_fetcher.rs:584 caps in-flight prefetches at pool_size - 1.
    // Knob to raise: the `parallelization` arg below (which the gate
    // reads), NOT the cache_capacity (advisor: previously-killed cap
    // changes never reached the saturation gate). Under
    // GZIPPY_BURST_PREFETCH=1 this raises the gate to pool_size * 2,
    // allowing 17 in-flight prefetches at T=9 instead of 8.
    let saturation_parallelization = if std::env::var_os("GZIPPY_BURST_PREFETCH").is_some() {
        pool_size * 2
    } else {
        pool_size
    };
    let block_fetcher: Arc<
        BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>,
    > = Arc::new(BlockFetcher::new(
        cache_capacity,
        prefetch_capacity,
        FetchMultiStream::new(FETCH_MEMORY_PER_STREAM, FETCH_MAX_STREAM_COUNT),
        saturation_parallelization,
    ));

    // ── m_unsplitBlocks (vendor GzipChunkFetcher.hpp:781) ───────────
    // Subchunk-offset → parent-chunk-key reverse map; populated when a
    // chunk decodes into multiple subchunks (vendor :380-396) and
    // queried on random-access reads (vendor :264-289). Single-pass
    // streaming never queries this; the map is scaffolding for the
    // future seekable-reader path. Vendor's GzipChunkFetcher is single-
    // threaded so vendor uses a bare `std::unordered_map`; gzippy
    // wraps in `Arc<Mutex<...>>` so the future seek consumer (which
    // may live on a different thread) can share the handle.
    let unsplit_blocks = new_unsplit_blocks();

    // ── m_threadPool (vendor BlockFetcher.hpp:686) ──────────────────
    // The single thread pool that backs both decode and post-process
    // submissions, mirroring vendor's single `BS::thread_pool` shared
    // through `submit` + `submitTaskWithHighPriority`. `Arc` so the
    // submit closure inside `BlockFetcher::get` (cloned per call) can
    // hold a reference for `ThreadPool::submit`.
    let thread_pool = Arc::new(ThreadPool::with_pinning_for_capacity(pool_size));

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

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "drive_begin",
            &format!(
                r#""input_bytes":{},"total_bits":{},"pool_size":{},"chunk_size":{}"#,
                input.len(),
                total_bits,
                pool_size,
                configuration.split_chunk_size,
            ),
        );
    }

    let drive_t0 = std::time::Instant::now();
    let consumer_result = consumer_loop(
        input_view,
        writer,
        total_bits,
        &block_finder,
        &block_fetcher,
        &block_map,
        &window_map,
        &unsplit_blocks,
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

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "drive_end",
            &format!(
                r#""duration_us":{},"decoded_bytes":{},"crc32":{}"#,
                drive_t0.elapsed().as_micros(),
                total_size,
                total_crc.crc32(),
            ),
        );
    }

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

    // Drain the prefetch cache before stats dump. Any entries still
    // sitting in the prefetch cache at end-of-decode were dispatched
    // but never consumed by the consumer — mirror of vendor's
    // destructor path at BlockFetcher.hpp:199-201 which counts those
    // as `cache_unused_entry`. The drain wires `record_cache_unused_entry`
    // once per remaining entry so the --verbose dump reports them.
    block_fetcher.clear_prefetch_cache();

    // --verbose stats dump. Mirror of vendor's destructor print at
    // GzipChunkFetcher.hpp:124-198 + BlockFetcher.hpp:73-124. Triggered
    // by GZIPPY_VERBOSE env var (matches the existing GZIPPY_DEBUG
    // pattern at single_member.rs::debug_enabled). The CLI sets this
    // when --verbose is passed; tests and other internal callers
    // ignore it.
    if std::env::var("GZIPPY_VERBOSE").is_ok() {
        let snap = block_fetcher.statistics.base.snapshot();
        let extra = block_fetcher.statistics.extra_snapshot();
        eprintln!("[gzippy --verbose] BlockFetcher statistics:");
        eprintln!("{}", snap);
        eprintln!("  Preemptive stops: {}", extra.preemptive_stop_count);
        eprintln!(
            "  Time queuing post-processing: {:.6} s",
            extra.queue_post_processing_duration
        );
        // Per-counter hot-path observations relevant to this branch's
        // optimization work. Mirror of vendor's --verbose details
        // adapted for gzippy-specific counters added this session.
        use std::sync::atomic::Ordering;
        eprintln!(
            "  Adjusted chunk size applied: {}",
            crate::decompress::parallel::single_member::ADJUSTED_CHUNK_SIZE_APPLIED
                .load(Ordering::Relaxed)
        );
        eprintln!(
            "  Prefetch next-offset filesize-accepts: {}",
            PREFETCH_NEXT_FILESIZE_ACCEPT.load(Ordering::Relaxed)
        );
        eprintln!(
            "  Unsplit blocks emplaced: {}",
            UNSPLIT_BLOCKS_EMPLACED.load(Ordering::Relaxed)
        );
        use crate::decompress::parallel::chunk_buffer_pool::*;
        eprintln!(
            "  Buffer pool u8: hits={} misses={} returns={}",
            TAKE_U8_HITS.load(Ordering::Relaxed),
            TAKE_U8_MISSES.load(Ordering::Relaxed),
            RETURN_U8_CALLS.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Buffer pool u16: hits={} misses={} returns={}",
            TAKE_U16_HITS.load(Ordering::Relaxed),
            TAKE_U16_MISSES.load(Ordering::Relaxed),
            RETURN_U16_CALLS.load(Ordering::Relaxed),
        );
        // Slow-path candidate iteration: ok + fail + no_candidate sum
        // to the slow-path call count.
        eprintln!(
            "  Slow-path decode: ok={} fail={} no_candidate={}",
            SLOW_PATH_FIRST_CANDIDATE_OK.load(Ordering::Relaxed),
            SLOW_PATH_FIRST_CANDIDATE_FAIL.load(Ordering::Relaxed),
            SLOW_PATH_NO_CANDIDATE.load(Ordering::Relaxed),
        );
        // V6: Was the boundary search (scoped BlockFinder spawn) needed
        // for most slow-path decodes, or did the first-candidate try
        // win without it? If runs == ok_count, every slow-path needed
        // a search; if runs << ok_count, the partition seed was usually
        // a real block boundary.
        eprintln!(
            "  BlockFinder coordinator spawns: {}",
            COORDINATOR_BOUNDARY_SEARCH_RUNS.load(Ordering::Relaxed),
        );
        // V1: Speculative-decode failure breakdown — picks Design 1
        // (precode pre-pass) vs Design 3 (coordinator amortization)
        // vs Design 4 (boundary-confirmed speculation).
        eprintln!(
            "  Speculation failure modes: header={} body={} inflate={} stop_missed={} other={}",
            SPEC_FAIL_HEADER.load(Ordering::Relaxed),
            SPEC_FAIL_BODY.load(Ordering::Relaxed),
            SPEC_FAIL_INFLATE.load(Ordering::Relaxed),
            SPEC_FAIL_STOP_MISSED.load(Ordering::Relaxed),
            SPEC_FAIL_OTHER.load(Ordering::Relaxed),
        );
        // Body-failure forensic detail — speculation-accuracy attack.
        // After disprove-advisor confirmed body failures are the dominant
        // re-decode cost, characterize them by error variant + waste size.
        use crate::decompress::parallel::gzip_chunk as gc;
        let body_count = gc::BODY_FAIL_COUNT.load(Ordering::Relaxed);
        let body_wasted = gc::BODY_FAIL_BYTES_WASTED.load(Ordering::Relaxed);
        let body_bits = gc::BODY_FAIL_BITS_INTO_BODY.load(Ordering::Relaxed);
        let avg_waste = if body_count > 0 {
            body_wasted as f64 / body_count as f64
        } else {
            0.0
        };
        let avg_bits = if body_count > 0 {
            body_bits as f64 / body_count as f64
        } else {
            0.0
        };
        eprintln!(
            "  Body-fail forensics: count={body_count} wasted_bytes={body_wasted} avg_waste={avg_waste:.0}B avg_bits_into_body={avg_bits:.0}",
        );
        eprintln!(
            "    by variant: invalid_huffman={} exceeded_window={} invalid_compression={} invalid_code_lengths={} other={}",
            gc::BODY_FAIL_INVALID_HUFFMAN.load(Ordering::Relaxed),
            gc::BODY_FAIL_EXCEEDED_WINDOW.load(Ordering::Relaxed),
            gc::BODY_FAIL_INVALID_COMPRESSION.load(Ordering::Relaxed),
            gc::BODY_FAIL_INVALID_CODE_LENGTHS.load(Ordering::Relaxed),
            gc::BODY_FAIL_OTHER_VARIANT.load(Ordering::Relaxed),
        );
        // B: BlockFinder per-spawn breakdown — scan vs consumer time.
        use crate::decompress::parallel::raw_block_finder as rbf;
        let bf_calls = rbf::BOUNDARY_SEARCH_CALLS.load(Ordering::Relaxed);
        let bf_total = rbf::BOUNDARY_SEARCH_TOTAL_US.load(Ordering::Relaxed);
        let bf_scan = rbf::BOUNDARY_SEARCH_SCAN_US.load(Ordering::Relaxed);
        let bf_consumer = rbf::CONSUMER_TIME_US.load(Ordering::Relaxed);
        eprintln!(
            "  BlockFinder spawn breakdown: calls={bf_calls} total_ms={:.1} scan_ms={:.1} consumer_ms={:.1} avg_total_us={}",
            bf_total as f64 / 1000.0,
            bf_scan as f64 / 1000.0,
            bf_consumer as f64 / 1000.0,
            if bf_calls > 0 { bf_total / bf_calls } else { 0 },
        );
        // C: bootstrap per-block header vs body cost.
        let bs_h_us = gc::BOOTSTRAP_HEADER_US.load(Ordering::Relaxed);
        let bs_h_calls = gc::BOOTSTRAP_HEADER_CALLS.load(Ordering::Relaxed);
        let bs_b_us = gc::BOOTSTRAP_BODY_US.load(Ordering::Relaxed);
        let bs_b_bytes = gc::BOOTSTRAP_BODY_BYTES.load(Ordering::Relaxed);
        let bs_h_avg = if bs_h_calls > 0 {
            bs_h_us as f64 / bs_h_calls as f64
        } else {
            0.0
        };
        let bs_b_rate = if bs_b_us > 0 {
            (bs_b_bytes as f64) / (bs_b_us as f64)
        } else {
            0.0
        };
        eprintln!(
            "  Bootstrap per-block: header_calls={bs_h_calls} header_ms={:.1} avg_header_us={:.1} body_ms={:.1} body_bytes={bs_b_bytes} body_rate_MB/s={:.0}",
            bs_h_us as f64 / 1000.0,
            bs_h_avg,
            bs_b_us as f64 / 1000.0,
            bs_b_rate,
        );
        // Per-fetch rejection cause: a prefetched chunk arrived but the
        // safety guard rejected it (chain invariant broken —
        // chunk.max != next_block_offset).
        eprintln!(
            "  Prefetch guard-rejects: {}",
            PREFETCH_REJECT_BY_GUARD.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Arc::try_unwrap hits/misses: {} / {}",
            ARC_TRY_UNWRAP_HITS.load(Ordering::Relaxed),
            ARC_TRY_UNWRAP_MISSES.load(Ordering::Relaxed),
        );
    }

    // Flush all per-thread trace_v2 buffers (consumer thread + any
    // already-joined worker threads).
    trace_v2::flush_all();
    Ok((total_crc.crc32(), total_size))
}

// ── Consumer: processNextChunk port ──────────────────────────────────────

/// Vendor-faithful port of `GzipChunkFetcher::processNextChunk`
/// (vendor/.../GzipChunkFetcher.hpp:311-362), wrapped in a `loop` that
/// plays the role of `ParallelGzipReader::read`'s outer loop
/// (ParallelGzipReader.hpp:702-810). Every per-iteration step is a
/// vendor primitive — no inline ring, no inline cache logic, no
/// inline take-from-prefetch.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#[allow(clippy::too_many_arguments)]
fn consumer_loop<W: std::io::Write>(
    input: InputSlice,
    writer: &mut W,
    total_bits: usize,
    block_finder: &GzipBlockFinder,
    block_fetcher: &Arc<BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>>,
    block_map: &BlockMap,
    window_map: &WindowMap,
    unsplit_blocks: &UnsplitBlocks,
    thread_pool: &Arc<ThreadPool>,
    pool_size: usize,
    configuration: ChunkConfiguration,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    // Vendor's `m_nextUnprocessedBlockIndex` (GzipChunkFetcher.hpp:318 + 410).
    let mut next_unprocessed_block_index: usize = 0;
    // Furthest compressed bit offset whose decoded bytes have been
    // handed to the consumer loop (chunk end, not partition guess).
    let mut furthest_decoded_bit: usize = 0;
    // Pending writes (post-process jobs in flight, output order).
    let mut pending: std::collections::VecDeque<PendingWrite> =
        std::collections::VecDeque::with_capacity(pool_size * 2);
    let post_process_inflight_cap = pool_size;

    // The vendor's `processNextChunk` returns one chunk per call; the
    // caller loops in `ParallelGzipReader::read`. We inline that loop
    // here so the local-state mutation (post-process queue + writer +
    // CRC) stays simple.
    #[allow(clippy::while_let_loop)] // faithful port of vendor processNextChunk loop
    let mut iter_us_sum: u128 = 0;
    let mut prefetch_us_sum: u128 = 0;
    let mut finder_us_sum: u128 = 0;
    let mut fetcher_get_us_sum: u128 = 0;
    let mut submit_us_sum: u128 = 0;
    let mut iter_count: usize = 0;
    loop {
        let _tv2 = trace_v2::SpanGuard::begin("consumer.iter");
        let t_iter = std::time::Instant::now();
        // BlockFetcher.hpp:427 — opportunistically promote ready prefetches
        // so workers don't idle while the consumer waits on a different key.
        let t_prefetch = std::time::Instant::now();
        {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.process_prefetches");
            block_fetcher.process_ready_prefetches();
        }
        prefetch_us_sum += t_prefetch.elapsed().as_micros();

        // Vendor GzipChunkFetcher.hpp:318 — `m_blockFinder->get(m_nextUnprocessedBlockIndex)`.
        let t_finder = std::time::Instant::now();
        let _tv2_finder = trace_v2::SpanGuard::begin("consumer.block_finder_get");
        let next_block_offset = match block_finder.get(next_unprocessed_block_index) {
            (Some(offset), GetReturnCode::Success) => offset,
            // Vendor GzipChunkFetcher.hpp:320-327 — EOF when no offset
            // or offset past end of file.
            _ => break,
        };
        finder_us_sum += t_finder.elapsed().as_micros();
        drop(_tv2_finder);
        if next_block_offset >= total_bits {
            break;
        }

        let block_is_confirmed = next_unprocessed_block_index < block_finder.size();
        // A spacing guess behind already-emitted bytes is stale — the
        // fast path jumped over this partition index in one chunk.
        if !block_is_confirmed && next_block_offset < furthest_decoded_bit {
            next_unprocessed_block_index += 1;
            continue;
        }
        // When the guess overshoots the last decoded bit (large fast-path
        // chunk ended before the next spacing multiple), resume from the
        // actual tail instead of leaving a compressed gap.
        let decode_start = if block_is_confirmed {
            next_block_offset
        } else if next_block_offset > furthest_decoded_bit {
            furthest_decoded_bit
        } else {
            next_block_offset
        };

        // A sub-byte span between `decode_start` and the end of the
        // deflate data is gzip end-of-stream byte-alignment padding
        // (0-7 zero bits before the 8-byte footer), not a deflate
        // block — the smallest complete block is ~10 bits, so this
        // fragment carries no decodable content. Handing it to the
        // parallel decode path spins forever: the ISA-L stopping-point
        // patch does not terminate `isal_inflate` on a sub-block
        // fragment. rapidgzip avoids this by never scheduling such a
        // chunk; mirror that here.
        if total_bits.saturating_sub(decode_start) < 8 {
            next_unprocessed_block_index += 1;
            continue;
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
            .unwrap_or(true)
            && std::env::var_os("GZIPPY_NO_PREFETCH").is_none();

        let stop_hint_bit = stop_hint_bit_for(
            block_finder,
            next_unprocessed_block_index,
            total_bits,
            decode_start,
        );
        let partition_idx_for_trace = next_unprocessed_block_index;
        let params = DecodeParams {
            start_bit: decode_start,
            stop_hint_bit,
            is_speculative_prefetch: false,
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
        // stop_hint_bit equal to `start_bit` for every prefetch past the
        // first one, causing the worker to immediately exit with an
        // empty chunk.
        let prefetch_submit = |offset: usize,
                               next_offset: usize|
         -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
            let prefetch_stop_hint_bit = next_offset.max(offset);
            let prefetch_params = DecodeParams {
                start_bit: offset,
                stop_hint_bit: prefetch_stop_hint_bit,
                is_speculative_prefetch: true,
                partition_idx: usize::MAX, // trace marker for "prefetch"
            };
            submit_decode_to_pool(
                thread_pool,
                input,
                prefetch_params,
                window_map,
                block_fetcher,
                configuration,
            )
        };

        // `lookup_block_offset` mirrors vendor's `m_blockFinder->get(idx, 0)`
        // at BlockFetcher.hpp:479 — non-blocking lookup, returns the
        // confirmed block offset for `idx` if known. SUCCESS-only:
        // the current-index lookup vendor uses at
        // BlockFetcher.hpp:533 rejects Failure outright.
        let lookup_block_offset = |idx: usize| -> Option<usize> {
            match block_finder.get(idx) {
                (Some(offset), GetReturnCode::Success) => Some(offset),
                _ => None,
            }
        };
        // `lookup_next_block_offset` — for the WORKER'S STOP HINT
        // (the `idx+1` lookup at BlockFetcher.hpp:535). Vendor accepts
        // `file_size_in_bits` as a valid stop hint for the LAST
        // chunk's prefetch. Without this asymmetry, the loop at
        // `prefetch_new_blocks` line 589 skips the last prefetch in
        // any file, leaving 1 prefetch dispatched instead of 3 on
        // the 221 MB / 3-partition fixture (bench-2026-05-18).
        let lookup_next_block_offset = |idx: usize| -> Option<usize> {
            match block_finder.get(idx) {
                (Some(offset), GetReturnCode::Success) => Some(offset),
                (Some(offset), GetReturnCode::Failure) if offset == total_bits => {
                    PREFETCH_NEXT_FILESIZE_ACCEPT
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Some(offset)
                }
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
        let _tv2_specf = trace_v2::SpanGuard::begin("consumer.try_take_prefetched");
        let partition_offset = partition_offset_for(&next_block_offset);
        let mut chunk_arc_from_partition: Option<Arc<ChunkData>> = None;
        if partition_offset != next_block_offset {
            // Lever H: while blocking on the in-flight prefetch's
            // `rx.recv`, pump `prefetch_new_blocks` every 1ms so the
            // prefetch horizon advances during the wait. Mirror of
            // vendor BlockFetcher.hpp:312-316.
            let pump = || {
                if should_drive_prefetch {
                    block_fetcher.prefetch_new_blocks(
                        &lookup_block_offset,
                        &lookup_next_block_offset,
                        &prefetch_submit,
                        &is_finalized_too_high,
                        &partition_offset_for,
                    );
                }
            };
            if let Some(Ok(arc)) =
                block_fetcher.try_take_prefetched_pumping(&partition_offset, pump)
            {
                // Vendor GzipChunkFetcher.hpp:646-648 — accept the chunk
                // only if `matchesEncodedOffset(blockOffset)`. If not,
                // discard and dispatch on-demand at the real offset.
                //
                // **gzippy-specific safety guard** — the speculative
                // slow path now publishes vendor-faithful metadata
                // (`encoded_offset = offset.first`, `max =
                // offset.second`, speculative_decode_find_boundary above). Vendor
                // safely uses a `matchesEncodedOffset` range hit
                // because chunk_N.actual_end == chunk_{N+1}.offset.second
                // by construction (chain invariant — fast-path
                // exactUntilOffset = prefetch's offset.second). gzippy
                // doesn't yet enforce that chain, so a `blockOffset`
                // strictly inside (encoded_offset, max) would shift
                // the chunk's claimed range past where its data
                // actually starts → missing decoded bytes for
                // `[blockOffset, max)` → output corruption.
                //
                // Restrict the trim to the ONE boundary case that is
                // safe without the chain invariant:
                //   - blockOffset == max_acceptable_start_bit → no
                //     trim needed because chunk's data starts AT
                //     `max` (slow path's offset.second). Writing
                //     chunk.data wholesale emits bytes for
                //     [max, end] = [blockOffset, end] exactly.
                //
                // Dropped the alternate clause `blockOffset ==
                // encoded_offset_bits` (audit 13 reflection): when
                // `encoded != max`, the chunk's data still starts at
                // `max`, so a blockOffset matching `encoded` means
                // bytes for [encoded, max) are missing from the
                // chunk — output corruption. The clause happened to
                // be safe in the common case (encoded == max after
                // slow path validated at the partition seed) but
                // unsafe in the general case. Keep only `max`.
                // Speculative chunks may claim a start RANGE
                // [encoded_offset_bits, max_acceptable_start_bit] but
                // decoded bytes only exist from max onward. Safe reuse
                // requires the consumer's confirmed start == max (vendor
                // chain: chunk_N.end == chunk_{N+1}.start at max).
                let handoff_at_decode_start = arc.max_acceptable_start_bit == next_block_offset;
                if arc.matches_encoded_offset(next_block_offset) && handoff_at_decode_start {
                    chunk_arc_from_partition = Some(arc.clone());
                    if trace::is_enabled() {
                        trace::emit(
                            "consumer",
                            "speculative_accept",
                            &format!(
                                r#""partition_idx":{partition_idx_for_trace},"expected_start":{next_block_offset},"speculative_start":{},"encoded_offset":{},"max_acceptable":{}"#,
                                arc.encoded_offset_bits,
                                arc.encoded_offset_bits,
                                arc.max_acceptable_start_bit,
                            ),
                        );
                    }
                } else {
                    // Diagnostic counter (added 2026-05-19): a prefetch
                    // arrived at the consumer (matches the partition
                    // lookup) but the safety guard rejected it. Bumps
                    // mean "chain invariant broken for this fetch."
                    // Distinguishes from "prefetch never arrived"
                    // (try_take_prefetched returned None), which is
                    // counted by `prefetch_cache_miss` and means a
                    // scheduling/timing miss, not a metadata mismatch.
                    PREFETCH_REJECT_BY_GUARD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if trace::is_enabled() {
                        trace::emit(
                            "consumer",
                            "speculative_mismatch",
                            &format!(
                                r#""partition_idx":{partition_idx_for_trace},"expected_start":{next_block_offset},"speculative_start":{},"encoded_offset":{},"max_acceptable":{}"#,
                                arc.encoded_offset_bits,
                                arc.encoded_offset_bits,
                                arc.max_acceptable_start_bit,
                            ),
                        );
                    }
                }
                // If !matches OR the safety guard rejected, the Arc
                // is dropped here (vendor "throws away" the
                // partition-keyed result and re-issues at real offset
                // below — line 654). Same for Some(Err(...)) —
                // vendor's `catch ( const NoBlockInRange& )` at
                // GzipChunkFetcher.hpp:604-609 silently discards
                // prefetch failures.
            } else if trace::is_enabled() {
                trace::emit(
                    "consumer",
                    "speculative_missing",
                    &format!(
                        r#""partition_idx":{partition_idx_for_trace},"partition_offset":{partition_offset},"expected_start":{next_block_offset}"#
                    ),
                );
            }
        }

        drop(_tv2_specf);
        let chunk_arc = match chunk_arc_from_partition {
            Some(arc) => arc,
            None => {
                let t_fg = std::time::Instant::now();
                let _tv2 = trace_v2::SpanGuard::begin_with(
                    "wait.block_fetcher_get",
                    &format!(
                        r#""chunk_id":{},"offset":{}"#,
                        partition_idx_for_trace, next_block_offset
                    ),
                );
                let (chunk_arc_result, _prefetched) = block_fetcher.get_with_prefetch(
                    next_block_offset,
                    |_key: usize| -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
                        submit_decode_to_pool(
                            thread_pool,
                            input,
                            params,
                            window_map,
                            block_fetcher,
                            configuration,
                        )
                    },
                    lookup_block_offset,
                    lookup_next_block_offset,
                    prefetch_submit,
                    is_finalized_too_high,
                    partition_offset_for,
                    should_drive_prefetch,
                );
                fetcher_get_us_sum += t_fg.elapsed().as_micros();
                chunk_arc_result.map_err(FetchError::Decode)?
            }
        };
        // Take ownership when we hold the only Arc; otherwise clone the
        // inner ChunkData. Mirror of rapidgzip's shared_ptr aliasing at
        // GzipChunkFetcher.hpp:329.
        let mut chunk: ChunkData = {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.arc_take_or_clone");
            match Arc::try_unwrap(chunk_arc) {
                Ok(c) => {
                    ARC_TRY_UNWRAP_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    c
                }
                Err(a) => {
                    ARC_TRY_UNWRAP_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    (*a).clone()
                }
            }
        };

        // Vendor GzipChunkFetcher.hpp:349 —
        //   `chunkData->setEncodedOffset(*nextBlockOffset);`
        // When we resumed from furthest_decoded_bit because the spacing
        // guess overshot, use the actual decode start — not the stale guess.
        let effective_start = decode_start;
        if chunk.encoded_offset_bits != effective_start {
            chunk.set_encoded_offset(effective_start);
        }

        // Vendor GzipChunkFetcher.hpp:350-355 — EOF mid-decode
        // (`encodedSizeInBits == 0`).
        if chunk.encoded_size_bits == 0 {
            break;
        }
        let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        furthest_decoded_bit = furthest_decoded_bit.max(chunk_end_bit);

        // Vendor GzipChunkFetcher.hpp:343 — `postProcessChunk(chunkData, lastWindow)`.
        // The window emplacement at lines 558-575 (tail-window publish on
        // consumer thread BEFORE handing off to the worker pool) is what
        // unblocks the next chunk's worker without serializing on this
        // chunk's apply_window. We mirror exactly:
        // `Window = Arc<CompressedVector>` — vendor's
        // `SharedWindow = shared_ptr<const CompressedVector>`
        // (WindowMap.hpp:24). Subchunk windows are published on the
        // consumer in `drain_one_pending` after post-process returns.
        #[allow(clippy::needless_late_init)] // assigned in two long branches below
        let predecessor_window_for_postprocess: Option<Window>;
        if chunk.data_with_markers.is_empty() {
            // No markers → apply_window is a no-op. Publish successor window
            // on the consumer only (vendor queueChunkForPostProcessing:558-575).
            // Mirror vendor `getLastWindow(*previousWindow)` — not `last_32kib`
            // alone, which ignores the predecessor chain at stream start.
            let _tv2 = trace_v2::SpanGuard::begin("consumer.window_publish_clean");
            if let Some((_pred_key, pred)) = window_map.get_predecessor(chunk.encoded_offset_bits) {
                let bytes = materialize_window(&pred);
                let tail = chunk
                    .last_32kib_window()
                    .unwrap_or_else(|| chunk.get_last_window(&bytes));
                window_map.insert_bytes_with_compression(
                    chunk_end_bit,
                    &tail,
                    CompressionType::None,
                );
            }
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
            // Presence-only spin: each iteration only needs to know
            // whether the predecessor's window has been published.
            // Vendor's `WindowMap::get` (WindowMap.hpp:79-90) returns
            // `shared_ptr<const CompressedVector>` — zero alloc on
            // miss — so a presence check is free. Gzippy's `get`
            // currently still allocates `Arc<[u8; 32768]>` on hit;
            // `contains` skips that path entirely.
            // Predecessor window is keyed at the prior chunk's *end* bit
            // offset. When a spacing guess overshoots that tail, the
            // published window lives at furthest_decoded_bit, not at
            // chunk.encoded_offset_bits.
            {
                let _tv2 = trace_v2::SpanGuard::begin("consumer.wait_replaced_markers");
                while !window_map.has_predecessor(chunk.encoded_offset_bits) {
                    if pending.is_empty() {
                        break;
                    }
                    drain_one_pending(
                        &mut pending,
                        window_map,
                        writer,
                        total_crc,
                        total_size,
                        block_fetcher,
                    )?;
                }
            }
            let _tv2_pred = trace_v2::SpanGuard::begin("consumer.window_publish_marker");
            let (pred_key, window) = window_map
                .get_predecessor(chunk.encoded_offset_bits)
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: chunk.encoded_offset_bits,
                    actual: furthest_decoded_bit,
                }))?;
            let _ = pred_key;
            // Vendor `GzipChunkFetcher.hpp:341`: `sharedLastWindow->
            // decompress()` materializes the bytes once. For
            // CompressionType::None this is a zero-alloc slice borrow.
            let window_bytes = materialize_window(&window);
            let tail = chunk.get_last_window(&window_bytes);
            window_map.insert_bytes_with_compression(chunk_end_bit, &tail, CompressionType::None);
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

        // Vendor `appendSubchunksToIndexes` continued
        // (GzipChunkFetcher.hpp:380-396): when a chunk produced multiple
        // subchunks, record each *internal* subchunk's offset → parent
        // chunk cache-key in `m_unsplitBlocks`. The seek consumer
        // (`getIndexedChunk` at `:264-289`) consults this map to
        // resolve a request for an internal subchunk offset back to
        // its parent chunk in cache. Single-pass streaming never
        // hits this lookup; the emplace is structural scaffolding for
        // the seekable-reader path.
        if chunk.subchunks.len() > 1 {
            // Vendor `:384-389` — `lookupKey` is the cache key under
            // which the parent chunk is findable, which can be EITHER
            // the chunk's actual encoded offset OR the partition
            // offset, depending on which one the cache holds:
            //   if !test(chunkOffset) && test(partitionOffset):
            //       partitionOffset
            //   else:
            //       chunkOffset
            // The prefetch-take path inserts under the partition
            // offset, so a chunk that came off the prefetch queue at a
            // speculative partition seed has its parent keyed under
            // partitionOffset, not its actual chunkOffset.
            let chunk_offset = chunk.encoded_offset_bits;
            let partition_offset = block_finder.partition_offset_containing_offset(chunk_offset);
            let lookup_key =
                if !block_fetcher.test(&chunk_offset) && block_fetcher.test(&partition_offset) {
                    partition_offset
                } else {
                    chunk_offset
                };
            let mut unsplit = unsplit_blocks.lock().unwrap();
            for sc in &chunk.subchunks {
                if sc.encoded_offset_bits != chunk_offset {
                    // Vendor uses `emplace` (insert-if-absent — does
                    // not overwrite). Match that semantic.
                    use std::collections::hash_map::Entry;
                    if let Entry::Vacant(v) = unsplit.entry(sc.encoded_offset_bits) {
                        v.insert(lookup_key);
                        UNSPLIT_BLOCKS_EMPLACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }

        // Vendor GzipChunkFetcher.hpp:359 — `m_statistics.merge(*chunkData)`.
        block_fetcher.statistics.base.record_get();

        // Vendor GzipChunkFetcher.hpp:410 — `m_nextUnprocessedBlockIndex += subchunks.size()`.
        let inserted = chunk.subchunks.len().max(1);
        next_unprocessed_block_index += inserted;

        // Hand off to post-process worker if there are markers; else
        // queue as immediately-ready for ordered write.
        {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.dispatch_post_process");
            match predecessor_window_for_postprocess {
                Some(window) => {
                    let rx = submit_post_process_to_pool(
                        thread_pool,
                        chunk,
                        window,
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
        }

        // Vendor parity: drain post-processes that are FIFO-ready to keep
        // the in-flight cap bounded. Vendor's
        // `waitForReplacedMarkers` (GzipChunkFetcher.hpp:478-518) does
        // this implicitly by blocking on the per-chunk future.
        while pending.len() > post_process_inflight_cap {
            drain_one_pending(
                &mut pending,
                window_map,
                writer,
                total_crc,
                total_size,
                block_fetcher,
            )?;
        }
        iter_us_sum += t_iter.elapsed().as_micros();
        iter_count += 1;
    }

    // Final drain — flush remaining post-processes in encoded order.
    while !pending.is_empty() {
        drain_one_pending(
            &mut pending,
            window_map,
            writer,
            total_crc,
            total_size,
            block_fetcher,
        )?;
    }
    let _ = submit_us_sum; // reserved for future submit-only timing

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "consumer_loop_summary",
            &format!(
                r#""iters":{iter_count},"iter_sum_us":{iter_us_sum},"prefetch_us":{prefetch_us_sum},"finder_us":{finder_us_sum},"fetcher_get_us":{fetcher_get_us_sum}"#,
            ),
        );
    }

    Ok(())
}

/// Compute the `stop_hint_bit` hint for the worker. Mirror of vendor's
/// `nextBlockOffset = m_blockFinder->get(validDataBlockIndex + 1)` at
/// BlockFetcher.hpp:268, with one gzippy guard for the confirmed-offset
/// / partition-guess interaction:
///
/// After chunk N finishes at a *confirmed* boundary B, `insert(B)` makes
/// `get(N+1)` return B, but `get(N+2)` may still be the old partition
/// guess P = B + spacing (e.g. only 5 bits past B on a 4 MiB spacing).
/// Using P as `until` caps the next worker to a handful of bits → ISA-L
/// `InvalidBlock`. Skip hints that are not meaningfully past `floor`.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn stop_hint_bit_for(
    block_finder: &GzipBlockFinder,
    block_index: usize,
    total_bits: usize,
    floor: usize,
) -> usize {
    let spacing = block_finder.spacing_in_bits();
    let min_gap = spacing.max(8);

    for delta in 1..=8 {
        let candidate = match block_finder.get(block_index + delta) {
            (Some(offset), GetReturnCode::Success) => offset.max(floor),
            (Some(offset), GetReturnCode::Failure) if offset == total_bits => {
                PREFETCH_NEXT_FILESIZE_ACCEPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                offset.max(floor)
            }
            (Some(offset), GetReturnCode::Failure) => offset.max(floor),
            _ => return total_bits,
        };
        if candidate > floor && candidate.saturating_sub(floor) >= min_gap {
            return candidate.min(total_bits);
        }
        if candidate >= total_bits {
            return total_bits;
        }
    }
    total_bits
}

/// Submit a decode task to the `ThreadPool`, returning the
/// `mpsc::Receiver` that `BlockFetcher::get` will wait on. Mirror of
/// vendor's `submitOnDemandTask(blockOffset, nextBlockOffset)` at
/// BlockFetcher.hpp:573-589 — both arrange for an asynchronous decode
/// whose `std::future<BlockData>` the caller waits on. Vendor calls
/// `m_threadPool.submit([this, ...] () { return decodeAndMeasureBlock(...); }, 0)`;
/// we mirror that with `thread_pool.submit(run_decode_task, /* priority */ 0)`.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn submit_decode_to_pool(
    thread_pool: &Arc<ThreadPool>,
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    block_fetcher: &Arc<BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> mpsc::Receiver<Result<Arc<ChunkData>, ChunkDecodeError>> {
    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "submit_decode",
            &format!(
                r#""partition_idx":{},"start_bit":{},"stop_hint_bit":{},"is_speculative_prefetch":{}"#,
                params.partition_idx,
                params.start_bit,
                params.stop_hint_bit,
                params.is_speculative_prefetch,
            ),
        );
    }
    let window_map = window_map.clone();
    let block_fetcher = block_fetcher.clone();
    let future = thread_pool.submit(
        move || run_decode_task(input, params, &window_map, &block_fetcher, configuration),
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
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn submit_post_process_to_pool(
    thread_pool: &Arc<ThreadPool>,
    chunk: ChunkData,
    predecessor_window: Window,
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
        move || run_post_process_task(chunk, predecessor_window),
        /* priority */ -1,
    );
    future.into_receiver()
}

/// Pool-side execution of a decode task (vendor `decodeBlock`,
/// GzipChunkFetcher.hpp:692-729). Routes to `decode_chunk_isal`
/// when the predecessor window is published, else
/// `speculative_decode_find_boundary` (marker bootstrap when no window).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn run_decode_task(
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    block_fetcher: &Arc<BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> Result<Arc<ChunkData>, ChunkDecodeError> {
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "worker.decode_chunk",
        &format!(
            r#""chunk_id":{},"start_bit":{},"stop_hint":{},"speculative":{}"#,
            params.partition_idx,
            params.start_bit,
            params.stop_hint_bit,
            params.is_speculative_prefetch
        ),
    );
    // SAFETY: `drive`'s contract — input outlives the thread pool.
    let input_bytes: &[u8] = unsafe { input.as_slice() };

    let label = trace::worker_label(params.partition_idx);
    let t0 = std::time::Instant::now();

    // Vendor's `decodeAndMeasureBlock` (BlockFetcher.hpp:649-672):
    // record decode start timestamp, decode wall time, decode end
    // timestamp. Drives the `Pool Efficiency` stat printed by
    // --verbose at end-of-decode. Without this, all four timer
    // entries in the stats dump are 0.0.
    let task_start_secs = std::time::UNIX_EPOCH
        .elapsed()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    block_fetcher
        .statistics
        .base
        .note_decode_block_start(task_start_secs);

    // Audit step 5: WindowMap is now Condvar-free (vendor
    // WindowMap.hpp:19-186 has no condition_variable). Workers do
    // NON-blocking `get`: if the predecessor's tail has been
    // published, use inexact ISA-L; otherwise speculative boundary
    // search (decode with markers, consumer's post-process
    // resolves them). This is the vendor model — workers never block
    // on the WindowMap; the dispatch order from `BlockFetcher::get`
    // and the post-process queue do the synchronization.
    // Chunk-0 special case: predecessor window is the zero sentinel.
    // Hold the materialized bytes on the stack so the
    // `decode_chunk_isal` slice borrow is valid for the call's
    // duration without going through WindowMap. Non-chunk-0 worker
    // gets the predecessor window via `window_map.get` (zero-alloc
    // Arc clone) and materializes bytes via `materialize_window`.
    let zero_window: [u8; 32768] = [0u8; 32768];
    let window: Option<Window> = if params.start_bit == 0 {
        None
    } else {
        window_map.get(params.start_bit)
    };

    let chunk_result = if params.start_bit == 0 {
        decode_chunk_isal(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            &zero_window[..],
            configuration,
        )
    } else if let Some(w) = window.as_ref() {
        let bytes = materialize_window(w);
        decode_chunk_isal(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            &bytes,
            configuration,
        )
    } else {
        speculative_decode_find_boundary(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            configuration,
        )
    };

    // Workers must not publish to WindowMap — consumer publishes fast-path
    // windows after decode (see consumer_loop marker-free branch).

    // Wrap in Arc to match BlockFetcher's `Value = Arc<ChunkData>`
    // (vendor's `std::shared_ptr<BlockData>` at BlockFetcher.hpp:46).
    let result = chunk_result.map(Arc::new);

    if trace::is_enabled() {
        let dur_us = t0.elapsed().as_micros();
        let path = if window_map.contains(params.start_bit) {
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
                    r#""partition_idx":{},"path":"{}","start_bit":{},"stop_hint_bit":{},"err":"{}","duration_us":{dur_us}"#,
                    params.partition_idx,
                    path,
                    params.start_bit,
                    params.stop_hint_bit,
                    trace::esc(&format!("{e:?}")),
                ),
            ),
        }
    }

    // Vendor's `decodeAndMeasureBlock` (BlockFetcher.hpp:660-672)
    // accumulates the decode wall time per task and records the end
    // timestamp.
    let decode_elapsed = t0.elapsed().as_secs_f64();
    block_fetcher
        .statistics
        .base
        .add_decode_block_time(decode_elapsed);
    let task_end_secs = std::time::UNIX_EPOCH
        .elapsed()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    block_fetcher
        .statistics
        .base
        .note_decode_block_end(task_end_secs);

    result
}

/// Pool-side execution of a post-process task. Mirror of the lambda
/// body at `GzipChunkFetcher::queueChunkForPostProcessing`
/// (vendor/.../GzipChunkFetcher.hpp:579-582).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn run_post_process_task(mut chunk: ChunkData, predecessor_window: Window) -> ChunkData {
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "post_process.task",
        &format!(
            r#""start_bit":{},"marker_bytes":{}"#,
            chunk.encoded_offset_bits,
            chunk.data_with_markers.len()
        ),
    );
    // Vendor lambda at GzipChunkFetcher.hpp:579-582 — `applyWindow` only.
    // WindowMap writes happen on the consumer (`publish_subchunk_windows`).
    let start_bit = chunk.encoded_offset_bits;
    let marker_bytes = chunk.data_with_markers.len();
    let t_materialize = std::time::Instant::now();
    let bytes = materialize_window(&predecessor_window);
    let materialize_us = t_materialize.elapsed().as_micros();
    // For chunks at or above the vendor LUT threshold (≥128 Ki u16
    // markers ≈ 256 KiB of marker prefix), fuse `replace_markers` +
    // `narrow_u16_to_u8` into one pass that reads u16 and writes the
    // resolved u8 directly into `chunk.narrowed`. Vendor parity:
    // `DecodedData.hpp:316-337` writes `target[i] = fullWindow[chunk[i]]`
    // in a single LUT pass. Below the threshold, the LUT construction
    // overhead (64 KiB stack zeroing + 32 KiB window copy ≈ 96 KiB)
    // dominates so we keep the existing two-pass path (AVX2 in-place
    // replace + scalar narrow).
    use crate::decompress::parallel::replace_markers::replace_markers_lut_narrow;
    const LUT_FUSE_THRESHOLD: usize = 128 * 1024;
    let dwm_len_pre = chunk.data_with_markers.len();
    let use_fused = dwm_len_pre >= LUT_FUSE_THRESHOLD && bytes.len() == 32768;
    let t_apply = std::time::Instant::now();
    let mut narrowed_buf: Option<crate::decompress::parallel::rpmalloc_alloc::types::U8> = None;
    let t_narrow;
    let narrow_us;
    if use_fused {
        // Fused path. `apply_window` is skipped entirely on this branch:
        // markers in `chunk.data_with_markers` are left in-place (no
        // downstream reader inspects them after this point — populate
        // takes `dwm_bytes` and the consumer writes `chunk.narrowed`).
        use crate::decompress::parallel::chunk_buffer_pool;
        let mut narrowed = chunk_buffer_pool::take_u8(dwm_len_pre);
        replace_markers_lut_narrow(&chunk.data_with_markers, &bytes, &mut narrowed);
        narrowed_buf = Some(narrowed);
        t_narrow = std::time::Instant::now();
        narrow_us = 0u128;
    } else {
        apply_window(&mut chunk, &bytes);
        t_narrow = std::time::Instant::now();
        if !chunk.data_with_markers.is_empty() {
            use crate::decompress::parallel::chunk_buffer_pool;
            let dwm_len = chunk.data_with_markers.len();
            let mut narrowed = chunk_buffer_pool::take_u8(dwm_len);
            narrow_u16_to_u8(&chunk.data_with_markers, &mut narrowed);
            narrowed_buf = Some(narrowed);
        }
        narrow_us = t_narrow.elapsed().as_micros();
    }
    let apply_us = t_apply.elapsed().as_micros();
    let _ = t_narrow;
    let t_pop = std::time::Instant::now();
    let dwm_slice: &[u8] = narrowed_buf.as_deref().unwrap_or(&[]);
    chunk.populate_subchunk_windows(&bytes, dwm_slice);
    let populate_us = t_pop.elapsed().as_micros();
    // Vendor parity: `ChunkData::applyWindow` (ChunkData.hpp:313-328)
    // CRC32s the resolved marker bytes on the worker. Previously this
    // ran on the consumer inside `drain_one_pending` and serialized
    // ~27 ms / 400 MB of pclmulqdq onto the wall-binding thread. Moving
    // it here parallelizes the work across the 16-thread pool — each
    // worker pays its own per-chunk CRC scan, and only the constant-time
    // polynomial `append` happens on the consumer.
    if let Some(narrowed) = narrowed_buf {
        {
            let _tv2 = trace_v2::SpanGuard::begin("post_process.crc_narrowed");
            chunk.narrowed_crc.update(&narrowed);
        }
        chunk.narrowed = narrowed;
    }
    if trace::is_enabled() {
        trace::emit(
            "post_process",
            "post_process_span",
            &format!(
                r#""start_bit":{start_bit},"materialize_us":{materialize_us},"apply_window_us":{apply_us},"populate_subchunk_windows_us":{populate_us},"narrow_us":{narrow_us},"marker_bytes":{marker_bytes}"#,
            ),
        );
    }
    chunk
}

/// Narrow `src: &[u16]` into `dst: &mut U8`, appending bytes. All values
/// in `src` MUST be < 256 (post-`apply_window` invariant). AVX2 path
/// uses `_mm256_packus_epi16` for 16-lane parallel narrowing
/// (saturating pack — since values are already < 256, saturation is
/// a no-op). Scalar tail handles the remainder.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn narrow_u16_to_u8(src: &[u16], dst: &mut crate::decompress::parallel::rpmalloc_alloc::types::U8) {
    dst.clear();
    dst.reserve(src.len());
    // AVX2 path disabled pending diagnostic: it produced a per-task
    // perf regression on neurotic (apply_window+populate were 3×
    // higher in the same trace) despite the AVX2 routine itself
    // being byte-correct (test_single_member_routing_multithread
    // passes). Suspected: AVX2 license keeps the post-process worker
    // downclocked through apply_window's SIMD too. Re-enable behind
    // GZIPPY_NARROW_AVX2 env once isolated.
    #[cfg(target_arch = "x86_64")]
    if std::env::var_os("GZIPPY_NARROW_AVX2").is_some()
        && std::arch::is_x86_feature_detected!("avx2")
    {
        // SAFETY: avx2 just detected at runtime.
        unsafe {
            narrow_u16_to_u8_avx2(src, dst);
        }
        return;
    }
    for &v in src {
        dst.push(v as u8);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#[target_feature(enable = "avx2")]
unsafe fn narrow_u16_to_u8_avx2(
    src: &[u16],
    dst: &mut crate::decompress::parallel::rpmalloc_alloc::types::U8,
) {
    use core::arch::x86_64::{
        _mm256_loadu_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_storeu_si256,
    };

    let n = src.len();
    let mut i = 0usize;
    // Each iteration consumes 32 u16s and produces 32 u8s.
    let chunk = 32usize;
    let simd_end = n & !(chunk - 1);
    // Use raw pointers into dst's spare capacity to avoid per-push
    // length checks. We bump len at the end.
    let dst_ptr = dst.as_mut_ptr();
    while i < simd_end {
        // Load two 256-bit vectors (16 u16 each).
        let a = _mm256_loadu_si256(src.as_ptr().add(i) as *const _);
        let b = _mm256_loadu_si256(src.as_ptr().add(i + 16) as *const _);
        // packus_epi16 packs across 128-bit lanes with saturation:
        //   out = [low(a)_lo, low(b)_lo, high(a)_lo, high(b)_lo]
        // (the 4 64-bit lanes are interleaved a/b within each 128-bit half).
        let packed = _mm256_packus_epi16(a, b);
        // Permute lanes to restore the natural order [a0..a15, b0..b15]:
        //   0b11_01_10_00 == permute mask (0, 2, 1, 3)
        let permuted = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        _mm256_storeu_si256(dst_ptr.add(i) as *mut _, permuted);
        i += chunk;
    }
    // Scalar tail.
    while i < n {
        *dst_ptr.add(i) = *src.get_unchecked(i) as u8;
        i += 1;
    }
    dst.set_len(n);
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
)))]
fn narrow_u16_to_u8(src: &[u16], dst: &mut crate::decompress::parallel::rpmalloc_alloc::types::U8) {
    dst.clear();
    dst.reserve(src.len());
    for &v in src {
        dst.push(v as u8);
    }
}

/// Consumer-thread publication of per-subchunk tail windows. Mirror of
/// vendor `appendSubchunksToIndexes` (GzipChunkFetcher.hpp:429-458).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn publish_subchunk_windows(window_map: &WindowMap, chunk: &ChunkData) {
    for sc in &chunk.subchunks {
        let sc_end_bit = sc.encoded_offset_bits + sc.encoded_size_bits;
        if sc_end_bit <= sc.encoded_offset_bits {
            continue;
        }
        let Some(w) = sc.window.as_ref() else {
            continue;
        };
        let existing = window_map.get(sc_end_bit);
        let may_insert = match existing {
            None => true,
            Some(ex) => !ex.is_empty(),
        };
        if may_insert {
            window_map.insert_bytes(sc_end_bit, &w[..]);
        }
    }
}

// ── Slow-path decoder (tryToDecode + BlockFinder iteration) ──────────────

/// Diagnostic counters for slow-path candidate iteration, surfaced in
/// `--verbose` stats: `NO_CANDIDATE` (no boundary found in the search
/// window) and the first-candidate trial-decode outcomes `OK` / `FAIL`.
pub static SLOW_PATH_NO_CANDIDATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static SLOW_PATH_FIRST_CANDIDATE_OK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static SLOW_PATH_FIRST_CANDIDATE_FAIL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Incremented when the slow-path boundary search uses the async
/// `RawBlockFinderCoordinator` (StreamedResults + single finder thread).
/// Proves production routes through the coordinator — see deletion-trap
/// test in `src/tests/routing.rs`.
pub static SPEC_FAIL_HEADER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SPEC_FAIL_BODY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SPEC_FAIL_INFLATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SPEC_FAIL_STOP_MISSED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static SPEC_FAIL_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static COORDINATOR_BOUNDARY_SEARCH_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumps once per consumer iter where the partition-keyed
/// `try_take_prefetched` returned a chunk but the safety guard
/// rejected it (mismatch between `chunk.max_acceptable_start_bit` and
/// consumer-requested `next_block_offset`) — distinct from
/// `prefetch_cache_miss`, which counts a prefetch being absent.
pub static PREFETCH_REJECT_BY_GUARD: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Lever G diagnostic: counts how often the consumer's
/// `Arc::try_unwrap(chunk_arc)` actually succeeded (HITS) vs fell back
/// to a deep clone (MISSES). Goal: HITS == chunk count, MISSES == 0.
pub static ARC_TRY_UNWRAP_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ARC_TRY_UNWRAP_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Speculative slow-path decode without a predecessor window. Must use
/// marker bootstrap — plain ISA-L with an empty dict resolves unknown
/// back-refs against zeros and corrupts output (Bug B, commit 4909ac7).
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn try_speculative_decode_candidate(
    input: &[u8],
    decode_start: usize,
    partition_seed: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let result = decode_chunk_marker_bootstrap_then_isal(
        input,
        decode_start,
        stop_hint_bit,
        &[],
        configuration,
    );
    // V1: classify failures so we know which fix to attack.
    //   header_fail  → "deflate header at bit X" — could be caught by
    //                  precode pre-pass (CountAllocatedLeaves port)
    //   body_fail    → "deflate body at bit X" — mid-stream Huffman
    //                  decode failed; precode wouldn't catch it
    //   inflate_fail → phase-2 IsalInflateWrapper / ResumableInflate2
    //   stop_missed  → chunk size cap hit (not really a "failure")
    if let Err(ref e) = result {
        use std::sync::atomic::Ordering;
        match e {
            ChunkDecodeError::BootstrapFailed(io_err) => {
                let msg = io_err.to_string();
                if msg.contains("deflate header") {
                    SPEC_FAIL_HEADER.fetch_add(1, Ordering::Relaxed);
                } else if msg.contains("deflate body") {
                    SPEC_FAIL_BODY.fetch_add(1, Ordering::Relaxed);
                } else {
                    SPEC_FAIL_OTHER.fetch_add(1, Ordering::Relaxed);
                }
            }
            ChunkDecodeError::InflateFailed(_) => {
                SPEC_FAIL_INFLATE.fetch_add(1, Ordering::Relaxed);
            }
            ChunkDecodeError::ExactStopMissed { .. } => {
                SPEC_FAIL_STOP_MISSED.fetch_add(1, Ordering::Relaxed);
            }
            ChunkDecodeError::UnsupportedPlatform => {
                SPEC_FAIL_OTHER.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let mut chunk = result?;
    // Vendor tryToDecode metadata (GzipChunk.hpp:716-722): encoded =
    // partition seed, max = actual decode start.
    if partition_seed < decode_start {
        let encoded_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        let first_sc_upper = chunk
            .subchunks
            .get(1)
            .map(|sc| sc.encoded_offset_bits)
            .unwrap_or(encoded_end);
        chunk.encoded_offset_bits = partition_seed;
        chunk.encoded_size_bits = encoded_end - partition_seed;
        chunk.max_acceptable_start_bit = decode_start;
        if let Some(first_sc) = chunk.subchunks.first_mut() {
            first_sc.encoded_offset_bits = partition_seed;
            first_sc.encoded_size_bits = first_sc_upper - partition_seed;
        }
    }
    Ok(chunk)
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn speculative_decode_find_boundary(
    input: &[u8],
    start_bit: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    const MAX_SCAN_BITS: usize = 512 * 1024 * 8;
    let input_bits = input.len() * 8;
    if start_bit >= input_bits {
        let mut chunk = ChunkData::new(start_bit, configuration);
        chunk.finalize(start_bit);
        return Ok(chunk);
    }
    let max_end = (start_bit + MAX_SCAN_BITS).min(input_bits);

    // Vendor `tryToDecode` (GzipChunk.hpp:712-841) attempts the partition
    // seed itself before walking BlockFinder candidates.
    if let Ok(chunk) =
        try_speculative_decode_candidate(input, start_bit, start_bit, stop_hint_bit, configuration)
    {
        SLOW_PATH_FIRST_CANDIDATE_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(chunk);
    }

    COORDINATOR_BOUNDARY_SEARCH_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Sync chunk-by-chunk boundary search — no thread spawn.
    // Per the bench-sm post-absorb-isal-tail profile, the prior
    // `with_scoped_boundary_search` path spent 77% of its
    // elapsed time in clone3 + 8 MiB stack page-faults +
    // scheduling + join overhead, and only 23% in the actual
    // BlockFinder scan. Verified empirically:
    //   Slow-path decode: ok=61 fail=0 no_candidate=0
    //   BlockFinder coordinator spawns: 33
    //   total_ms=3010 / scan_ms=702 / avg_total_us=91236
    // → ~70 ms / spawn of pure overhead.
    //
    // Sync trades the ~5 ms "streaming overlap" (consumer trying
    // candidate N while scan finds candidate N+1) for ~70 ms
    // saved per call. For 33 slow-path chunks per single run,
    // that's ~2.3 s of CPU-time saved per iteration across all
    // workers.
    if let Some(chunk) = RawBlockFinderCoordinator::with_sync_boundary_search(
        input,
        start_bit,
        max_end,
        |cand_bit| match try_speculative_decode_candidate(
            input,
            cand_bit,
            start_bit,
            stop_hint_bit,
            configuration,
        ) {
            Ok(chunk) => {
                SLOW_PATH_FIRST_CANDIDATE_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(chunk)
            }
            Err(_) => {
                SLOW_PATH_FIRST_CANDIDATE_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        },
    ) {
        return Ok(chunk);
    }

    // Near-EOF tail: when the 512 KiB scan window reaches the end of
    // the deflate stream and every BlockFinder candidate failed trial
    // decode (common for pipelined L9 random data — tail is a few KiB
    // of fixed/stored blocks with few valid dynamic boundaries), walk
    // byte-aligned offsets in the remaining tail. Cost is bounded:
    // e.g. 4 KiB tail → 4096 cheap read_header rejects.
    if max_end >= input_bits {
        let tail_start_byte = start_bit / 8;
        for byte_off in tail_start_byte..input.len() {
            let bit = byte_off * 8;
            if let Ok(chunk) = try_speculative_decode_candidate(
                input,
                bit,
                start_bit,
                stop_hint_bit,
                configuration,
            ) {
                SLOW_PATH_FIRST_CANDIDATE_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(chunk);
            }
        }
    }

    SLOW_PATH_NO_CANDIDATE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Err(ChunkDecodeError::ExactStopMissed {
        requested: start_bit,
        actual: max_end,
    })
}

// ── Pending-write queue (order-preserved output) ─────────────────────────

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#[allow(clippy::large_enum_variant)] // boxing ChunkData would add an alloc on the per-chunk write path
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
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
fn drain_one_pending<W: std::io::Write>(
    pending: &mut std::collections::VecDeque<PendingWrite>,
    window_map: &WindowMap,
    writer: &mut W,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>,
) -> Result<(), FetchError> {
    let _tv2 = trace_v2::SpanGuard::begin("consumer.drain");
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    let t_chunk = std::time::Instant::now();
    let t_recv = std::time::Instant::now();
    let (idx, mut chunk, cache_key) = match head {
        PendingWrite::Ready {
            idx,
            chunk,
            cache_key,
        } => (idx, chunk, cache_key),
        PendingWrite::Async { idx, rx, cache_key } => {
            let _tv2_wait = trace_v2::SpanGuard::begin_with(
                "wait.future_recv",
                &format!(r#""chunk_id":{idx}"#),
            );
            let chunk = rx.recv().map_err(|_| {
                FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: idx,
                    actual: 0,
                })
            })?;
            (idx, chunk, cache_key)
        }
    };
    let recv_us = t_recv.elapsed().as_micros();

    // Vendor `appendSubchunksToIndexes` window emplace — orchestrator only.
    let t_pub = std::time::Instant::now();
    {
        let _tv2 = trace_v2::SpanGuard::begin("consumer.publish_windows");
        publish_subchunk_windows(window_map, &chunk);
    }
    let publish_us = t_pub.elapsed().as_micros();

    // Mirror of vendor's per-chunk write loop (GzipChunkFetcher.hpp:333-342).
    // The post-process worker pre-narrows `data_with_markers` into
    // `chunk.narrowed` so the consumer thread does not pay the scalar
    // u16→u8 cast loop (~24ms/3.3MiB chunk previously dominated wall
    // time on real silesia). When the post-process step was skipped
    // (chunks with no markers), `narrowed` is empty and there is
    // nothing to write here — the clean tail comes through `chunk.data`
    // below.
    // CRC pipeline split:
    //   - chunk.data: bytes were CRC'd on the WORKER (every `append_clean` /
    //     `append_owned_buffer` updates `chunk.crc32s.last_mut()`). Re-CRCing
    //     them on the consumer was a ~5 MB pclmulqdq pass per chunk that
    //     dominated the consumer thread (~55% of E-core cycles per
    //     `target/tooling/profile-single-member-decompression-x86_64/9bbf17c-...`).
    //   - chunk.narrowed: resolved marker bytes — the WORKER could NOT CRC
    //     these during decode because markers in `data_with_markers` were
    //     unresolved u16 values; CRC happens here, on the consumer, for
    //     just the marker-prefix bytes (small relative to the full chunk).
    //
    // Vendor parity: rapidgzip combines per-chunk CRCs via constant-time
    // polynomial combine (crc32.hpp:214-258) and never re-CRCs decoded
    // bytes on the writer thread. The narrowed-bytes CRC is now computed
    // on the post-process worker (`run_post_process_task` stores it on
    // `chunk.narrowed_crc`) so the consumer just consumes the result —
    // no second scan of `chunk.narrowed` here.
    let t_crc_write = std::time::Instant::now();
    if !chunk.narrowed.is_empty() {
        {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.write_narrowed");
            writer
                .write_all(&chunk.narrowed)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        }
        *total_size += chunk.narrowed.len();
    } else if !chunk.data_with_markers.is_empty() {
        // Defensive fallback when `apply_window` ran but the narrow
        // step was somehow skipped — narrows scalar inline. Should
        // not happen on the production path (post_process always
        // runs narrow after apply_window). Keeps the consumer
        // correctness invariant: all marker bytes written before
        // `chunk.data`. This path computes CRC inline because the
        // worker's path didn't (no `chunk.narrowed` was produced).
        let mut narrowed: Vec<u8> = Vec::with_capacity(chunk.data_with_markers.len());
        for v in &chunk.data_with_markers {
            narrowed.push(*v as u8);
        }
        chunk.narrowed_crc.update(&narrowed);
        writer
            .write_all(&narrowed)
            .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        *total_size += narrowed.len();
    }
    if !chunk.data.is_empty() {
        {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.write_data");
            writer
                .write_all(&chunk.data)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        }
        *total_size += chunk.data.len();
    }
    let crc_write_us = t_crc_write.elapsed().as_micros();
    let t_combine = std::time::Instant::now();
    {
        let _tv2 = trace_v2::SpanGuard::begin("consumer.combine_crc");
        // Concatenated chunk output is (narrowed | data), so we combine
        // in that order. `chunk.narrowed_crc` covers narrowed bytes
        // (now computed on the post-process worker — vendor parity);
        // the worker-computed `chunk.crc32s` cover the data-stream bytes.
        // Multiple stream entries (multi-member inside a chunk) are
        // appended in stream order to match the output byte order.
        total_crc.append(&chunk.narrowed_crc);
        for stream_crc in &chunk.crc32s {
            total_crc.append(stream_crc);
        }
    }
    let combine_us = t_combine.elapsed().as_micros();
    let total_us = t_chunk.elapsed().as_micros();

    if trace::is_enabled() {
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"decoded":{},"recv_us":{recv_us},"publish_us":{publish_us},"crc_write_us":{crc_write_us},"combine_us":{combine_us},"total_us":{total_us}"#,
                chunk.decoded_size()
            ),
        );
    }
    // Lever G: do NOT re-insert the consumed chunk into the cache.
    // Single-pass forward decode never queries the same key twice, so
    // the post-consume re-insert was strictly wasted bookkeeping (and
    // an extra Arc allocation per chunk).
    let _ = cache_key;
    let _ = block_fetcher;
    Ok(())
}

// Non-x86_64 / non-isal stub.
#[cfg(not(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
)))]
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
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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
    #[ignore = "slow integration test; run with --ignored"]
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
    fn stop_hint_bit_for_skips_partition_guess_immediately_after_confirmed_end() {
        let spacing_bytes = 4 * 1024 * 1024;
        let spacing_bits = spacing_bytes * 8;
        let total_bits = spacing_bits * 10;
        let finder = GzipBlockFinder::new(0, spacing_bytes, Some(total_bits));
        // Chunk 0 confirmed end one byte into the next partition slot.
        let confirmed_end = spacing_bits - 5;
        finder.insert(confirmed_end);
        let until = super::stop_hint_bit_for(&finder, 1, total_bits, confirmed_end);
        assert!(
            until.saturating_sub(confirmed_end) >= spacing_bits,
            "until {until} must skip stale get(idx+1) guess {spacing_bits} (only 5b past end)"
        );
    }

    /// Regression for neurotic `make profile-decompression-x86_64` (64 MiB
    /// silesia → **gzip(1) -9 -c**, T=2). Skipped when benchmark_data is absent.
    #[test]
    #[ignore = "requires benchmark_data/silesia-large.bin; run with --ignored"]
    fn drive_silesia_head_gzip9_t2() {
        use crate::decompress::parallel::sm_driver::read_parallel_sm;
        use std::io::Read;

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
        std::io::Write::write_all(child.stdin.as_mut().expect("stdin"), head).expect("write");
        let mut gz = Vec::new();
        child
            .stdout
            .as_mut()
            .expect("stdout")
            .read_to_end(&mut gz)
            .expect("read gzip stdout");
        let _ = child.wait();

        let chunk_size = 4 * 1024 * 1024;
        let mut out = Vec::new();
        let result = read_parallel_sm(&gz, &mut out, 2, chunk_size);
        assert_eq!(result.expect("read_parallel_sm T=2").total_size, head.len());
        assert_eq!(out, head);
    }

    #[test]
    #[ignore = "slow integration test (~60 MiB); run with --ignored"]
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
