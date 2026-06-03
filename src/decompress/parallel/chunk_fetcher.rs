#![cfg(parallel_sm)]

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
#[cfg(parallel_sm)]
use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
#[cfg(parallel_sm)]
use std::sync::Arc;

#[cfg(parallel_sm)]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(parallel_sm)]
use crate::decompress::parallel::block_fetcher::BlockFetcher;
#[cfg(parallel_sm)]
use crate::decompress::parallel::block_map::{append_subchunks_to_block_map, BlockMap};
#[cfg(parallel_sm)]
use crate::decompress::parallel::compressed_vector::CompressionType;
#[cfg(parallel_sm)]
use crate::decompress::parallel::crc32::CRC32Calculator;
#[cfg(parallel_sm)]
use crate::decompress::parallel::gzip_block_finder::{GetReturnCode, GzipBlockFinder};
#[cfg(parallel_sm)]
use crate::decompress::parallel::gzip_chunk::{
    decode_chunk_isal, decode_chunk_marker_bootstrap_then_isal,
};
#[cfg(parallel_sm)]
use crate::decompress::parallel::prefetcher::FetchMultiStream;
#[cfg(parallel_sm)]
use crate::decompress::parallel::raw_block_finder::RawBlockFinderCoordinator;
#[cfg(parallel_sm)]
use crate::decompress::parallel::streamed_results::StreamedGetReturnCode;
#[cfg(parallel_sm)]
use crate::decompress::parallel::thread_pool::ThreadPool;
#[cfg(parallel_sm)]
use crate::decompress::parallel::trace;
use crate::decompress::parallel::trace_v2;
#[cfg(parallel_sm)]
use crate::decompress::parallel::window_map::{Window, WindowMap};
#[cfg(parallel_sm)]
use std::borrow::Cow;
#[cfg(parallel_sm)]
use std::sync::mpsc;
#[cfg(parallel_sm)]
use std::time::Duration;

/// Materialize a `Window`'s raw bytes. Mirror of vendor's
/// `sharedLastWindow->decompress()` call at
/// GzipChunkFetcher.hpp:341. For `CompressionType::None` (the
/// single-pass single-member production case) this is a zero-alloc
/// slice borrow into the existing `CompressedVector`. For Zlib
/// (seekable-reader path) it allocates a fresh `Vec<u8>`.
#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
pub type UnsplitBlocks = Arc<std::sync::Mutex<std::collections::HashMap<usize, usize>>>;

/// Construct a fresh, empty `UnsplitBlocks`. Mirror of the default
/// construction of `m_unsplitBlocks` as a `GzipChunkFetcher` member.
#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
const FETCH_MEMORY_PER_STREAM: usize = 3;
#[cfg(parallel_sm)]
const FETCH_MAX_STREAM_COUNT: usize = 16;

// ── Worker pool job parameters ───────────────────────────────────────────

/// Static descriptor for a single decode task — replaces the prior
/// `DecodeJob` struct now that dispatch goes through
/// `ThreadPool::submit(closure, priority)` directly (no enum-tagged
/// work channel). Mirror of the arguments captured by vendor's
/// `decodeAndMeasureBlock` task lambda body at
/// vendor/.../core/BlockFetcher.hpp:555-558.
#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
#[derive(Clone, Copy)]
struct InputSlice {
    ptr: *const u8,
    len: usize,
}

#[cfg(parallel_sm)]
unsafe impl Send for InputSlice {}
#[cfg(parallel_sm)]
unsafe impl Sync for InputSlice {}

#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
pub fn drive<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    drive_impl(input, writer, out_fd, parallelization, configuration, None)
}

/// Clean-window oracle (advisor 2026-05-29): the cheapest experiment that
/// discriminates "is the marker/copy/bootstrap pipeline the rapidgzip gap"
/// from "the gap is inner-loop / scan". Decode every chunk with its TRUE
/// predecessor window so NO chunk takes the speculative bootstrap path —
/// no markers, hence no `append_markered` / `absorb_isal_tail` /
/// `narrow_u16_to_u8` copies. Pass 1 runs the normal speculative decode to
/// a sink, populating a shared `WindowMap` with every chunk-boundary
/// window; pass 2 re-runs with that map pre-seeded (every
/// `window_map.get(start_bit)` HITS → `decode_chunk_isal`) and is the
/// TIMED measurement. Reuses the entire production pool/consumer/decoder —
/// apples-to-apples with production minus speculation. Output is correct
/// (windows from pass 1 are real), so byte-exactness is verifiable.
/// Triggered by `GZIPPY_CLEAN_WINDOW_ORACLE=1`.
pub fn drive_clean_window_oracle<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    let seed = WindowMap::with_compression(
        crate::decompress::parallel::compressed_vector::CompressionType::None,
    );
    // Pass 1: normal speculative decode to a sink — its ONLY purpose is to
    // populate `seed` with a real 32 KiB dict at every published block
    // boundary (the consumer publishes one per chunk/subchunk end).
    let mut sink = std::io::sink();
    drive_impl(
        input,
        &mut sink,
        None,
        parallelization,
        configuration,
        Some(seed.clone()),
    )?;

    let _ = &seed; // (pass-1 window keys are NOT trusted for boundaries; see below)
    let total_bits = input.len() * 8;
    let pool_size = parallelization.max(1).min(num_cpus::get_physical().max(1));

    // --- Phase A (UNTIMED): self-derive REAL block boundaries + correct dicts. ---
    // Do NOT trust the speculative pass's published window keys for span starts:
    // some are not confirmed block boundaries, so decode_chunk_isal starting
    // there misreads random bits as a block header ("Stored block len=0"). Chain
    // from the decoder's OWN returned end bit instead — decode_chunk_isal stops
    // at a real block boundary at-or-past stop_hint and reports it
    // (encoded_offset_bits + encoded_size_bits), so the next span begins at a
    // guaranteed-valid boundary with the true trailing 32 KiB as its dict. This
    // is the honest "clean window known a priori" partition.
    const STRIDE_BITS: usize = 4 * 1024 * 1024 * 8; // ~4 MiB compressed per span
    let mut starts: Vec<usize> = Vec::new();
    let mut dicts: Vec<Vec<u8>> = Vec::new();
    {
        let mut prev_tail = vec![0u8; 32768];
        let mut cur = 0usize;
        while cur < total_bits {
            starts.push(cur);
            dicts.push(prev_tail.clone());
            let stop_hint = (cur + STRIDE_BITS).min(total_bits);
            let c = crate::decompress::parallel::gzip_chunk::decode_chunk_isal(
                input,
                cur,
                stop_hint,
                &prev_tail,
                configuration,
            )
            .map_err(FetchError::Decode)?;
            let end_bit = c.encoded_offset_bits + c.encoded_size_bits;
            let data_len = c.data.len();
            if data_len >= 32768 {
                let mut t = vec![0u8; 32768];
                c.data.copy_last_into(&mut t);
                prev_tail = t;
            } else {
                let mut nt = prev_tail.clone();
                let mut tail_bytes = vec![0u8; data_len];
                c.data.copy_range_into(0, &mut tail_bytes);
                nt.extend_from_slice(&tail_bytes);
                let n = nt.len();
                prev_tail = nt[n.saturating_sub(32768)..].to_vec();
            }
            if end_bit <= cur {
                break; // no forward progress — stop (last span ran to EOF/BFINAL)
            }
            cur = end_bit;
        }
    }
    let n_spans = starts.len();

    // --- Phase B (TIMED): clean-parallel decode with the correct dicts. ---
    // A purpose-built clean-parallel driver that BYPASSES the speculative
    // scheduler (block_finder / prefetcher / block_map) entirely — those
    // carry tight invariants a hand-seeded partition violates. We dispatch
    // one clean `decode_chunk_isal` per span across `pool_size` workers and
    // write results in order. This isolates EXACTLY the clean-parallel
    // ceiling: pure-Rust inner decode × parallelism + the in-order consumer
    // write, with ZERO marker / bootstrap / narrow / absorb cost. It is the
    // markers-vs-scaling discriminator.
    let mut results: Vec<Option<Result<ChunkData, ChunkDecodeError>>> =
        (0..n_spans).map(|_| None).collect();

    let t = std::time::Instant::now();
    {
        // Atomic-index work queue (NO wave barriers): spawn `pool_size`
        // persistent workers that each pull the next span from a shared counter
        // and decode it. The previous wave-of-pool_size design serialized on a
        // per-wave join — a single straggler stalled the whole wave, giving only
        // ~1.2× effective parallelism and making this oracle's wall dispatch-
        // limited rather than decode-limited. A pull queue keeps every worker
        // busy to the end, isolating the true clean-parallel decode ceiling.
        // The brief mutex is held only to store a finished result, never during
        // decode, so it adds no decode contention.
        use std::sync::atomic::{AtomicUsize, Ordering as AOrd};
        let next = AtomicUsize::new(0);
        let results_mtx = std::sync::Mutex::new(&mut results);
        std::thread::scope(|scope| {
            for _ in 0..pool_size {
                scope.spawn(|| loop {
                    let span_idx = next.fetch_add(1, AOrd::Relaxed);
                    if span_idx >= n_spans {
                        break;
                    }
                    let start_bit = starts[span_idx];
                    let stop_hint = starts.get(span_idx + 1).copied().unwrap_or(total_bits);
                    let dict: &[u8] = &dicts[span_idx];
                    let r = crate::decompress::parallel::gzip_chunk::decode_chunk_isal(
                        input,
                        start_bit,
                        stop_hint,
                        dict,
                        configuration,
                    );
                    results_mtx.lock().unwrap()[span_idx] = Some(r);
                });
            }
        });
    }

    // In-order write + CRC fold (mirror of the consumer's clean branch).
    let mut total_crc = CRC32Calculator::new();
    let mut total_size = 0usize;
    for (i, slot) in results.into_iter().enumerate() {
        let chunk = slot
            .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                requested: starts[i],
                actual: 0,
            }))?
            .map_err(FetchError::Decode)?;
        // Clean chunks carry no markers: output is the segmented `data`,
        // CRC is the per-stream crc32s folded in order.
        chunk
            .data
            .write_payload_skipping_prefix(chunk.data_prefix_len, writer)
            .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        total_size += chunk.decoded_size();
        for stream_crc in &chunk.crc32s {
            total_crc.append(stream_crc);
        }
        if std::env::var_os("GZIPPY_ORACLE_TRACE").is_some() {
            eprintln!(
                "ORACLE_SPAN i={i} start_bit={} stop_hint={} decoded_bytes={} out_cum={total_size} end_bit={} prefix_len={}",
                starts[i],
                starts.get(i + 1).copied().unwrap_or(total_bits),
                chunk.decoded_size(),
                chunk.encoded_offset_bits + chunk.encoded_size_bits,
                chunk.data_prefix_len,
            );
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let mb = total_size as f64 / secs / 1e6;
    eprintln!(
        "CLEAN_WINDOW_ORACLE pass2 wall={secs:.4}s out={total_size} bytes {mb:.0} MB/s ({parallelization} workers, {n_spans} spans)"
    );
    Ok((total_crc.crc32(), total_size))
}

fn drive_impl<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    configuration: ChunkConfiguration,
    seed_window_map: Option<WindowMap>,
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
    // Oracle mode injects a pre-populated map (clean-window experiment);
    // production passes None and builds a fresh one.
    let window_map = seed_window_map.unwrap_or_else(|| {
        WindowMap::with_compression(
            crate::decompress::parallel::compressed_vector::CompressionType::None,
        )
    });
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
    // (GZIPPY_CACHE_CAP sweep removed 2026-05-28: shrinking the LRU cap did
    // NOT bound in-flight depth — chunks are held by the prefetch cache +
    // pending reorder queue, not the LRU — and the wall delta was ordering
    // noise. See project_real_gap_pinned_2026_05_28 memory.)
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
        out_fd,
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

    // Report replay stats even on a (CRC) error so the bypass experiment
    // can see hit/miss counts when the run aborts.
    crate::decompress::parallel::decode_bypass::report_replay_stats();
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
        eprintln!(
            "  Eager post-process: runs={} runs_nonempty={} inspected={} submitted={} max/run={} reused={}",
            EAGER_PROBE_RUNS.load(Ordering::Relaxed),
            EAGER_PROBE_RUNS_NONEMPTY.load(Ordering::Relaxed),
            EAGER_PROBE_INSPECTED.load(Ordering::Relaxed),
            EAGER_PROBE_SUBMITTED.load(Ordering::Relaxed),
            EAGER_PROBE_MAX_PER_RUN.load(Ordering::Relaxed),
            EAGER_PROBE_REUSED.load(Ordering::Relaxed),
        );
        use crate::decompress::parallel::chunk_buffer_pool::*;
        eprintln!(
            "  Max concurrently-live ChunkData (in-flight depth): {}  (live now: {})",
            crate::decompress::parallel::chunk_data::MAX_LIVE_CHUNKS.load(Ordering::Relaxed),
            crate::decompress::parallel::chunk_data::LIVE_CHUNKS.load(Ordering::Relaxed),
        );
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
        // Per-worker distribution — disambiguates "16 workers each
        // cold-start" (first-touch dominance) vs "worker 0 is the
        // hotspot" (consumer-thread-on-wrong-bucket).
        // Shows only workers with any activity to keep the dump
        // readable when MAX_WORKERS = 64.
        eprintln!("  Per-worker buffer pool activity (worker: u8 h/m/r | u16 h/m/r):");
        for i in 0..TAKE_U8_HITS_BY_WORKER.len() {
            let u8h = TAKE_U8_HITS_BY_WORKER[i].load(Ordering::Relaxed);
            let u8m = TAKE_U8_MISSES_BY_WORKER[i].load(Ordering::Relaxed);
            let u8r = RETURN_U8_BY_WORKER[i].load(Ordering::Relaxed);
            let u16h = TAKE_U16_HITS_BY_WORKER[i].load(Ordering::Relaxed);
            let u16m = TAKE_U16_MISSES_BY_WORKER[i].load(Ordering::Relaxed);
            let u16r = RETURN_U16_BY_WORKER[i].load(Ordering::Relaxed);
            if u8h + u8m + u8r + u16h + u16m + u16r > 0 {
                eprintln!(
                    "    w{:>2}:  {}/{}/{}  |  {}/{}/{}",
                    i, u8h, u8m, u8r, u16h, u16m, u16r
                );
            }
        }
        use crate::decompress::parallel::gzip_chunk::{
            BOOTSTRAP_OUTPUT_ALLOCS, BOOTSTRAP_OUTPUT_DROPPED, BOOTSTRAP_OUTPUT_RETURNS,
            BOOTSTRAP_OUTPUT_REUSED_BYTES, BOOTSTRAP_OUTPUT_TAKES,
        };
        let takes = BOOTSTRAP_OUTPUT_TAKES.load(Ordering::Relaxed);
        let allocs = BOOTSTRAP_OUTPUT_ALLOCS.load(Ordering::Relaxed);
        let reuse_pct = if takes > 0 {
            100.0 * (takes - allocs) as f64 / takes as f64
        } else {
            0.0
        };
        eprintln!(
            "  Bootstrap pool: takes={} allocs={} returns={} dropped={} reused_MB={:.1} reuse_rate={:.1}%",
            takes,
            allocs,
            BOOTSTRAP_OUTPUT_RETURNS.load(Ordering::Relaxed),
            BOOTSTRAP_OUTPUT_DROPPED.load(Ordering::Relaxed),
            BOOTSTRAP_OUTPUT_REUSED_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            reuse_pct,
        );
        // Slow-path candidate iteration: ok + fail + no_candidate sum
        // to the slow-path call count.
        eprintln!(
            "  Slow-path decode: ok={} fail={} no_candidate={}",
            SLOW_PATH_FIRST_CANDIDATE_OK.load(Ordering::Relaxed),
            SLOW_PATH_FIRST_CANDIDATE_FAIL.load(Ordering::Relaxed),
            SLOW_PATH_NO_CANDIDATE.load(Ordering::Relaxed),
        );
        // Early window publish: did the worker publish provably-clean
        // tail windows off the consumer's serial path (the lever)? A high
        // `published` count with `range_speculative` accounting for the
        // rest is the success signal; `tail_not_clean` are chunks that
        // fell back to the consumer's serial `get_last_window_vec(pred)`.
        eprintln!(
            "  Early window publish: published={} tail_not_clean={} range_speculative={}",
            EARLY_WINDOW_PUBLISHED.load(Ordering::Relaxed),
            EARLY_WINDOW_TAIL_NOT_CLEAN.load(Ordering::Relaxed),
            EARLY_WINDOW_RANGE_SPECULATIVE.load(Ordering::Relaxed),
        );
        // (Design B) Speculative side-slot: did early-published speculative
        // windows get promoted (accept) off the consumer's serial chain, or
        // evicted (reject)? High promoted = the serial window-resolution
        // chain was broken for those chunks (the lever).
        eprintln!(
            "  Speculative window slot: spec_published={} promoted={} evicted={}",
            EARLY_SPEC_PUBLISHED.load(Ordering::Relaxed),
            EARLY_SPEC_PROMOTED.load(Ordering::Relaxed),
            EARLY_SPEC_EVICTED.load(Ordering::Relaxed),
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
        // Post-process path mix: tells us if the AVX2 narrow re-enable
        // is bench-relevant for this fixture. Small-markers path is the
        // only one that calls `narrow_u16_to_u8`.
        eprintln!(
            "  Post-process path: fused_lut={} small_markers={}",
            POST_PROCESS_FUSED_PATH.load(Ordering::Relaxed),
            POST_PROCESS_SMALL_MARKERS_PATH.load(Ordering::Relaxed),
        );
        // Prefetch dispatcher outcomes: disambiguates worker idleness.
        // If `called` is high but `saturated` + `zero_submitted` together
        // dominate, dispatch is firing but the gates short-circuit it.
        // If `called` is low, the dispatcher just isn't being invoked
        // often enough — and the fix is hoisting the call site.
        use crate::decompress::parallel::block_fetcher::{
            PREFETCH_NEW_BLOCKS_CALLED, PREFETCH_RETURN_SATURATED, PREFETCH_RETURN_SUBMITTED_ANY,
            PREFETCH_RETURN_ZERO_SUBMITTED, PREFETCH_TOTAL_SUBMITTED,
        };
        eprintln!(
            "  Prefetch dispatch: called={} saturated={} zero_submitted={} any_submitted={} total_submitted={}",
            PREFETCH_NEW_BLOCKS_CALLED.load(Ordering::Relaxed),
            PREFETCH_RETURN_SATURATED.load(Ordering::Relaxed),
            PREFETCH_RETURN_ZERO_SUBMITTED.load(Ordering::Relaxed),
            PREFETCH_RETURN_SUBMITTED_ANY.load(Ordering::Relaxed),
            PREFETCH_TOTAL_SUBMITTED.load(Ordering::Relaxed),
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
        let bs_postflip = gc::BOOTSTRAP_POST_FLIP_U16_BYTES.load(Ordering::Relaxed);
        let postflip_pct = if bs_b_bytes > 0 {
            100.0 * bs_postflip as f64 / bs_b_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "  Bootstrap per-block: header_calls={bs_h_calls} header_ms={:.1} avg_header_us={:.1} body_ms={:.1} body_bytes={bs_b_bytes} body_rate_MB/s={:.0} post_flip_u16_bytes={bs_postflip} ({postflip_pct:.1}% of body = Design-B1 prize)",
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
    // DECODE-BYPASS: serialize the capture map (no-op unless capture on).
    crate::decompress::parallel::decode_bypass::flush_capture();
    Ok((total_crc.crc32(), total_size))
}

// ── Consumer: processNextChunk port ──────────────────────────────────────

/// Vendor-faithful port of `GzipChunkFetcher::processNextChunk`
/// (vendor/.../GzipChunkFetcher.hpp:311-362), wrapped in a `loop` that
/// plays the role of `ParallelGzipReader::read`'s outer loop
/// (ParallelGzipReader.hpp:702-810). Every per-iteration step is a
/// vendor primitive — no inline ring, no inline cache logic, no
/// inline take-from-prefetch.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn consumer_loop<W: std::io::Write>(
    input: InputSlice,
    writer: &mut W,
    out_fd: Option<i32>,
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

    // Eager post-process probe state (GZIPPY_EAGER_POSTPROC=1). Maps a
    // chunk's real encoded_offset_bits → an in-flight apply_window
    // receiver submitted eagerly during a stall. The consumer reuses the
    // receiver when it reaches that offset (see dispatch site below).
    // Default OFF so it's a clean A/B against the in-order baseline.
    let eager_enabled = eager_postproc_enabled();
    let mut eager_submitted: EagerSubmitted = std::collections::HashMap::new();

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
                // (Design B) The chunk's `encoded_end` — preserved by
                // `set_encoded_offset` through the accept rewrite, so it
                // equals both the consumer's post-accept `chunk_end_bit`
                // AND the worker's speculative side-slot key. Used to
                // promote (accept) / evict (reject) the speculative window.
                let spec_key = arc.encoded_offset_bits + arc.encoded_size_bits;
                if arc.matches_encoded_offset(next_block_offset) && handoff_at_decode_start {
                    chunk_arc_from_partition = Some(arc.clone());
                    // ACCEPT: confirmed_start == max == decode_origin, so
                    // the worker's speculative key == this chunk's
                    // accept-time `chunk_end_bit`. Promote (move spec→real)
                    // synchronously, BEFORE the consumer advances to any
                    // successor — so no successor can ever resolve against
                    // an un-promoted window. No-op if the worker skipped
                    // early-publish (unclean tail); the serial publish at
                    // ~line 1361 then handles the key.
                    if window_map.promote_speculative(spec_key) {
                        EARLY_SPEC_PROMOTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
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
                    // REJECT: this speculative chunk is discarded and
                    // re-decoded at the real offset below. Evict its
                    // speculative window so a stale unconfirmed tail can
                    // never be promoted later (the re-decode publishes a
                    // fresh, confirmed window). No-op if the worker skipped
                    // early-publish for this chunk.
                    if window_map.evict_speculative(spec_key) {
                        EARLY_SPEC_EVICTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
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
                // The head chunk is NOT in the prefetch cache → the
                // consumer is about to BLOCK on its decode (the
                // `wait.block_fetcher_get` span, the other real serial wait
                // besides `wait.future_recv`). Vendor's
                // `queuePrefetchedChunkPostProcessing` overlaps post-process
                // of ready successors with exactly this kind of stall —
                // submit it NOW so it runs on the pool while we block.
                if eager_enabled {
                    eager_postprocess_prefetched(
                        block_fetcher,
                        window_map,
                        thread_pool,
                        &mut eager_submitted,
                    );
                }
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

        // Real offset the eager probe would have keyed this chunk under
        // (its `encoded_offset_bits` BEFORE set_encoded_offset rewrites
        // it). Used to reuse an eager-submitted post-process below.
        let chunk_offset_pre_set = chunk.encoded_offset_bits;

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
        // Predecessor-window key the consumer resolved against (marker
        // branch only). Used to validate an eager-submitted result is
        // byte-identical before reusing it.
        let mut consumer_pred_key: Option<usize> = None;
        if chunk.data_with_markers.is_empty() {
            // No markers → apply_window is a no-op. Publish successor window
            // on the consumer only (vendor queueChunkForPostProcessing:558-575).
            // Mirror vendor `getLastWindow(*previousWindow)` — not `last_32kib`
            // alone, which ignores the predecessor chain at stream start.
            // B/E span: its DURATION is the INDEPENDENT per-link serial
            // resolve/publish work Fulcrum's `model` view reads as L_resolve
            // (NOT the inter-publish gap — that conflation is the tautology).
            // `end_bit` keys the publish so the model orders + de-dups links.
            let _tv2 = trace_v2::SpanGuard::begin_with(
                "consumer.window_publish_clean",
                &format!(r#""end_bit":{chunk_end_bit},"had_markers":false"#),
            );
            if let Some((_pred_key, pred)) = window_map.get_predecessor(chunk.encoded_offset_bits) {
                let bytes = materialize_window(&pred);
                let tail = chunk
                    .last_32kib_window_vec()
                    .unwrap_or_else(|| chunk.get_last_window_vec(&bytes));
                window_map.insert_owned_none(chunk_end_bit, tail);
                trace_v2::emit_instant(
                    "causal.window_publish",
                    &format!(
                        r#""start_bit":{},"end_bit":{chunk_end_bit},"site":"consumer_clean""#,
                        chunk.encoded_offset_bits
                    ),
                    "t",
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
                    // Vendor GzipChunkFetcher.hpp:513 — `drain_one_pending`
                    // queues successor post-processing during its
                    // `wait.future_recv` block (see `eager_ctx`). This spin
                    // is empirically near-0× (has_predecessor stays true
                    // after chunk 0), but keep the eager hook wired through
                    // for the rare overshoot case.
                    drain_one_pending(
                        &mut pending,
                        window_map,
                        writer,
                        out_fd,
                        total_crc,
                        total_size,
                        block_fetcher,
                        if eager_enabled {
                            Some((thread_pool, &mut eager_submitted))
                        } else {
                            None
                        },
                    )?;
                }
            }
            // B/E span: DURATION = INDEPENDENT serial marker-resolve/publish
            // work (L_resolve) for the model view; `end_bit` keys the link.
            // The blocking WAIT for the predecessor is the SEPARATE
            // `consumer.wait_replaced_markers` span above, so this span is the
            // resolve WORK only, not the wait.
            let _tv2_pred = trace_v2::SpanGuard::begin_with(
                "consumer.window_publish_marker",
                &format!(r#""end_bit":{chunk_end_bit},"had_markers":true"#),
            );
            let (pred_key, window) = window_map
                .get_predecessor(chunk.encoded_offset_bits)
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: chunk.encoded_offset_bits,
                    actual: furthest_decoded_bit,
                }))?;
            consumer_pred_key = Some(pred_key);
            // Vendor `GzipChunkFetcher.hpp:341`: `sharedLastWindow->
            // decompress()` materializes the bytes once. For
            // CompressionType::None this is a zero-alloc slice borrow.
            let window_bytes = materialize_window(&window);
            let tail = chunk.get_last_window_vec(&window_bytes);
            window_map.insert_owned_none(chunk_end_bit, tail);
            trace_v2::emit_instant(
                "causal.window_publish",
                &format!(
                    r#""start_bit":{},"end_bit":{chunk_end_bit},"site":"consumer_marker""#,
                    chunk.encoded_offset_bits
                ),
                "t",
            );
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
        // Vendor GzipChunkFetcher.hpp:380-382 — when the chunk produced
        // multiple subchunks, the fetching strategy's index accounting
        // (in CHUNK units) needs to be re-stretched into SUBCHUNK units
        // so it matches `block_offsets.len()`. Without this, the
        // strategy's `prefetch()` returns chunk-indexes that are ALREADY
        // in `block_offsets` — the BlockFinder returns them as
        // confirmed sub-partition offsets and `prefetch_new_blocks`
        // emits sub-partition prefetches that vendor never emits.
        // See `BlockFetcher::split_index` doc comment for the full
        // chain. Empirically: 26 sub-partition emits per silesia-large
        // run worth ~50 ms wall.
        if chunk.subchunks.len() > 1 {
            block_fetcher.split_index(next_unprocessed_block_index, chunk.subchunks.len());
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
                    // Reuse an eager-submitted apply_window result if one
                    // exists for this chunk AND it resolved against the
                    // SAME predecessor window (byte-identity guard). The
                    // eager probe keyed on the chunk's pre-set offset;
                    // post-processing the same chunk+window earlier vs.
                    // now yields identical bytes.
                    let reuse = if eager_enabled {
                        use std::sync::atomic::Ordering;
                        match eager_submitted.remove(&chunk_offset_pre_set) {
                            Some((eager_pred_key, eager_rx))
                                if consumer_pred_key == Some(eager_pred_key) =>
                            {
                                EAGER_PROBE_REUSED.fetch_add(1, Ordering::Relaxed);
                                Some(eager_rx)
                            }
                            Some(_) => {
                                // Found, but it resolved against a
                                // different predecessor window → rejecting
                                // is mandatory for byte-identity.
                                EAGER_PROBE_REUSE_PRED_MISMATCH.fetch_add(1, Ordering::Relaxed);
                                None
                            }
                            None => {
                                EAGER_PROBE_REUSE_KEY_ABSENT.fetch_add(1, Ordering::Relaxed);
                                None
                            }
                        }
                    } else {
                        None
                    };
                    let rx = match reuse {
                        Some(eager_rx) => eager_rx,
                        None => submit_post_process_to_pool(
                            thread_pool,
                            chunk,
                            window,
                            partition_idx_for_trace,
                        ),
                    };
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
                out_fd,
                total_crc,
                total_size,
                block_fetcher,
                if eager_enabled {
                    Some((thread_pool, &mut eager_submitted))
                } else {
                    None
                },
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
            out_fd,
            total_crc,
            total_size,
            block_fetcher,
            if eager_enabled {
                Some((thread_pool, &mut eager_submitted))
            } else {
                None
            },
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

    // Eager post-process probe report — the deliverable. Always printed
    // to stderr when the probe is enabled so an A/B run records whether
    // the lever actually fired. If `submitted ≈ 0`, the prefetch cache is
    // empty during the stall and the real lever is prefetch DEPTH, not
    // post-process eagerness.
    if eager_enabled {
        use std::sync::atomic::Ordering;
        let runs = EAGER_PROBE_RUNS.load(Ordering::Relaxed);
        let inspected = EAGER_PROBE_INSPECTED.load(Ordering::Relaxed);
        let submitted = EAGER_PROBE_SUBMITTED.load(Ordering::Relaxed);
        let nonempty = EAGER_PROBE_RUNS_NONEMPTY.load(Ordering::Relaxed);
        let max_per_run = EAGER_PROBE_MAX_PER_RUN.load(Ordering::Relaxed);
        let reused = EAGER_PROBE_REUSED.load(Ordering::Relaxed);
        let avg_inspected = if runs > 0 {
            inspected as f64 / runs as f64
        } else {
            0.0
        };
        let key_absent = EAGER_PROBE_REUSE_KEY_ABSENT.load(Ordering::Relaxed);
        let pred_mismatch = EAGER_PROBE_REUSE_PRED_MISMATCH.load(Ordering::Relaxed);
        eprintln!(
            "[gzippy EAGER_POSTPROC] probe_runs={runs} runs_with_ready_successor={nonempty} \
             cache_chunks_inspected={inspected} (avg {avg_inspected:.2}/run) \
             eager_submitted={submitted} max_per_run={max_per_run} reused_by_consumer={reused} \
             reuse_miss[key_absent={key_absent} pred_mismatch={pred_mismatch}]"
        );
        if submitted == 0 {
            eprintln!(
                "[gzippy EAGER_POSTPROC] KEY FINDING: 0 ready successors during stalls — \
                 the prefetch cache holds no post-processable successor whose predecessor \
                 window is published. The lever is prefetch DEPTH, not post-process eagerness."
            );
        }
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
#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
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
    // FRONTIER SCHEDULING (gzippy deviation from vendor's flat /* priority */ 0
    // at BlockFetcher.hpp:557 prefetch AND :586 submitOnDemandTask).
    //
    // Vendor submits both the on-demand frontier decode and speculative
    // prefetches at priority 0, relying on the on-demand task being ENQUEUED
    // (line 276) before prefetchNewBlocks (line 298) in the same FIFO bucket,
    // so a worker picks the frontier chunk first. gzippy's Lever-H pump
    // (try_take_prefetched_pumping) and the prefetch driver can enqueue FAR
    // speculative successors into the priority-0 bucket BEFORE the consumer
    // reaches the on-demand dispatch for the chunk it is blocked on NOW — so
    // the frontier chunk sits behind unrelated speculation in FIFO order and
    // the consumer waits (measured: ~85% of T8 wall blocked on the next
    // in-order chunk's decode while speculative successors were already done).
    //
    // Fix: the ON-DEMAND frontier decode (is_speculative_prefetch == false)
    // gets priority -1 so a worker picks it ahead of any priority-0
    // speculative prefetch. Speculative prefetches keep priority 0 (vendor
    // value). Post-process is bumped to -2 (see submit_post_process_to_pool)
    // so it still strictly outranks the frontier decode — the consumer waits
    // on post-process results directly, so they must never be starved. A
    // worker can always make progress on the highest-priority pending band,
    // so no deadlock: frontier (-1) and prefetch (0) work both eventually
    // complete, and post-process (-2) — which the consumer blocks on — is
    // never blocked by lower-priority decode work.
    let priority = if params.is_speculative_prefetch {
        /* speculative prefetch — vendor priority */
        0
    } else {
        /* on-demand frontier decode — favored ahead of prefetch */
        -1
    };
    let future = thread_pool.submit(
        move || run_decode_task(input, params, &window_map, &block_fetcher, configuration),
        priority,
    );
    future.into_receiver()
}

/// Submit a post-process task to the `ThreadPool`, returning the
/// `mpsc::Receiver` that the consumer's pending-write queue will wait
/// on. Mirror of vendor's `submitTaskWithHighPriority(applyWindow)` at
/// GzipChunkFetcher.hpp:579 (which forwards to
/// `m_threadPool.submit(task, /* priority */ -1)` at
/// BlockFetcher.hpp:606-611).
/// Eager post-process probe — gated port of vendor's
/// `queuePrefetchedChunkPostProcessing` (GzipChunkFetcher.hpp:520-551).
///
/// Called during the consumer's stall (the `while !has_predecessor` wait
/// in `drive_impl`). Walks the prefetch cache contents in offset order
/// and, for each prefetched SUCCESSOR chunk whose predecessor window is
/// already published, submits its `apply_window` post-processing to the
/// pool NOW instead of waiting for the consumer to reach it in order.
///
/// Structural note vs. vendor: rapidgzip mutates the shared
/// `ChunkData` in place (shared_ptr), so the consumer later picks up the
/// already-resolved chunk. gzippy's post-process CONSUMES a `ChunkData`
/// (moves it into the pool task). To stay byte-identical AND not perturb
/// the consumer's fetch/ordering, we eager-post-process a CLONE of the
/// cached chunk and stash the resulting receiver in `eager_submitted`,
/// keyed by the chunk's real `encoded_offset_bits`. The original Arc
/// stays in the prefetch cache so the consumer's `get_if_available`
/// still hits. When the consumer later reaches that offset it REUSES the
/// stashed receiver (see the dispatch site in `drive_impl`) instead of
/// re-submitting — so the apply_window runs exactly once and the bytes
/// are identical regardless of WHEN it ran.
///
/// Returns the number of eager tasks submitted in this run.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn eager_postprocess_prefetched(
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>,
    window_map: &WindowMap,
    thread_pool: &Arc<ThreadPool>,
    eager_submitted: &mut EagerSubmitted,
) -> usize {
    use std::sync::atomic::Ordering;
    let _tv2 = trace_v2::SpanGuard::begin("consumer.eager_postproc");
    EAGER_PROBE_RUNS.fetch_add(1, Ordering::Relaxed);

    // Vendor BlockFetcher.hpp:280 — drain ready in-flight prefetches into
    // the prefetch cache first so `contents()` sees them.
    block_fetcher.process_ready_prefetches();

    // Vendor GzipChunkFetcher.hpp:524-528 — `prefetchCache().contents()`,
    // sorted by offset. Non-evicting snapshot: the consumer still finds
    // these entries on its own fetch.
    let contents = block_fetcher.prefetch_cache_contents_sorted();
    EAGER_PROBE_INSPECTED.fetch_add(contents.len() as u64, Ordering::Relaxed);

    let mut submitted_this_run = 0usize;
    for (_partition_key, arc) in contents {
        let real_offset = arc.encoded_offset_bits;

        // Vendor :533 — skip blocks already enqueued for marker
        // replacement (here: already eager-submitted by a prior probe).
        if eager_submitted.contains_key(&real_offset) {
            continue;
        }

        // Vendor :539 — skip blocks that have no markers to resolve.
        // apply_window is a no-op for a clean chunk; the consumer handles
        // it on the cheap `PendingWrite::Ready` path. (Vendor's
        // `hasBeenPostProcessed()` is the analogous "nothing to do" gate.)
        if arc.data_with_markers.is_empty() {
            continue;
        }

        // Vendor :544-547 — require the predecessor window.
        //
        // CORRECTNESS — which offset to resolve the predecessor at. The
        // consumer accepts a speculative/prefetched chunk only when
        // `next_block_offset == arc.max_acceptable_start_bit` (the handoff
        // guard above), then rewrites the chunk's start to that value
        // (`set_encoded_offset(effective_start)`, effective_start ==
        // decode_start == max_acceptable_start_bit) and resolves
        // `get_predecessor(chunk.encoded_offset_bits)` == `get_predecessor(
        // max_acceptable_start_bit)`. The chunk's `encoded_offset_bits`
        // here is the PARTITION SEED (offset.first), which for a markered
        // bootstrap chunk is strictly BEFORE where its decoded bytes begin
        // — resolving the predecessor there would (a) pick a wrong/earlier
        // window and (b) never match the consumer's key. Resolve at
        // `max_acceptable_start_bit` so the eager apply_window uses the
        // SAME window the in-order consumer will → byte-identical, and the
        // recorded `pred_key` matches the consumer's `consumer_pred_key`
        // so the result is actually REUSED (not rejected by the guard).
        let resolve_offset = arc.max_acceptable_start_bit;

        // STRICT GATE — only eager-process when the predecessor window is
        // CONFIRMED-published at the EXACT key the consumer will use
        // (`contains(resolve_offset)`), not merely "some earlier window
        // exists" (`has_predecessor`). Design B promotes a confirmed
        // chunk's clean tail at its `chunk_end_bit` (== the successor's
        // `max_acceptable_start_bit`) on accept; the serial publish does
        // the same. Requiring an exact key here guarantees we NEVER
        // eager-resolve against an unconfirmed/earlier predecessor — the
        // crux correctness invariant. Range-lookup `get_predecessor` would
        // silently return a stale earlier window when the true predecessor
        // hasn't been published yet, corrupting the chunk.
        if !window_map.contains(resolve_offset) {
            continue;
        }
        let (pred_key, predecessor_window) = match window_map.get_predecessor(resolve_offset) {
            Some((k, w)) => (k, w),
            None => continue,
        };

        // Vendor :549 — submit apply_window. Operate on a CLONE so the
        // cached Arc is untouched and the consumer's fetch still hits.
        // Record `pred_key` so the consumer can verify it would resolve
        // against the SAME predecessor window before reusing this result
        // (byte-identity guard for the speculative/overshoot case where
        // the consumer rewrites the chunk's start offset).
        let chunk_clone: ChunkData = (*arc).clone();
        let rx =
            submit_post_process_to_pool(thread_pool, chunk_clone, predecessor_window, real_offset);
        eager_submitted.insert(real_offset, (pred_key, rx));
        submitted_this_run += 1;
    }

    EAGER_PROBE_SUBMITTED.fetch_add(submitted_this_run as u64, Ordering::Relaxed);
    if submitted_this_run > 0 {
        EAGER_PROBE_RUNS_NONEMPTY.fetch_add(1, Ordering::Relaxed);
        EAGER_PROBE_MAX_PER_RUN.fetch_max(submitted_this_run as u64, Ordering::Relaxed);
    }
    submitted_this_run
}

#[cfg(parallel_sm)]
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
    // FRONTIER SCHEDULING (gzippy deviation from vendor's flat -1):
    // post-process is bumped to -2 (highest) so it strictly outranks the
    // on-demand frontier decode (-1 below). The consumer's pending-write
    // queue waits DIRECTLY on these post-process results, so they must
    // never be starved by a frontier decode that the consumer is only
    // about to need. Vendor keeps both decode (0) and post-process (-1)
    // and relies on the on-demand task being ENQUEUED before prefetches in
    // the same priority bucket; gzippy's pump can enqueue far prefetches
    // ahead of the frontier on-demand task in the 0 bucket, so we promote
    // the frontier decode to its own band instead (see submit_decode_to_pool).
    let future = thread_pool.submit(
        move || run_post_process_task(chunk, predecessor_window),
        /* priority */ -2,
    );
    future.into_receiver()
}

/// Pool-side execution of a decode task (vendor `decodeBlock`,
/// GzipChunkFetcher.hpp:692-729). Routes to `decode_chunk_isal`
/// when the predecessor window is published, else
/// `speculative_decode_find_boundary` (marker bootstrap when no window).
#[cfg(parallel_sm)]
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

    // DECODE-BYPASS replay (campaign instrument, GZIPPY_BYPASS_DECODE).
    // Return the precomputed ChunkData for this (start_bit, stop_hint)
    // via memcpy — ~zero inner-Huffman CPU — keeping the FULL downstream
    // coordination chain. On a cache miss fall through to real decode so
    // output stays byte-correct.
    // FIXED-SLEEP coordination-isolation (GZIPPY_SLEEP_DECODE_NS). Sleep a
    // fixed duration instead of decoding, return a correct-size clean
    // zero-filled chunk. Equalizes per-chunk "decode" cost to a constant
    // identical to the rapidgzip sleep patch so wall delta = pure
    // coordination. Output is garbage; CRC/size verification is gated off in
    // sm_driver. Uses the bypass capture file for size/boundary metadata.
    if crate::decompress::parallel::decode_bypass::sleep_decode_enabled() {
        if let Some(chunk) = crate::decompress::parallel::decode_bypass::sleep_replay(
            params.start_bit,
            params.stop_hint_bit,
            configuration,
        ) {
            if chunk.max_acceptable_start_bit == chunk.encoded_offset_bits
                && chunk.encoded_size_bits > 0
            {
                if let Some(tail) = chunk.last_32kib_window_vec() {
                    let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                    window_map.insert_owned_none(chunk_end_bit, tail);
                }
            } else if chunk.encoded_size_bits > 0 {
                if let Some(tail) = chunk.last_32kib_window_vec() {
                    let key = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                    window_map.insert_speculative_owned_none(key, tail);
                }
            }
            return Ok(Arc::new(chunk));
        }
    }

    if crate::decompress::parallel::decode_bypass::replay_enabled() {
        if let Some(chunk) = crate::decompress::parallel::decode_bypass::replay(
            params.start_bit,
            params.stop_hint_bit,
            configuration,
        ) {
            // Preserve the early-window-publish behavior the real decode
            // path performs so successor resolution is unchanged.
            if chunk.max_acceptable_start_bit == chunk.encoded_offset_bits
                && chunk.encoded_size_bits > 0
            {
                if let Some(tail) = chunk.last_32kib_window_vec() {
                    let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                    window_map.insert_owned_none(chunk_end_bit, tail);
                }
            } else if chunk.encoded_size_bits > 0 {
                if let Some(tail) = chunk.last_32kib_window_vec() {
                    let key = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                    window_map.insert_speculative_owned_none(key, tail);
                }
            }
            return Ok(Arc::new(chunk));
        }
    }

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

    // INDEPENDENT decode-mode signal for Fulcrum's `model` view. The mode is
    // the ACTUAL window-present-at-decode-start predicate (clean iff
    // start_bit==0 or the predecessor window is published), NOT the dispatch
    // `is_speculative_prefetch` intent (a prefetch can race the publish and
    // run clean). Emitted as an instant keyed by start_bit so the model joins
    // it to this chunk's `worker.decode_chunk` span and splits d_c / d_w
    // honestly.
    let decode_mode_clean = params.start_bit == 0 || window.is_some();
    let mode_str = if decode_mode_clean {
        "clean"
    } else {
        "window_absent"
    };
    trace_v2::emit_instant(
        "worker.decode_mode",
        &format!(r#""start_bit":{},"mode":"{mode_str}""#, params.start_bit),
        "t",
    );
    trace_v2::emit_instant(
        "causal.decode_decision",
        &format!(
            r#""start_bit":{},"window_present":{},"mode":"{mode_str}","stop_hint":{},"speculative":{}"#,
            params.start_bit,
            decode_mode_clean,
            params.stop_hint_bit,
            params.is_speculative_prefetch
        ),
        "t",
    );

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

    // EARLY WINDOW PUBLISH (break the serial window-resolution chain).
    //
    // The in-order consumer is the wall: a successor chunk that decoded
    // with markers can only be RESOLVED (apply_window) once its
    // predecessor's tail-window is in the WindowMap, and historically
    // that window appeared only as the consumer serially advanced to and
    // resolved the predecessor (a strict N-1 → N chain). A probe (this
    // branch's parent commit) confirmed ready prefetched successors sit
    // idle because `get_predecessor` misses.
    //
    // The lever: a chunk's last 32 KiB is the exact window the NEXT chunk
    // needs, and the worker has it RIGHT AFTER decode — long before the
    // consumer reaches this chunk. Publish it now so successors'
    // predecessor windows become available off the consumer's serial path
    // (both the consumer's `has_predecessor` wait and
    // `eager_postprocess_prefetched`'s `get_predecessor` then succeed).
    //
    // CORRECTNESS — three invariants make this byte-identical to the
    // consumer's eventual publish:
    //
    //  1. KEY STABILITY. We key at `chunk_end_bit = encoded_offset_bits +
    //     encoded_size_bits`. The consumer publishes at the same
    //     expression AFTER `set_encoded_offset`, which holds that sum
    //     invariant (ChunkData::set_encoded_offset: new_size = end - off,
    //     new_off = off ⇒ off + new_size == end). So the worker's
    //     `chunk_end_bit` equals the consumer's for any chunk the
    //     consumer accepts.
    //
    //  2. ACCEPT DETERMINISM. We publish ONLY when
    //     `max_acceptable_start_bit == encoded_offset_bits` — i.e. the
    //     chunk decoded at an EXACT confirmed offset, not a speculative
    //     RANGE. These are (a) fast-path `decode_chunk_isal` chunks
    //     (window known) and (b) marker-bootstrap chunks whose partition
    //     seed already WAS the real boundary (`try_speculative_decode_
    //     candidate` widens `max` past `encoded` only when
    //     `partition_seed < decode_start`). Such chunks are always
    //     accepted by the consumer's `matches_encoded_offset` +
    //     `max == next_block_offset` guard with the SAME end, so the
    //     window we publish is never stale/rejected. Range-speculative
    //     chunks (`max > encoded`) may be rejected and re-decoded at a
    //     different start/end — we never publish for those; they take the
    //     existing serial consumer publish unchanged.
    //
    //  3. PROVABLY-CLEAN TAIL. We publish only `last_32kib_window_vec()`,
    //     which returns `Some` IFF the trailing 32 KiB contains no
    //     markers (resolved purely from clean bytes, no predecessor
    //     needed). On a clean tail it is byte-identical to the consumer's
    //     `get_last_window_vec(pred)` (which only consults the
    //     predecessor for marker bytes in the tail — of which there are
    //     none). When the tail straddles markers it returns `None` and we
    //     publish nothing → the chunk falls back to the consumer's serial
    //     `get_last_window_vec(pred)` publish, unchanged.
    //
    // `insert` uses overwrite semantics (WindowMap.hpp:65-76
    // insert_or_assign), so the consumer's later publish at the same key
    // is a harmless idempotent overwrite with identical bytes.
    // DECODE-BYPASS capture (campaign instrument, GZIPPY_BYPASS_CAPTURE).
    // Record this decode result keyed by (start_bit, stop_hint_bit) so a
    // later replay run can memcpy it back. No-op unless capture is on.
    if let Ok(ref chunk) = chunk_result {
        crate::decompress::parallel::decode_bypass::record(
            params.start_bit,
            params.stop_hint_bit,
            chunk,
        );
    }

    if let Ok(ref chunk) = chunk_result {
        if chunk.max_acceptable_start_bit == chunk.encoded_offset_bits
            && chunk.encoded_size_bits > 0
        {
            if let Some(tail) = chunk.last_32kib_window_vec() {
                let chunk_end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                window_map.insert_owned_none(chunk_end_bit, tail);
                trace_v2::emit_instant(
                    "causal.window_publish",
                    &format!(
                        r#""start_bit":{},"end_bit":{chunk_end_bit},"site":"worker_early""#,
                        chunk.encoded_offset_bits
                    ),
                    "t",
                );
                EARLY_WINDOW_PUBLISHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                EARLY_WINDOW_TAIL_NOT_CLEAN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else if chunk.encoded_size_bits > 0 {
            // RANGE-SPECULATIVE chunk (`max > encoded`): its end is not yet
            // authoritative (the consumer may accept or reject the
            // speculative start). Design B: publish the provably-clean tail
            // into the SPECULATIVE side-slot so it is INVISIBLE to successor
            // resolution until the consumer confirms the start. The consumer
            // promotes it (accept) or evicts it (reject).
            //
            // KEY PROOF: on accept the consumer sets `effective_start =
            // decode_start = max_acceptable_start_bit` and calls
            // `set_encoded_offset(effective_start)`, which preserves
            // `encoded_offset_bits + encoded_size_bits`. So the consumer's
            // `chunk_end_bit` (computed post-rewrite) equals THIS expression
            // (`chunk.encoded_offset_bits + chunk.encoded_size_bits`,
            // pre-rewrite) exactly — both equal the chunk's `encoded_end`.
            // Hence the speculative key matches the consumer's accept-time
            // key, making `promote_speculative` an identity-key move.
            // (NOTE: `decode_origin_bit + encoded_size_bits` would be WRONG
            // — after the partition-seed rewrite `encoded_size_bits` is
            // measured from `partition_seed`, not from `decode_origin_bit`.)
            if let Some(tail) = chunk.last_32kib_window_vec() {
                let key = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                window_map.insert_speculative_owned_none(key, tail);
                EARLY_SPEC_PUBLISHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            EARLY_WINDOW_RANGE_SPECULATIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

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
#[cfg(parallel_sm)]
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
    // For chunks at or above the LUT threshold, fuse `replace_markers` +
    // `narrow_u16_to_u8` into one pass that reads u16 and writes the
    // resolved u8 directly into `chunk.narrowed`. Vendor parity:
    // `DecodedData.hpp:316-337` writes `target[i] = fullWindow[chunk[i]]`
    // in a single LUT pass.
    //
    // Threshold lowered 128 KiB → 16 KiB (2026-05-31, consumer-path
    // copy-elimination): the two-pass path costs ~3 buffer passes
    // (AVX2 u16 in-place replace = read+write 2·N bytes, then narrow =
    // read 2·N + write N), while the fused path costs ~96 KiB of LUT
    // setup (64 KiB zero-init + 32 KiB window copy) plus ONE write of N
    // bytes. The LUT is window-dependent (every chunk has a different
    // predecessor window), so it cannot be reused across chunks — the
    // ~96 KiB setup is paid per fused call regardless. Break-even is
    // around N ≈ 16 KiB of markers; below that the LUT setup dominates,
    // so 16 KiB is the conservative floor (not 0). Above it the fused
    // single-pass wins. The fused path already produces byte-identical
    // output (it is the production large-marker path), so only the
    // threshold moves.
    // FOOTPRINT-ALIGN: resolve markers IN PLACE inside `data_with_markers`,
    // then narrow IN PLACE (u16→u8 within the same backing store). No
    // separate `narrowed` buffer is allocated — the resolved u8 output is the
    // first `narrowed_len` bytes of `data_with_markers`. Faithful port of
    // vendor's `applyWindow` (DecodedData.hpp:325-388). This replaces the
    // former fused-LUT-into-separate-buffer path; the LUT fusion saved a
    // pass but at the cost of a whole extra resident u8 buffer per chunk
    // (memlife: 1.88 GB alloc / a buffer rapidgzip does not have), which is
    // exactly the footprint divergence this change targets.
    let dwm_len_pre = chunk.data_with_markers.len();
    let t_apply = std::time::Instant::now();
    if dwm_len_pre >= 16 * 1024 {
        POST_PROCESS_FUSED_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        POST_PROCESS_SMALL_MARKERS_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    apply_window(&mut chunk, &bytes);
    let t_narrow = std::time::Instant::now();
    chunk.narrow_markers_in_place();
    let narrow_us = t_narrow.elapsed().as_micros();
    let apply_us = t_apply.elapsed().as_micros();
    let t_pop = std::time::Instant::now();
    // `narrowed_bytes()` aliases `data_with_markers`'s store; copy out the
    // borrow boundary by computing windows against it directly. The borrow is
    // immutable and `populate_subchunk_windows` only reads `&self.data`/args.
    chunk.populate_subchunk_windows(&bytes);
    let populate_us = t_pop.elapsed().as_micros();
    // CRC the narrowed bytes on the worker (vendor parity: applyWindow CRCs
    // resolved marker bytes on the worker, not the serial consumer).
    {
        let _tv2 = trace_v2::SpanGuard::begin("post_process.crc_narrowed");
        chunk.update_narrowed_crc();
    }
    let resolve_us = apply_us as f64 + narrow_us as f64;
    let fused = dwm_len_pre >= 16 * 1024;
    trace_v2::emit_instant(
        "causal.tax",
        &format!(
            r#""start_bit":{start_bit},"marker_bytes":{marker_bytes},"resolve_us":{resolve_us},"narrow_us":{narrow_us},"materialize_us":{materialize_us},"populate_us":{populate_us},"fused":{fused}"#,
        ),
        "t",
    );
    if trace::is_enabled() {
        trace::emit(
            "post_process",
            "post_process_span",
            &format!(
                r#""start_bit":{start_bit},"materialize_us":{materialize_us},"apply_window_us":{apply_us},"populate_subchunk_windows_us":{populate_us},"narrow_us":{narrow_us},"marker_bytes":{marker_bytes}"#,
            ),
        );
    }
    // The narrowed bytes live INSIDE data_with_markers until the consumer
    // writes them, so we no longer eagerly recycle that buffer here (it would
    // free the bytes the consumer still needs). Drop returns it post-consume.
    chunk.recycle_markers_after_resolution();
    chunk
}

/// Narrow `src: &[u16]` into `dst: &mut U8`, appending bytes. All values
/// in `src` MUST be < 256 (post-`apply_window` invariant). AVX2 path
/// uses `_mm256_packus_epi16` for 16-lane parallel narrowing
/// (saturating pack — since values are already < 256, saturation is
/// a no-op). Scalar tail handles the remainder. AVX-256 downclock
/// concern from an earlier session was empirically refuted on neurotic
/// via injection probe (see `plans/rust-rapidgzip.md` §4).
#[cfg(parallel_sm)]
fn narrow_u16_to_u8(src: &[u16], dst: &mut crate::decompress::parallel::rpmalloc_alloc::types::U8) {
    dst.clear();
    dst.reserve(src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 just detected at runtime.
            unsafe {
                narrow_u16_to_u8_avx2(src, dst);
            }
            return;
        }
    }
    // Scalar fallback (universal). On aarch64 with `-C target-cpu=native`
    // LLVM auto-vectorizes this tight `v as u8` store loop to NEON; there is
    // no truncation/saturation subtlety because the post-`apply_window`
    // invariant guarantees every value is < 256.
    for &v in src {
        dst.push(v as u8);
    }
}

#[cfg(all(parallel_sm, target_arch = "x86_64"))]
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

#[cfg(not(parallel_sm))]
fn narrow_u16_to_u8(src: &[u16], dst: &mut crate::decompress::parallel::rpmalloc_alloc::types::U8) {
    dst.clear();
    dst.reserve(src.len());
    for &v in src {
        dst.push(v as u8);
    }
}

/// Consumer-thread publication of per-subchunk tail windows. Mirror of
/// vendor `appendSubchunksToIndexes` (GzipChunkFetcher.hpp:429-458).
#[cfg(parallel_sm)]
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

/// Early-window-publish telemetry (run_decode_task). PUBLISHED: worker
/// published a provably-clean 32 KiB tail at chunk_end_bit, off the
/// consumer's serial path. TAIL_NOT_CLEAN: chunk had an exact confirmed
/// start but its trailing 32 KiB straddled markers (falls back to the
/// consumer's serial `get_last_window_vec(pred)` publish).
/// RANGE_SPECULATIVE: chunk decoded over a speculative start RANGE
/// (`max > encoded`) so its end is not yet authoritative — never early-
/// published (the consumer publishes it after accept + set_encoded_offset).
pub static EARLY_WINDOW_PUBLISHED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static EARLY_WINDOW_TAIL_NOT_CLEAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static EARLY_WINDOW_RANGE_SPECULATIVE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// (Design B) Speculative side-slot telemetry.
/// PUBLISHED: worker published a range-speculative chunk's provably-clean
/// tail into the speculative side-slot (subset of RANGE_SPECULATIVE that
/// had a clean tail).
/// PROMOTED: consumer accepted the speculative chunk and moved its window
/// into the real map (identity-key move).
/// EVICTED: consumer rejected the speculative chunk and dropped its
/// stale window. PUBLISHED should ≈ PROMOTED + EVICTED (+ a small residue
/// for entries whose chunk the consumer never reached, e.g. EOF).
pub static EARLY_SPEC_PUBLISHED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static EARLY_SPEC_PROMOTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static EARLY_SPEC_EVICTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static COORDINATOR_BOUNDARY_SEARCH_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumps once per consumer iter where the partition-keyed
/// `try_take_prefetched` returned a chunk but the safety guard
/// rejected it (mismatch between `chunk.max_acceptable_start_bit` and
/// consumer-requested `next_block_offset`) — distinct from
/// `prefetch_cache_miss`, which counts a prefetch being absent.
pub static PREFETCH_REJECT_BY_GUARD: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Count of post-process tasks that took the fused
/// `replace_markers_lut_narrow` path (markers ≥ 128 KiB, 32 KiB window).
pub static POST_PROCESS_FUSED_PATH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Count of post-process tasks that took the small-markers path
/// (`apply_window` + separate `narrow_u16_to_u8`). This is the path
/// that benefits from the AVX2 narrow re-enable. If this is ~0 on
/// silesia bench-sm, the AVX2-narrow change is benchmark-invisible
/// for that fixture (but may still help BGZF / small-marker workloads).
pub static POST_PROCESS_SMALL_MARKERS_PATH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Lever G diagnostic: counts how often the consumer's
/// `Arc::try_unwrap(chunk_arc)` actually succeeded (HITS) vs fell back
/// to a deep clone (MISSES). Goal: HITS == chunk count, MISSES == 0.
pub static ARC_TRY_UNWRAP_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ARC_TRY_UNWRAP_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Eager post-process bookkeeping: maps a prefetched chunk's real
/// `encoded_offset_bits` (partition-seed key, stable across the consumer's
/// `set_encoded_offset` rewrite) → (predecessor-window key it resolved
/// against, in-flight apply_window receiver). The consumer reuses the
/// receiver when it reaches that offset, after validating the predecessor
/// key still matches (byte-identity guard).
#[cfg(parallel_sm)]
type EagerSubmitted = std::collections::HashMap<usize, (usize, mpsc::Receiver<ChunkData>)>;

// ── Eager post-process probe (GZIPPY_EAGER_POSTPROC=1) ────────────────────
// Port of rapidgzip's `queuePrefetchedChunkPostProcessing`
// (GzipChunkFetcher.hpp:520-551): during a consumer stall, submit
// apply_window for prefetched SUCCESSOR chunks whose predecessor window
// is already published, instead of waiting to reach them in order.
//
// These counters are the deliverable: a prior pump attempt TIED because
// it found 0 ready successors. We MUST distinguish "the lever works"
// (many eager submits) from "the cache is empty during the stall" (the
// real lever is prefetch DEPTH, not eagerness).
/// How many times the eager probe ran (once per stall iteration where it
/// was invoked).
pub static EAGER_PROBE_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Total prefetched-cache chunks the probe inspected across all runs.
pub static EAGER_PROBE_INSPECTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Total eager post-process tasks the probe actually submitted (ready
/// successors: had markers + a published predecessor window + not yet
/// submitted). This is THE number that tells us whether the lever fires.
pub static EAGER_PROBE_SUBMITTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Probe runs that submitted at least one eager task (so we can report
/// "N of M stalls found ready successors").
pub static EAGER_PROBE_RUNS_NONEMPTY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Max ready successors found in a single probe run (per-stall peak).
pub static EAGER_PROBE_MAX_PER_RUN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Eager-submitted results the consumer actually reused (vs. submitted
/// but never consumed — e.g. evicted before the consumer reached them).
pub static EAGER_PROBE_REUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Consumer reached a markered chunk but found NO eager entry under its
/// pre-set offset key (the eager probe keyed it differently, or it was
/// never eagerly submitted).
pub static EAGER_PROBE_REUSE_KEY_ABSENT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Consumer found an eager entry but its predecessor-window key did not
/// match the consumer's (would have changed bytes — correctly rejected).
pub static EAGER_PROBE_REUSE_PRED_MISMATCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// True iff `GZIPPY_EAGER_POSTPROC=1`. Read once and cached.
#[cfg(parallel_sm)]
fn eager_postproc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0); // 0=unknown, 1=off, 2=on
    match CACHED.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("GZIPPY_EAGER_POSTPROC")
                .map(|v| v == "1")
                .unwrap_or(false);
            CACHED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Speculative slow-path decode without a predecessor window. Must use
/// marker bootstrap — plain ISA-L with an empty dict resolves unknown
/// back-refs against zeros and corrupts output (Bug B, commit 4909ac7).
#[cfg(parallel_sm)]
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
    // (Design B) Record the encoded bit the worker decoded byte 0 from.
    // `decode_chunk_marker_bootstrap_then_isal` anchored the chunk at
    // `decode_start`, so that is the true origin regardless of the
    // partition-seed rewrite below.
    chunk.decode_origin_bit = decode_start;
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

#[cfg(parallel_sm)]
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
    // worker.seed_first span: wraps the seed-exact attempt; matched on
    // the vendor side by the trace patch around
    // `tryToDecode({ blockOffset, blockOffset })` at GzipChunk.hpp:739.
    {
        let _tv2 = trace_v2::SpanGuard::begin("worker.seed_first");
        if let Ok(chunk) = try_speculative_decode_candidate(
            input,
            start_bit,
            start_bit,
            stop_hint_bit,
            configuration,
        ) {
            SLOW_PATH_FIRST_CANDIDATE_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(chunk);
        }
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
    // worker.scan_run span: the bit-by-bit scan + candidate-try loop.
    // Matched on vendor side by the trace patch around the alternating
    // findNextDynamic/findNextUncompressed loop at GzipChunk.hpp:803+.
    let _tv2_scan = trace_v2::SpanGuard::begin("worker.scan_run");
    if let Some(chunk) = RawBlockFinderCoordinator::with_sync_boundary_search(
        input,
        start_bit,
        max_end,
        |cand_bit| {
            // worker.scan_candidate span: one full-decode attempt at a
            // single bit position the scan flagged as a valid block
            // candidate. Matched on vendor side by the trace patch
            // around each tryToDecode call inside the alternating loop.
            let _tv2_cand = trace_v2::SpanGuard::begin("worker.scan_candidate");
            match try_speculative_decode_candidate(
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
                    SLOW_PATH_FIRST_CANDIDATE_FAIL
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        },
    ) {
        return Ok(chunk);
    }
    drop(_tv2_scan);

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

#[cfg(parallel_sm)]
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
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn drain_one_pending<W: std::io::Write>(
    pending: &mut std::collections::VecDeque<PendingWrite>,
    window_map: &WindowMap,
    writer: &mut W,
    out_fd: Option<i32>,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
    block_fetcher: &BlockFetcher<usize, Arc<ChunkData>, FetchMultiStream, ChunkDecodeError>,
    // Eager post-process context (GZIPPY_EAGER_POSTPROC=1). When the head
    // pending write is an in-flight post-process (`Async`), the consumer is
    // about to BLOCK on `rx.recv()` (the `wait.future_recv` span — the
    // consumer's real serial wait once Design B has promoted predecessor
    // windows early). Vendor GzipChunkFetcher.hpp:513 queues successor
    // post-processing DURING exactly this wait. Mirror that: submit
    // apply_window for ready prefetched successors whose CONFIRMED
    // predecessor window is published, so they run on the pool while the
    // consumer blocks on the head. `None` (eager disabled) skips it.
    eager_ctx: Option<(&Arc<ThreadPool>, &mut EagerSubmitted)>,
) -> Result<(), FetchError> {
    let _tv2 = trace_v2::SpanGuard::begin("consumer.drain");
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    // Vendor GzipChunkFetcher.hpp:513 — queue successor post-processing
    // BEFORE blocking on the head chunk's future, but only when the head is
    // actually an in-flight `Async` (else there is no wait to overlap).
    if let (PendingWrite::Async { .. }, Some((thread_pool, eager_submitted))) = (&head, eager_ctx) {
        eager_postprocess_prefetched(block_fetcher, window_map, thread_pool, eager_submitted);
    }
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
    let decoded_data_len = chunk.data.len().saturating_sub(chunk.data_prefix_len);
    let payload_bytes = chunk.narrowed_len + decoded_data_len;
    let mut wrote_via_fd = false;
    #[cfg(unix)]
    if let Some(fd) = out_fd {
        if std::env::var_os("GZIPPY_DISABLE_WRITEV").is_none() && payload_bytes > 0 {
            use crate::decompress::parallel::fd_vectored_write;
            let _tv2 = trace_v2::SpanGuard::begin("consumer.writev");
            let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
            let mut left = chunk.narrowed_len;
            for seg in chunk.data_with_markers.segments() {
                if left == 0 {
                    break;
                }
                let n = left.min(seg.len());
                // SAFETY: in-place narrow wrote `n` u8 at the front of this segment.
                let sl = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, n) };
                parts.push(sl);
                left -= n;
            }
            let mut skip = chunk.data_prefix_len;
            for seg in chunk.data.segments() {
                if skip >= seg.len() {
                    skip -= seg.len();
                    continue;
                }
                parts.push(&seg[skip..]);
                skip = 0;
            }
            // CRC combine before boxing `chunk` for vmsplice lifetime.
            total_crc.append(&chunk.narrowed_crc);
            for stream_crc in &chunk.crc32s {
                total_crc.append(stream_crc);
            }
            let mut iovs = fd_vectored_write::to_io_vec(parts.iter().copied(), &[]);
            let owner: Box<dyn std::any::Any + Send> = Box::new(chunk);
            fd_vectored_write::write_all_to_fd(fd, &mut iovs, owner)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += payload_bytes;
            wrote_via_fd = true;
            let crc_write_us = t_crc_write.elapsed().as_micros();
            let combine_us = 0usize;
            let total_us = t_chunk.elapsed().as_micros();
            if trace::is_enabled() {
                trace::emit(
                    "consumer",
                    "consume_done",
                    &format!(
                        r#""partition_idx":{idx},"decoded":{payload_bytes},"recv_us":{recv_us},"publish_us":{publish_us},"crc_write_us":{crc_write_us},"combine_us":{combine_us},"total_us":{total_us}"#,
                    ),
                );
            }
            let _ = cache_key;
            let _ = block_fetcher;
            crate::coz_probe::progress("chunk_emitted");
            return Ok(());
        }
    }
    #[cfg(not(unix))]
    let _ = out_fd;
    if !wrote_via_fd {
        if chunk.narrowed_len > 0 {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.write_narrowed");
            let mut left = chunk.narrowed_len;
            for seg in chunk.data_with_markers.segments() {
                if left == 0 {
                    break;
                }
                let n = left.min(seg.len());
                let sl = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, n) };
                writer
                    .write_all(sl)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                left -= n;
            }
            *total_size += chunk.narrowed_len;
        }
        if decoded_data_len > 0 {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.write_data");
            chunk
                .data
                .write_payload_skipping_prefix(chunk.data_prefix_len, writer)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += decoded_data_len;
        }
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
    // Coz throughput marker: one in-order chunk just left the consumer. Coz
    // measures a virtual speedup's effect as the change in THIS visit-rate —
    // the program's true end-to-end throughput. No-op without --features coz.
    crate::coz_probe::progress("chunk_emitted");
    Ok(())
}

// Non-x86_64 / non-isal stub.
#[cfg(not(parallel_sm))]
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
#[cfg(parallel_sm)]
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
        let (_crc, size) = drive(&deflate, &mut out, None, 4, cfg).expect("drive");
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
        let (_crc, size) = drive(&deflate, &mut out, None, 8, cfg).expect("drive");
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
        // Feed stdin on a worker thread while draining stdout here, else
        // gzip's stdout pipe fills mid-write and deadlocks (see
        // gzip_chunk::cross_chunk_resume for the same fix).
        let mut stdin = child.stdin.take().expect("stdin");
        let head_owned = head.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = std::io::Write::write_all(&mut stdin, &head_owned);
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

        let chunk_size = 4 * 1024 * 1024;
        let mut out = Vec::new();
        let result = read_parallel_sm(&gz, &mut out, None, 2, chunk_size);
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
        let (_crc, size) = drive(&deflate, &mut out, None, 8, cfg).expect("drive");
        assert_eq!(size, payload.len(), "size mismatch (suggests early break)");
        assert_eq!(out, payload, "byte mismatch");
    }
}
