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
//! - Worker decode: `decode_chunk` (window known or chunk 0) or
//!   `speculative_decode_find_boundary` → marker bootstrap (prefetch, no window).
//! - `postProcessChunk` / `applyWindow` → `apply_window` on the pool
//!   (priority −1). **WindowMap publishes stay on the consumer** (vendor
//!   orchestrator thread): tail before post-process, subchunk windows
//!   after `apply_window` completes (`appendSubchunksToIndexes`).

use crate::decompress::parallel::chunk_data::ChunkConfiguration;
#[cfg(parallel_sm)]
use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::chunk_decode::ChunkDecodeError;
#[cfg(parallel_sm)]
use crate::decompress::parallel::chunk_handle::{ChunkArc, SharedChunkData};
#[cfg(parallel_sm)]
use std::sync::Arc;

#[cfg(parallel_sm)]
use crate::decompress::parallel::async_block_finder::RawBlockFinderCoordinator;
#[cfg(parallel_sm)]
#[cfg(parallel_sm)]
use crate::decompress::parallel::block_fetcher::BlockFetcher;
#[cfg(parallel_sm)]
use crate::decompress::parallel::block_map::{append_subchunks_to_block_map, BlockMap};
#[cfg(parallel_sm)]
use crate::decompress::parallel::chunk_decode::decode_chunk_window_absent;
#[cfg(parallel_sm)]
use crate::decompress::parallel::compressed_vector::CompressionType;
#[cfg(parallel_sm)]
use crate::decompress::parallel::crc32::CRC32Calculator;
#[cfg(parallel_sm)]
use crate::decompress::parallel::gzip_block_finder::{GetReturnCode, GzipBlockFinder};
#[cfg(all(unix, parallel_sm))]
use crate::decompress::parallel::output_writer;
#[cfg(parallel_sm)]
use crate::decompress::parallel::prefetcher::FetchMultiStream;
#[cfg(parallel_sm)]
use crate::decompress::parallel::stall_residency;
#[cfg(parallel_sm)]
use crate::decompress::parallel::thread_pool::ThreadPool;
use crate::decompress::parallel::trace;
use crate::decompress::parallel::trace_v2;
use crate::decompress::parallel::window_map::{Window, WindowMap};

#[cfg(parallel_sm)]
use std::borrow::Cow;
#[cfg(parallel_sm)]
use std::sync::mpsc;

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

/// Counter incremented each time the consumer skips a CONFIRMED block
/// (an entry whose start bit falls behind `furthest_decoded_bit`).
/// COMPONENT RECORD-CORRECTION: `GzipBlockFinder` (gzip_block_finder.rs) is
/// spacing/bookkeeping only and has NO scanner; confirmed entries enter it
/// solely via `consumer_append_subchunks_vendor` insertions (and the
/// test-only oracle pre-seed). The deflate-candidate scanner is the raw
/// block finder (blockfinder_validation.rs), which can hand stored-payload false
/// positives to trial decodes; a stale confirmed entry behind the frontier
/// arises via subchunk insertion around such regions, not via any
/// GzipBlockFinder scan (98fd618c's message attributed this to the wrong
/// component).
///
/// Vendor divergence: rapidgzip's GzipChunkFetcher.hpp:411-427 throws a
/// `std::logic_error` when `block_finder->get(next_unprocessed_block_index)`
/// does not equal `blockOffsetAfterNext` after advancing the index.
/// In gzippy the skip guard (removing `!block_is_confirmed &&`) handles this
/// case silently and correctly — the confirmed false-positive is discarded
/// rather than decoded.  This counter provides the same signal for
/// diagnostics without aborting: a non-zero value means GzipBlockFinder
/// produced at least one false-positive confirmed entry inside a stored
/// block's raw payload.
pub static STALE_CONFIRMED_BLOCK_SKIP: std::sync::atomic::AtomicU64 =
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
    /// NOT a hard byte cap on the ISA-L reader — see `decode_chunk`.
    stop_hint_bit: usize,
    /// True when `stop_hint_bit` is already known to be the exact chain
    /// end and the window-present decode may take the vendor
    /// `untilOffsetIsExact` path.
    stop_hint_is_exact: bool,
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
    drive_impl(
        input,
        writer,
        out_fd,
        parallelization,
        configuration,
        None,
        None,
    )
}

/// Like [`drive`] but also reports how many output bytes were written to the
/// sink — even when the decode fails. The multi-member driver
/// ([`crate::decompress::parallel::sm_driver::read_parallel_sm_multi`]) uses
/// the byte count to RESUME the remaining members past the prefix already
/// streamed, so a misrouted multi-member stream (second member past the 16 MiB
/// detection window) decodes in full instead of erroring/truncating.
#[cfg(parallel_sm)]
pub fn drive_capturing<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    configuration: ChunkConfiguration,
    bytes_written_out: &mut usize,
) -> Result<(u32, usize), FetchError> {
    drive_impl(
        input,
        writer,
        out_fd,
        parallelization,
        configuration,
        None,
        Some(bytes_written_out),
    )
}

/// Clean-window oracle (advisor 2026-05-29): the cheapest experiment that
/// discriminates "is the marker/copy/bootstrap pipeline the rapidgzip gap"
/// from "the gap is inner-loop / scan". Decode every chunk with its TRUE
/// predecessor window so NO chunk takes the speculative bootstrap path —
/// no markers, hence no `append_markered` /
/// `narrow_u16_to_u8` copies. Pass 1 runs the normal speculative decode to
/// a sink, populating a shared `WindowMap` with every chunk-boundary
/// window; pass 2 re-runs with that map pre-seeded (every
/// `window_map.get(start_bit)` HITS → `decode_chunk`) and is the
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
        None,
    )?;

    let _ = &seed; // (pass-1 window keys are NOT trusted for boundaries; see below)
    let total_bits = input.len() * 8;
    let pool_size = parallelization.max(1);

    // --- Phase A (UNTIMED): self-derive REAL block boundaries + correct dicts. ---
    // Do NOT trust the speculative pass's published window keys for span starts:
    // some are not confirmed block boundaries, so decode_chunk starting
    // there misreads random bits as a block header ("Stored block len=0"). Chain
    // from the decoder's OWN returned end bit instead — decode_chunk stops
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
            let c = crate::decompress::parallel::chunk_decode::decode_chunk(
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
    // one clean `decode_chunk` per span across `pool_size` workers and
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
                    let r = crate::decompress::parallel::chunk_decode::decode_chunk(
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
    }
    let secs = t.elapsed().as_secs_f64();
    let mb = total_size as f64 / secs / 1e6;
    eprintln!(
        "CLEAN_WINDOW_ORACLE pass2 wall={secs:.4}s out={total_size} bytes {mb:.0} MB/s ({parallelization} workers, {n_spans} spans)"
    );
    Ok((total_crc.crc32(), total_size))
}

/// THIN-T1-DRIVER removal-oracle (`GZIPPY_THIN_T1_ORACLE=1`, default OFF).
///
/// The MINIMAL clean removal-oracle the BEAT-IGZIP TASK 2 brief calls for: a
/// single SERIAL rolling-window loop over chunks calling the SHARED
/// `decode_chunk` kernel with NO parallel bulk whatsoever — no `block_finder`,
/// no `BlockFetcher`/prefetch, no `WindowMap` publish/handoff, no marker arming
/// (never fires — every chunk gets a full 32 KiB predecessor window), no
/// `ThreadPool`/`std::thread::scope`, no `Arc` chunk lifecycle, no pending
/// queue. It is the candidate THIN-T1-DRIVER spine measured in isolation: the
/// ENTIRE process (perf-stat over the whole binary) is this clean serial decode
/// plus the same CLI/mmap/header prologue both arms share — so cyc/B over the
/// process attributes to the shared kernel + serial driver, not the scaffold.
///
/// Unlike `drive_clean_window_oracle` this does NOT run a speculative Pass 1 and
/// has NO separate untimed Phase A: the rolling window IS derived inline as the
/// loop advances (each chunk's true trailing 32 KiB becomes the next chunk's
/// dict), so there is exactly ONE decode of the stream and nothing the perf
/// counter sees that is not the thin-T1 path. Output is byte-correct (real
/// rolling windows) and markers never fire — both verified by the caller's
/// CRC/size check (Gate-0 non-inert + markers==0).
#[cfg(parallel_sm)]
pub fn drive_thin_t1_oracle<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    configuration: ChunkConfiguration,
    mut bytes_written_out: Option<&mut usize>,
) -> Result<(u32, usize), FetchError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    // Cache-residency lever (b): activate the T1 resident-pool scope for THIS
    // (single, serial) decode thread. While active, the chunk output buffer is
    // taken from / returned to the manual per-thread pool AND its reserve is
    // pinned to the fixed resident cap, so consecutive chunks decode into ONE
    // resident buffer instead of a fresh ratio-sized alloc per chunk — the
    // GATED `GZIPPY_RESIDENT_OUTPUT_POOL` mechanism (minor-faults drop toward
    // igzip, monorepo wall 1.39→1.29 both arches; plans/T1-CACHE-RESIDENCY-RESULTS).
    // SCOPED to this thread (RAII): the T>1 parallel workers never set it, so the
    // faithful per-chunk reserve/pool path is unchanged at T>1.
    let _resident = crate::decompress::parallel::chunk_buffer_pool::T1ResidentScope::enter();

    let total_bits = input.len() * 8;
    // Compressed-bytes-per-chunk stride from the chunk configuration (T1 default
    // = 1 MiB target, warm output-buffer recycling; GZIPPY_CHUNK_KIB overrides).
    // Clamp to ≥64 KiB so a pathological tiny config can't thrash the loop.
    let stride_bits = configuration.split_chunk_size.max(64 * 1024) * 8;

    let mut total_crc = CRC32Calculator::new();
    let mut total_size = 0usize;
    let mut marker_chunks = 0usize;

    // Rolling 32 KiB window. First chunk starts with a zero window (chunk-0
    // semantics: the first deflate block references no back-history past output
    // start, so a zero dict is correct for chunk 0). All subsequent chunks get
    // the TRUE trailing 32 KiB, so they take the window-PRESENT clean path.
    let mut prev_tail = vec![0u8; MAX_WINDOW_SIZE];
    let mut cur = 0usize;
    while cur < total_bits {
        let stop_hint = (cur + stride_bits).min(total_bits);
        let chunk = crate::decompress::parallel::chunk_decode::decode_chunk(
            input,
            cur,
            stop_hint,
            &prev_tail,
            configuration,
        )
        .map_err(FetchError::Decode)?;

        // Gate-0(d): a clean window-present decode must produce NO markers.
        // (Clean output lands in `data: SegmentedU8`; markers would show as a
        // non-zero `marker_count` / non-empty `data_with_markers`.)
        if chunk.statistics.marker_count != 0 || chunk.data_with_markers.len() != 0 {
            marker_chunks += 1;
        }

        // Advance the rolling window from this chunk's decoded tail.
        let data_len = chunk.data.len();
        if data_len >= MAX_WINDOW_SIZE {
            let mut t = vec![0u8; MAX_WINDOW_SIZE];
            chunk.data.copy_last_into(&mut t);
            prev_tail = t;
        } else {
            let mut nt = prev_tail.clone();
            let mut tail_bytes = vec![0u8; data_len];
            chunk.data.copy_range_into(0, &mut tail_bytes);
            nt.extend_from_slice(&tail_bytes);
            let n = nt.len();
            prev_tail = nt[n.saturating_sub(MAX_WINDOW_SIZE)..].to_vec();
        }

        // In-order write + CRC fold (mirror of the consumer's clean branch).
        chunk
            .data
            .write_payload_skipping_prefix(chunk.data_prefix_len, writer)
            .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
        total_size += chunk.decoded_size();
        // Track bytes streamed (for the multi-member-misroute resume net: a
        // stream the classifier called single-member but whose 2nd member began
        // past the detection window decodes member 1 here, then fails the
        // trailer check; resume re-decodes members 2+ past this prefix).
        if let Some(out) = bytes_written_out.as_deref_mut() {
            *out = total_size;
        }
        for stream_crc in &chunk.crc32s {
            total_crc.append(stream_crc);
        }

        // Stop once the gzip footer (trailer) was reached: the deflate stream
        // is complete and the remaining bits are the 8-byte CRC/ISIZE trailer
        // (single-member corpora). Decoding past it would misread the trailer
        // as a deflate header.
        if !chunk.footers.is_empty() {
            break;
        }

        let end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        if end_bit <= cur {
            break; // no forward progress (last chunk ran to EOF/BFINAL)
        }
        cur = end_bit;
        // The final deflate block of a single-member stream leaves only the
        // 8-byte gzip trailer (CRC32 + ISIZE = 64 bits). The smallest possible
        // gzip member HEADER is 10 bytes + footer, so < 18 bytes (144 bits) of
        // remaining input cannot hold another member — it is the trailer. Stop
        // rather than feed the trailer to decode_chunk as a deflate header.
        // (The corpora here — silesia/nasa/monorepo — are single-member; a real
        // multi-member next header always leaves >> 144 bits.)
        if total_bits.saturating_sub(cur) < 18 * 8 {
            break;
        }
    }

    if std::env::var_os("GZIPPY_DEBUG").is_some() {
        eprintln!(
            "[parallel_sm] thin-T1 out={total_size} bytes marker_chunks={marker_chunks} \
             stride={}KiB (clean serial, no parallel bulk)",
            stride_bits / 8 / 1024
        );
    }
    THIN_T1_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((total_crc.crc32(), total_size))
}

/// Production-routing proof: incremented once per [`drive_thin_t1_oracle`] call
/// (the thin single-chunk/T1 production path). A test snapshots it around a T1
/// decode to assert the thin spine — not the parallel scaffold — handled it.
#[cfg(parallel_sm)]
pub static THIN_T1_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// T1-MONOLITH driver — a deliberate, T1-gated DIVERGENCE from the rapidgzip
/// chunk pipeline TOWARD the igzip serial monolith (see
/// `plans/T1-MONOLITH-DIVERGENCE-LEDGER.md`). Decodes the ENTIRE single-member
/// deflate body as ONE chunk into ONE contiguous buffer reserved upfront to the
/// whole-member ISIZE, then writes it once. No per-chunk alloc, no per-chunk
/// rolling-window clone+re-seed, no per-block boundary record — the single
/// buffer IS the implicit history (igzip `tmp_out`/`out` shape).
///
/// Output is byte-identical to `drive_thin_t1_oracle` (same kernel, same clean
/// window-present decode, chunk-0 zero-window seed). CRC + ISIZE are verified by
/// the caller. T>1 NEVER reaches here (the caller gates strictly on `T==1`).
#[cfg(parallel_sm)]
pub fn drive_monolith_t1<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    configuration: ChunkConfiguration,
    expected_isize: usize,
    bytes_written_out: Option<&mut usize>,
) -> Result<(u32, usize), FetchError> {
    use crate::decompress::parallel::chunk_decode;

    #[cfg(not(isal_clean_tail))]
    let chunk = chunk_decode::decode_monolith_native(input, expected_isize, configuration)
        .map_err(FetchError::Decode)?;
    #[cfg(isal_clean_tail)]
    let chunk = chunk_decode::decode_monolith_isal(input, expected_isize, configuration)
        .map_err(FetchError::Decode)?;

    // ONE write of the whole output buffer (skip the 32 KiB dictionary prefix).
    chunk
        .data
        .write_payload_skipping_prefix(chunk.data_prefix_len, writer)
        .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
    let total_size = chunk.decoded_size();
    // Multi-member-misroute resume net: report bytes streamed so a stream the
    // classifier called single-member but whose 2nd member began past the
    // detection window decodes member 1 here, then fails the trailer check;
    // resume re-decodes members 2+ past this prefix.
    if let Some(out) = bytes_written_out {
        *out = total_size;
    }
    let mut total_crc = CRC32Calculator::new();
    for stream_crc in &chunk.crc32s {
        total_crc.append(stream_crc);
    }

    if std::env::var_os("GZIPPY_DEBUG").is_some() {
        eprintln!(
            "[parallel_sm] T1-MONOLITH out={total_size} bytes (single chunk, igzip-shaped: \
             one ISIZE-reserved buffer, no per-chunk alloc/window/boundary)"
        );
    }
    MONOLITH_T1_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((total_crc.crc32(), total_size))
}

/// Production-routing proof for the T1-monolith path (deletion-trap pattern).
#[cfg(parallel_sm)]
pub static MONOLITH_T1_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// T1-MONOLITH-STREAMING driver — the gzippy-native production T==1 path.
///
/// Decodes the WHOLE single-member deflate body in ONE continuous serial pass
/// (one `Block`, one marker ctx, one `set_initial_window`) that STREAMS output
/// through ONE small fixed-size resident buffer — shedding the per-chunk DRIVER
/// SCAFFOLD (the gated +21–30% T1-vs-igzip gap, F-6ce077591bb5/F-8bd982118f3d)
/// WITHOUT the prior full-ISIZE monolith's page-fault storm (the buffer is
/// reserved once and recycled in place via `retain_tail`; see
/// `decode_and_stream_monolith_native`). The decode runs inside the
/// `T1ResidentScope` (validated cache-residency lever: warm pooled pages,
/// scoped to this single serial thread; T>1 workers never enter it).
///
/// Byte-identical output to `drive_thin_t1_oracle`. CRC + ISIZE are verified by
/// the caller; errors are terminal (NO fallback). T>1 NEVER reaches here.
#[cfg(all(parallel_sm, not(isal_clean_tail)))]
pub fn drive_monolith_streaming_t1<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    configuration: ChunkConfiguration,
    expected_isize: usize,
    bytes_written_out: Option<&mut usize>,
) -> Result<(u32, usize), FetchError> {
    let _resident = crate::decompress::parallel::chunk_buffer_pool::T1ResidentScope::enter();
    let result = crate::decompress::parallel::chunk_decode::decode_and_stream_monolith_native(
        input,
        expected_isize,
        configuration,
        writer,
        bytes_written_out,
    )
    .map_err(FetchError::Decode)?;
    Ok(result)
}

fn drive_impl<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    configuration: ChunkConfiguration,
    seed_window_map: Option<WindowMap>,
    // Multi-member trailing-member support: when `Some`, the count of output
    // bytes already written to the sink is stored here even on the ERROR path.
    // A misrouted multi-member stream (member 2 past the 16 MiB detection
    // window) makes the single-stream finder error near the member boundary;
    // the driver uses this count to resume the remaining members past the
    // bytes already streamed, instead of silently truncating. `None` for every
    // existing caller (zero behavior change).
    mut bytes_written_out: Option<&mut usize>,
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
    // Oracle mode injects a pre-populated map (clean-window experiment);
    // production passes None and builds a fresh one.
    let window_map = seed_window_map.unwrap_or_else(|| {
        WindowMap::with_compression(
            crate::decompress::parallel::compressed_vector::CompressionType::None,
        )
    });
    // CLEAN-ONLY ENGINE ORACLE: eagerly load the seed side store (if
    // GZIPPY_SEED_WINDOWS is set) so the on-disk read is out of the workers'
    // decode time. No-op otherwise.
    crate::decompress::parallel::seed_windows::preload();
    // CLEAN-ONLY ENGINE ORACLE: pre-seed the block_finder with the REAL per-chunk
    // boundaries that have a captured predecessor window, so every dispatch lands
    // on a boundary whose window is seedable. A clean decode is only possible at a
    // real boundary; this + the seeded-window fallback forces every chunk down the
    // CLEAN path while the production consumer / publish chain runs unchanged.
    // No-op unless seed mode is on. Skip boundaries within one chunk of EOF (an
    // EOF-adjacent window key is a stream-end artifact, not a real chunk start).
    // DECOMPOSE knob (GZIPPY_SEED_NO_BOUNDARIES=1, measurement-only): skip the
    // block_finder pre-seed so dispatch lands on prod partition-GUESS boundaries
    // even when windows are seeded. Isolates the boundary-ALIGNMENT sub-lever from
    // the marker-COMPUTE sub-lever. OFF==identity. Byte-correct (partition guesses
    // are corrected at the confirmed offset, as in unseeded prod).
    if !crate::decompress::parallel::seed_windows::seed_no_boundaries() {
        let starts = crate::decompress::parallel::seed_windows::seedable_chunk_starts();
        for off in starts {
            if off + (32 * 1024 * 8) < total_bits {
                block_finder.insert(off);
            }
        }
    }

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
    // TRANSLITERATION (vendor BlockFetcher.hpp:181-183, :185): match rapidgzip's
    // constructor sizing exactly.
    //   m_cache( std::max( size_t( 16 ), m_parallelization ) )
    //   m_prefetchCache( 2 * m_parallelization )
    //   threadPoolSaturated: m_prefetching.size() + 1 >= m_parallelization
    // gzippy previously diverged: cache_capacity = pool_size*2 (vendor is
    // max(16,pool)) and a GZIPPY_BURST_PREFETCH lever on the saturation arg with
    // no vendor counterpart. Both deleted; sizing now matches vendor.
    let mut cache_capacity = std::cmp::max(16, pool_size);
    let mut prefetch_capacity = pool_size * 2;
    // PERFECT_OVERLAP oracle (GZIPPY_PERFECT_OVERLAP=1): NO cache-cap bump.
    // The corrected overlap dispatch parks IN-FLIGHT receivers in the unbounded
    // `prefetching` map (submit_prefetch), NOT in the size-capped caches, so the
    // oracle does not need a larger cache. The PRIOR (backwards) oracle bumped
    // BOTH caps to 4096 with no eviction, which held the whole member's u16
    // (2×-width) markered working set resident (~400MB vs production's ~190MB) —
    // an advisor-identified memory-bandwidth CONFOUND that added ~5-7% wall
    // (git history (campaign plan, removed) §C1.2). Vendor sizing kept so
    // the oracle is production-shaped and isolates ONLY the dispatch-depth term.
    // STEP-0 discriminator (a) POSITIVE CONTROL (gated on the probe being on, so
    // production is untouched / OFF==identity): GZIPPY_PREFETCH_CACHE_CAP=N
    // overrides BOTH cache capacities. Shrinking to a tiny value MUST force the
    // containing parent to be evicted before the lagging consumer reaches it ⇒
    // NOT_RESIDENT must rise (validates the probe observes residency). A LARGE
    // value should DROP NOT_RESIDENT. See git history (campaign plan, removed).
    if stall_residency::enabled() {
        if let Some(v) = std::env::var_os("GZIPPY_PREFETCH_CACHE_CAP") {
            if let Ok(n) = v.to_string_lossy().parse::<usize>() {
                let n = n.max(1);
                cache_capacity = n;
                prefetch_capacity = n;
                eprintln!("  STALL_RESIDENCY_PROBE: POSITIVE-CONTROL cache caps overridden to {n}");
            }
        }
    }
    let block_fetcher: Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>> =
        Arc::new(BlockFetcher::new(
            cache_capacity,
            prefetch_capacity,
            FetchMultiStream::new(FETCH_MEMORY_PER_STREAM, FETCH_MAX_STREAM_COUNT),
            pool_size,
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
    // TRANSLITERATION (vendor BlockFetcher.hpp:185): `m_threadPool( m_parallelization
    // == 1 ? 0 : m_parallelization )`. At a single thread, vendor uses ZERO pool
    // threads so submitted tasks run INLINE (deferred), avoiding a cross-thread
    // handoff that can't be overlapped. gzippy passed `pool_size` straight through
    // (one real worker), which made the eager full-scan's apply_window submits pure
    // overhead at T1 (+16% regression). Apply the `==1 ? 0` ONLY at pool construction;
    // the cache/prefetch sizing above keeps reading the true `pool_size` (vendor sizes
    // m_prefetchCache off m_parallelization, which stays 1 → 2, not 0).
    let pool_threads = if pool_size == 1 { 0 } else { pool_size };
    // Build the decode pool with an EMPTY pinning map — faithful to rapidgzip's
    // decode pool (`BlockFetcher.hpp:185` constructs the pool with a thread COUNT
    // only, empty pinning → the OS scheduler places threads). gzippy previously
    // added speculative worker-pinning (`with_pinning_for_capacity`) rapidgzip
    // never had; on an SMT box the default `get_core_ids()` cycling order packed
    // workers onto SMT-sibling logical cores of the same physical core, costing
    // +18-20% on silesia-T4. The PIN-DISCRIMINATOR measurement (FROZEN Intel,
    // N=13, plans/PIN-DISCRIMINATOR-2026-06-21.md) proved the unpinned pool ties
    // rapidgzip (silesia-T4 1.028 vs pinned 1.198) and the OS spreads the workers
    // across distinct physical cores on its own — so the pinning is deleted.
    let thread_pool = Arc::new(ThreadPool::new(
        pool_threads,
        crate::decompress::parallel::thread_pool::ThreadPinning::new(),
    ));

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

    // ORACLE-C (decode-bypass FLOOR): warm the prebuilt replay map BEFORE the
    // timed drive so the one-time capture-load + ChunkData reconstruction (the
    // ~656MB rebuild that CONTAMINATED the prior leverB floor at 3.667s, see
    // git history (campaign plan, removed)) is NOT counted as decode-floor time. Emits the
    // build duration to stderr (GZIPPY_BYPASS_DECODE replay only) so the harness
    // can report floor = wall − warm. No-op unless replay is enabled.
    crate::decompress::parallel::decode_bypass::warm_prebuilt();
    // REMOVAL-ORACLE NODECODE: same out-of-wall warm for the symbol-stream
    // replay map (no-op unless GZIPPY_ORACLE_NODECODE is set).
    crate::decompress::parallel::removal_oracle::warm_replay();

    // Phase-timing: all scaffold (block_finder, window_map, block_fetcher,
    // thread_pool spawn) is built — this is the block-finder-bootstrap boundary.
    crate::decompress::parallel::phase_timing::mark("scaffold_built");

    let drive_t0 = std::time::Instant::now();
    // PERFECT_OVERLAP oracle (GZIPPY_PERFECT_OVERLAP=1): CORRECTED overlap
    // dispatch — submit EVERY chunk's decode as an in-flight prefetch
    // (non-blocking) so they decode CONCURRENTLY with the timed consumer below.
    // The consumer drains in order (resolve + write) while later chunks still
    // decode = decode↔drain OVERLAP (the schedule production / rg use). Removes
    // ONLY the prefetch-dispatch-DEPTH gap; KEEPS the marker engine, serial
    // resolve chain, drain, write. This dispatch is cheap (no recv); the whole
    // wall is timed by drive_t0 below. No-op unless enabled.
    let warm_t0 = std::time::Instant::now();
    if crate::decompress::parallel::perfect_overlap::enabled() {
        perfect_overlap_warm(
            input_view,
            total_bits,
            &block_finder,
            &block_fetcher,
            &window_map,
            &thread_pool,
            pool_size,
            configuration,
        );
        eprintln!(
            "  PERFECT_OVERLAP: dispatch phase {:.4}s (all chunks in flight; consumer overlaps)",
            warm_t0.elapsed().as_secs_f64()
        );
    }
    // The in-order consumer (this thread runs `consumer_loop` inline) is left
    // unpinned — faithful to rapidgzip, which never pins its reader thread; the
    // OS schedules it alongside the unpinned decode workers (see the decode-pool
    // construction above and plans/PIN-DISCRIMINATOR-2026-06-21.md).
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

    // Phase-timing: consumer_loop returned — steady-state parallel decode +
    // in-order drain done. What follows (pool stop + flush + finalize) is the
    // finalize phase.
    crate::decompress::parallel::phase_timing::mark("consumer_done");

    // Stop the pool BEFORE returning so any straggler prefetch tasks
    // are joined and `InputSlice` is no longer reachable. Mirror of
    // vendor's `stopThreadPool()` at BlockFetcher.hpp:600-604, called
    // from `~GzipChunkFetcher` before member destruction (line 686).
    thread_pool.stop();

    // Faithful overlap writer: drain + join the background output writer so
    // ALL bytes are on the fd and any write error becomes terminal BEFORE the
    // trailer CRC/ISIZE is returned for verification. No-op when the overlap
    // writer was never used (inline writev path). Join even on a consumer
    // error so the writer thread does not outlive the run.
    #[cfg(all(unix, parallel_sm))]
    let writer_result = crate::decompress::parallel::output_writer::finish();
    // Report replay stats even on a (CRC) error so the bypass experiment
    // can see hit/miss counts when the run aborts.
    crate::decompress::parallel::decode_bypass::report_replay_stats();
    crate::decompress::parallel::removal_oracle::report_replay_stats();
    if crate::decompress::parallel::perfect_overlap::enabled() {
        eprintln!(
            "  PERFECT_OVERLAP: overlapped wall (dispatch+consumer) {:.4}s",
            drive_t0.elapsed().as_secs_f64()
        );
        crate::decompress::parallel::perfect_overlap::report_stats();
    }
    // Record bytes streamed so far for the multi-member resume path. The
    // consumer writes chunks in order and only after they fully validate, so on
    // a finder/decode error at a member boundary this count is exactly the
    // contiguous, correct prefix already on the sink. Set on BOTH paths.
    if let Some(out) = bytes_written_out.as_deref_mut() {
        *out = total_size;
    }
    consumer_result?;
    // Surface any background-writer error as a terminal decode error (the
    // bytes are on the fd / partially written, same contract as the inline
    // writev path which returns Err on a failed write).
    #[cfg(all(unix, parallel_sm))]
    writer_result.map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;

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

    // Phase-timing: pool stopped, output flushed, block_map/finder finalized,
    // prefetch cache drained — drive() finalize bookkeeping complete. The
    // trailer CRC32+ISIZE verify (crc_verified mark) runs next in sm_driver.
    crate::decompress::parallel::phase_timing::mark("finalize_done");

    // --verbose stats dump. Mirror of vendor's destructor print at
    // GzipChunkFetcher.hpp:124-198 + BlockFetcher.hpp:73-124. Triggered
    // by GZIPPY_VERBOSE env var (matches the existing GZIPPY_DEBUG
    // pattern at single_member.rs::debug_enabled). The CLI sets this
    // when --verbose is passed; tests and other internal callers
    // ignore it.
    if std::env::var("GZIPPY_VERBOSE").is_ok() {
        // P3.1 cycle-profile dump (no-op unless GZIPPY_CONTIG_PROF=1).
        crate::decompress::parallel::contig_prof::dump_if_enabled();
        // ASM-kernel effect counters (no-op unless GZIPPY_ASM_STATS=1).
        crate::decompress::parallel::asm_kernel::dump_if_enabled();
        // Rung-(d) marker-loop dist-arm effect counters (no-op unless
        // GZIPPY_MARKER_DIST_STATS=1).
        #[cfg(pure_inflate_decode)]
        crate::decompress::parallel::marker_inflate::marker_dist_stats::dump_if_enabled();
        // mfast-phase0 probe: cycle/event breakdown for `'mfast` vs careful loop
        // (no-op unless GZIPPY_MFAST_PROF=1).
        crate::decompress::parallel::marker_inflate::mfast_prof::dump_if_enabled();
        // AMD-residual region partition (no-op unless GZIPPY_REGION_PROF=1).
        crate::decompress::parallel::region_prof::dump_if_enabled();
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
        eprintln!(
            "  Eager harvest ready: {}",
            EAGER_HARVEST_READY.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Prefetch post-process promoted: {} harvest-promoted: {} already-resolved take: {}",
            PREFETCH_POST_PROCESS_PROMOTED.load(Ordering::Relaxed),
            PREFETCH_HARVEST_PROMOTED.load(Ordering::Relaxed),
            PREFETCH_ALREADY_RESOLVED.load(Ordering::Relaxed),
        );
        // Unified-decoder migration counters (reimplement-isa-l):
        //   handoff_window_grows ≈ num_worker_threads (one per thread, then
        //     flat) PROVES the per-chunk `clean_window: Vec<u8>` alloc is gone.
        //   resumable_fallback MUST be 0 with a 32 KiB window (no decline into
        //     slow ResumableInflate2).
        {
            use crate::decompress::parallel::chunk_decode::{
                BAD_SEED_RESYNC, BULK_TAIL_RESUMABLE_FALLBACK, EXACT_BLOCK_CHUNKS,
                EXACT_WRAPPER_CHUNKS, FINISHED_NO_FLIP_CHUNKS, FINISH_DECODE_ENTRIES,
                FLIP_TO_CLEAN_CHUNKS, HANDOFF_WINDOW_BUF_GROWS, INFLATE_WRAPPER_CHUNKS,
                ISAL_BFINAL_EXACT_LANDING_ACCEPTED, ISAL_ENGINE_ORACLE_CHUNKS,
                ISAL_ENGINE_ORACLE_FALLBACKS, ISAL_INEXACT_FALLBACKS, ISAL_UNTIL_EXACT_FALLBACKS,
                SEEDED_BLOCK_CHUNKS, SEEDED_WRAPPER_CHUNKS, WINDOW_SEEDED_CHUNKS,
            };
            eprintln!(
            "  Unified decoder: flip_to_clean={} finished_no_flip={} finish_decode={} inflate_wrapper={} window_seeded={} seeded_block={} seeded_wrapper={} exact_block={} exact_wrapper={} bad_seed_resync={} resumable_resync_calls={} handoff_window_grows={}",
            FLIP_TO_CLEAN_CHUNKS.load(Ordering::Relaxed),
            FINISHED_NO_FLIP_CHUNKS.load(Ordering::Relaxed),
            FINISH_DECODE_ENTRIES.load(Ordering::Relaxed),
            INFLATE_WRAPPER_CHUNKS.load(Ordering::Relaxed),
            WINDOW_SEEDED_CHUNKS.load(Ordering::Relaxed),
            SEEDED_BLOCK_CHUNKS.load(Ordering::Relaxed),
            SEEDED_WRAPPER_CHUNKS.load(Ordering::Relaxed),
            EXACT_BLOCK_CHUNKS.load(Ordering::Relaxed),
            EXACT_WRAPPER_CHUNKS.load(Ordering::Relaxed),
            BAD_SEED_RESYNC.load(Ordering::Relaxed),
            BULK_TAIL_RESUMABLE_FALLBACK.load(Ordering::Relaxed),
            HANDOFF_WINDOW_BUF_GROWS.load(Ordering::Relaxed),
        );
            // On gzippy-isal (`isal_clean_tail`) these are PRODUCTION counters: the
            // ISA-L clean-tail engine (faithful rapidgzip WITH_ISAL). On gzippy-native
            // they are non-zero only under the GZIPPY_ISAL_ENGINE_ORACLE measurement
            // knob. `isal_oracle_fallbacks` MUST be 0 — any non-zero means a clean tail
            // fell back to pure-Rust (a counted correctness net, not a silent diverge).
            eprintln!(
                "  ISA-L clean-tail engine (production on gzippy-isal): isal_chunks={} isal_fallbacks={} bfinal_exact_accepted={} until_exact_fb={} inexact_fb={}",
                ISAL_ENGINE_ORACLE_CHUNKS.load(Ordering::Relaxed),
                ISAL_ENGINE_ORACLE_FALLBACKS.load(Ordering::Relaxed),
                ISAL_BFINAL_EXACT_LANDING_ACCEPTED.load(Ordering::Relaxed),
                ISAL_UNTIL_EXACT_FALLBACKS.load(Ordering::Relaxed),
                ISAL_INEXACT_FALLBACKS.load(Ordering::Relaxed),
            );
        }
        // StoredParallel demotion counter: non-zero means a stored-prefix+Huffman-tail
        // stream was routed back to ParallelSM because the stored prefix < 50% of output.
        // A stored-dominated file (e.g. random100.gz with ~8% prefix) should show
        // stored_demoted=1 per run; a pure-stored or >50% stored stream stays 0.
        eprintln!(
            "  StoredParallel demoted to ParallelSM (Huffman tail > 50% of output): {}",
            crate::decompress::parallel::stored_split::STORED_DEMOTE_TO_PARALLEL_SM
                .load(Ordering::Relaxed),
        );
        // Pure-stored chunked-streaming path: non-zero = the no-monolithic-buffer
        // streaming path ran (Gate-0 non-inert witness; the 100 MB zero-init Vec
        // is gone, runs stream straight from input → sink).
        eprintln!(
            "  StoredParallel pure-stored chunked-streaming runs (no monolithic buffer): {}",
            crate::decompress::parallel::stored_split::STORED_STREAM_RUNS.load(Ordering::Relaxed),
        );
        // Window-sparsity effect counter: 0 = keepIndex=false faithful port (default),
        // non-zero = GZIPPY_WINDOW_SPARSITY=1 kill-switch active (old always-on path).
        // Each count = one 32 KiB getUsedWindowSymbols scan run at chunk finalize.
        eprintln!(
            "  Window-sparsity decode runs (0=off=vendor-default, >0=kill-switch-active): {}",
            crate::decompress::parallel::chunk_data::SPARSITY_DECODE_COUNT.load(Ordering::Relaxed),
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
        use crate::decompress::parallel::chunk_decode::{
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
        // Out-of-sync confirmed-block skips (vendor GzipChunkFetcher.hpp:411-427
        // throws std::logic_error; gzippy skips silently and counts here).
        eprintln!(
            "  Stale confirmed-block skips: {}",
            STALE_CONFIRMED_BLOCK_SKIP.load(Ordering::Relaxed),
        );
        // Early window publish: did the worker publish provably-clean
        // tail windows off the consumer's serial path (the lever)? A high
        // `published` count with `range_speculative` accounting for the
        // rest is the success signal; `tail_not_clean` are chunks that
        // fell back to the consumer's serial `get_last_window_vec(pred)`.
        eprintln!(
            "  Early window publish: published={} handoff_key={} tail_not_clean={} range_speculative={}",
            EARLY_WINDOW_PUBLISHED.load(Ordering::Relaxed),
            HANDOFF_WINDOW_PUBLISHED.load(Ordering::Relaxed),
            EARLY_WINDOW_TAIL_NOT_CLEAN.load(Ordering::Relaxed),
            EARLY_WINDOW_RANGE_SPECULATIVE.load(Ordering::Relaxed),
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
        eprintln!(
            "  Speculation phantom-EOS rejects: {}",
            SPECULATIVE_PHANTOM_EOS_REJECTS.load(Ordering::Relaxed),
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
            PREFETCH_CACHE_POLLUTION_STOPS, PREFETCH_NEW_BLOCKS_CALLED, PREFETCH_RETURN_SATURATED,
            PREFETCH_RETURN_SUBMITTED_ANY, PREFETCH_RETURN_ZERO_SUBMITTED,
            PREFETCH_TOTAL_SUBMITTED,
        };
        eprintln!(
            "  Prefetch dispatch: called={} saturated={} zero_submitted={} any_submitted={} total_submitted={} pollution_stops={}",
            PREFETCH_NEW_BLOCKS_CALLED.load(Ordering::Relaxed),
            PREFETCH_RETURN_SATURATED.load(Ordering::Relaxed),
            PREFETCH_RETURN_ZERO_SUBMITTED.load(Ordering::Relaxed),
            PREFETCH_RETURN_SUBMITTED_ANY.load(Ordering::Relaxed),
            PREFETCH_TOTAL_SUBMITTED.load(Ordering::Relaxed),
            PREFETCH_CACHE_POLLUTION_STOPS.load(Ordering::Relaxed),
        );
        // Body-failure forensic detail — speculation-accuracy attack.
        // After disprove-advisor confirmed body failures are the dominant
        // re-decode cost, characterize them by error variant + waste size.
        use crate::decompress::parallel::chunk_decode as gc;
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
        use crate::decompress::parallel::async_block_finder as rbf;
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
        let bs_clean_flipped = gc::BOOTSTRAP_CLEAN_FLIPPED_BYTES.load(Ordering::Relaxed);
        let clean_flipped_pct = if bs_b_bytes > 0 {
            100.0 * bs_clean_flipped as f64 / bs_b_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "  Bootstrap per-block: header_calls={bs_h_calls} header_ms={:.1} avg_header_us={:.1} body_ms={:.1} body_bytes={bs_b_bytes} body_rate_MB/s={:.0} clean_flipped_bytes={bs_clean_flipped} ({clean_flipped_pct:.1}% of body = marker-FREE complement; marker loop owns the other {:.1}%)",
            bs_h_us as f64 / 1000.0,
            bs_h_avg,
            bs_b_us as f64 / 1000.0,
            bs_b_rate,
            100.0 - clean_flipped_pct,
        );
        // Per-fetch rejection cause: a prefetched chunk arrived but the
        // safety guard rejected it (chain invariant broken —
        // chunk.max != next_block_offset).
        eprintln!(
            "  Prefetch guard-rejects: {}",
            PREFETCH_REJECT_BY_GUARD.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Clean decode (pred@key / pred@seed / handoff@stop / boundary@seed / candidate): {} / {} / {} / {} / {}",
            CLEAN_DECODE_VIA_PRED_KEY.load(Ordering::Relaxed),
            CLEAN_DECODE_VIA_PREDECESSOR.load(Ordering::Relaxed),
            HANDOFF_DECODE_CLEAN_OK.load(Ordering::Relaxed),
            SPEC_PRED_BOUNDARY_CLEAN.load(Ordering::Relaxed),
            SPEC_DECODE_CLEAN_OK.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Worker resolve-ahead: ok={} / attempts={}",
            RESOLVE_AHEAD_OK.load(Ordering::Relaxed),
            RESOLVE_AHEAD_ATTEMPTS.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Arc::try_unwrap hits/misses: {} / {}",
            ARC_TRY_UNWRAP_HITS.load(Ordering::Relaxed),
            ARC_TRY_UNWRAP_MISSES.load(Ordering::Relaxed),
        );
        eprintln!(
            "  Consumer pair-ready drains: {}",
            DRAIN_READY_IMMEDIATE.load(Ordering::Relaxed),
        );
    }

    // Flush all per-thread trace_v2 buffers (consumer thread + any
    // already-joined worker threads).
    trace_v2::flush_all();
    // DECODE-BYPASS: serialize the capture map (no-op unless capture on).
    crate::decompress::parallel::decode_bypass::flush_capture();
    // REMOVAL-ORACLE NODECODE: serialize the symbol-stream record (no-op
    // unless GZIPPY_ORACLE_RECORD is set).
    crate::decompress::parallel::removal_oracle::flush_record();
    // CLEAN-ONLY ENGINE ORACLE: snapshot every published window (capture mode) /
    // report seed hit-miss (seed mode). Both no-ops unless the env vars are set.
    if crate::decompress::parallel::seed_windows::capture_enabled() {
        crate::decompress::parallel::seed_windows::write_capture();
    }
    crate::decompress::parallel::seed_windows::report_seed_stats();
    Ok((total_crc.crc32(), total_size))
}

// ── Consumer: processNextChunk port ──────────────────────────────────────

/// Kill-switch for the cache-hit-path prefetch drive (vendor
/// BlockFetcher.hpp:297-299 parity). `GZIPPY_NO_HIT_DRIVE=1` reverts to
/// the pre-fix cadence (drive on miss + blocked-pump only) so the drive
/// can be A/B'd within a single binary (constant code layout). Read once.
#[cfg(parallel_sm)]
fn hit_drive_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("GZIPPY_NO_HIT_DRIVE").is_some_and(|v| v == "1"))
}

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
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
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
    // One below pool capacity: forces a head drain (often `[Ready, Async]`)
    // one iteration sooner so warm decode buffers return to the worker pool
    // without mid-stream lone-Ready drain (CRC-unsafe at len==1).
    let post_process_inflight_cap = pool_size.saturating_sub(1).max(2);

    // Vendor `m_markersBeingReplaced` + resolve-ahead: pool post-process
    // for prefetched successors, keyed by partition `encoded_offset_bits`.
    let mut prefetch_post_inflight: EagerSubmitted = std::collections::HashMap::new();
    // Vendor `waitForReplacedMarkers` (:497-511) non-blocking harvest of ready
    // marker-replace futures while blocked on the current chunk's future.
    let mut eager_completed: EagerCompleted = std::collections::HashMap::new();
    // Keep the last N drained chunks' buffers off the pool until the
    // pipeline moves on (lone-drain CRC bisect 2026-06-05: 1-chunk defer
    // insufficient at T>1; byte diff at chunk 4 boundary when lone emit
    // races worker fill of successor buffers). The depth + lone-drain
    // policy is T1-tuned (depth 0 + drain-lone at pool_size==1, where the
    // pool runs inline with no racing workers); see `RecycleDeferral`.
    let mut recycle_deferral = RecycleDeferral::new(pool_size);
    // Optional duplicate probe (GZIPPY_EAGER_POSTPROC=1) — full-cache scan
    // on every stall; production uses handoff-triggered resolve-ahead only.
    let eager_probe_enabled = eager_postproc_enabled();

    // The vendor's `processNextChunk` returns one chunk per call; the
    // caller loops in `ParallelGzipReader::read`. We inline that loop
    // here so the local-state mutation (post-process queue + writer +
    // CRC) stays simple.
    #[allow(clippy::while_let_loop)] // faithful port of vendor processNextChunk loop
    let mut iter_us_sum: u128 = 0;
    let mut prefetch_us_sum: u128 = 0;
    let mut finder_us_sum: u128 = 0;
    let mut fetcher_get_us_sum: u128 = 0;
    let submit_us_sum: u128 = 0;
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
        // Vendor `waitForReplacedMarkers` (:497-511): non-blocking harvest of
        // ready marker-replace futures on every consumer iteration.
        harvest_ready_postprocess(&mut prefetch_post_inflight, &mut eager_completed);

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
        // Any block — confirmed or spacing-guess — whose start bit falls
        // within already-decoded territory is stale: either a fast-path
        // chunk consumed it as an intermediate subchunk (spacing guess)
        // or subchunk insertion around a raw-finder (blockfinder_validation.rs)
        // stored-payload false-positive region left a confirmed entry
        // behind the frontier.  In both cases skip immediately.
        // Invariant: a legitimate confirmed block is ALWAYS at or after
        // furthest_decoded_bit when it reaches next_unprocessed_block_index,
        // because consumer_append_subchunks_vendor's `inserted` count
        // advances the index past every subchunk within the decoded chunk.
        if next_block_offset < furthest_decoded_bit {
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
        let stop_hint_is_exact = stop_hint_is_exact_for(
            block_finder,
            next_unprocessed_block_index,
            total_bits,
            decode_start,
            stop_hint_bit,
            false,
        );
        let partition_idx_for_trace = next_unprocessed_block_index;
        let params = DecodeParams {
            start_bit: decode_start,
            stop_hint_bit,
            stop_hint_is_exact,
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
         -> mpsc::Receiver<Result<ChunkArc, ChunkDecodeError>> {
            let prefetch_stop_hint_bit = next_offset.max(offset);
            let prefetch_params = DecodeParams {
                start_bit: offset,
                stop_hint_bit: prefetch_stop_hint_bit,
                stop_hint_is_exact: false,
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
        let mut chunk_arc_from_partition: Option<ChunkArc> = None;
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
                // Vendor GzipChunkFetcher.hpp:646-648,670-684 — accept when
                // `matchesEncodedOffset(blockOffset)`; else discard and
                // re-issue at the real offset via `get(blockOffset, ...)`.
                if arc.matches_encoded_offset(decode_start) {
                    if trace::is_enabled() {
                        trace::emit(
                            "consumer",
                            "speculative_accept",
                            &format!(
                                r#""partition_idx":{partition_idx_for_trace},"decode_start":{decode_start},"spacing_guess":{next_block_offset},"speculative_start":{},"encoded_offset":{},"max_acceptable":{}"#,
                                arc.encoded_offset_bits,
                                arc.encoded_offset_bits,
                                arc.max_acceptable_start_bit,
                            ),
                        );
                    }
                    // ACCEPT: move the `Arc` (no extra refcount) into the consumer path.
                    // PERFECT_OVERLAP self-test: a warm-cache hit = head-of-line
                    // decode wait removed for this chunk. Gated so production is
                    // perf-transparent (OFF == identity).
                    if crate::decompress::parallel::perfect_overlap::enabled() {
                        crate::decompress::parallel::perfect_overlap::record_warm_hit();
                    }
                    chunk_arc_from_partition = Some(arc);
                } else {
                    // Vendor GzipChunkFetcher.hpp:646-648 — partition-keyed
                    // prefetch present but `!matchesEncodedOffset(blockOffset)`.
                    PREFETCH_REJECT_BY_GUARD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if trace::is_enabled() {
                        trace::emit(
                            "consumer",
                            "speculative_mismatch",
                            &format!(
                                r#""partition_idx":{partition_idx_for_trace},"decode_start":{decode_start},"spacing_guess":{next_block_offset},"speculative_start":{},"encoded_offset":{},"max_acceptable":{}"#,
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
            Some(arc) => {
                // Vendor BlockFetcher.hpp:297-299 — `prefetchNewBlocks` runs on
                // EVERY index-changed `get()`, INCLUDING when the result was
                // already cached/prefetched: vendor's call sits BEFORE the
                // cached-result return at BlockFetcher.hpp:302-309. gzippy's
                // hit path (the partition-keyed take above) previously skipped
                // it — the dispatch drive only existed on the miss path
                // (inside `get_with_prefetch`) and in the 1ms blocked-pump
                // ticks — so during a drain of already-ready prefetched chunks
                // the dispatcher went COMPLETELY silent (masked trace
                // 2026-06-09, model.gz T8: zero dispatch calls for 53ms of
                // drain, then a ~17ms on-demand stall at the first
                // un-prefetched partition, repeating every ~13 chunks; 5
                // on-demand fetches == the 5 consumer stalls; pool fill
                // factor 53%). Driving here keeps the prefetch horizon
                // advancing one call per consumed chunk, exactly like vendor.
                // Miss-path ordering is untouched: on-demand submit still
                // precedes the dispatch drive (vendor BlockFetcher.hpp:276
                // before :297), so the head-of-line task cannot queue behind
                // fresh prefetches.
                // Kill-switch (A/B discriminator): GZIPPY_NO_HIT_DRIVE=1
                // disables ONLY this hit-path drive, reverting to the prior
                // miss-only/blocked-pump cadence. Verifiable: with the switch
                // set, the consumer.drive_prefetch_on_hit span count is 0 and
                // the unfrozen model-T8 Fetched-On-demand counter reverts to
                // its pre-fix value (~5).
                if should_drive_prefetch && !hit_drive_disabled() {
                    let _tv2 = trace_v2::SpanGuard::begin("consumer.drive_prefetch_on_hit");
                    block_fetcher.prefetch_new_blocks(
                        &lookup_block_offset,
                        &lookup_next_block_offset,
                        &prefetch_submit,
                        &is_finalized_too_high,
                        &partition_offset_for,
                    );
                }
                arc
            }
            None => {
                // PERFECT_OVERLAP self-test: a warm-cache MISS = a residual
                // head-of-line decode wait the oracle failed to remove. If this
                // fires often the warm_hit_frac drops below 1.0 and the number
                // is void (Rule 4). No-op unless the oracle is enabled.
                if crate::decompress::parallel::perfect_overlap::enabled() {
                    crate::decompress::parallel::perfect_overlap::record_warm_miss();
                }
                // STEP-0 discriminator (a): PARENT-CACHED-AT-STALL probe
                // (git history (campaign plan, removed)). This `None`
                // branch IS the genuine head-of-line cold-get stall. Before
                // blocking, classify whether the chunk CONTAINING `decode_start`
                // is resident (interior-reuse fixable) or evicted (re-scope).
                // Read-only snapshots; env-gated (OFF == identity).
                if stall_residency::enabled() {
                    // Snapshot each resident chunk's (start, max_start_tolerance,
                    // encoded_END). The advisor (step0-advisor-verdict.md) caught
                    // that [enc, max] is the speculative START-tolerance window
                    // (decoded bytes BEGIN at max, chunk_data.rs:143-145), so the
                    // containing-parent test must use the chunk's encoded RANGE
                    // [enc, enc+encoded_size_bits) that its decoded content spans,
                    // not [enc, max] (which collapses to enc==max after re-anchor).
                    let mut cached: Vec<(usize, usize, usize)> = Vec::new();
                    for (_k, arc) in block_fetcher.prefetch_cache_contents_sorted() {
                        cached.push((
                            arc.encoded_offset_bits,
                            arc.max_acceptable_start_bit,
                            arc.encoded_offset_bits
                                .saturating_add(arc.encoded_size_bits),
                        ));
                    }
                    for (_k, arc) in block_fetcher.cache_contents_sorted() {
                        cached.push((
                            arc.encoded_offset_bits,
                            arc.max_acceptable_start_bit,
                            arc.encoded_offset_bits
                                .saturating_add(arc.encoded_size_bits),
                        ));
                    }
                    let in_flight_keys = block_fetcher.prefetching_keys();
                    let partition_span = configuration.split_chunk_size.saturating_mul(8);
                    stall_residency::classify_stall(
                        decode_start,
                        &cached,
                        &in_flight_keys,
                        partition_span,
                    );
                    // SATURATION vs HORIZON occupancy snapshot at the stall
                    // instant (git history (campaign plan, removed)). Excludes the
                    // startup stall (decode_start == 0, unavoidable). Worker
                    // occupancy: lazy-spawn means an un-spawned slot is available
                    // capacity, so idle_capacity = parked idle + (cap - spawned).
                    if decode_start != 0 {
                        let cap = thread_pool.capacity();
                        let spawned = thread_pool.spawned_threads();
                        let idle = thread_pool.idle_thread_count();
                        let busy = spawned.saturating_sub(idle);
                        let idle_capacity = idle.saturating_add(cap.saturating_sub(spawned));
                        // `enqueued`: an in-flight prefetch key covers decode_start
                        // (the index WAS dispatched, just not finished) — same
                        // keyed estimate the residency probe uses for in-flight.
                        let enqueued = in_flight_keys.iter().any(|&k| {
                            k <= decode_start && decode_start < k.saturating_add(partition_span)
                        });
                        stall_residency::classify_occupancy(busy, idle_capacity, enqueued);
                    }
                }
                // The head chunk is NOT in the prefetch cache → the
                // consumer is about to BLOCK on its decode (the
                // `wait.block_fetcher_get` span, the other real serial wait
                // besides `wait.future_recv`). Vendor's
                // `queuePrefetchedChunkPostProcessing` overlaps post-process
                // of ready successors with exactly this kind of stall —
                // submit it NOW so it runs on the pool while we block.
                {
                    let partition_key = partition_offset_for(&next_block_offset);
                    let skip_keys: &[usize] = if partition_key != next_block_offset {
                        &[partition_key, next_block_offset]
                    } else {
                        &[partition_key]
                    };
                    queue_prefetched_marker_postprocess(
                        block_fetcher,
                        window_map,
                        thread_pool,
                        &mut prefetch_post_inflight,
                        &[],
                        skip_keys,
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
                    |_key: usize| -> mpsc::Receiver<Result<ChunkArc, ChunkDecodeError>> {
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
        // inner ChunkData unless resolve-ahead already submitted post-process
        // (borrow the `Arc` through window publish, `recv` at dispatch).
        let chunk_holder = {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.arc_take_or_clone");
            match Arc::try_unwrap(chunk_arc) {
                Ok(shared) => {
                    ARC_TRY_UNWRAP_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    ConsumerChunkHold::Owned(SharedChunkData::into_inner(shared))
                }
                Err(arc) => {
                    let real_offset = arc.encoded_offset_bits;
                    let resolved_pred_matches = arc.markers_resolved
                        && confirmed_predecessor_window(window_map, decode_start)
                            .map(|(pred_key, _)| arc.resolved_pred_key == Some(pred_key))
                            .unwrap_or(false);
                    if resolved_pred_matches {
                        use std::sync::atomic::Ordering;
                        PREFETCH_ALREADY_RESOLVED.fetch_add(1, Ordering::Relaxed);
                        trace_v2::emit_instant(
                            "causal.has_been_post_processed",
                            &format!(r#""start_bit":{},"site":"consumer_take""#, real_offset),
                            "t",
                        );
                    }
                    if resolved_pred_matches
                        || prefetch_post_inflight.contains_key(&real_offset)
                        || eager_completed.contains_key(&real_offset)
                    {
                        ARC_DEFERRED_BORROW.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ConsumerChunkHold::Deferred { arc }
                    } else {
                        ARC_TRY_UNWRAP_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ConsumerChunkHold::Owned(SharedChunkData::take_or_clone(arc))
                    }
                }
            }
        };

        // Real offset the eager probe would have keyed this chunk under
        // (its `encoded_offset_bits` BEFORE set_encoded_offset rewrites
        // it). Used to reuse an eager-submitted post-process below.
        let chunk_offset_pre_set = chunk_holder.encoded_offset_bits();
        let chunk = chunk_holder.as_ref();

        // Vendor GzipChunkFetcher.hpp:334-349 — `get(*nextBlockOffset)` then
        // `postProcessChunk`, then `setEncodedOffset(*nextBlockOffset)`.
        // Re-anchor happens in `drain_one_pending` after post-process returns.
        let handoff_bit = decode_start;

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
        // (WindowMap.hpp:24). Subchunk windows are published in
        // `drain_one_pending` after post-process + `set_encoded_offset`.
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
            if let Some((_pred_key, pred)) = confirmed_predecessor_window(window_map, handoff_bit) {
                let bytes = materialize_window(&pred);
                publish_end_window_before_post_process(window_map, chunk, bytes.as_ref());
                trace_v2::emit_instant(
                    "causal.window_publish",
                    &format!(
                        r#""start_bit":{},"end_bit":{chunk_end_bit},"site":"consumer_clean""#,
                        chunk.encoded_offset_bits
                    ),
                    "t",
                );
                // TRANSLITERATION (vendor `queuePrefetchedChunkPostProcessing`,
                // GzipChunkFetcher.hpp:520-551, called from waitForReplacedMarkers:513
                // on EVERY consumed chunk): full sorted scan of the prefetch cache,
                // each chunk checked against ITS OWN predecessor independently. gzippy
                // diverged with `Some(chunk_end_bit)` — a single chain-follow that
                // SKIPS any chunk whose key ≠ the running handoff (lut_bulk_inflate.rs
                // :2551 `continue`), so the chain breaks at the first speculative/
                // overshoot mismatch (resolved 5/12). `None` = vendor's robust full
                // scan; it still propagates the chain (publishes each end-window as it
                // goes) AND resolves the off-chain chunks one-pass in parallel.
                queue_prefetched_marker_postprocess(
                    block_fetcher,
                    window_map,
                    thread_pool,
                    &mut prefetch_post_inflight,
                    &[],
                    &[],
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
                while !window_map.contains(handoff_bit) {
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
                        Some((
                            thread_pool,
                            &mut prefetch_post_inflight,
                            &mut eager_completed,
                        )),
                        &mut recycle_deferral,
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
            let (pred_key, window) = confirmed_predecessor_window(window_map, handoff_bit).ok_or(
                FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: handoff_bit,
                    actual: furthest_decoded_bit,
                }),
            )?;
            consumer_pred_key = Some(pred_key);
            // Vendor `GzipChunkFetcher.hpp:341`: `sharedLastWindow->
            // decompress()` materializes the bytes once. For
            // CompressionType::None this is a zero-alloc slice borrow.
            // TRANSLITERATION (vendor queueChunkForPostProcessing, GzipChunkFetcher.hpp:558
            // `if ( !m_windowMap->get( windowOffset ) )`): only compute+publish this chunk's
            // end window if it is NOT already published. With the eager full-scan
            // (queuePrefetchedChunkPostProcessing) publishing end-windows ahead, the consumer's
            // recompute was REDUNDANT — measured get_last_window calls=102 for ~39 chunks
            // (~1.5ms each). get_last_window is deterministic given the same predecessor, so
            // the eager-published window is byte-identical; skipping the recompute is byte-exact.
            {
                let _tv2 = trace_v2::SpanGuard::begin("consumer.get_last_window");
                let window_bytes = materialize_window(&window);
                publish_end_window_before_post_process(window_map, chunk, window_bytes.as_ref());
            }
            trace_v2::emit_instant(
                "causal.window_publish",
                &format!(
                    r#""start_bit":{},"end_bit":{chunk_end_bit},"site":"consumer_marker""#,
                    chunk.encoded_offset_bits
                ),
                "t",
            );
            // TRANSLITERATION (vendor queuePrefetchedChunkPostProcessing, full sorted
            // scan on every consumed chunk; see clean-branch note above). `None`
            // replaces the divergent `Some(chunk_end_bit)` chain-follow.
            queue_prefetched_marker_postprocess(
                block_fetcher,
                window_map,
                thread_pool,
                &mut prefetch_post_inflight,
                &[],
                &[],
            );
            predecessor_window_for_postprocess = Some(window);
        }

        // Vendor GzipChunkFetcher.hpp:343-357 — `postProcessChunk` (blocking
        // `future.get()` on pool `applyWindow`), then `setEncodedOffset`, then
        // `appendSubchunksToIndexes`. Previously gzippy appended BEFORE
        // post-process and queued Async writes — ordering skew vs vendor and
        // measured publish-chain inflation (Fulcrum L_resolve / dispatch_recv).
        {
            let _tv2 = trace_v2::SpanGuard::begin("consumer.dispatch_post_process");
            harvest_ready_postprocess(&mut prefetch_post_inflight, &mut eager_completed);
            let (mut chunk, eager_already_done) = chunk_holder.into_chunk_data(
                &mut prefetch_post_inflight,
                &mut eager_completed,
                consumer_pred_key,
            );
            // STEP-1 U16-preserving ceiling oracle (consumer-serial arm): a ceiling
            // chunk was decoded through the seeded-clean path (dwm empty). Inject the
            // u16 write + resolve traffic HERE, serially on the consumer thread (the
            // pessimistic resolve-location arm), exactly ONCE per chunk (owned chunk;
            // flag cleared after firing). Wrong bytes are expected (terminal CRC fail);
            // this only adds the asm-irremovable resolve cost to the wall.
            if chunk.phantom_ceiling_len > 0 {
                crate::decompress::parallel::slow_knob::phantom_marker_resolve_traffic(
                    chunk.phantom_ceiling_len,
                    false,
                );
                chunk.phantom_ceiling_len = 0;
            }
            let pool_resolved = chunk_matches_consumer_pred(&chunk, consumer_pred_key);
            if let Some(window) = predecessor_window_for_postprocess {
                if !(eager_already_done || pool_resolved)
                    && !chunk.data_with_markers.is_empty()
                    && !chunk.markers_resolved
                {
                    use std::sync::atomic::Ordering;
                    let reuse = match prefetch_post_inflight.remove(&chunk_offset_pre_set) {
                        Some((eager_pred_key, _, eager_rx))
                            if consumer_pred_key == Some(eager_pred_key) =>
                        {
                            EAGER_PROBE_REUSED.fetch_add(1, Ordering::Relaxed);
                            Some(eager_rx)
                        }
                        Some(_) => {
                            EAGER_PROBE_REUSE_PRED_MISMATCH.fetch_add(1, Ordering::Relaxed);
                            None
                        }
                        None => {
                            EAGER_PROBE_REUSE_KEY_ABSENT.fetch_add(1, Ordering::Relaxed);
                            None
                        }
                    };
                    let _tv2_wait = trace_v2::SpanGuard::begin_with(
                        "wait.future_recv",
                        &format!(r#""chunk_id":{partition_idx_for_trace}"#),
                    );
                    let overlap = Some((
                        block_fetcher,
                        window_map,
                        thread_pool,
                        &mut prefetch_post_inflight,
                        &mut eager_completed,
                    ));
                    if let Some(eager_rx) = reuse {
                        recv_post_process_blocking(eager_rx, partition_idx_for_trace, overlap)?;
                    } else {
                        let rx = submit_post_process_to_pool(
                            thread_pool,
                            chunk,
                            consumer_pred_key.expect("marker branch sets pred_key"),
                            window,
                            partition_idx_for_trace,
                        );
                        chunk = recv_post_process_blocking(rx, partition_idx_for_trace, overlap)?;
                    }
                }
            }
            // Vendor `setEncodedOffset(*nextBlockOffset)` after post-process.
            if chunk.encoded_offset_bits != handoff_bit && chunk.matches_encoded_offset(handoff_bit)
            {
                use std::sync::atomic::Ordering;
                REANCHOR_AFTER_POSTPROCESS.fetch_add(1, Ordering::Relaxed);
                chunk.set_encoded_offset(handoff_bit);
            }
            let inserted = consumer_append_subchunks_vendor(
                block_map,
                block_finder,
                block_fetcher,
                &chunk,
                handoff_bit,
                chunk_end_bit,
                next_unprocessed_block_index,
                unsplit_blocks,
            );
            next_unprocessed_block_index += inserted;
            // Out-of-sync detection — vendor GzipChunkFetcher.hpp:411-427.
            // rapidgzip throws std::logic_error when
            //   block_finder->get(next_unprocessed_block_index) != blockOffsetAfterNext
            // (where blockOffsetAfterNext = chunk.encoded_offset_bits + chunk.encoded_size_bits).
            // gzippy diverges: the skip guard above handles confirmed false-positives
            // silently rather than aborting.  Here we detect the out-of-sync condition
            // and count it for diagnostics: a non-zero STALE_CONFIRMED_BLOCK_SKIP means
            // a confirmed entry fell behind the decode frontier (subchunk
            // insertion around a raw-finder false-positive region — NOT a
            // GzipBlockFinder scan; that component has no scanner. See the
            // record-correction note on STALE_CONFIRMED_BLOCK_SKIP and 98fd618c).
            {
                use std::sync::atomic::Ordering;
                let (got_next, _) = block_finder.get(next_unprocessed_block_index);
                if !block_finder.finalized() && got_next != Some(chunk_end_bit) {
                    STALE_CONFIRMED_BLOCK_SKIP.fetch_add(1, Ordering::Relaxed);
                }
            }
            pending.push_back(PendingWrite::Ready {
                idx: partition_idx_for_trace,
                chunk,
                cache_key: next_block_offset,
                handoff_bit,
            });
            crate::decompress::parallel::chunk_data::lc_set(
                &crate::decompress::parallel::chunk_data::LC_G_PENDING,
                pending.len(),
            );
        }

        // Vendor parity: write each post-process-complete chunk as soon as
        // it reaches the FIFO head (GzipChunkFetcher.hpp:333-342 writes
        // inline after `postProcessChunk` returns). Previously gzippy only
        // drained when `pending.len() > pool_size`, leaving Ready chunks
        // holding warm rpmalloc buffers until the cap filled — the measured
        // buffer-return-latency gap at T16 (chunk_buffer_pool.rs:176-179).
        drain_ready_pending_heads(
            &mut pending,
            window_map,
            writer,
            out_fd,
            total_crc,
            total_size,
            block_fetcher,
            thread_pool,
            &mut prefetch_post_inflight,
            &mut eager_completed,
            &mut recycle_deferral,
        )?;
        // Bound in-flight Async post-processes when the head is still
        // waiting on a worker (vendor blocks on the per-chunk future).
        while pending.len() > post_process_inflight_cap {
            drain_one_pending(
                &mut pending,
                window_map,
                writer,
                out_fd,
                total_crc,
                total_size,
                block_fetcher,
                Some((
                    thread_pool,
                    &mut prefetch_post_inflight,
                    &mut eager_completed,
                )),
                &mut recycle_deferral,
            )?;
            drain_ready_pending_heads(
                &mut pending,
                window_map,
                writer,
                out_fd,
                total_crc,
                total_size,
                block_fetcher,
                thread_pool,
                &mut prefetch_post_inflight,
                &mut eager_completed,
                &mut recycle_deferral,
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
            Some((
                thread_pool,
                &mut prefetch_post_inflight,
                &mut eager_completed,
            )),
            &mut recycle_deferral,
        )?;
    }
    while let Some(mut held) = recycle_deferral.queue.pop_front() {
        held.recycle_decoded_buffers();
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
    if eager_probe_enabled {
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

    // STEP-0 discriminator (a) tally — see stall_residency::report.
    stall_residency::report();

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
    // OVERSHOOT FIX (task #12). `min_gap` rejects a stop-hint candidate too close
    // to `floor` so we don't decode a degenerate tiny chunk. It was `spacing`,
    // which ALSO rejected the next partition boundary P+1 when `floor` is an
    // OVERSHOOT TAIL just past P (gap to P+1 = spacing − overshoot ≈ 0.99·spacing
    // < spacing) — bumping the on-demand decode to P+2, a 2× chunk (~16-21MB,
    // ~130ms) decoded SYNCHRONOUSLY on the consumer's critical path (the measured
    // head-of-line stalls, ≈40% of the T8 wall). Half a spacing still skips the
    // genuinely-tiny gaps (the 5-bit test case) while accepting a near-full P+1.
    let min_gap = (spacing / 2).max(8);

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

/// Conservative `untilOffsetIsExact` gate for the window-present path.
/// We only mark on-demand decodes exact when the chosen stop hint is
/// already in the confirmed block chain or is the EOF sentinel.
#[cfg(parallel_sm)]
fn stop_hint_is_exact_for(
    block_finder: &GzipBlockFinder,
    block_index: usize,
    total_bits: usize,
    start_bit: usize,
    stop_hint_bit: usize,
    is_speculative_prefetch: bool,
) -> bool {
    if is_speculative_prefetch || stop_hint_bit <= start_bit {
        return false;
    }
    if stop_hint_bit >= total_bits {
        return true;
    }

    let confirmed_count = block_finder.size();
    for delta in 1..=8 {
        let idx = block_index + delta;
        let (candidate, code) = block_finder.get(idx);
        let Some(candidate) = candidate else {
            return false;
        };
        let candidate = candidate.max(start_bit).min(total_bits);
        if candidate != stop_hint_bit {
            continue;
        }
        return idx < confirmed_count
            || (matches!(code, GetReturnCode::Failure) && candidate == total_bits);
    }
    false
}

#[cfg(parallel_sm)]
#[inline]
fn decode_chunk_with_until_exact(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    until_exact: bool,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    crate::decompress::parallel::chunk_decode::decode_chunk_until_exact(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
        until_exact,
    )
}

/// Submit a decode task to the `ThreadPool`, returning the
/// `mpsc::Receiver` that `BlockFetcher::get` will wait on. Mirror of
/// vendor's `submitOnDemandTask(blockOffset, nextBlockOffset)` at
/// BlockFetcher.hpp:573-589 — both arrange for an asynchronous decode
/// whose `std::future<BlockData>` the caller waits on. Vendor calls
/// `m_threadPool.submit([this, ...] () { return decodeAndMeasureBlock(...); }, 0)`;
/// we mirror that with `thread_pool.submit(run_decode_task, /* priority */ 0)`.
#[cfg(parallel_sm)]
/// PERFECT_OVERLAP oracle dispatch phase (GZIPPY_PERFECT_OVERLAP=1).
///
/// CORRECTED (2026-06-07): dispatch EVERY chunk's decode as an IN-FLIGHT
/// prefetch up-front (non-blocking — `submit_prefetch`, NOT `recv()`),
/// then return IMMEDIATELY so the timed `consumer_loop` runs CONCURRENTLY
/// with the still-running decodes. The consumer drains chunk i in order
/// (resolve markers off chunk i-1's published window + write) WHILE chunks
/// i+1.. are still decoding on the pool. That is decode↔drain OVERLAP — the
/// schedule production and rapidgzip actually use.
///
/// The PRIOR version (warm-all-then-drain) blocked on `recv()` for every
/// chunk before the consumer started, SERIALIZING the two phases production
/// overlaps — an ANTI-overlap whose wall was a pessimistic SUM (advisor
/// REFUTED it: git history (campaign plan, removed)). It could not
/// decide F1. This version removes ONLY the prefetch-DEPTH gap (every chunk
/// is in flight from t0, so the consumer never head-of-line stalls on a
/// not-yet-DISPATCHED chunk — the named project_confirmed_offset_prefetch_gap
/// dispatch-TIMING term), while KEEPING the real marker engine, the serial
/// marker-resolution window chain, drain, and write.
///
/// Faithfulness: each dispatched decode is the SAME `submit_decode_to_pool`
/// task the prefetcher would have run, at the SAME partition-guess start
/// (`is_speculative_prefetch=true`), keyed in the in-flight map under the
/// SAME partition offset the consumer queries (`submit_prefetch` mirrors
/// vendor `m_prefetching.emplace`, BlockFetcher.hpp:558). The window map is
/// NOT pre-seeded, so each chunk decodes markered at its true rate.
/// Output is therefore byte-identical (self-test: sha unchanged).
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn perfect_overlap_warm(
    input: InputSlice,
    total_bits: usize,
    block_finder: &GzipBlockFinder,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    window_map: &WindowMap,
    thread_pool: &Arc<ThreadPool>,
    _pool_size: usize,
    configuration: ChunkConfiguration,
) {
    use crate::decompress::parallel::perfect_overlap;

    let spacing = block_finder.spacing_in_bits().max(1);
    let mut block_index: usize = 0;
    let mut guess_offset: usize = 0;

    loop {
        // Partition-guess offset for this index. The block_finder yields the
        // confirmed offset once known, else the partition spacing multiple.
        let start = match block_finder.get(block_index) {
            (Some(off), GetReturnCode::Success) => off,
            _ => guess_offset,
        };
        if start >= total_bits || total_bits.saturating_sub(start) < 8 {
            break;
        }
        let part_key = block_finder.partition_offset_containing_offset(start);
        // Skip if a prefetch for this partition key is already in flight
        // (e.g. a confirmed offset and its partition-guess collapse to the
        // same key) — avoid clobbering an existing receiver.
        if block_fetcher.prefetch_in_flight(&part_key) {
            block_index += 1;
            guess_offset = guess_offset.saturating_add(spacing);
            if guess_offset >= total_bits {
                break;
            }
            continue;
        }
        // Worker stop hint: next partition boundary (or EOF). Same derivation
        // as the prefetch path (next_offset.max(offset)).
        let next_guess = guess_offset.saturating_add(spacing).min(total_bits);
        let stop_hint = next_guess.max(start);
        let params = DecodeParams {
            start_bit: start,
            stop_hint_bit: stop_hint,
            stop_hint_is_exact: false,
            is_speculative_prefetch: true,
            partition_idx: usize::MAX,
        };
        let rx = submit_decode_to_pool(
            thread_pool,
            input,
            params,
            window_map,
            block_fetcher,
            configuration,
        );
        // OVERLAP: park the IN-FLIGHT receiver (do NOT block on recv). The
        // consumer's `try_take_prefetched_pumping(&part_key)` will
        // `take_prefetch` it and block on the head chunk while the rest keep
        // decoding on the pool. Mirror of vendor `m_prefetching.emplace`.
        block_fetcher.submit_prefetch(part_key, rx);
        perfect_overlap::record_warm_chunk();

        block_index += 1;
        guess_offset = guess_offset.saturating_add(spacing);
        if guess_offset >= total_bits {
            break;
        }
        // Safety bound: never loop past a reasonable chunk count.
        if block_index > (total_bits / spacing) + 64 {
            break;
        }
    }
}

#[cfg(parallel_sm)]
fn submit_decode_to_pool(
    thread_pool: &Arc<ThreadPool>,
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> mpsc::Receiver<Result<ChunkArc, ChunkDecodeError>> {
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
    // Vendor keeps both speculative prefetch and on-demand decode in the same
    // priority band.
    let priority = if params.is_speculative_prefetch { 0 } else { 0 };
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
#[allow(dead_code)]
fn eager_postprocess_prefetched(
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    window_map: &WindowMap,
    thread_pool: &Arc<ThreadPool>,
    in_flight: &mut EagerSubmitted,
) -> usize {
    let _tv2 = trace_v2::SpanGuard::begin("consumer.eager_postproc");
    queue_prefetched_marker_postprocess(block_fetcher, window_map, thread_pool, in_flight, &[], &[])
}

#[cfg(parallel_sm)]
fn submit_post_process_to_pool(
    thread_pool: &Arc<ThreadPool>,
    chunk: ChunkData,
    pred_key: usize,
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
    submit_post_process_task(thread_pool, move || {
        run_post_process_task(chunk, pred_key, predecessor_window)
    })
}

/// Post-process a prefetched cache entry without deep-cloning [`ChunkData`] on
/// the consumer thread (vendor queues work; the clone happens on the pool).
#[cfg(parallel_sm)]
fn submit_post_process_from_prefetch(
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    thread_pool: &Arc<ThreadPool>,
    cache_key: usize,
    arc: ChunkArc,
    pred_key: usize,
    predecessor_window: Window,
) -> mpsc::Receiver<()> {
    let _ = (block_fetcher, cache_key);
    submit_post_process_void(thread_pool, move || {
        run_post_process_in_place(&arc, pred_key, predecessor_window);
    })
}

/// Vendor `submitTaskWithHighPriority` completion signal — pool mutates the
/// shared chunk in place; no `ChunkData` move through the channel.
#[cfg(parallel_sm)]
fn submit_post_process_void(
    thread_pool: &Arc<ThreadPool>,
    task: impl FnOnce() + Send + 'static,
) -> mpsc::Receiver<()> {
    let future = thread_pool.submit(task, /* priority */ -1);
    future.into_receiver()
}

#[cfg(parallel_sm)]
fn submit_post_process_task(
    thread_pool: &Arc<ThreadPool>,
    task: impl FnOnce() -> ChunkData + Send + 'static,
) -> mpsc::Receiver<ChunkData> {
    // Vendor queues post-process one band above decode.
    let future = thread_pool.submit(task, /* priority */ -1);
    future.into_receiver()
}

/// Pool-side execution of a decode task (vendor `decodeBlock`,
/// GzipChunkFetcher.hpp:692-729). Routes to `decode_chunk`
/// when the predecessor window is published, else
/// `speculative_decode_find_boundary` (marker bootstrap when no window).
#[cfg(parallel_sm)]
fn run_decode_task(
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> Result<ChunkArc, ChunkDecodeError> {
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
            return Ok(SharedChunkData::new(chunk));
        }
    }

    if crate::decompress::parallel::decode_bypass::replay_enabled() {
        if let Some(chunk) = crate::decompress::parallel::decode_bypass::replay(
            params.start_bit,
            params.stop_hint_bit,
            configuration,
        ) {
            return Ok(SharedChunkData::new(chunk));
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
    // `decode_chunk` slice borrow is valid for the call's
    // duration without going through WindowMap. Non-chunk-0 worker
    // Vendor `decodeBlock` (GzipChunkFetcher.hpp:712-729): exact
    // `m_windowMap->get(blockOffset)` only — no predecessor scan, no
    // speculative side-slot merge. Windows are published on the consumer
    // path (`queueChunkForPostProcessing`, GzipChunkFetcher.hpp:557-574).
    let zero_window: [u8; 32768] = [0u8; 32768];
    let window_at_offset: Option<Window> = if params.start_bit == 0 {
        None
    } else {
        window_map.get(params.start_bit)
    };
    // CLEAN-ONLY ENGINE ORACLE (GZIPPY_SEED_WINDOWS): when the live WindowMap has
    // NOT yet published this chunk's predecessor window, fall back to the captured
    // correct window from the seed store (forcing the CLEAN decode path) instead of
    // speculation. No-op (None) unless seed mode is on; on a seed miss this returns
    // None and the chunk takes its normal path (output stays byte-correct). Does NOT
    // shadow a real window: only consulted when window_at_offset is None.
    let seeded_window: Option<Vec<u8>> = if params.start_bit != 0 && window_at_offset.is_none() {
        crate::decompress::parallel::seed_windows::seed_window_for(params.start_bit)
    } else {
        None
    };
    let until_exact =
        (window_at_offset.is_some() || seeded_window.is_some()) && params.stop_hint_is_exact;

    let decode_mode_clean =
        params.start_bit == 0 || window_at_offset.is_some() || seeded_window.is_some();
    // Fulcrum `model` reads `worker.decode` span mode tags (rapidgzip patch uses
    // clean | window_absent). Keep boundary_search on causal instants only.
    let worker_decode_mode = if decode_mode_clean {
        "clean"
    } else {
        "window_absent"
    };
    let mode_str = if decode_mode_clean {
        "clean"
    } else {
        "boundary_search"
    };
    let _tv2_decode = trace_v2::SpanGuard::begin_with(
        "worker.decode",
        &format!(
            r#""start_bit":{},"mode":"{worker_decode_mode}""#,
            params.start_bit
        ),
    );
    trace_v2::emit_instant(
        "worker.decode_mode",
        &format!(
            r#""start_bit":{},"mode":"{worker_decode_mode}","pred_available":false"#,
            params.start_bit
        ),
        "t",
    );
    trace_v2::emit_instant(
        "causal.decode_decision",
        &format!(
            r#""start_bit":{},"window_present":{},"window_exact":{},"until_exact":{},"predecessor_available":false,"mode":"{mode_str}","stop_hint":{},"speculative":{}"#,
            params.start_bit,
            decode_mode_clean,
            window_at_offset.is_some(),
            until_exact,
            params.stop_hint_bit,
            params.is_speculative_prefetch
        ),
        "t",
    );

    // AMD-residual R_WORKER region: wraps the WHOLE per-chunk decode dispatch (all 4
    // branches incl. window-absent speculative) so every chunk (on-demand + prefetched)
    // is counted. ONE rdtsc pair per chunk on this worker thread.
    let _worker_t0 = crate::decompress::parallel::region_prof::rdtsc(
        crate::decompress::parallel::region_prof::enabled(),
    );
    if crate::decompress::parallel::region_prof::enabled() {
        crate::decompress::parallel::region_prof::region_enter();
    }
    let chunk_result = if params.start_bit == 0 {
        decode_chunk_with_until_exact(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            &zero_window[..],
            false,
            configuration,
        )
    } else if let Some(w) = window_at_offset.as_ref() {
        let bytes = materialize_window(w);
        // CLEAN-ONLY ENGINE ORACLE capture: this is the NATURAL clean path — the
        // window is correctly aligned to this real boundary. Record the aligned
        // (start_bit -> window) pair (no-op unless capture mode is on). A p=1
        // capture records every chunk this way → perfectly-aligned seed.
        crate::decompress::parallel::seed_windows::record_clean_window(
            params.start_bit,
            bytes.as_ref(),
        );
        decode_chunk_with_until_exact(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            &bytes,
            until_exact,
            configuration,
        )
    } else if let Some(seed_bytes) = seeded_window.as_ref() {
        // CLEAN-ONLY ENGINE ORACLE: decode with the captured correct predecessor
        // window — same clean path the published-window arm takes, just sourced
        // from the seed store instead of the live WindowMap.
        decode_chunk_with_until_exact(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            seed_bytes.as_slice(),
            until_exact,
            configuration,
        )
    } else {
        speculative_decode_find_boundary(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            configuration,
            window_map,
        )
    };
    // RING/FOLD-DRAIN perturbation (GZIPPY_RING_INJECT_NS, Gate-2). Fires INSIDE
    // the R_WORKER rdtsc window but OUTSIDE the Block-method R_TABLE/R_DECODE
    // spans, so the injected per-chunk work lands in `ring_other` — the residual
    // fulcrum F-9c5ca01d020d flagged. OFF (unset) ⇒ a single hoistable branch.
    crate::decompress::parallel::slow_knob::ring_inject();
    if crate::decompress::parallel::region_prof::enabled() {
        use std::sync::atomic::Ordering;
        crate::decompress::parallel::region_prof::region_exit();
        let dt = crate::decompress::parallel::region_prof::rdtsc(true).wrapping_sub(_worker_t0);
        crate::decompress::parallel::region_prof::R_WORKER_CYC.fetch_add(dt, Ordering::Relaxed);
        crate::decompress::parallel::region_prof::R_WORKER_N.fetch_add(1, Ordering::Relaxed);
        if let Ok(ref c) = chunk_result {
            crate::decompress::parallel::region_prof::R_WORKER_BYTES
                .fetch_add(c.decoded_size() as u64, Ordering::Relaxed);
        }
    }

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

    // Wrap in `ChunkArc` to match BlockFetcher's `Value` type (vendor's
    // `std::shared_ptr<BlockData>` at BlockFetcher.hpp:46).
    let result = chunk_result.map(SharedChunkData::new);

    if trace::is_enabled() {
        let dur_us = t0.elapsed().as_micros();
        let path = if params.start_bit == 0 || window_at_offset.is_some() {
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

/// True when pool post-process finished for this handoff (vendor
/// `ChunkData::hasBeenPostProcessed`, ChunkData.hpp:496-501).
#[cfg(parallel_sm)]
fn chunk_matches_consumer_pred(chunk: &ChunkData, consumer_pred_key: Option<usize>) -> bool {
    chunk.has_been_post_processed(false) && consumer_pred_key == chunk.resolved_pred_key
}

/// Shared marker resolve body (`applyWindow` + narrow + subchunk windows + CRC).
#[cfg(parallel_sm)]
fn resolve_chunk_markers_on_chunk(
    chunk: &mut ChunkData,
    pred_key: usize,
    predecessor_window: &[u8],
) {
    let dwm_len_pre = chunk.data_with_markers.len();
    // AMD-residual R_MARKERPP region (resolve+narrow+narrowed-crc+subwin == rg applyWindow).
    let _rp_span = crate::decompress::parallel::region_prof::Span::new(
        &crate::decompress::parallel::region_prof::R_MARKERPP_CYC,
        &crate::decompress::parallel::region_prof::R_MARKERPP_N,
    );
    if crate::decompress::parallel::region_prof::enabled() {
        crate::decompress::parallel::region_prof::R_MARKERPP_BYTES
            .fetch_add(dwm_len_pre as u64, std::sync::atomic::Ordering::Relaxed);
    }
    if dwm_len_pre > 0 {
        // Always fused resolve+narrow (64 KiB u8 LUT, one pass). The old
        // sub-128Ki-element branch ran `resolve_markers_u16` (128 KiB u16 LUT)
        // plus a second narrow pass — Fulcrum T8 showed +411ms wall-critical
        // vs rapidgzip in marker-resolve despite only 6/35 chunks on that path.
        POST_PROCESS_FUSED_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        chunk.resolve_and_narrow_markers_in_place(predecessor_window);
    }
    chunk.update_narrowed_crc();
    // Vendor `applyWindow` = narrow (DecodedData.hpp:325-363) → swap + in-place
    // VectorViews (:365-388). There is NO output-size copy: rapidgzip's narrowed
    // marker buffers ARE the output views, recycled when `writeAll` completes.
    // gzippy mirrors this — the narrowed marker bytes stay in `data_with_markers`
    // (the u8 view of the u16 backing) with `narrowed_len` set, and the consumer
    // emits them zero-copy via `append_narrowed_iovecs` (chunk_data.rs:1609).
    // We therefore DROP the redundant `merge_resolved_markers_into_data` full-
    // output memcpy AND the eager `recycle_markers_after_resolution`: the marker
    // segments must outlive the consumer's writev, so recycling is DEFERRED
    // behind the write via `defer_chunk_recycle` → `recycle_decoded_buffers`
    // (chunk_fetcher.rs:3486 / chunk_data.rs:1621, which frees BOTH `data` and
    // `data_with_markers`). `populate_subchunk_windows` runs against the un-merged
    // markers — `copy_window_at_chunk_offset` already branches on `narrowed_len>0`.
    chunk.populate_subchunk_windows(predecessor_window);
    // RSS convergence (GZIPPY_FREE_MARKERS): after narrow + CRC + subchunk-
    // window extraction have read the low-half narrowed bytes, release the
    // dead upper half of each marker segment back to the OS (faithful to rg
    // applyWindow @todo, DecodedData.hpp:374-379). The narrowed low half
    // stays resident for the consumer's zero-copy iovec write.
    if dwm_len_pre > 0 && crate::decompress::parallel::segmented_markers::free_markers_enabled() {
        chunk.data_with_markers.release_narrowed_upper_pages();
    }
    chunk.markers_resolved = true;
    chunk.resolved_pred_key = Some(pred_key);
}

/// Predecessor window for marker `apply_window` — vendor `WindowMap::get(*nextBlockOffset)`
/// (GzipChunkFetcher.hpp:478-518), exact confirmed handoff key only.
#[cfg(parallel_sm)]
fn confirmed_predecessor_window(
    window_map: &WindowMap,
    handoff_bit: usize,
) -> Option<(usize, Window)> {
    window_map.get(handoff_bit).map(|w| (handoff_bit, w))
}

/// Bit offset the consumer will use as `handoff_bit` / predecessor lookup
/// when this chunk reaches the head of the publish chain. For range-speculative
/// prefetches `encoded_offset_bits` stays at the partition seed while the
/// predecessor window is keyed at the worker's decode start
/// (`max_acceptable_start_bit` == vendor `offset.second`).
#[cfg(parallel_sm)]
fn chunk_consumer_handoff_bit(chunk: &ChunkData) -> usize {
    chunk.max_acceptable_start_bit
}

/// Marker chunks eligible for prefetch post-process (vendor
/// `queuePrefetchedChunkPostProcessing` + `hasBeenPostProcessed` gate,
/// GzipChunkFetcher.hpp:539-546).
#[cfg(parallel_sm)]
fn chunk_may_resolve_markers_early(chunk: &ChunkData, window_map: &WindowMap) -> bool {
    // Vendor GzipChunkFetcher.hpp:539 — skip `hasBeenPostProcessed()` chunks.
    if chunk.has_been_post_processed(false) {
        return false;
    }
    window_map.get(chunk_consumer_handoff_bit(chunk)).is_some()
}

/// Vendor `queueChunkForPostProcessing` footer empty-window branch
/// (GzipChunkFetcher.hpp:562-570).
#[cfg(parallel_sm)]
fn chunk_end_uses_empty_footer_window(chunk: &ChunkData) -> bool {
    chunk
        .footers
        .last()
        .is_some_and(|footer| footer.decoded_end_offset == chunk.decoded_size())
}

/// Vendor `queueChunkForPostProcessing` publish half (GzipChunkFetcher.hpp:557-575):
/// caller thread inserts `getLastWindow(*previousWindow)` — or an empty window
/// when the chunk ends on a gzip footer — at `encodedOffsetInBits +
/// encodedSizeInBits` BEFORE pool `applyWindow`.
#[cfg(parallel_sm)]
fn publish_end_window_before_post_process(
    window_map: &WindowMap,
    chunk: &ChunkData,
    predecessor_window: &[u8],
) {
    use std::sync::atomic::Ordering;
    if std::env::var_os("GZIPPY_NO_PUBLISH_AHEAD").is_some() {
        return;
    }
    if chunk.encoded_size_bits == 0 {
        return;
    }
    let window_offset = chunk.encoded_offset_bits + chunk.encoded_size_bits;
    if window_map.contains(window_offset) {
        return;
    }
    if chunk_end_uses_empty_footer_window(chunk) {
        window_map.insert_owned_none(window_offset, vec![0u8; 32768]);
    } else {
        let end_window = chunk.get_last_window_vec(predecessor_window);
        window_map.insert_owned_none(window_offset, end_window);
    }
    PUBLISH_AHEAD_WINDOWS.fetch_add(1, Ordering::Relaxed);
    EARLY_WINDOW_PUBLISHED.fetch_add(1, Ordering::Relaxed);
}

/// Vendor `queuePrefetchedChunkPostProcessing` (GzipChunkFetcher.hpp:520-551).
/// Full sorted prefetch-cache scan; predecessor via exact
/// `m_windowMap->get(chunkData->encodedOffsetInBits)` only.
#[cfg(parallel_sm)]
fn queue_prefetched_marker_postprocess(
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    window_map: &WindowMap,
    thread_pool: &Arc<ThreadPool>,
    in_flight: &mut EagerSubmitted,
    skip_real_offsets: &[usize],
    skip_cache_keys: &[usize],
) -> usize {
    use std::sync::atomic::Ordering;
    EAGER_PROBE_RUNS.fetch_add(1, Ordering::Relaxed);
    let _tv2 = trace_v2::SpanGuard::begin("consumer.queue_prefetched_postproc");
    block_fetcher.process_ready_prefetches();
    let contents = block_fetcher.prefetch_cache_contents_sorted();
    let n_inspected = contents.len();
    let mut submitted = 0usize;
    for (cache_key, arc) in contents {
        if skip_cache_keys.contains(&cache_key) {
            continue;
        }
        let real_offset = arc.encoded_offset_bits;
        if skip_real_offsets.contains(&real_offset) {
            continue;
        }
        if !chunk_may_resolve_markers_early(arc.as_ref(), window_map) {
            continue;
        }
        if in_flight.contains_key(&real_offset) {
            continue;
        }
        if arc.data_with_markers.is_empty() {
            continue;
        }
        // The window that resolves THIS chunk's markers must be the one valid at the
        // chunk's OWN decode-start (`max_acceptable_start_bit`). Re-keying this lookup
        // to the offset-sorted predecessor's published end (`prev.encoded_offset_bits
        // + prev.encoded_size_bits`, the §3.1 experiment) was REJECTED: for range-
        // speculative chunks that key names a DIFFERENT published window, and the
        // dense window map usually contains it, so the chunk gets resolved against the
        // wrong predecessor window → CRC32 mismatch (test_coalesce_fixed_huffman /
        // silesia parallel-SM, green on the seed key, red on the predecessor key).
        let handoff_bit = chunk_consumer_handoff_bit(arc.as_ref());
        // F1 instrument: RESOLVE_AHEAD_* / HANDOFF_WINDOW_PUBLISHED were declared but
        // NEVER incremented (the dead-counter trap behind `--verbose`'s "Worker
        // resolve-ahead" / "handoff_key" lines, which always read 0). Count one
        // attempt per chunk entering the predecessor lookup; the eligibility gate
        // above (`chunk_may_resolve_markers_early`) pre-screens window presence, so OK
        // tracks ATTEMPTS, but both are now LIVE on the production resolve-ahead path.
        RESOLVE_AHEAD_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        let Some((pred_key, predecessor_window)) =
            confirmed_predecessor_window(window_map, handoff_bit)
        else {
            continue;
        };
        RESOLVE_AHEAD_OK.fetch_add(1, Ordering::Relaxed);
        // Vendor queueChunkForPostProcessing:557-575 — publish end-window on
        // caller thread before pool applyWindow (marker chunks included).
        let pred_bytes = materialize_window(&predecessor_window);
        publish_end_window_before_post_process(window_map, arc.as_ref(), pred_bytes.as_ref());
        let rx = submit_post_process_from_prefetch(
            block_fetcher,
            thread_pool,
            cache_key,
            Arc::clone(&arc),
            pred_key,
            predecessor_window,
        );
        in_flight.insert(real_offset, (pred_key, cache_key, rx));
        // F1 instrument: resolve-ahead submitted ahead-work for this chunk (beside the
        // batch EAGER_PROBE_SUBMITTED below).
        HANDOFF_WINDOW_PUBLISHED.fetch_add(1, Ordering::Relaxed);
        submitted += 1;
    }
    EAGER_PROBE_INSPECTED.fetch_add(n_inspected as u64, Ordering::Relaxed);
    EAGER_PROBE_SUBMITTED.fetch_add(submitted as u64, Ordering::Relaxed);
    if submitted > 0 {
        EAGER_PROBE_RUNS_NONEMPTY.fetch_add(1, Ordering::Relaxed);
        EAGER_PROBE_MAX_PER_RUN.fetch_max(submitted as u64, Ordering::Relaxed);
    }
    submitted
}

/// In-place pool post-process on a cache-shared chunk (vendor
/// `GzipChunkFetcher.hpp:579-582` `chunkData->applyWindow`).
#[cfg(parallel_sm)]
fn run_post_process_in_place(arc: &SharedChunkData, pred_key: usize, predecessor_window: Window) {
    let start_bit = arc.encoded_offset_bits;
    let marker_bytes = arc.data_with_markers.len();
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "post_process.task",
        &format!(r#""start_bit":{start_bit},"marker_bytes":{marker_bytes},"in_place":true"#,),
    );
    let t_materialize = std::time::Instant::now();
    let bytes = materialize_window(&predecessor_window);
    let materialize_us = t_materialize.elapsed().as_micros();
    let t_apply = std::time::Instant::now();
    {
        let _tv2 = trace_v2::SpanGuard::begin("post_process.apply_window");
        arc.with_mut(|chunk| {
            resolve_chunk_markers_on_chunk(chunk, pred_key, bytes.as_ref());
        });
    }
    let apply_us = t_apply.elapsed().as_micros();
    trace_v2::emit_instant(
        "causal.tax",
        &format!(
            r#""start_bit":{start_bit},"marker_bytes":{marker_bytes},"resolve_us":{apply_us},"materialize_us":{materialize_us},"fused":{fused},"in_place":true"#,
            fused = marker_bytes > 0,
        ),
        "t",
    );
    if trace::is_enabled() {
        trace::emit(
            "post_process",
            "post_process_span",
            &format!(
                r#""start_bit":{start_bit},"materialize_us":{materialize_us},"apply_window_us":{apply_us},"marker_bytes":{marker_bytes},"in_place":true"#,
            ),
        );
    }
}

/// Pool-side execution of a uniquely-owned post-process task (consumer head
/// chunk where `Arc::try_unwrap` succeeded on `get()`).
#[cfg(parallel_sm)]
fn run_post_process_task(
    mut chunk: ChunkData,
    pred_key: usize,
    predecessor_window: Window,
) -> ChunkData {
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "post_process.task",
        &format!(
            r#""start_bit":{},"marker_bytes":{}"#,
            chunk.encoded_offset_bits,
            chunk.data_with_markers.len()
        ),
    );
    let start_bit = chunk.encoded_offset_bits;
    let marker_bytes = chunk.data_with_markers.len();
    let t_materialize = std::time::Instant::now();
    let bytes = materialize_window(&predecessor_window);
    let materialize_us = t_materialize.elapsed().as_micros();
    let t_apply = std::time::Instant::now();
    {
        let _tv2 = trace_v2::SpanGuard::begin("post_process.apply_window");
        resolve_chunk_markers_on_chunk(&mut chunk, pred_key, bytes.as_ref());
    }
    let apply_us = t_apply.elapsed().as_micros();
    let resolve_us = apply_us as f64;
    let fused = marker_bytes > 0;
    trace_v2::emit_instant(
        "causal.tax",
        &format!(
            r#""start_bit":{start_bit},"marker_bytes":{marker_bytes},"resolve_us":{resolve_us},"materialize_us":{materialize_us},"fused":{fused}"#,
        ),
        "t",
    );
    if trace::is_enabled() {
        trace::emit(
            "post_process",
            "post_process_span",
            &format!(
                r#""start_bit":{start_bit},"materialize_us":{materialize_us},"apply_window_us":{apply_us},"marker_bytes":{marker_bytes}"#,
            ),
        );
    }
    chunk
}

/// Narrow `src: &[u16]` into `dst: &mut U8`, appending bytes. All values
/// in `src` MUST be < 256 (post-`apply_window` invariant). AVX2 path
/// uses `_mm256_packus_epi16` for 16-lane parallel narrowing
/// (saturating pack — since values are already < 256, saturation is
/// a no-op). Scalar tail handles the remainder. AVX-256 downclock
/// concern from an earlier session was empirically refuted on neurotic
/// via injection probe (see `git history (campaign plan, removed)` §4).
#[cfg(parallel_sm)]
#[allow(dead_code)]
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

/// Vendor `appendSubchunksToIndexes` body (GzipChunkFetcher.hpp:357-396):
/// BlockMap push, BlockFinder insert, split_index, unsplitBlocks emplace.
/// Must run AFTER `postProcessChunk` and `setEncodedOffset` return.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn consumer_append_subchunks_vendor(
    block_map: &BlockMap,
    block_finder: &GzipBlockFinder,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    chunk: &ChunkData,
    handoff_bit: usize,
    chunk_end_bit: usize,
    next_unprocessed_block_index: usize,
    unsplit_blocks: &std::sync::Arc<std::sync::Mutex<std::collections::HashMap<usize, usize>>>,
) -> usize {
    append_subchunks_to_block_map(block_map, chunk);
    if chunk.subchunks.is_empty() {
        block_finder.insert(chunk_end_bit);
    } else {
        for sc in &chunk.subchunks {
            block_finder.insert(sc.encoded_offset_bits + sc.encoded_size_bits);
        }
    }
    if chunk.subchunks.len() > 1 {
        block_fetcher.split_index(next_unprocessed_block_index, chunk.subchunks.len());
    }
    if chunk.subchunks.len() > 1 {
        let chunk_offset = handoff_bit;
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
                use std::collections::hash_map::Entry;
                if let Entry::Vacant(v) = unsplit.entry(sc.encoded_offset_bits) {
                    v.insert(lookup_key);
                    UNSPLIT_BLOCKS_EMPLACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
    block_fetcher.statistics.base.record_get();
    chunk.subchunks.len().max(1)
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
            window_map.insert(sc_end_bit, Arc::clone(w));
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
/// Counter: speculative candidates rejected because the DEFLATE stream ended at a
/// BFINAL block mid-input and the bytes after the gzip footer (8 bytes: CRC32 + ISIZE)
/// are not EOF and do not start with valid gzip magic (0x1f 0x8b).
///
/// Vendor equivalent: after `block->isLastBlock()` in `decodeChunkWithRapidgzip`,
/// the code reads the gzip footer (GzipChunk.hpp:626-627), sets `isAtStreamEnd=true`
/// (GzipChunk.hpp:645), and on the next loop iteration calls `gzip::readHeader` at
/// GzipChunk.hpp:481. A non-EOF header-read error throws `std::domain_error`
/// (GzipChunk.hpp:491-498), which `tryToDecode`'s catch (GzipChunk.hpp:728-732)
/// turns into a `std::nullopt` — the candidate is silently discarded and the next
/// candidate is tried. `SPECULATIVE_PHANTOM_EOS_REJECTS` counts the gzippy-side
/// equivalent rejections, observable in `GZIPPY_VERBOSE` output.
#[cfg(parallel_sm)]
pub static SPECULATIVE_PHANTOM_EOS_REJECTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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

/// Worker published a confirmed window at `max_acceptable_start_bit` (handoff key).
pub static HANDOFF_WINDOW_PUBLISHED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Worker resolve-ahead: attempts / successes.
pub static RESOLVE_AHEAD_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static RESOLVE_AHEAD_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Vendor queueChunkForPostProcessing eager end-window publishes on the consumer
/// (the window-chain propagation that lets prefetched chunks resolve in parallel).
/// Chunks drained immediately when the FIFO head is `Ready` (post-process
/// complete) instead of waiting for `pending.len() > pool_size`. Returns
/// decode buffers to the warm pool sooner — vendor writes each chunk as
/// soon as its future completes rather than batching behind the cap.
pub static DRAIN_READY_IMMEDIATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// `set_encoded_offset(handoff_bit)` at drain after post-process (vendor order).
pub static REANCHOR_AFTER_POSTPROCESS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PUBLISH_AHEAD_WINDOWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub static COORDINATOR_BOUNDARY_SEARCH_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumps once per consumer iter where the partition-keyed
/// `try_take_prefetched` returned a chunk but the safety guard
/// rejected it (mismatch between `chunk.max_acceptable_start_bit` and
/// consumer-requested `next_block_offset`) — distinct from
/// `prefetch_cache_miss`, which counts a prefetch being absent.
/// Worker took the clean `decode_chunk` path using
/// `WindowMap::get_predecessor` (predecessor published at prior chunk end,
/// not at this chunk's partition seed).
pub static CLEAN_DECODE_VIA_PREDECESSOR: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Design H′: clean decode starting at confirmed `pred_key` before partition seed.
pub static CLEAN_DECODE_VIA_PRED_KEY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Speculative partition prefetch: clean `decode_chunk` at the handoff key
/// (`get_predecessor(stop_hint)`), not at the partition seed.
pub static HANDOFF_DECODE_CLEAN_OK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// KEY-MISMATCH oracle: clean decode at partition seed using predecessor tail,
/// accepting a real boundary inside the partition (Fulcrum causal perturbation).
pub static SPEC_PRED_BOUNDARY_CLEAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `try_speculative_decode_candidate` skipped marker bootstrap via clean
/// `decode_chunk` at a real boundary with the predecessor dict.
pub static SPEC_DECODE_CLEAN_OK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
/// `try_unwrap` missed but resolve-ahead is in flight — borrowed `Arc` through publish.
pub static ARC_DEFERRED_BORROW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Deferred borrow finished via in-flight post-process `recv` at dispatch.
pub static ARC_DEFERRED_INFLIGHT_RECV: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Eager post-process bookkeeping: maps a prefetched chunk's real
/// `encoded_offset_bits` (partition-seed key, stable across the consumer's
/// `set_encoded_offset` rewrite) → (predecessor-window key it resolved
/// against, in-flight apply_window receiver). The consumer reuses the
/// receiver when it reaches that offset, after validating the predecessor
/// key still matches (byte-identity guard).
#[cfg(parallel_sm)]
type EagerSubmitted = std::collections::HashMap<usize, (usize, usize, mpsc::Receiver<()>)>;
/// Harvested eager post-process completions (vendor non-blocking `future.get()`).
/// Resolved bytes live on the shared cache `ChunkArc` after in-place apply.
#[cfg(parallel_sm)]
type EagerCompleted = std::collections::HashMap<usize, usize>;

/// Vendor `waitForReplacedMarkers` (:497-511): poll ready marker-replace
/// futures without blocking the consumer on non-head chunks.
#[cfg(parallel_sm)]
fn harvest_ready_postprocess(in_flight: &mut EagerSubmitted, completed: &mut EagerCompleted) {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;
    let offsets: Vec<usize> = in_flight.keys().copied().collect();
    for offset in offsets {
        let Some((pred_key, cache_key, rx)) = in_flight.remove(&offset) else {
            continue;
        };
        match rx.try_recv() {
            Ok(()) => {
                completed.insert(offset, pred_key);
                EAGER_HARVEST_READY.fetch_add(1, Ordering::Relaxed);
            }
            Err(TryRecvError::Empty) => {
                in_flight.insert(offset, (pred_key, cache_key, rx));
            }
            Err(TryRecvError::Disconnected) => {}
        }
    }
}

/// Vendor `waitForReplacedMarkers` (:497-516): while blocking on the head
/// post-process future, harvest ready non-head futures and queue successor
/// prefetch post-processing (`queuePrefetchedChunkPostProcessing`).
#[cfg(parallel_sm)]
fn recv_post_process_blocking<F>(
    rx: mpsc::Receiver<F>,
    partition_idx: usize,
    mut overlap: Option<(
        &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
        &WindowMap,
        &Arc<ThreadPool>,
        &mut EagerSubmitted,
        &mut EagerCompleted,
    )>,
) -> Result<F, FetchError> {
    use std::sync::mpsc::{RecvTimeoutError, TryRecvError};
    use std::time::Duration;
    loop {
        match rx.try_recv() {
            Ok(v) => return Ok(v),
            Err(TryRecvError::Disconnected) => {
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: partition_idx,
                    actual: 0,
                }));
            }
            Err(TryRecvError::Empty) => {}
        }
        if let Some((block_fetcher, window_map, thread_pool, in_flight, completed)) =
            overlap.as_mut()
        {
            harvest_ready_postprocess(in_flight, completed);
            queue_prefetched_marker_postprocess(
                block_fetcher,
                window_map,
                thread_pool,
                in_flight,
                &[],
                &[],
            );
        }
        match rx.recv_timeout(Duration::from_micros(100)) {
            Ok(v) => return Ok(v),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: partition_idx,
                    actual: 0,
                }));
            }
        }
    }
}

/// Consumer chunk ownership across window publish vs post-process dispatch.
#[cfg(parallel_sm)]
enum ConsumerChunkHold {
    Owned(ChunkData),
    /// Resolve-ahead holds another `Arc` ref; borrow for publish, `recv` at dispatch.
    Deferred {
        arc: ChunkArc,
    },
}

#[cfg(parallel_sm)]
impl ConsumerChunkHold {
    fn as_ref(&self) -> &ChunkData {
        match self {
            Self::Owned(c) => c,
            Self::Deferred { arc } => arc.as_ref(),
        }
    }

    fn encoded_offset_bits(&self) -> usize {
        self.as_ref().encoded_offset_bits
    }

    /// Returns `(chunk, already_postprocessed)` — when true, dispatch must not
    /// re-submit or re-`remove` from `prefetch_post_inflight`.
    fn into_chunk_data(
        self,
        inflight: &mut EagerSubmitted,
        completed: &mut EagerCompleted,
        pred_key: Option<usize>,
    ) -> (ChunkData, bool) {
        match self {
            Self::Owned(c) => {
                let already = chunk_matches_consumer_pred(&c, pred_key);
                (c, already)
            }
            Self::Deferred { arc } => {
                let real_offset = arc.encoded_offset_bits;
                if chunk_matches_consumer_pred(arc.get(), pred_key) {
                    return (SharedChunkData::take_or_clone(arc), true);
                }
                if let Some(eager_pred_key) = completed.remove(&real_offset) {
                    if pred_key == Some(eager_pred_key) {
                        EAGER_PROBE_REUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return (SharedChunkData::take_or_clone(arc), true);
                    }
                    completed.insert(real_offset, eager_pred_key);
                }
                match inflight.remove(&real_offset) {
                    Some((eager_pred_key, _, rx)) if pred_key == Some(eager_pred_key) => {
                        ARC_DEFERRED_INFLIGHT_RECV
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        EAGER_PROBE_REUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let _tv2 = trace_v2::SpanGuard::begin("consumer.dispatch_recv");
                        if rx.recv().is_err() {
                            ARC_TRY_UNWRAP_MISSES
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        (SharedChunkData::take_or_clone(arc), true)
                    }
                    Some((eager_pred_key, cache_key, rx)) => {
                        EAGER_PROBE_REUSE_PRED_MISMATCH
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        inflight.insert(real_offset, (eager_pred_key, cache_key, rx));
                        ARC_TRY_UNWRAP_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        (SharedChunkData::take_or_clone(arc), false)
                    }
                    None => {
                        EAGER_PROBE_REUSE_KEY_ABSENT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ARC_TRY_UNWRAP_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        (SharedChunkData::take_or_clone(arc), false)
                    }
                }
            }
        }
    }
}

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
/// Non-blocking harvest of ready eager post-process futures (vendor :497-511).
pub static EAGER_HARVEST_READY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Pool post-process promoted resolved chunk into prefetch cache (vendor
/// `hasBeenPostProcessed` in-place mutation equivalent).
pub static PREFETCH_POST_PROCESS_PROMOTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Harvest path promoted a ready eager future into the prefetch cache.
pub static PREFETCH_HARVEST_PROMOTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Consumer took an already-resolved prefetched `Arc` (no dispatch needed).
pub static PREFETCH_ALREADY_RESOLVED: std::sync::atomic::AtomicU64 =
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

/// Speculative try-decode at a block-finder candidate (vendor `tryToDecode`,
/// GzipChunk.hpp:712-734): always `decodeChunkWithRapidgzip` with
/// `initialWindow = nullopt` at the actual decode seek bit (`offset.second`),
/// then rewrite metadata so `encoded_offset_bits = offset.first` and
/// `max_acceptable_start_bit = offset.second`. No `WindowMap::get` on the
/// candidate — only `decodeBlock` does an exact-key lookup before entering
/// the no-window path.
#[cfg(parallel_sm)]
fn try_speculative_decode_candidate(
    input: &[u8],
    decode_start: usize,
    partition_seed: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
    _window_map: &WindowMap,
) -> Result<ChunkData, ChunkDecodeError> {
    let result = decode_chunk_window_absent(input, decode_start, stop_hint_bit, configuration);
    // V1: classify failures so we know which fix to attack.
    //   header_fail  → "deflate header at bit X" — could be caught by
    //                  precode pre-pass (CountAllocatedLeaves port)
    //   body_fail    → "deflate body at bit X" — mid-stream Huffman
    //                  decode failed; precode wouldn't catch it
    //   inflate_fail → phase-2 StreamingInflateWrapper / ResumableInflate2
    //   stop_missed  → chunk size cap hit (not really a "failure")
    if let Err(ref e) = result {
        use std::sync::atomic::Ordering;
        let fail_kind = match e {
            ChunkDecodeError::BootstrapFailed(io_err) => {
                let msg = io_err.to_string();
                if msg.contains("deflate header") {
                    SPEC_FAIL_HEADER.fetch_add(1, Ordering::Relaxed);
                    "header"
                } else if msg.contains("deflate body") {
                    SPEC_FAIL_BODY.fetch_add(1, Ordering::Relaxed);
                    "body"
                } else {
                    SPEC_FAIL_OTHER.fetch_add(1, Ordering::Relaxed);
                    "other"
                }
            }
            ChunkDecodeError::InflateFailed(_) => {
                SPEC_FAIL_INFLATE.fetch_add(1, Ordering::Relaxed);
                "inflate"
            }
            ChunkDecodeError::ExactStopMissed { .. } => {
                SPEC_FAIL_STOP_MISSED.fetch_add(1, Ordering::Relaxed);
                "stop_missed"
            }
            ChunkDecodeError::OutputCeilingExceeded { .. } => {
                SPEC_FAIL_OTHER.fetch_add(1, Ordering::Relaxed);
                "output_ceiling_exceeded"
            }
            ChunkDecodeError::UnsupportedPlatform => {
                SPEC_FAIL_OTHER.fetch_add(1, Ordering::Relaxed);
                "other"
            }
        };
        trace_v2::emit_instant(
            "worker.try_to_decode",
            &format!(
                r#""partition_seed":{partition_seed},"decode_start":{decode_start},"ok":false,"fail_kind":"{fail_kind}""#
            ),
            "t",
        );
        return result;
    }
    trace_v2::emit_instant(
        "worker.try_to_decode",
        &format!(r#""partition_seed":{partition_seed},"decode_start":{decode_start},"ok":true"#),
        "t",
    );
    let mut chunk = result?;
    // (Design B) Record the encoded bit the worker decoded byte 0 from.
    // `decode_chunk_window_absent` anchored the chunk at `decode_start`
    // (`offset.second`), so that is the true origin regardless of the
    // range rewrite below.
    chunk.decode_origin_bit = decode_start;
    // Vendor tryToDecode metadata (GzipChunk.hpp:716-722):
    //   encoded = offset.first
    //   max     = offset.second
    debug_assert!(
        partition_seed <= decode_start,
        "candidate first={} must not exceed seek={}",
        partition_seed,
        decode_start
    );
    let encoded_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;

    // Vendor GzipChunk.hpp:470-499 + tryToDecode catch GzipChunk.hpp:728-732:
    // after the BFINAL block `decodeChunkWithRapidgzip` reads the gzip footer
    // (GzipChunk.hpp:626-627 `gzip::readFooter`), sets `isAtStreamEnd=true`
    // (GzipChunk.hpp:645), and on the next iteration tries `gzip::readHeader`
    // (GzipChunk.hpp:481). A non-EOF failure throws `std::domain_error`
    // (GzipChunk.hpp:491-498), which the `tryToDecode` catch at
    // GzipChunk.hpp:728-732 catches and returns `std::nullopt` — the candidate
    // is discarded. We replicate that rejection here: if the chunk ended early
    // (BFINAL before stop_hint_bit) and the bytes immediately after the 8-byte
    // gzip footer are neither EOF nor valid gzip magic (0x1f 0x8b), reject.
    // The check is SPECULATIVE PATH ONLY — confirmed/window-seeded paths do not
    // call this function.
    if encoded_end < stop_hint_bit {
        // Byte-align: deflate pads to a byte boundary before the gzip footer.
        let end_byte = (encoded_end + 7) / 8;
        // Gzip footer is 8 bytes (CRC32 LE + ISIZE LE, RFC 1952 §2.3.1).
        let footer_end = end_byte.saturating_add(8);
        if footer_end + 2 <= input.len() {
            // At least 2 bytes after the footer — check for valid gzip magic,
            // and when a 3rd byte is in-bounds also require CM=8 (deflate),
            // matching vendor readHeader more closely (advisor hardening:
            // cuts magic false-accept odds 256x; false-accept only reverts
            // to the pre-fix consume-time discard, so this is perf-only).
            let looks_like_gzip = input[footer_end] == 0x1f
                && input[footer_end + 1] == 0x8b
                && input.get(footer_end + 2).is_none_or(|&cm| cm == 0x08);
            if !looks_like_gzip {
                SPECULATIVE_PHANTOM_EOS_REJECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "phantom EOS at bit {encoded_end} (< stop_hint {stop_hint_bit}): \
                         post-footer bytes are not gzip magic \
                         (vendor GzipChunk.hpp:481,491-498,728-732)"
                    ),
                )));
            }
            // Bytes are 0x1f 0x8b: genuine multi-member stream — accept.
        }
        // footer_end + 2 > input.len(): near EOF, legitimate stream end — accept.
    }

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
    Ok(chunk)
}

/// Test-only re-export of `try_speculative_decode_candidate` so that
/// `src/tests/phantom_eos_probe.rs` can call the speculative path directly.
/// Not a public API — compiled only in `#[cfg(test)]` builds.
#[cfg(all(parallel_sm, test))]
pub(crate) fn try_speculative_decode_candidate_test_hook(
    input: &[u8],
    decode_start: usize,
    partition_seed: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let wm = WindowMap::new();
    try_speculative_decode_candidate(
        input,
        decode_start,
        partition_seed,
        stop_hint_bit,
        configuration,
        &wm,
    )
}

#[cfg(parallel_sm)]
fn speculative_decode_find_boundary(
    input: &[u8],
    start_bit: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
    window_map: &WindowMap,
) -> Result<ChunkData, ChunkDecodeError> {
    const MAX_SCAN_BITS: usize = 512 * 1024 * 8;
    let input_bits = input.len() * 8;
    if start_bit >= input_bits {
        let mut chunk = ChunkData::new(start_bit, configuration);
        chunk.finalize_with_deflate(start_bit, Some(input));
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
            window_map,
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
        |candidate| {
            // worker.scan_candidate span: one full-decode attempt at a
            // single bit position the scan flagged as a valid block
            // candidate. Matched on vendor side by the trace patch
            // around each tryToDecode call inside the alternating loop.
            let _tv2_cand = trace_v2::SpanGuard::begin("worker.scan_candidate");
            match try_speculative_decode_candidate(
                input,
                candidate.seek_bit,
                candidate.bit_offset,
                stop_hint_bit,
                configuration,
                window_map,
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
            if !super::blockfinder_validation::plausible_trial_decode_offset(input, bit) {
                continue;
            }
            if let Ok(chunk) = try_speculative_decode_candidate(
                input,
                bit,
                start_bit,
                stop_hint_bit,
                configuration,
                window_map,
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
        /// Vendor `setEncodedOffset(*nextBlockOffset)` after post-process.
        handoff_bit: usize,
    },
    Async {
        idx: usize,
        rx: mpsc::Receiver<ChunkData>,
        cache_key: usize,
        handoff_bit: usize,
    },
}

/// Recycle-deferral queue + the thread-count-tuned T1 in-flight-depth config.
///
/// Return drained chunk buffers to the pool only after `depth` newer chunks
/// have drained, and (at T1) drain a lone head Ready immediately.
///
/// At `parallelization == 1` the thread pool runs INLINE (`pool_threads == 0`,
/// see `drive_impl`) with NO concurrent workers, so the per-chunk output buffer
/// can be recycled immediately (`depth == 0`) and a lone Ready can drain without
/// the worker-fill race that forces defer-2 + lone-hold at T>1 (the 2026-06-05
/// lone-drain CRC bisect: "byte diff at chunk 4 boundary when lone emit races
/// worker fill of successor buffers"). Shrinking the T1 live-ChunkData working
/// set 4→1 is byte-exact at T1 and a measured win (BEAT-IGZIP-T1 night4, Intel:
/// nasa −22.9% cyc/B / −21% wall, silesia −3.2% cyc/B / −4.4% wall, RSS
/// −64%/−30%, both p<0.005, sha-identical) — the smaller resident output
/// working-set cuts page-faults (nasa −73%, silesia −42%) that feed the kernel.
/// It is INCORRECT at T>1 (a global depth-1 corrupts silesia at T4/T8) so the
/// T1 tuning is GATED on `pool_size == 1`. Sweep-only env overrides
/// (`GZIPPY_RECYCLE_DEFER_DEPTH`, `GZIPPY_DRAIN_LONE`) take precedence for
/// measurement; unset == identity.
#[cfg(parallel_sm)]
struct RecycleDeferral {
    queue: std::collections::VecDeque<ChunkData>,
    depth: usize,
    drain_lone: bool,
}

#[cfg(parallel_sm)]
impl RecycleDeferral {
    fn new(pool_size: usize) -> Self {
        let t1 = pool_size == 1;
        // T>1 default: defer 2, hold lone (correctness — worker-fill race).
        // T1: defer 0, drain lone (single inline worker, no race; measured win).
        let mut depth = if t1 { 0 } else { 2 };
        let mut drain_lone = t1;
        if let Some(v) = std::env::var("GZIPPY_RECYCLE_DEFER_DEPTH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            depth = v;
        }
        if std::env::var_os("GZIPPY_DRAIN_LONE").is_some() {
            drain_lone = true;
        }
        Self {
            queue: std::collections::VecDeque::with_capacity(depth + 1),
            depth,
            drain_lone,
        }
    }
}

#[cfg(parallel_sm)]
fn defer_chunk_recycle(deferral: &mut RecycleDeferral, chunk: ChunkData) {
    deferral.queue.push_back(chunk);
    while deferral.queue.len() > deferral.depth {
        if let Some(mut old) = deferral.queue.pop_front() {
            old.recycle_decoded_buffers();
        }
    }
    crate::decompress::parallel::chunk_data::lc_set(
        &crate::decompress::parallel::chunk_data::LC_G_RECYCLE,
        deferral.queue.len(),
    );
}

/// Drain consecutive `Ready` entries at the FIFO head when two or more
/// are queued (including `[Ready, Async]`). A sole `Ready` is flushed
/// only via the marker-branch `wait_replaced_markers` drain (predecessor
/// window needed), cap pressure, or the final flush — NOT at iteration
/// end, post-`block_finder.get`, post-`block_fetcher.get`, or when
/// eager-idle (all CRC-fail at T≥2; not buffer-pool UAF).
/// Stops when the head is `Async` (post-process still in flight on a worker).
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn drain_ready_pending_heads<W: std::io::Write>(
    pending: &mut std::collections::VecDeque<PendingWrite>,
    window_map: &WindowMap,
    writer: &mut W,
    out_fd: Option<i32>,
    total_crc: &mut CRC32Calculator,
    total_size: &mut usize,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    thread_pool: &Arc<ThreadPool>,
    eager_inflight: &mut EagerSubmitted,
    eager_completed: &mut EagerCompleted,
    recycle_deferral: &mut RecycleDeferral,
) -> Result<(), FetchError> {
    use std::sync::atomic::Ordering;
    let debug_lone = recycle_deferral.drain_lone;
    while matches!(pending.front(), Some(PendingWrite::Ready { .. }))
        && (debug_lone
            || pending.len() >= 2
            || matches!(pending.get(1), Some(PendingWrite::Async { .. })))
    {
        DRAIN_READY_IMMEDIATE.fetch_add(1, Ordering::Relaxed);
        drain_one_pending(
            pending,
            window_map,
            writer,
            out_fd,
            total_crc,
            total_size,
            block_fetcher,
            Some((thread_pool, eager_inflight, eager_completed)),
            recycle_deferral,
        )?;
    }
    Ok(())
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
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    // Eager post-process context (GZIPPY_EAGER_POSTPROC=1). When the head
    // pending write is an in-flight post-process (`Async`), the consumer is
    // about to BLOCK on `rx.recv()` (the `wait.future_recv` span — the
    // consumer's real serial wait once Design B has promoted predecessor
    // windows early). Vendor GzipChunkFetcher.hpp:513 queues successor
    // post-processing DURING exactly this wait. Mirror that: submit
    // apply_window for ready prefetched successors whose CONFIRMED
    // predecessor window is published, so they run on the pool while the
    // consumer blocks on the head. `None` (eager disabled) skips it.
    mut eager_ctx: Option<(&Arc<ThreadPool>, &mut EagerSubmitted, &mut EagerCompleted)>,
    recycle_deferral: &mut RecycleDeferral,
) -> Result<(), FetchError> {
    let _tv2 = trace_v2::SpanGuard::begin("consumer.drain");
    if let Some((_, in_flight, completed)) = eager_ctx.as_mut() {
        harvest_ready_postprocess(in_flight, completed);
    }
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    crate::decompress::parallel::chunk_data::lc_set(
        &crate::decompress::parallel::chunk_data::LC_G_PENDING,
        pending.len(),
    );
    // Phase-timing: first real chunk dequeued for write — first-chunk-to-first-
    // output latency boundary (the consumer's block-for-first-chunk wait ends
    // here). mark_once → only the FIRST chunk records it.
    crate::decompress::parallel::phase_timing::mark_once("first_output");
    let t_chunk = std::time::Instant::now();
    let t_recv = std::time::Instant::now();
    let (idx, mut chunk, cache_key, handoff_bit) = match head {
        PendingWrite::Ready {
            idx,
            chunk,
            cache_key,
            handoff_bit,
        } => (idx, chunk, cache_key, handoff_bit),
        PendingWrite::Async {
            idx,
            rx,
            cache_key,
            handoff_bit,
        } => {
            let _tv2_wait = trace_v2::SpanGuard::begin_with(
                "wait.future_recv",
                &format!(r#""chunk_id":{idx}"#),
            );
            let overlap = eager_ctx
                .as_mut()
                .map(|ctx| (block_fetcher, window_map, &*ctx.0, &mut *ctx.1, &mut *ctx.2));
            let chunk = recv_post_process_blocking(rx, idx, overlap)?;
            (idx, chunk, cache_key, handoff_bit)
        }
    };
    let recv_us = t_recv.elapsed().as_micros();

    if std::env::var_os("GZIPPY_TRACE_DRAIN").is_some() {
        eprintln!(
            "[trace_drain] idx={idx} enc={} handoff={} narrowed_len={} \
             markers_resolved={} data_len={}",
            chunk.encoded_offset_bits,
            handoff_bit,
            chunk.narrowed_len,
            chunk.markers_resolved,
            chunk.data.len().saturating_sub(chunk.data_prefix_len),
        );
    }

    // Subchunk window publish at drain (post-process + setEncodedOffset +
    // appendSubchunks already ran in `consumer_loop`).
    let t_pub = std::time::Instant::now();
    {
        let _tv2 = trace_v2::SpanGuard::begin("consumer.publish_windows");
        publish_subchunk_windows(window_map, &chunk);
    }
    let publish_us = t_pub.elapsed().as_micros();

    // Mirror of vendor's per-chunk write loop (GzipChunkFetcher.hpp:333-342).
    // Post-process narrows markers in-place then `merge_resolved_markers_into_data`
    // swaps them into `chunk.data` (DecodedData.hpp:365-388), so the consumer
    // gathers iovecs from unified `data` only.
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
    if crate::decompress::parallel::chunk_data::rss_split_enabled() {
        crate::decompress::parallel::chunk_data::rss_split_account(
            chunk.narrowed_len,
            decoded_data_len,
        );
    }
    let mut wrote_via_fd = false;
    #[cfg(unix)]
    if let Some(fd) = out_fd {
        if std::env::var_os("GZIPPY_DISABLE_WRITEV").is_none() && payload_bytes > 0 {
            use crate::decompress::parallel::fd_vectored_write;
            let _tv2 = trace_v2::SpanGuard::begin("consumer.writev");
            // AMD-residual R_OUTPUT region (iovec assembly + crc-combine + writev == rg write).
            let _ro_span = crate::decompress::parallel::region_prof::Span::new(
                &crate::decompress::parallel::region_prof::R_OUTPUT_CYC,
                &crate::decompress::parallel::region_prof::R_OUTPUT_N,
            );
            if crate::decompress::parallel::region_prof::enabled() {
                crate::decompress::parallel::region_prof::R_OUTPUT_BYTES
                    .fetch_add(payload_bytes as u64, std::sync::atomic::Ordering::Relaxed);
            }
            // Zero-copy gather: narrowed marker segments + clean payload
            // (vendor DecodedData.hpp:529 toIoVec). No memcpy — iovecs borrow
            // decode buffers until writev/vmsplice completes.
            let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
            chunk.append_output_iovecs(&mut parts);
            // CRC combine before boxing `chunk` for vmsplice lifetime.
            total_crc.append(&chunk.narrowed_crc);
            for stream_crc in &chunk.crc32s {
                total_crc.append(stream_crc);
            }
            let mut iovs = fd_vectored_write::to_io_vec(parts.iter().copied(), &[]);
            // MEASUREMENT-ONLY output-removal oracle. GZIPPY_SKIP_WRITEV_SYSCALL=1
            // skips ONLY the kernel writev (the 211 MB tmpfs page-cache copy on the
            // serial in-order consumer thread) while keeping CRC combine, recycle,
            // and all accounting. WRONG bytes ON (terminal trailer CRC still
            // verifies the decode ran) -> the off->on wall delta is the CAUSAL
            // removable share of the OUTPUT term that fulcrum_total localized on the
            // consumer timeline. OFF == identity (byte-exact). Never on a prod build.
            // The read is gated to linux because its only consumer (the `if
            // skip_writev` branch below) is linux-only; on other targets the env
            // var has no effect, so reading it there is dead (unused-var warning).
            #[cfg(target_os = "linux")]
            let skip_writev = std::env::var_os("GZIPPY_SKIP_WRITEV_SYSCALL").is_some();
            #[cfg(target_os = "linux")]
            if skip_writev {
                defer_chunk_recycle(recycle_deferral, chunk);
            } else if fd_vectored_write::is_pipe_fd(fd) {
                fd_vectored_write::write_chunk_payload_to_fd(fd, &mut iovs, chunk)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            } else if output_writer::enabled() {
                // FAITHFUL OVERLAP (rapidgzip writeFunctor, ParallelGzipReader.hpp:521):
                // hand the owned chunk to a single in-order writer thread so the
                // 211 MiB serial writev overlaps the next chunk's decode WAIT. Windows
                // are already published and the CRC already combined above, so output
                // order + trailer correctness are preserved. The writer owns the chunk
                // for the write lifetime (iovecs stay valid) and recycles it after.
                // `iovs`/`parts` borrow `chunk`; drop them before moving it.
                drop(iovs);
                drop(parts);
                output_writer::submit_chunk(fd, chunk);
            } else {
                fd_vectored_write::writev_all_to_fd(fd, &mut iovs)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                defer_chunk_recycle(recycle_deferral, chunk);
            }
            #[cfg(not(target_os = "linux"))]
            {
                fd_vectored_write::writev_all_to_fd(fd, &mut iovs)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                defer_chunk_recycle(recycle_deferral, chunk);
            }
            *total_size += payload_bytes;
            #[allow(unused_assignments)]
            {
                wrote_via_fd = true;
            }
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
        let _tv2 = trace_v2::SpanGuard::begin("consumer.write_buffered");
        let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
        chunk.append_output_iovecs(&mut parts);
        for sl in parts {
            writer
                .write_all(sl)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += sl.len();
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
    defer_chunk_recycle(recycle_deferral, chunk);
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

    /// Low-entropy deflate forces window-absent marker bootstrap; worker
    /// resolve-ahead must still produce byte-exact output.
    #[test]
    fn drive_handoff_marker_resolve_low_entropy() {
        let payload = vec![0x42u8; 32 * 1024 * 1024];
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 64 * 1024 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        let (_crc, n) = drive(&deflate, &mut out, None, 8, cfg).expect("drive");
        assert_eq!(n, payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn has_been_post_processed_blocks_prefetch_eager_queue() {
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 64 * 1024 * 1024,
            crc32_enabled: false,
            ..Default::default()
        };
        let mut chunk = ChunkData::new(1_000, cfg);
        chunk.encoded_offset_bits = 4_000_000;
        chunk.data_with_markers.push_slice(&[0u16, 1, 2]);
        chunk.subchunks[0].window = Some(Arc::new(
            crate::decompress::parallel::compressed_vector::CompressedVector::from_bytes(
                &[0u8; 32768],
                crate::decompress::parallel::compressed_vector::CompressionType::None,
            ),
        ));
        chunk.markers_resolved = true;
        let window_map = super::WindowMap::new();
        window_map.insert_owned_none(4_000_000, vec![0u8; 32768]);
        assert!(!super::chunk_may_resolve_markers_early(&chunk, &window_map));
    }

    #[test]
    fn chunk_may_resolve_markers_early_uses_consumer_handoff_bit() {
        // Range-speculative chunk: predecessor window is at decode start
        // (`max_acceptable_start_bit`), not partition seed.
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 64 * 1024 * 1024,
            crc32_enabled: false,
            ..Default::default()
        };
        let mut chunk = ChunkData::new(1_000, cfg);
        chunk.decode_origin_bit = 5_000_000;
        chunk.max_acceptable_start_bit = 5_000_000;
        chunk.encoded_offset_bits = 4_000_000;
        chunk.data_with_markers.push_slice(&[0u16, 1, 2]);
        let window_map = super::WindowMap::new();
        window_map.insert_owned_none(4_000_000, vec![0u8; 32768]);
        assert!(!super::chunk_may_resolve_markers_early(&chunk, &window_map));
        window_map.insert_owned_none(5_000_000, vec![0u8; 32768]);
        assert!(super::chunk_may_resolve_markers_early(&chunk, &window_map));
    }

    /// Cache-residency lever (b): the thin-T1 driver runs inside the resident-
    /// pool scope (manual buffer pool + fixed resident reserve, scoped to the T1
    /// thread). It must be byte-EXACT vs a flate2 reference across MANY chunk
    /// sizes (so the pool take/return cycle engages over multiple iterations) and
    /// several payload shapes (compressible / back-ref-heavy / mixed). A
    /// regression in the pooled-buffer reuse or the resident reserve would
    /// corrupt a later-chunk window or write into stale pages — caught here.
    #[test]
    fn thin_t1_recycled_byte_exact_multi_chunk_sizes() {
        use std::io::Read;
        // Three shapes: long literals, repetitive (back-ref heavy), and mixed.
        let mut mixed = Vec::new();
        for i in 0..(6 * 1024 * 1024u32) {
            mixed.push((i.wrapping_mul(2654435761) >> 24) as u8);
        }
        let payloads: Vec<Vec<u8>> = vec![
            b"abcdefghijklmnopqrst".repeat(350_000), // ~7 MiB, compressible
            vec![0x5au8; 5 * 1024 * 1024],           // pure run, back-ref heavy
            mixed,                                   // ~6 MiB, low-redundancy
        ];
        for payload in &payloads {
            for level in [1u32, 6, 9] {
                let deflate = make_deflate(payload, level);
                // flate2 reference decode of the SAME deflate body.
                let mut reference = Vec::new();
                flate2::read::DeflateDecoder::new(&deflate[..])
                    .read_to_end(&mut reference)
                    .expect("flate2 reference");
                assert_eq!(&reference, payload, "flate2 self-check");
                // Force MULTIPLE chunks at each size so recycling spans iterations.
                for kib in [64usize, 256, 1024, 2048] {
                    let cfg = ChunkConfiguration {
                        split_chunk_size: kib * 1024,
                        max_decoded_chunk_size: 20 * kib * 1024,
                        crc32_enabled: true,
                        ..Default::default()
                    };
                    let mut out = Vec::new();
                    let mut bytes_written = 0usize;
                    let (_crc, n) =
                        drive_thin_t1_oracle(&deflate, &mut out, cfg, Some(&mut bytes_written))
                            .expect("thin-T1 resident-pool drive");
                    assert_eq!(n, payload.len(), "size at {kib}KiB level{level}");
                    assert_eq!(bytes_written, payload.len(), "bytes_written at {kib}KiB");
                    assert_eq!(&out, payload, "bytes at {kib}KiB level{level}");
                }
            }
        }
    }

    #[test]
    fn drive_round_trips_2mb_level6() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
            ..Default::default()
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
            ..Default::default()
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

    #[test]
    fn stop_hint_is_exact_for_confirmed_successor() {
        let spacing_bytes = 4 * 1024 * 1024;
        let spacing_bits = spacing_bytes * 8;
        let total_bits = spacing_bits * 10;
        let finder = GzipBlockFinder::new(0, spacing_bytes, Some(total_bits));
        let confirmed_end = spacing_bits - 5;
        let confirmed_next = 2 * spacing_bits;
        finder.insert(confirmed_end);
        finder.insert(confirmed_next);

        let until = super::stop_hint_bit_for(&finder, 1, total_bits, confirmed_end);
        assert_eq!(until, confirmed_next);
        assert!(super::stop_hint_is_exact_for(
            &finder,
            1,
            total_bits,
            confirmed_end,
            until,
            false,
        ));
    }

    #[test]
    fn stop_hint_is_exact_for_rejects_partition_guess() {
        let spacing_bytes = 4 * 1024 * 1024;
        let spacing_bits = spacing_bytes * 8;
        let total_bits = spacing_bits * 10;
        let finder = GzipBlockFinder::new(0, spacing_bytes, Some(total_bits));
        let confirmed_end = spacing_bits - 5;
        finder.insert(confirmed_end);

        let until = super::stop_hint_bit_for(&finder, 1, total_bits, confirmed_end);
        assert!(until > confirmed_end);
        assert!(!super::stop_hint_is_exact_for(
            &finder,
            1,
            total_bits,
            confirmed_end,
            until,
            false,
        ));
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
        // chunk_decode::cross_chunk_resume for the same fix).
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
            ..Default::default()
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, None, 8, cfg).expect("drive");
        assert_eq!(size, payload.len(), "size mismatch (suggests early break)");
        assert_eq!(out, payload, "byte mismatch");
    }
}
