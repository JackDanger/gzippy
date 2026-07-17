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
#[cfg(parallel_sm)]
use crate::decompress::parallel::prefetcher::FetchMultiStream;
#[cfg(parallel_sm)]
use crate::decompress::parallel::thread_pool::ThreadPool;
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

/// Minimum input length for the STEP-3 progressive input-page reaper to arm.
/// Small inputs cost ~nothing to tear down, so they stay byte-for-byte on the
/// old path (no madvise). storedheavy/silesia/nasa/… all clear this.
#[cfg(parallel_sm)]
const INPUT_RELEASE_MIN_LEN: usize = 16 * 1024 * 1024;
/// Keep this many bytes immediately behind the in-order decode frontier
/// resident — a slack band so an in-flight worker's read cursor (or a
/// speculative prefetch that reaches slightly back) never trips a re-fault.
#[cfg(parallel_sm)]
const INPUT_RELEASE_MARGIN: usize = 4 * 1024 * 1024;
/// Release in stride-sized batches (amortize `madvise` calls on the consumer
/// thread, not one per chunk). The batch cap directly bounds the behind-frontier
/// resident lag to `MARGIN + STRIDE`. The high-thread default is 8 MiB (≤ 12 MiB
/// behind-frontier resident); the low/mid-thread value is 1 MiB (≤ 5 MiB) — a
/// byte-transparent peak-RSS reduction. The safety MARGIN is
/// UNCHANGED in both, so no worker/prefetch read cursor gains any new re-fault
/// exposure; only the release cadence tightens.
///
/// WHY POOL-SIZE-GATED (`reaper_stride_for`): `MADV_DONTNEED` on a shared
/// file-backed mmap triggers a TLB shootdown IPI to every thread mapping it, so
/// the tighter (more frequent) cadence costs O(pool_size) per call. At low/mid T
/// the shootdown is cheap and the consumer slack absorbs it. At
/// T16 the tighter 1 MiB cadence's 16-way shootdown storm outweighs the RSS gain,
/// so T > 8 keeps the coarse 8 MiB stride. The residual gap
/// to rapidgzip is the AHEAD-of-frontier read-ahead inherent to gz's
/// zero-copy whole-file mmap decode (rg streams input into bounded copies and
/// holds a small file-backed set) — NOT reachable by tightening a behind-frontier
/// reaper.
#[cfg(parallel_sm)]
const INPUT_RELEASE_STRIDE: usize = 8 * 1024 * 1024;
/// Low/mid-thread stride (see [`INPUT_RELEASE_STRIDE`] doc): tighter release
/// cadence for the peak-RSS reduction where the TLB-shootdown cost is cheap.
#[cfg(parallel_sm)]
const INPUT_RELEASE_STRIDE_LOWT: usize = 1024 * 1024;
/// Pool-size threshold at/below which the tighter [`INPUT_RELEASE_STRIDE_LOWT`]
/// applies. Above it, the shootdown storm dominates, so
/// the coarse [`INPUT_RELEASE_STRIDE`] is used.
#[cfg(parallel_sm)]
const INPUT_RELEASE_LOWT_MAX_POOL: usize = 8;

/// The behind-frontier release stride for a given decode `pool_size` (see
/// [`INPUT_RELEASE_STRIDE`] doc for the pool-size gate rationale).
#[cfg(parallel_sm)]
#[inline]
fn reaper_stride_for(pool_size: usize) -> usize {
    if pool_size <= INPUT_RELEASE_LOWT_MAX_POOL {
        INPUT_RELEASE_STRIDE_LOWT
    } else {
        INPUT_RELEASE_STRIDE
    }
}

#[cfg(parallel_sm)]
#[inline]
fn reaper_page_size() -> usize {
    #[cfg(unix)]
    {
        let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if v > 0 {
            v as usize
        } else {
            4096
        }
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

/// STEP-3. On a large single-member decode gz holds the ENTIRE input mmap
/// resident until the `Mmap::drop` at the end of `decompress_file` — where a
/// single serial `munmap` of the large input costs a few ms at process exit.
/// rapidgzip pays ~0 there because it keeps a bounded resident set. This reaper
/// releases consumed input pages DURING decode, overlapped with the parallel
/// decode work, so the resident set — and thus the final teardown — shrinks
/// toward rapidgzip's.
///
/// Correctness: the input is a file-backed mmap, so `MADV_DONTNEED` only drops
/// resident pages; any later access re-faults them from the page cache with the
/// original bytes. It therefore CANNOT change decoded output. We only release
/// pages behind `furthest_decoded_bit − MARGIN`, which are past every worker's
/// read cursor (decode and the block-finder only advance forward; a block that
/// falls behind the frontier is skipped, never re-decoded — see the
/// `next_block_offset < furthest_decoded_bit` guard). This preserves gz's
/// eager teardown (no deferred munmap is introduced).
#[cfg(parallel_sm)]
struct InputReaper {
    base: *const u8,
    len: usize,
    released: usize,
    page: usize,
    enabled: bool,
    /// Pool-size-gated behind-frontier release stride (see `reaper_stride_for`).
    stride: usize,
}

#[cfg(parallel_sm)]
impl InputReaper {
    fn new(input: InputSlice, pool_size: usize) -> Self {
        // Arm ONLY when the input is a FILE-BACKED mapping. `MADV_DONTNEED` on a
        // file-backed range merely drops resident pages (re-fault restores the
        // bytes from the page cache), but on ANONYMOUS memory it is DESTRUCTIVE
        // (pages zero-fill on re-fault). Library / stdin-pipe / test callers pass
        // a `Vec`-backed (anonymous) input, so the reaper MUST stay off there; the
        // production `-dc file` path maps the input file and is safe. Verified by
        // parsing this address's mapping in `/proc/self/maps` (inode != 0).
        let enabled = input.len >= INPUT_RELEASE_MIN_LEN && Self::input_is_file_backed(input.ptr);
        Self {
            base: input.ptr,
            len: input.len,
            released: 0,
            page: reaper_page_size(),
            enabled,
            stride: reaper_stride_for(pool_size),
        }
    }

    /// True iff `ptr` lies in a file-backed VMA (non-zero inode in
    /// `/proc/self/maps`). Conservative: any parse/read failure ⇒ `false`
    /// (reaper disabled ⇒ byte-safe fallback to the old path).
    #[cfg(unix)]
    fn input_is_file_backed(ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let maps = match std::fs::read_to_string("/proc/self/maps") {
            Ok(m) => m,
            Err(_) => return false,
        };
        for line in maps.lines() {
            // `start-end perms offset dev inode pathname`
            let mut it = line.split_whitespace();
            let range = match it.next() {
                Some(r) => r,
                None => continue,
            };
            let mut rp = range.split('-');
            let start = rp.next().and_then(|x| usize::from_str_radix(x, 16).ok());
            let end = rp.next().and_then(|x| usize::from_str_radix(x, 16).ok());
            let (start, end) = match (start, end) {
                (Some(s), Some(e)) => (s, e),
                _ => continue,
            };
            if addr >= start && addr < end {
                // remaining tokens: perms(0) offset(1) dev(2) inode(3) path(4)
                let inode = it.nth(3).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
                return inode != 0;
            }
        }
        false
    }

    #[cfg(not(unix))]
    fn input_is_file_backed(_ptr: *const u8) -> bool {
        false
    }

    /// Release consumed input pages up to `frontier_bit − MARGIN` (page-aligned,
    /// stride-batched). Called from the in-order consumer after the frontier
    /// advances. No-op when disabled, off-unix, or below the stride threshold.
    #[inline]
    fn advance(&mut self, frontier_bit: usize) {
        if !self.enabled {
            return;
        }
        let frontier = (frontier_bit / 8).min(self.len);
        let safe = frontier.saturating_sub(INPUT_RELEASE_MARGIN);
        if safe <= self.released || safe - self.released < self.stride {
            return;
        }
        #[cfg(unix)]
        {
            let start_addr = self.base as usize + self.released;
            let aligned_start = (start_addr + self.page - 1) & !(self.page - 1);
            let aligned_end = (self.base as usize + safe) & !(self.page - 1);
            if aligned_end > aligned_start {
                // SAFETY: [base, base+len) is the live input mmap held by `drive`
                // for the whole decode. MADV_DONTNEED on a file-backed range drops
                // only resident pages (re-access re-faults from the page cache) —
                // it frees no Rust-owned memory and cannot alter decoded bytes.
                unsafe {
                    libc::madvise(
                        aligned_start as *mut libc::c_void,
                        aligned_end - aligned_start,
                        libc::MADV_DONTNEED,
                    );
                }
            }
        }
        self.released = safe;
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
    verbose: bool,
) -> Result<(u32, usize), FetchError> {
    drive_impl(
        input,
        writer,
        out_fd,
        parallelization,
        configuration,
        None,
        None,
        verbose,
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
    verbose: bool,
) -> Result<(u32, usize), FetchError> {
    drive_impl(
        input,
        writer,
        out_fd,
        parallelization,
        configuration,
        None,
        Some(bytes_written_out),
        verbose,
    )
}

/// THIN-T1 serial driver — the production T1 single-member path.
///
/// A minimal clean driver: a
/// single SERIAL rolling-window loop over chunks calling the SHARED
/// `decode_chunk` kernel with NO parallel bulk whatsoever — no `block_finder`,
/// no `BlockFetcher`/prefetch, no `WindowMap` publish/handoff, no marker arming
/// (never fires — every chunk gets a full 32 KiB predecessor window), no
/// `ThreadPool`/`std::thread::scope`, no `Arc` chunk lifecycle, no pending
/// queue.
///
/// It does NOT run a speculative Pass 1 and
/// has NO separate untimed Phase A: the rolling window IS derived inline as the
/// loop advances (each chunk's true trailing 32 KiB becomes the next chunk's
/// dict), so there is exactly ONE decode of the stream. Output is byte-correct (real
/// rolling windows) and markers never fire — both verified by the caller's
/// CRC/size check.
#[cfg(parallel_sm)]
pub fn drive_thin_t1_oracle<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    configuration: ChunkConfiguration,
    mut bytes_written_out: Option<&mut usize>,
) -> Result<(u32, usize), FetchError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    // Activate the T1 resident-pool scope for THIS
    // (single, serial) decode thread. While active, the chunk output buffer is
    // taken from / returned to the manual per-thread pool AND its reserve is
    // pinned to the fixed resident cap, so consecutive chunks decode into ONE
    // resident buffer instead of a fresh ratio-sized alloc per chunk.
    // SCOPED to this thread (RAII): the T>1 parallel workers never set it, so the
    // faithful per-chunk reserve/pool path is unchanged at T>1.
    let _resident = crate::decompress::parallel::chunk_buffer_pool::T1ResidentScope::enter();

    let total_bits = input.len() * 8;
    // STREAMING RSS FIX (thin-T1): the serial T1 driver reads the mmap'd input
    // strictly front-to-back and never re-reads behind the decode cursor (no
    // block-finder, no prefetch; the rolling window is a private 32 KiB copy),
    // so consumed input pages can be released the moment `cur` passes them — the
    // SAME mechanism the parallel consumer uses (`InputReaper` in `consumer_loop`).
    // Without it a T1 decode holds the ENTIRE input mmap resident until process
    // exit: peak RSS grows to ~= input size (~= output size for near-incompressible
    // corpora), ~20-24x rapidgzip which streams a bounded reader window.
    // Byte-transparent: MADV_DONTNEED on the file-backed mapping only drops
    // resident pages (re-fault restores the identical page-cache bytes) and cannot
    // change decoded output; auto-disabled for Vec-backed (anonymous) library/test
    // inputs and for inputs below the arm threshold — see `InputReaper::new`.
    // T1 serial driver: single-thread → 1-way TLB shootdown, so the tight low-T
    // stride is free here (pool_size = 1).
    let mut input_reaper = InputReaper::new(unsafe { InputSlice::from_slice(input) }, 1);
    // Compressed-bytes-per-chunk stride from the chunk configuration (T1 default
    // = 1 MiB target, warm output-buffer recycling). Clamp to ≥64 KiB so a
    // pathological tiny config can't thrash the loop.
    let stride_bits = configuration.split_chunk_size.max(64 * 1024) * 8;

    let mut total_crc = CRC32Calculator::new();
    let mut total_size = 0usize;

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
        input_reaper.advance(cur);
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

    THIN_T1_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((total_crc.crc32(), total_size))
}

/// Production-routing proof: incremented once per [`drive_thin_t1_oracle`] call
/// (the thin single-chunk/T1 production path). A test snapshots it around a T1
/// decode to assert the thin spine — not the parallel scaffold — handled it.
#[cfg(parallel_sm)]
pub static THIN_T1_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    bytes_written_out: Option<&mut usize>,
    // `--verbose` flag threaded from the CLI's `args.verbose` (main.rs →
    // decompress::io → down the call graph). Retained for call-site
    // compatibility; the end-of-decode internal-stats dump it used to gate
    // has been removed.
    _verbose: bool,
) -> Result<(u32, usize), FetchError> {
    let total_bits = input.len() * 8;
    // Vendor `availableCores()` (AffinityHelpers.hpp:18-21): avoid pinning
    // more workers than physical cores — SMT siblings collide on cache.
    let pool_size = parallelization.max(1);

    // ── m_blockFinder (vendor GzipChunkFetcher.hpp:283) ─────────────
    // AMD/Zen2 T3 TAIL-TAPER (byte-transparent; gated). At 3 workers the in-order
    // pull-queue's makespan is set by a STRAGGLER: the last decode round leaves 2
    // of 3 workers idle while one finishes a full 4 MiB chunk. A GLOBAL finer chunk
    // size costs more (its own optimum is 4 MiB; at 3 workers whole-file per-chunk
    // overhead > tail saved), so we subdivide ONLY the trailing region: the bulk
    // keeps 4 MiB, the last few chunks run short so the final round's straggler is
    // small. NO vendor counterpart — rapidgzip's GzipBlockFinder uses one uniform
    // spacing (its last chunk is just the remainder). Scoped to cpu_is_amd() &&
    // pool_size==3; every other cell/arch takes the byte-identical uniform finder.
    let block_finder = Arc::new(build_block_finder(
        configuration.split_chunk_size,
        total_bits,
        pool_size,
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
    // TRANSLITERATION (vendor BlockFetcher.hpp:181-183, :185): match rapidgzip's
    // constructor sizing exactly.
    //   m_cache( std::max( size_t( 16 ), m_parallelization ) )
    //   m_prefetchCache( 2 * m_parallelization )
    //   threadPoolSaturated: m_prefetching.size() + 1 >= m_parallelization
    // gzippy previously diverged: cache_capacity = pool_size*2 (vendor is
    // max(16,pool)) and a prefetch-saturation lever with no vendor counterpart.
    // Both deleted; sizing now matches vendor.
    let cache_capacity = std::cmp::max(16, pool_size);
    let prefetch_capacity = pool_size * 2;
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
    // overhead at T1. Apply the `==1 ? 0` ONLY at pool construction;
    // the cache/prefetch sizing above keeps reading the true `pool_size` (vendor sizes
    // m_prefetchCache off m_parallelization, which stays 1 → 2, not 0).
    let pool_threads = if pool_size == 1 { 0 } else { pool_size };
    // Build the decode pool with an EMPTY pinning map — faithful to rapidgzip's
    // decode pool (`BlockFetcher.hpp:185` constructs the pool with a thread COUNT
    // only, empty pinning → the OS scheduler places threads). gzippy previously
    // added speculative worker-pinning (`with_pinning_for_capacity`) rapidgzip
    // never had; on an SMT box the default `get_core_ids()` cycling order packed
    // workers onto SMT-sibling logical cores of the same physical core. The
    // unpinned pool lets the OS spread the workers across distinct
    // physical cores on its own — so the pinning is deleted.
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

    // The in-order consumer (this thread runs `consumer_loop` inline) is left
    // unpinned — faithful to rapidgzip, which never pins its reader thread; the
    // OS schedules it alongside the unpinned decode workers (see the decode-pool
    // construction above).
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

    // Faithful overlap writer: drain + join the background output writer so
    // ALL bytes are on the fd and any write error becomes terminal BEFORE the
    // trailer CRC/ISIZE is returned for verification. No-op when the overlap
    // writer was never used (inline writev path). Join even on a consumer
    // error so the writer thread does not outlive the run.
    #[cfg(all(unix, parallel_sm))]
    let writer_result = crate::decompress::parallel::output_writer::finish();
    // Record bytes streamed so far for the multi-member resume path. The
    // consumer writes chunks in order and only after they fully validate, so on
    // a finder/decode error at a member boundary this count is exactly the
    // contiguous, correct prefix already on the sink. Set on BOTH paths.
    if let Some(out) = bytes_written_out {
        *out = total_size;
    }
    consumer_result?;
    // Surface any background-writer error as a terminal decode error (the
    // bytes are on the fd / partially written, same contract as the inline
    // writev path which returns Err on a failed write).
    #[cfg(all(unix, parallel_sm))]
    writer_result.map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;

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

    // Drain the prefetch cache at end-of-decode. Any entries still sitting in
    // the prefetch cache were dispatched but never consumed by the consumer —
    // mirror of vendor's destructor path at BlockFetcher.hpp:199-201 which
    // counts those as `cache_unused_entry`.
    block_fetcher.clear_prefetch_cache();

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
    // STEP-3: progressively release consumed input mmap pages behind the
    // in-order frontier so process-exit teardown of the input is cheap
    // (overlapped with decode). Byte-transparent — see `InputReaper`.
    let mut input_reaper = InputReaper::new(input, pool_size);
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
    // pipeline moves on (a 1-chunk defer is insufficient at T>1: a byte diff
    // appears at a chunk boundary when a lone emit races worker fill of
    // successor buffers). The depth + lone-drain
    // policy is T1-tuned (depth 0 + drain-lone at pool_size==1, where the
    // pool runs inline with no racing workers); see `RecycleDeferral`.
    let mut recycle_deferral = RecycleDeferral::new(pool_size);
    // STAGE-2d MULTI-MEMBER GRID: arm the per-member CRC32 + ISIZE verifier so
    // `drain_one_pending` runs the vendor `processCRC32` steps (design §4) over
    // resolved chunks in decode order. Single-member drive leaves this None.
    if configuration.multi_member {
        recycle_deferral.member_verifier = Some(
            crate::decompress::parallel::chunk_data::MemberVerifier::new(
                configuration.crc32_enabled,
            ),
        );
    }
    // The vendor's `processNextChunk` returns one chunk per call; the
    // caller loops in `ParallelGzipReader::read`. We inline that loop
    // here so the local-state mutation (post-process queue + writer +
    // CRC) stays simple.
    #[allow(clippy::while_let_loop)] // faithful port of vendor processNextChunk loop
    loop {
        // BlockFetcher.hpp:427 — opportunistically promote ready prefetches
        // so workers don't idle while the consumer waits on a different key.
        {
            block_fetcher.process_ready_prefetches();
        }
        // Vendor `waitForReplacedMarkers` (:497-511): non-blocking harvest of
        // ready marker-replace futures on every consumer iteration.
        harvest_ready_postprocess(&mut prefetch_post_inflight, &mut eager_completed);

        // Vendor GzipChunkFetcher.hpp:318 — `m_blockFinder->get(m_nextUnprocessedBlockIndex)`.
        let phase_blockfind_result = block_finder.get(next_unprocessed_block_index);
        let next_block_offset = match phase_blockfind_result {
            (Some(offset), GetReturnCode::Success) => offset,
            // Vendor GzipChunkFetcher.hpp:320-327 — EOF when no offset
            // or offset past end of file.
            _ => break,
        };
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
            .unwrap_or(true);

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
        // the 221 MB / 3-partition fixture.
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
            // EVENT-DRIVEN wakeup: block on the pool's task-completion event
            // (with a 1ms liveness fallback) instead of a fixed 1ms poll, so the
            // consumer re-pumps the prefetch horizon at ~0 latency whenever a
            // worker frees up. Monotonic gen ⇒ a completion during `pump` makes
            // the wait return immediately (no lost wakeup).
            let mut last_completion_gen = thread_pool.completion_gen();
            let wait_ready = || {
                last_completion_gen = thread_pool
                    .wait_for_completion(last_completion_gen, std::time::Duration::from_millis(1));
            };
            let phase_decode_wait_result =
                block_fetcher.try_take_prefetched_pumping(&partition_offset, pump, wait_ready);
            if let Some(Ok(arc)) = phase_decode_wait_result {
                // Vendor GzipChunkFetcher.hpp:646-648,670-684 — accept when
                // `matchesEncodedOffset(blockOffset)`; else discard and
                // re-issue at the real offset via `get(blockOffset, ...)`.
                if arc.matches_encoded_offset(decode_start) {
                    // ACCEPT: move the `Arc` (no extra refcount) into the consumer path.
                    chunk_arc_from_partition = Some(arc);
                } else {
                    // Vendor GzipChunkFetcher.hpp:646-648 — partition-keyed
                    // prefetch present but `!matchesEncodedOffset(blockOffset)`.
                }
                // If !matches OR the safety guard rejected, the Arc
                // is dropped here (vendor "throws away" the
                // partition-keyed result and re-issues at real offset
                // below — line 654). Same for Some(Err(...)) —
                // vendor's `catch ( const NoBlockInRange& )` at
                // GzipChunkFetcher.hpp:604-609 silently discards
                // prefetch failures.
            }
        }

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
                // the dispatcher went COMPLETELY silent, stalling on-demand at
                // the first un-prefetched partition. Driving here keeps the
                // prefetch horizon advancing one call per consumed chunk,
                // exactly like vendor.
                // Miss-path ordering is untouched: on-demand submit still
                // precedes the dispatch drive (vendor BlockFetcher.hpp:276
                // before :297), so the head-of-line task cannot queue behind
                // fresh prefetches.
                if should_drive_prefetch {
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
                let (chunk_arc_result, _prefetched) = {
                    // EVENT-DRIVEN wakeup while blocked on the on-demand decode
                    // (see the prefetch-hit branch above): wake on the pool's
                    // task-completion event, not a fixed 1ms poll.
                    let mut last_completion_gen = thread_pool.completion_gen();
                    let wait_ready = || {
                        last_completion_gen = thread_pool.wait_for_completion(
                            last_completion_gen,
                            std::time::Duration::from_millis(1),
                        );
                    };
                    block_fetcher.get_with_prefetch(
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
                        wait_ready,
                    )
                };
                chunk_arc_result.map_err(FetchError::Decode)?
            }
        };
        // Take ownership when we hold the only Arc; otherwise clone the
        // inner ChunkData unless resolve-ahead already submitted post-process
        // (borrow the `Arc` through window publish, `recv` at dispatch).
        let chunk_holder = {
            match Arc::try_unwrap(chunk_arc) {
                Ok(shared) => ConsumerChunkHold::Owned(SharedChunkData::into_inner(shared)),
                Err(arc) => {
                    let real_offset = arc.encoded_offset_bits;
                    let resolved_pred_matches = arc.markers_resolved
                        && confirmed_predecessor_window(window_map, decode_start)
                            .map(|(pred_key, _)| arc.resolved_pred_key == Some(pred_key))
                            .unwrap_or(false);
                    if resolved_pred_matches
                        || prefetch_post_inflight.contains_key(&real_offset)
                        || eager_completed.contains_key(&real_offset)
                    {
                        ConsumerChunkHold::Deferred { arc }
                    } else {
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
        input_reaper.advance(furthest_decoded_bit);

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
            if let Some((_pred_key, pred)) = confirmed_predecessor_window(window_map, handoff_bit) {
                let bytes = materialize_window(&pred);
                publish_end_window_before_post_process(window_map, chunk, bytes.as_ref());
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
            // recompute was REDUNDANT. get_last_window is deterministic given the same predecessor, so
            // the eager-published window is byte-identical; skipping the recompute is byte-exact.
            {
                let window_bytes = materialize_window(&window);
                publish_end_window_before_post_process(window_map, chunk, window_bytes.as_ref());
            }
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
        // post-process and queued Async writes — ordering skew vs vendor.
        {
            harvest_ready_postprocess(&mut prefetch_post_inflight, &mut eager_completed);
            let (mut chunk, eager_already_done) = chunk_holder.into_chunk_data(
                &mut prefetch_post_inflight,
                &mut eager_completed,
                consumer_pred_key,
            );
            let pool_resolved = chunk_matches_consumer_pred(&chunk, consumer_pred_key);
            if let Some(window) = predecessor_window_for_postprocess {
                if !(eager_already_done || pool_resolved)
                    && !chunk.data_with_markers.is_empty()
                    && !chunk.markers_resolved
                {
                    let reuse = match prefetch_post_inflight.remove(&chunk_offset_pre_set) {
                        Some((eager_pred_key, _, eager_rx))
                            if consumer_pred_key == Some(eager_pred_key) =>
                        {
                            Some(eager_rx)
                        }
                        Some(_) => None,
                        None => None,
                    };
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
                        );
                        chunk = recv_post_process_blocking(rx, partition_idx_for_trace, overlap)?;
                    }
                }
            }
            // Vendor `setEncodedOffset(*nextBlockOffset)` after post-process.
            if chunk.encoded_offset_bits != handoff_bit && chunk.matches_encoded_offset(handoff_bit)
            {
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
            pending.push_back(PendingWrite::Ready {
                idx: partition_idx_for_trace,
                chunk,
                cache_key: next_block_offset,
                handoff_bit,
            });
        }

        // Vendor parity: write each post-process-complete chunk as soon as
        // it reaches the FIFO head (GzipChunkFetcher.hpp:333-342 writes
        // inline after `postProcessChunk` returns). Previously gzippy only
        // drained when `pending.len() > pool_size`, leaving Ready chunks
        // holding warm rpmalloc buffers until the cap filled — a
        // buffer-return-latency gap (chunk_buffer_pool.rs:176-179).
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
    // STAGE-2d MULTI-MEMBER GRID: every chunk has now been fed to the per-member
    // verifier in decode order. The stream must have ended exactly at a member
    // footer (running member carries zero pending bytes) — a non-zero remainder
    // is a torn final member. No-op for single-member drive (verifier is None).
    if let Some(v) = recycle_deferral.member_verifier.take() {
        v.finish().map_err(|e| {
            FetchError::Decode(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("multi-member verify: {e}"),
            )))
        })?;
    }
    Ok(())
}

/// Build the chunk-partitioning [`GzipBlockFinder`], applying the AMD/Zen2 T3
/// TAIL-TAPER when gated. Returns the byte-identical uniform vendor finder for
/// every non-AMD-T3 caller (taper disabled). See the call site in `drive` for the
/// mechanism.
#[cfg(parallel_sm)]
fn build_block_finder(
    spacing_in_bytes: usize,
    total_bits: usize,
    pool_size: usize,
) -> GzipBlockFinder {
    use crate::decompress::parallel::single_member::cpu_is_amd;
    // (chunks, divisor). chunks = # of trailing full spacings to subdivide;
    // divisor = spacing / divisor for that region. Gated default for AMD T3.
    // Per-T tail-taper schedule (AMD/Zen2 only): T3=5:2, T4=8:2, T7=8:2.
    // Non-monotonic optimum: 3:2 under-tapers (leaves a full straggler), 10:2
    // over-tapers (per-chunk overhead at low workers > tail saved) — so the
    // taper depth is gated per-T, not a smooth formula. Every other T and every
    // non-AMD arch takes the byte-identical uniform finder.
    let gated: Option<(usize, usize)> = if cpu_is_amd() {
        match pool_size {
            3 => Some((TAIL_TAPER_CHUNKS_DEFAULT, TAIL_TAPER_DIVISOR_DEFAULT)),
            4 | 7 => Some((TAIL_TAPER_CHUNKS_T4_T7, TAIL_TAPER_DIVISOR_DEFAULT)),
            _ => None,
        }
    } else {
        None
    };
    let (taper_chunks, divisor) = gated.unwrap_or((0, 0));
    GzipBlockFinder::new_tail_tapered(
        /* first_block_offset_in_bits = */ 0,
        spacing_in_bytes,
        Some(total_bits),
        taper_chunks,
        divisor,
    )
}

/// AMD/Zen2 T3 tail-taper gated defaults (# trailing full spacings subdivided;
/// spacing divisor): subdividing the last 5 spacings by 2 (4 MiB → 2 MiB in the
/// trailing region). Fewer tapered chunks leave a full-4 MiB straggler in the
/// final round; more pay per-chunk overhead that at 3 workers exceeds the tail
/// saved (the same wall a GLOBAL finer-chunk configuration hits).
#[cfg(parallel_sm)]
const TAIL_TAPER_CHUNKS_DEFAULT: usize = 5;
#[cfg(parallel_sm)]
const TAIL_TAPER_DIVISOR_DEFAULT: usize = 2;

/// AMD/Zen2 T4 & T7 tail-taper chunk count (divisor reuses
/// [`TAIL_TAPER_DIVISOR_DEFAULT`] = 2, i.e. 8:2): subdividing the last 8 spacings
/// by 2 for both the T4 (4 MiB base) and T7 (2.5 MiB base) cells. Response is
/// non-monotonic — 3:2 under-tapers (a full straggler survives the final round),
/// 10:2 over-tapers (per-chunk overhead at 4/7 workers exceeds the tail saved).
/// These two cells carry the largest last-wave tail imbalance; the taper shrinks
/// the trailing chunks so the final decode round finishes fast.
#[cfg(parallel_sm)]
const TAIL_TAPER_CHUNKS_T4_T7: usize = 8;

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
    // OVERSHOOT FIX (task #12). `min_gap` rejects a stop-hint candidate too close
    // to `floor` so we don't decode a degenerate tiny chunk. It was `spacing`,
    // which ALSO rejected the next partition boundary P+1 when `floor` is an
    // OVERSHOOT TAIL just past P (gap to P+1 = spacing − overshoot ≈ 0.99·spacing
    // < spacing) — bumping the on-demand decode to P+2, a 2× chunk (~16-21MB)
    // decoded SYNCHRONOUSLY on the consumer's critical path (head-of-line
    // stalls). Half a spacing still skips the
    // genuinely-tiny gaps (the 5-bit test case) while accepting a near-full P+1.
    // Size off the SMALLEST guess stride (the tapered tail stride when the AMD-T3
    // tail-taper is on) so a legitimately-short tapered chunk is not rejected and
    // merged back to full size. Reduces to spacing/2 when the taper is off.
    let min_gap = (block_finder.min_guess_spacing_bits() / 2).max(8);

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
fn submit_decode_to_pool(
    thread_pool: &Arc<ThreadPool>,
    input: InputSlice,
    params: DecodeParams,
    window_map: &WindowMap,
    block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> mpsc::Receiver<Result<ChunkArc, ChunkDecodeError>> {
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
/// ⚠️ SUPERSEDED / DEAD CODE — do NOT read the "CLONE" note below as the
/// live behavior. The production resolve-ahead path is
/// `queue_prefetched_marker_postprocess` → `submit_post_process_from_prefetch`
/// → `run_post_process_in_place`, which mutates the SHARED `ChunkData` IN
/// PLACE via `SharedChunkData` (`Arc<UnsafeCell<ChunkData>>` = vendor's
/// `shared_ptr`), byte-for-byte faithful to vendor `GzipChunkFetcher.hpp:579-582`
/// — NO clone. The residual `SharedChunkData::take_or_clone` is a cold
/// correctness fallback. This function's CLONE approach was retired; kept only
/// as a reference for the abandoned eager-probe shape.
///
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
    queue_prefetched_marker_postprocess(block_fetcher, window_map, thread_pool, in_flight, &[], &[])
}

#[cfg(parallel_sm)]
fn submit_post_process_to_pool(
    thread_pool: &Arc<ThreadPool>,
    chunk: ChunkData,
    pred_key: usize,
    predecessor_window: Window,
) -> mpsc::Receiver<ChunkData> {
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
    // Retained for call-site/signature parity with vendor `decodeBlock`; the
    // former `decodeAndMeasureBlock` timing that consumed it was removed.
    _block_fetcher: &Arc<BlockFetcher<usize, ChunkArc, FetchMultiStream, ChunkDecodeError>>,
    configuration: ChunkConfiguration,
) -> Result<ChunkArc, ChunkDecodeError> {
    // SAFETY: `drive`'s contract — input outlives the thread pool.
    let input_bytes: &[u8] = unsafe { input.as_slice() };

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
    let until_exact = window_at_offset.is_some() && params.stop_hint_is_exact;

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
        decode_chunk_with_until_exact(
            input_bytes,
            params.start_bit,
            params.stop_hint_bit,
            &bytes,
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

    // Wrap in `ChunkArc` to match BlockFetcher's `Value` type (vendor's
    // `std::shared_ptr<BlockData>` at BlockFetcher.hpp:46).
    chunk_result.map(SharedChunkData::new)
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
    if dwm_len_pre > 0 {
        // Always fused resolve+narrow+CRC (64 KiB u8 LUT, ONE traversal). The
        // old sub-128Ki-element branch ran `resolve_markers_u16` (128 KiB u16
        // LUT) plus a second narrow pass — Fulcrum T8 showed +411ms
        // wall-critical vs rapidgzip in marker-resolve despite only 6/35 chunks
        // on that path. The narrowed-CRC second touch is now folded INTO the
        // resolve+narrow pass (byte-exact, universal, strict work reduction):
        // each segment's narrowed bytes are CRC'd while hot in cache instead of
        // re-read in a separate `update_narrowed_crc` whole-buffer pass.
        chunk.resolve_and_narrow_markers_in_place_crc(predecessor_window);
    }
    // CRC folded above for the marker path; when `dwm_len_pre == 0` there are no
    // narrowed bytes (`narrowed_len == 0`), so there is nothing to CRC here.
    // Vendor `applyWindow` = narrow (DecodedData.hpp:325-363) → swap + in-place
    // VectorViews (:365-388). There is NO output-size copy: rapidgzip's narrowed
    // marker buffers ARE the output views, recycled when `writeAll` completes.
    // gzippy mirrors this — the narrowed marker bytes stay in `data_with_markers`
    // (the u8 view of the u16 backing) with `narrowed_len` set, and the consumer
    // emits them zero-copy via `append_narrowed_iovecs` (chunk_data.rs:1609).
    // We therefore DROP the redundant `merge_resolved_markers_into_data` full-
    // output memcpy AND the eager `recycle_markers_after_resolution`: the marker
    // segments must outlive the consumer's writev, so recycling is DEFERRED
    // behind the write via the fd write path → `recycle_decoded_buffers`
    // (fd_vectored_write.rs / chunk_data.rs:1621, which frees BOTH `data` and
    // `data_with_markers`). `populate_subchunk_windows` runs against the un-merged
    // markers — `copy_window_at_chunk_offset` already branches on `narrowed_len>0`.
    chunk.populate_subchunk_windows(predecessor_window);
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
    block_fetcher.process_ready_prefetches();
    let contents = block_fetcher.prefetch_cache_contents_sorted();
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
        // + prev.encoded_size_bits`) is WRONG: for range-speculative chunks that key
        // names a DIFFERENT published window, and the dense window map usually
        // contains it, so the chunk would resolve against the wrong predecessor
        // window → CRC32 mismatch.
        let handoff_bit = chunk_consumer_handoff_bit(arc.as_ref());
        // The RESOLVE_AHEAD_* / HANDOFF_WINDOW_PUBLISHED counters track one
        // attempt per chunk entering the predecessor lookup; the eligibility gate
        // above (`chunk_may_resolve_markers_early`) pre-screens window presence, so
        // they track ATTEMPTS. Both are live on the production resolve-ahead path.
        let Some((pred_key, predecessor_window)) =
            confirmed_predecessor_window(window_map, handoff_bit)
        else {
            continue;
        };
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
        // resolve-ahead submitted ahead-work for this chunk.
        submitted += 1;
    }
    submitted
}

/// In-place pool post-process on a cache-shared chunk (vendor
/// `GzipChunkFetcher.hpp:579-582` `chunkData->applyWindow`).
#[cfg(parallel_sm)]
fn run_post_process_in_place(arc: &SharedChunkData, pred_key: usize, predecessor_window: Window) {
    let bytes = materialize_window(&predecessor_window);
    arc.with_mut(|chunk| {
        resolve_chunk_markers_on_chunk(chunk, pred_key, bytes.as_ref());
    });
}

/// Pool-side execution of a uniquely-owned post-process task (consumer head
/// chunk where `Arc::try_unwrap` succeeded on `get()`).
#[cfg(parallel_sm)]
fn run_post_process_task(
    mut chunk: ChunkData,
    pred_key: usize,
    predecessor_window: Window,
) -> ChunkData {
    let bytes = materialize_window(&predecessor_window);
    resolve_chunk_markers_on_chunk(&mut chunk, pred_key, bytes.as_ref());
    chunk
}

/// Narrow `src: &[u16]` into `dst: &mut U8`, appending bytes. All values
/// in `src` MUST be < 256 (post-`apply_window` invariant). AVX2 path
/// uses `_mm256_packus_epi16` for 16-lane parallel narrowing
/// (saturating pack — since values are already < 256, saturation is
/// a no-op). Scalar tail handles the remainder.
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
/// equivalent rejections, observable in `--verbose` output.
#[cfg(parallel_sm)]
pub static SPECULATIVE_PHANTOM_EOS_REJECTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub static COORDINATOR_BOUNDARY_SEARCH_RUNS: std::sync::atomic::AtomicU64 =
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
    use std::sync::mpsc::TryRecvError;
    let offsets: Vec<usize> = in_flight.keys().copied().collect();
    for offset in offsets {
        let Some((pred_key, cache_key, rx)) = in_flight.remove(&offset) else {
            continue;
        };
        match rx.try_recv() {
            Ok(()) => {
                completed.insert(offset, pred_key);
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
                        return (SharedChunkData::take_or_clone(arc), true);
                    }
                    completed.insert(real_offset, eager_pred_key);
                }
                match inflight.remove(&real_offset) {
                    Some((eager_pred_key, _, rx)) if pred_key == Some(eager_pred_key) => {
                        let _ = rx.recv();
                        (SharedChunkData::take_or_clone(arc), true)
                    }
                    Some((eager_pred_key, cache_key, rx)) => {
                        inflight.insert(real_offset, (eager_pred_key, cache_key, rx));
                        (SharedChunkData::take_or_clone(arc), false)
                    }
                    None => (SharedChunkData::take_or_clone(arc), false),
                }
            }
        }
    }
}

// ── Eager post-process probe ────────────────────────────────────────────
// Port of rapidgzip's `queuePrefetchedChunkPostProcessing`
// (GzipChunkFetcher.hpp:520-551): during a consumer stall, submit
// apply_window for prefetched SUCCESSOR chunks whose predecessor window
// is already published, instead of waiting to reach them in order.
//
// These counters distinguish "many eager submits" from "the cache is empty
// during the stall" — i.e. whether the prefetch depth reaches ahead far enough
// for eager post-processing to find ready successors.

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
    let mut chunk = decode_chunk_window_absent(input, decode_start, stop_hint_bit, configuration)?;
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
    // STAGE-2d: in the multi-member grid the decoder INTENTIONALLY walks past a
    // member-final BFINAL (footer → next header → continue), so an early
    // `encoded_end < stop_hint_bit` is a legitimate clean member-boundary / EOF
    // stop, not a phantom end-of-stream to reject. The `_multi` decode already
    // consumed the footer and (if present) the next header; discarding here would
    // wrongly drop a valid boundary-crossing chunk. Skip the single-member reject.
    if !configuration.multi_member && encoded_end < stop_hint_bit {
        // Byte-align: deflate pads to a byte boundary before the gzip footer.
        let end_byte = encoded_end.div_ceil(8);
        // Gzip footer is 8 bytes (CRC32 LE + ISIZE LE, RFC 1952 §2.3.1).
        let footer_end = end_byte.saturating_add(8);
        if footer_end + 2 <= input.len() {
            // At least 2 bytes after the footer — check for valid gzip magic,
            // and when a 3rd byte is in-bounds also require CM=8 (deflate),
            // matching vendor readHeader more closely. A false-accept only
            // reverts to the consume-time discard, so this is perf-only.
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
        if let Ok(chunk) = try_speculative_decode_candidate(
            input,
            start_bit,
            start_bit,
            stop_hint_bit,
            configuration,
            window_map,
        ) {
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
    if let Some(chunk) = RawBlockFinderCoordinator::with_sync_boundary_search(
        input,
        start_bit,
        max_end,
        |candidate| {
            // worker.scan_candidate span: one full-decode attempt at a
            // single bit position the scan flagged as a valid block
            // candidate. Matched on vendor side by the trace patch
            // around each tryToDecode call inside the alternating loop.
            try_speculative_decode_candidate(
                input,
                candidate.seek_bit,
                candidate.bit_offset,
                stop_hint_bit,
                configuration,
                window_map,
            )
            .ok()
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
                return Ok(chunk);
            }
        }
    }

    Err(ChunkDecodeError::ExactStopMissed {
        requested: start_bit,
        actual: max_end,
    })
}

// ── Pending-write queue (order-preserved output) ─────────────────────────

#[cfg(parallel_sm)]
#[allow(clippy::large_enum_variant)] // boxing ChunkData would add an alloc on the per-chunk write path
#[allow(dead_code)] // the `Async` prefetch-write variant is retained scaffolding, not yet constructed
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

/// The thread-count-tuned T1 lone-drain config (+ the multi-member verifier).
///
/// At T1, drain a lone head Ready immediately; at T>1, hold it.
///
/// At `parallelization == 1` the thread pool runs INLINE (`pool_threads == 0`,
/// see `drive_impl`) with NO concurrent workers, so a lone Ready can drain without
/// the worker-fill race that forces lone-hold at T>1 (a byte diff appears at a
/// chunk boundary when a lone emit races worker fill of successor buffers).
/// Shrinking the T1 live-ChunkData working set 4→1 is byte-exact at T1: the
/// smaller resident output working-set cuts page-faults that feed the kernel.
/// It is INCORRECT at T>1 (a global lone-drain corrupts silesia at T4/T8) so the
/// T1 tuning is GATED on `pool_size == 1`.
#[cfg(parallel_sm)]
struct RecycleDeferral {
    drain_lone: bool,
    /// STAGE-2d MULTI-MEMBER GRID: the running per-member CRC32 + ISIZE
    /// verifier, fed each chunk in the in-order drain (`drain_one_pending`).
    /// `Some` only when the grid driver set `configuration.multi_member`; the
    /// single-member drive leaves it `None` (zero behavior change — the feed is
    /// skipped and the whole-stream `total_crc` remains the single-trailer
    /// oracle). Threaded here because `RecycleDeferral` already reaches every
    /// drain call site, avoiding signature churn on the hot drain path.
    member_verifier: Option<crate::decompress::parallel::chunk_data::MemberVerifier>,
}

#[cfg(parallel_sm)]
impl RecycleDeferral {
    fn new(pool_size: usize) -> Self {
        let t1 = pool_size == 1;
        // T1: drain lone (single inline worker, no worker-fill race).
        // T>1: hold lone (correctness — worker-fill race).
        let drain_lone = t1;
        Self {
            drain_lone,
            member_verifier: None,
        }
    }
}

/// Release a just-written chunk's decoded buffers to the allocator immediately
/// (peak-RSS reduction). The chunk's output has already been synchronously
/// written and its window published, so the buffers are dead (see
/// `ChunkData::free_decoded_buffers`). `chunk` drops at end of scope,
/// releasing it now instead of after the 2-chunk deferral hold —
/// lowering both peak resident bytes AND peak live-ChunkData depth.
#[cfg(parallel_sm)]
fn release_written_chunk(mut chunk: ChunkData) {
    chunk.free_decoded_buffers();
    // chunk drops here, backing Vecs already freed.
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
    let debug_lone = recycle_deferral.drain_lone;
    while matches!(pending.front(), Some(PendingWrite::Ready { .. }))
        && (debug_lone
            || pending.len() >= 2
            || matches!(pending.get(1), Some(PendingWrite::Async { .. })))
    {
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
    // Eager post-process context. When the head
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
    if let Some((_, in_flight, completed)) = eager_ctx.as_mut() {
        harvest_ready_postprocess(in_flight, completed);
    }
    let head = match pending.pop_front() {
        Some(h) => h,
        None => return Ok(()),
    };
    let (_idx, chunk, cache_key, _handoff_bit) = match head {
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
            let overlap = eager_ctx
                .as_mut()
                .map(|ctx| (block_fetcher, window_map, ctx.0, &mut *ctx.1, &mut *ctx.2));
            let chunk = recv_post_process_blocking(rx, idx, overlap)?;
            (idx, chunk, cache_key, handoff_bit)
        }
    };

    // Subchunk window publish at drain (post-process + setEncodedOffset +
    // appendSubchunks already ran in `consumer_loop`).
    publish_subchunk_windows(window_map, &chunk);

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
    let decoded_data_len = chunk.data.len().saturating_sub(chunk.data_prefix_len);
    let payload_bytes = chunk.narrowed_len + decoded_data_len;
    let mut wrote_via_fd = false;
    #[cfg(unix)]
    if let Some(fd) = out_fd {
        if payload_bytes > 0 {
            use crate::decompress::parallel::fd_vectored_write;
            // Zero-copy gather: narrowed marker segments + clean payload
            // (vendor DecodedData.hpp:529 toIoVec). No memcpy — iovecs borrow
            // decode buffers until writev/vmsplice completes.
            let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
            chunk.append_output_iovecs(&mut parts);
            // STAGE-2d MULTI-MEMBER GRID: feed this chunk (in decode order,
            // fully resolved) to the per-member verifier BEFORE it is moved to
            // the writer / recycled. No-op for single-member drive.
            if let Some(v) = recycle_deferral.member_verifier.as_mut() {
                v.feed(&chunk).map_err(|e| {
                    FetchError::Decode(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("multi-member verify: {e}"),
                    )))
                })?;
            }
            // CRC combine before boxing `chunk` for vmsplice lifetime.
            total_crc.append(&chunk.narrowed_crc);
            for stream_crc in &chunk.crc32s {
                total_crc.append(stream_crc);
            }
            let mut iovs = fd_vectored_write::to_io_vec(parts.iter().copied(), &[]);
            #[cfg(target_os = "linux")]
            if fd_vectored_write::is_pipe_fd(fd) {
                fd_vectored_write::write_chunk_payload_to_fd(fd, &mut iovs, chunk)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            } else {
                fd_vectored_write::writev_all_to_fd(fd, &mut iovs)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                release_written_chunk(chunk);
            }
            #[cfg(not(target_os = "linux"))]
            {
                fd_vectored_write::writev_all_to_fd(fd, &mut iovs)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                release_written_chunk(chunk);
            }
            *total_size += payload_bytes;
            #[allow(unused_assignments)]
            {
                wrote_via_fd = true;
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
        let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
        chunk.append_output_iovecs(&mut parts);
        for sl in parts {
            writer
                .write_all(sl)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += sl.len();
        }
    }
    // STAGE-2d MULTI-MEMBER GRID: feed this chunk (buffered-writer path) to the
    // per-member verifier in decode order. No-op for single-member drive.
    if let Some(v) = recycle_deferral.member_verifier.as_mut() {
        v.feed(&chunk).map_err(|e| {
            FetchError::Decode(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("multi-member verify: {e}"),
            )))
        })?;
    }
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

    // Lever G: do NOT re-insert the consumed chunk into the cache.
    // Single-pass forward decode never queries the same key twice, so
    // the post-consume re-insert was strictly wasted bookkeeping (and
    // an extra Arc allocation per chunk).
    let _ = cache_key;
    let _ = block_fetcher;
    release_written_chunk(chunk);
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
        let (_crc, n) = drive(&deflate, &mut out, None, 8, cfg, false).expect("drive");
        assert_eq!(n, payload.len());
        assert_eq!(out, payload);
    }

    /// InputReaper differential on a REAL file-backed mmap (the only path where
    /// the reaper arms — `input_is_file_backed`; Vec-backed tests keep it OFF).
    /// Exercises the low-T 1 MiB stride (`reaper_stride_for(4)`) across ~40
    /// release batches so a mis-tuned stride that dropped a still-referenced page
    /// would corrupt a later chunk. Byte-exact vs flate2 + the drive CRC/size.
    #[test]
    fn reaper_file_backed_byte_exact_multi_stride() {
        use std::io::{Read, Write};
        // ~40 MiB low-redundancy payload (> INPUT_RELEASE_MIN_LEN=16 MiB so the
        // reaper arms; ratio≈1 so the compressed input is also large → many
        // 1 MiB strides fire on the file-backed input).
        let mut payload = Vec::with_capacity(40 * 1024 * 1024);
        for i in 0..(40 * 1024 * 1024u32 / 4) {
            payload.extend_from_slice(&i.wrapping_mul(2654435761).to_le_bytes());
        }
        let deflate = make_deflate(&payload, 6);
        assert!(
            deflate.len() > super::INPUT_RELEASE_MIN_LEN,
            "input must exceed reaper arm threshold to exercise the stride"
        );
        // Write the deflate to a temp file and mmap it: the mmap'd pointer lies
        // in a file-backed VMA (non-zero inode), so `InputReaper::new` arms and
        // MADV_DONTNEED fires on real strides during the decode.
        let mut tf = tempfile::NamedTempFile::new().expect("tempfile");
        tf.write_all(&deflate).expect("write deflate");
        tf.flush().unwrap();
        let file = std::fs::File::open(tf.path()).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let cfg = ChunkConfiguration {
            split_chunk_size: 1024 * 1024,
            max_decoded_chunk_size: 8 * 1024 * 1024,
            crc32_enabled: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        let (_crc, n) = drive(&mmap[..], &mut out, None, 4, cfg, false).expect("drive");
        assert_eq!(n, payload.len(), "decoded size");
        assert_eq!(out, payload, "reaper-active decode must be byte-exact");
        // Independent oracle cross-check.
        let mut reference = Vec::new();
        flate2::read::DeflateDecoder::new(&deflate[..])
            .read_to_end(&mut reference)
            .unwrap();
        assert_eq!(reference, payload, "flate2 self-check");
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
        let (_crc, size) = drive(&deflate, &mut out, None, 4, cfg, false).expect("drive");
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
        let (_crc, size) = drive(&deflate, &mut out, None, 8, cfg, false).expect("drive");
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

    /// Regression for `make profile-decompression-x86_64` (64 MiB
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
        let result = read_parallel_sm(&gz, &mut out, None, 2, chunk_size, false);
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
        let (_crc, size) = drive(&deflate, &mut out, None, 8, cfg, false).expect("drive");
        assert_eq!(size, payload.len(), "size mismatch (suggests early break)");
        assert_eq!(out, payload, "byte mismatch");
    }
}
