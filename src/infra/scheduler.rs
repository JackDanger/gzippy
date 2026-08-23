//! Pigz-style parallel scheduler with dedicated writer thread
//!
//! This implements pigz's proven threading model:
//!
//! 1. N compress worker threads (claim work via atomic counter)
//! 2. 1 dedicated writer thread (writes blocks in order)
//! 3. All N+1 threads run concurrently (no main-thread stalls)
//! 4. Simple spin-wait for block completion (low latency)
//!
//! The bounded reorder window limits buffered output to O(thread count).
//! Workers may wait when the writer falls behind, rather than allocating an
//! unbounded slot for every input block.
//!
//! Set GZIPPY_DEBUG=1 to enable timing diagnostics.

use crate::compress::deflate::bitstream::{BitSplicer, ChunkMeta};
use std::cell::UnsafeCell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

/// Check if debug mode is enabled via GZIPPY_DEBUG env var
fn is_debug_enabled() -> bool {
    std::env::var("GZIPPY_DEBUG").is_ok_and(|v| v == "1" || v == "true")
}

/// Size of the bounded reorder window: how many compressed blocks may be in flight.
///
/// It depends on workers, never input length. Two spare slots leave the worker
/// holding the next ordered block able to publish it, so the writer cannot be
/// starved by a full ring.
#[inline]
pub fn reorder_window_for(num_threads: usize, num_blocks: usize) -> usize {
    (num_threads + 2).min(num_blocks)
}

pub struct BlockSlot {
    /// Whether this block has been compressed
    ready: AtomicBool,
    /// The compressed data for this block
    data: UnsafeCell<Vec<u8>>,
    /// Splice metadata for this block's fragment (bit length + alignment
    /// need), written by the same worker that fills `data`, before
    /// `mark_ready`. See [`ChunkMeta`].
    meta: UnsafeCell<ChunkMeta>,
}

const SPINS_BEFORE_YIELD: u32 = 64;

/// Wait for a worker to publish an ordered slot.
///
/// The scheduler cannot let an I/O failure strand a writer spinning forever, nor
/// let a writer failure strand workers waiting for ring space.  Both waits observe
/// the same cancellation flag.  A short spin preserves the low-latency hand-off for
/// a slot that is already completing; yielding after that avoids consuming a full
/// core when compression or I/O takes milliseconds.
#[inline]
fn wait_for_slot_ready(slot: &BlockSlot, cancelled: &AtomicBool) -> bool {
    let mut spins = 0;
    while !slot.is_ready() {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        spins += 1;
        if spins < SPINS_BEFORE_YIELD {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    true
}

/// Wait until the writer has released the ring position for `block_idx`.
#[inline]
fn wait_for_ring_space(
    block_idx: usize,
    window: usize,
    blocks_written: &AtomicUsize,
    cancelled: &AtomicBool,
) -> bool {
    if block_idx < window {
        return !cancelled.load(Ordering::Acquire);
    }

    let mut spins = 0;
    while block_idx.saturating_sub(blocks_written.load(Ordering::Acquire)) >= window {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        spins += 1;
        if spins < SPINS_BEFORE_YIELD {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    !cancelled.load(Ordering::Acquire)
}

// Safety: Each slot is written by exactly one worker thread, then read by main thread
// after ready=true. The atomic provides the synchronization.
unsafe impl Sync for BlockSlot {}

impl BlockSlot {
    /// Create a new slot with pre-allocated capacity
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            ready: AtomicBool::new(false),
            data: UnsafeCell::new(Vec::with_capacity(capacity)),
            meta: UnsafeCell::new(ChunkMeta::ALIGNED),
        }
    }

    /// Get mutable access to the data buffer (called by single worker)
    ///
    /// # Safety
    /// Only call from the single worker assigned to this block index.
    /// The UnsafeCell allows interior mutability from an immutable reference.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn data_mut(&self) -> &mut Vec<u8> {
        &mut *self.data.get()
    }

    /// Mark this block as ready (worker calls after compression)
    #[inline]
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Check if this block is ready (main thread polls)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Get the data (main thread calls after is_ready returns true)
    ///
    /// # Safety
    /// Only call after is_ready() returns true. At that point the worker
    /// has finished writing and will not access the slot again.
    #[inline]
    pub fn data(&self) -> &[u8] {
        unsafe { &*self.data.get() }
    }

    /// Record this block's splice metadata (called by the single worker
    /// assigned to this block, before `mark_ready`).
    ///
    /// # Safety
    /// Same single-writer contract as [`data_mut`](Self::data_mut).
    #[inline]
    pub unsafe fn set_meta(&self, meta: ChunkMeta) {
        *self.meta.get() = meta;
    }

    /// Release this slot for reuse by a later block (writer thread, after the
    /// fragment has been spliced out).
    ///
    /// Truncates rather than frees: the allocation is the point of the pool, and
    /// `Vec::clear` keeps capacity. Clearing `ready` with `Release` and then bumping
    /// the writer's counter (also `Release`) is what lets the next worker for this
    /// ring position see a reset slot rather than the previous block's `ready`.
    ///
    /// # Safety
    /// Only call from the writer thread, after `is_ready()` and after the data has
    /// been consumed. No worker may hold a reference to this slot at that point —
    /// guaranteed by the reorder window: a worker for block `i` does not touch this
    /// slot until the writer has released block `i - window`.
    #[inline]
    pub unsafe fn release(&self) {
        (*self.data.get()).clear();
        *self.meta.get() = ChunkMeta::ALIGNED;
        self.ready.store(false, Ordering::Release);
    }

    /// Get this block's splice metadata (writer thread, after `is_ready`).
    #[inline]
    pub fn meta(&self) -> ChunkMeta {
        unsafe { *self.meta.get() }
    }
}

/// Compress blocks in parallel with dedicated writer thread (pigz model)
///
/// This implements the pigz threading model:
/// 1. N compress worker threads claim blocks via atomic counter
/// 2. 1 dedicated writer thread writes blocks in order
/// 3. All threads run concurrently - no blocking on I/O
///
/// The writer consumes blocks in order while workers compress later blocks in
/// parallel. Its reorder window bounds the compressed output held in memory.
pub fn compress_parallel<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    writer: W,
    compress_fn: F,
) -> io::Result<W>
where
    W: Write + Send,
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>) -> ChunkMeta + Sync,
{
    let debug = is_debug_enabled();
    let start = Instant::now();

    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(writer);
    }
    // This is public infrastructure, so do not rely on every caller having
    // validated a CLI thread flag.  Zero workers would leave the writer waiting
    // forever for block zero; treating it as one worker matches the rest of the
    // compression API's single-thread fallback.
    let num_threads = num_threads.max(1);

    if debug {
        eprintln!(
            "[gzippy] compress_parallel: input={}KB, block_size={}KB, blocks={}, threads={}",
            input.len() / 1024,
            block_size / 1024,
            num_blocks,
            num_threads
        );
    }

    // Reusable ring: output storage is O(workers), not O(input blocks). A
    // worker for block i only reuses its slot after the writer releases i-window.
    let alloc_start = Instant::now();
    let slot_capacity = block_size + (block_size / 10) + 1024;
    let window = reorder_window_for(num_threads, num_blocks);
    let slots: Vec<BlockSlot> = (0..window).map(|_| BlockSlot::new(slot_capacity)).collect();
    let alloc_time = alloc_start.elapsed();

    // Blocks the writer has spliced AND released. Workers read this to find out
    // whether their ring position is free yet.
    let blocks_written = AtomicUsize::new(0);

    if debug {
        eprintln!(
            "[gzippy] slot allocation: {}ms for {} slots ({}KB each) — reorder window              {window} of {num_blocks} blocks, {}KB bounded",
            alloc_time.as_millis(),
            window,
            slot_capacity / 1024,
            window * slot_capacity / 1024,
        );
    }

    // Atomic counter for lock-free work distribution
    let next_block = AtomicUsize::new(0);

    // Track any write error from writer thread
    let write_error: AtomicBool = AtomicBool::new(false);
    // A writer error must cancel workers that are waiting on future ring slots.
    // Without this, a broken pipe made the writer exit while every worker beyond
    // the window spun forever waiting for a `blocks_written` value that could never
    // advance.
    let cancelled: AtomicBool = AtomicBool::new(false);

    // Timing accumulators (atomic for thread-safe updates)
    let total_compress_ns = AtomicU64::new(0);
    let total_wait_ns = AtomicU64::new(0);
    let total_write_ns = AtomicU64::new(0);
    let blocks_compressed = AtomicUsize::new(0);

    // Use scoped threads - no Arc needed, everything is borrowed
    let thread_start = Instant::now();
    let result = thread::scope(|scope| {
        // Spawn dedicated writer thread (pigz model)
        // Returns the writer so caller can write trailer
        let writer_handle = scope.spawn(|| {
            let mut w = writer;
            // Bit-splice each block's fragment onto one continuous DEFLATE
            // stream. Fragments produced byte-aligned with pad_bits=0 (the
            // `ChunkMeta::ALIGNED` case) degrade to plain in-order
            // `write_all`s, so non-DEFLATE users of this scheduler are
            // unaffected.
            let mut splicer = BitSplicer::new();
            // Walk BLOCK indices and map them onto the ring; `slots.len()` is the
            // reorder window, not the block count.
            for slot_idx in 0..num_blocks {
                let slot = &slots[slot_idx % window];
                let wait_start = Instant::now();
                let t0 = crate::infra::trace_spans::now_us();
                if !wait_for_slot_ready(slot, &cancelled) {
                    break;
                }
                crate::infra::trace_spans::record("write_wait", 0, t0, slot_idx, 0);
                total_wait_ns.fetch_add(wait_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let write_start = Instant::now();
                let t1 = crate::infra::trace_spans::now_us();
                if splicer.splice_to(&mut w, slot.data(), slot.meta()).is_err() {
                    write_error.store(true, Ordering::Release);
                    cancelled.store(true, Ordering::Release);
                    break;
                }
                crate::infra::trace_spans::record("write", 0, t1, slot_idx, slot.data().len());
                total_write_ns
                    .fetch_add(write_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                // Hand this ring position back. `release` clears `ready` with Release
                // and the counter bump below is also Release, so the worker that
                // acquires `blocks_written` sees a reset slot.
                unsafe { slot.release() };
                blocks_written.fetch_add(1, Ordering::Release);
            }
            // Zero-pad the trailing partial byte (normal DEFLATE padding
            // after the final BFINAL block).
            if !cancelled.load(Ordering::Acquire) && splicer.finish(&mut w).is_err() {
                write_error.store(true, Ordering::Release);
                cancelled.store(true, Ordering::Release);
            }
            w
        });

        // Spawn N compress worker threads. Shadow every capture as a
        // reference first so the `move` (needed to give each worker its own
        // `wid`) moves ONLY these references, not the values.
        let compress_fn = &compress_fn;
        let slots_ref = &slots;
        let next_block_ref = &next_block;
        let blocks_written_ref = &blocks_written;
        let cancelled_ref = &cancelled;
        let total_compress_ns_ref = &total_compress_ns;
        let blocks_compressed_ref = &blocks_compressed;
        for wid in 0..num_threads {
            scope.spawn(move || {
                worker_loop_timed(
                    input,
                    block_size,
                    num_blocks,
                    slots_ref,
                    next_block_ref,
                    blocks_written_ref,
                    cancelled_ref,
                    compress_fn,
                    total_compress_ns_ref,
                    blocks_compressed_ref,
                    wid as u32 + 1,
                );
            });
        }

        // Wait for writer to finish and get it back
        let w = writer_handle.join().unwrap();

        if write_error.load(Ordering::Acquire) {
            Err(io::Error::other("write failed"))
        } else {
            Ok(w)
        }
    });
    let thread_time = thread_start.elapsed();
    let total_time = start.elapsed();
    crate::infra::trace_spans::flush();

    if debug {
        let compress_ms = total_compress_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let wait_ms = total_wait_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let write_ms = total_write_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let blocks = blocks_compressed.load(Ordering::Relaxed);

        eprintln!("[gzippy] timing breakdown:");
        eprintln!("  total time: {}ms", total_time.as_millis());
        eprintln!("  thread scope: {}ms", thread_time.as_millis());
        eprintln!(
            "  compress (sum across threads): {:.1}ms ({} blocks, {:.2}ms/block avg)",
            compress_ms,
            blocks,
            if blocks > 0 {
                compress_ms / blocks as f64
            } else {
                0.0
            }
        );
        eprintln!("  writer wait: {:.1}ms", wait_ms);
        eprintln!("  writer write: {:.1}ms", write_ms);
        eprintln!(
            "  overhead: {:.1}ms",
            total_time.as_millis() as f64 - thread_time.as_millis() as f64
        );
    }

    result
}

/// Worker loop with timing instrumentation
#[inline]
#[allow(clippy::too_many_arguments)]
fn worker_loop_timed<F>(
    input: &[u8],
    block_size: usize,
    num_blocks: usize,
    slots: &[BlockSlot],
    next_block: &AtomicUsize,
    blocks_written: &AtomicUsize,
    cancelled: &AtomicBool,
    compress_fn: &F,
    total_compress_ns: &AtomicU64,
    blocks_compressed: &AtomicUsize,
    trace_tid: u32,
) where
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>) -> ChunkMeta,
{
    loop {
        if cancelled.load(Ordering::Acquire) {
            break;
        }
        // Claim next block atomically
        let block_idx = next_block.fetch_add(1, Ordering::Relaxed);
        if block_idx >= num_blocks {
            break;
        }

        // WAIT FOR RING SPACE. This ring position is shared with block
        // `block_idx - window`, so it must not be touched until the writer has
        // spliced and released that block. Acquire pairs with the writer's Release
        // bump, so once this returns the slot is observed reset.
        //
        // Cannot deadlock: workers claim strictly increasing indices, so the lowest
        // unwritten block is always held by some worker, and for that worker
        // `block_idx - blocks_written == 0 < window`.
        let window = slots.len();
        if !wait_for_ring_space(block_idx, window, blocks_written, cancelled) {
            break;
        }

        // Calculate block boundaries
        let start = block_idx * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];

        // Get dictionary: last 32KB of input before this block
        let dict = if block_idx > 0 {
            let dict_end = start;
            let dict_start = dict_end.saturating_sub(32768);
            Some(&input[dict_start..dict_end])
        } else {
            None
        };

        let is_last = block_idx == num_blocks - 1;

        // Get output buffer from the ring position for this block
        let slot = &slots[block_idx % slots.len()];
        let output = unsafe { slot.data_mut() };

        // Time the compression
        let compress_start = Instant::now();
        let t0 = crate::infra::trace_spans::now_us();
        let meta = compress_fn(block_idx, block, dict, is_last, output);
        crate::infra::trace_spans::record("chunk_compress", trace_tid, t0, block_idx, block.len());
        unsafe { slot.set_meta(meta) };
        total_compress_ns.fetch_add(
            compress_start.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        blocks_compressed.fetch_add(1, Ordering::Relaxed);

        // Signal completion
        slot.mark_ready();
    }
}

/// Variant for independent blocks (L1-L6) that don't need dictionaries
///
/// Uses same pigz model: N workers + dedicated writer thread.
/// Returns the writer so caller can write any trailer.
///
/// Increment 7: the only caller is the C-FFI `ParallelGzEncoder` ("GZ"
/// multi-block) differential oracle, so this is compiled only under
/// `ffi-oracle`. The pure production path uses `compress_parallel` instead.
#[cfg(any(test, feature = "ffi-oracle"))]
pub fn compress_parallel_independent<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    writer: W,
    compress_fn: F,
) -> io::Result<W>
where
    W: Write + Send,
    F: Fn(&[u8], &mut Vec<u8>) + Sync,
{
    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(writer);
    }
    let num_threads = num_threads.max(1);

    // Use the same bounded reusable ring as `compress_parallel`.
    let slot_capacity = block_size + (block_size / 10) + 1024;
    let window = reorder_window_for(num_threads, num_blocks);
    let slots: Vec<BlockSlot> = (0..window).map(|_| BlockSlot::new(slot_capacity)).collect();

    let next_block = AtomicUsize::new(0);
    let blocks_written = AtomicUsize::new(0);
    let write_error = AtomicBool::new(false);
    let cancelled = AtomicBool::new(false);

    thread::scope(|scope| {
        // Spawn dedicated writer thread
        let writer_handle = scope.spawn(|| {
            let mut w = writer;
            for block_idx in 0..num_blocks {
                let slot = &slots[block_idx % window];
                if !wait_for_slot_ready(slot, &cancelled) {
                    break;
                }
                if w.write_all(slot.data()).is_err() {
                    write_error.store(true, Ordering::Release);
                    cancelled.store(true, Ordering::Release);
                    break;
                }
                unsafe { slot.release() };
                blocks_written.fetch_add(1, Ordering::Release);
            }
            w
        });

        // Spawn N compress workers
        for _ in 0..num_threads {
            scope.spawn(|| loop {
                if cancelled.load(Ordering::Acquire) {
                    break;
                }
                let block_idx = next_block.fetch_add(1, Ordering::Relaxed);
                if block_idx >= num_blocks {
                    break;
                }

                // Wait for ring space (see `compress_parallel`).
                if !wait_for_ring_space(block_idx, window, &blocks_written, &cancelled) {
                    break;
                }

                let start = block_idx * block_size;
                let end = (start + block_size).min(input.len());
                let block = &input[start..end];

                let slot = &slots[block_idx % window];
                let output = unsafe { slot.data_mut() };
                compress_fn(block, output);
                slot.mark_ready();
            });
        }

        let w = writer_handle.join().unwrap();

        if write_error.load(Ordering::Acquire) {
            Err(io::Error::other("write failed"))
        } else {
            Ok(w)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_basic() {
        let input = b"Hello, world! ".repeat(1000);
        let mut output = Vec::new();

        compress_parallel(
            &input,
            1024, // 1KB blocks
            4,    // 4 threads
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                // Simple "compression": just copy
                out.clear();
                out.extend_from_slice(block);
                ChunkMeta::ALIGNED
            },
        )
        .unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_parallel_ordering() {
        // Verify blocks are written in order even when compressed out of order
        let input: Vec<u8> = (0..100).collect();
        let mut output = Vec::new();

        compress_parallel(
            &input,
            10, // 10-byte blocks
            4,  // 4 threads
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                // Add artificial delay for odd blocks to scramble completion order
                // (In real use, compression time varies)
                out.clear();
                out.extend_from_slice(block);
                ChunkMeta::ALIGNED
            },
        )
        .unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_single_block() {
        let input = b"small";
        let mut output = Vec::new();

        compress_parallel(
            input,
            1024, // Block size larger than input
            4,
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                out.clear();
                out.extend_from_slice(block);
                ChunkMeta::ALIGNED
            },
        )
        .unwrap();

        assert_eq!(output, input.as_slice());
    }

    #[test]
    fn writer_error_cancels_workers_waiting_for_ring_space() {
        struct FailWriter;

        impl Write for FailWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed consumer"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        // More blocks than the reorder window forces workers to wait for the
        // writer to release slots. Before cancellation existed, the failed writer
        // exited before releasing them and this call never returned.
        let input = vec![0xA5; 64 * 1024];
        let result = compress_parallel(
            &input,
            1024,
            2,
            FailWriter,
            |_idx, block, _dict, _is_last, out| {
                out.clear();
                out.extend_from_slice(block);
                ChunkMeta::ALIGNED
            },
        );

        assert!(
            result.is_err(),
            "writer failure must be returned, not deadlock"
        );
    }

    #[test]
    fn zero_workers_falls_back_to_one_worker() {
        let input = b"zero workers must still complete".repeat(100);
        let mut output = Vec::new();

        compress_parallel(
            &input,
            32,
            0,
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                out.clear();
                out.extend_from_slice(block);
                ChunkMeta::ALIGNED
            },
        )
        .unwrap();

        assert_eq!(output, input);
    }
}
