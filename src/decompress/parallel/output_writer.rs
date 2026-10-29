//! Background output writer thread — faithful overlap of the per-chunk write
//! with the in-order consumer's next-chunk decode WAIT.
//!
//! ## Why
//! At high T the per-thread engine work parallelizes ~Tx but the serial
//! 211 MiB output materialization (writev into the page cache) on the ONE
//! in-order consumer thread does NOT — it is an Amdahl serial tail. A
//! validated removal-oracle measurement showed it is the
//! single largest T8 binder: removing it moved silesia T8 from ~0.79x to
//! ~0.98x rapidgzip. The instant-feed discriminator proved the cost is
//! engine-INDEPENDENT (it did not shrink when the engine sped up 38ms), i.e.
//! a genuine serial batched-write cost, not a feed-rate artifact.
//!
//! ## What (faithful to rapidgzip)
//! rapidgzip writes each chunk the instant it resolves in the consumer read
//! loop (ParallelGzipReader.hpp:521 `writeFunctor`), so each write overlaps
//! the next chunk's decode. The consumer is 64-67% WAIT — ample slack to
//! hide a copy whose irreducible floor is ~14ms. gzippy batches into a few
//! large writev that land EXPOSED on the serial consumer. This module takes
//! the writev OFF the consumer's critical path: after the consumer has done
//! the order-sensitive work (window publish + CRC combine), it HANDS the
//! owned chunk to a single dedicated writer thread which performs the writev
//! in receive (= output) order while the consumer immediately proceeds to
//! wait on / drain the next chunk. The chunk is owned by the writer for the
//! duration of the write, so the zero-copy iovecs stay valid; dropping the
//! chunk after the write recycles its buffers (vendor FasterVector
//! auto-recycle on writeAll completion).
//!
//! Ordering: a single writer thread consuming an MPSC channel preserves the
//! consumer's send order = output order. CRC + total_size accounting stay on
//! the consumer (O(1) combine), so correctness of the trailer check is
//! unaffected. A write error is captured and surfaced at `join`.
//!
//! WIRE STATE (wave-B lever): production-wired. `drain_one_pending` routes the
//! Linux regular-fd writev through `submit_chunk` whenever the consumer runs
//! with pool threads (T>1); the T1 thin path and every in-memory writer keep
//! the inline write/`Writer` path (the `Writer` trait cannot writev, and the
//! fast path is fd-specific). The vmsplice/pipe path is unchanged — splice
//! page-lifetime accounting needs the inline path.
//!
//! Channel: bounded `sync_channel(MAX_INFLIGHT_JOBS)`. Backpressure, not
//! unbounded queueing: a lagging writer (slow/blocking sink) stalls the
//! consumer at the bound instead of growing the live decoded-buffer set past
//! the working set the decode already holds (BlockFetcher cache = max(16, T)
//! + prefetch = 2·T chunks). At the intended T8 steady state the writer keeps
//! up and the bound never binds.
//!
//! Buffer lifetime / pool accounting (verified against chunk_buffer_pool.rs):
//! the chunk's buffers recycle on the writer thread AFTER its writev
//! (`ChunkData::Drop` → `recycle_decoded_buffers`, chunk_data.rs:1862-1866;
//! contract anticipated at chunk_data.rs:1667/1852). Returns
//! (`return_u8_to_worker`, chunk_buffer_pool.rs:276) are Mutex-guarded statics
//! keyed by owner worker — thread-safe from a foreign thread — and at T>1 the
//! manual pool is DISABLED (`manual_buffer_pool_enabled` is the T1-scoped
//! resident flag), so a writer-side drop frees to the allocator exactly like
//! the inline `release_written_chunk` path it replaces (which is why the T>1
//! gate also keeps the T1 resident-pool configuration on its synchronous
//! inline path where deferral would race the single worker's refill).

#![cfg(all(unix, parallel_sm))]
// The overlap-writer machinery (`enabled`/`submit_chunk`/`OverlapWriter`/
// `WriteJob`) is reached only from the Linux-only writev branch in
// `chunk_fetcher`; on non-Linux unix it compiles but is unused. Allow dead_code
// here so the macOS lint gate stays clean without `cfg`-splitting every item.
#![allow(dead_code)]

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Mutex;
use std::thread::JoinHandle;

use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::fd_vectored_write;

/// Route-assertion counter: one increment per chunk handed to the background
/// overlap writer instead of the consumer-thread inline writev. A production
/// fd decode at T>1 must bump this (see the routing test; same precedent as
/// `THIN_T1_RUNS` in chunk_fetcher.rs).
pub static OVERLAP_WRITER_SUBMITS: AtomicU64 = AtomicU64::new(0);

/// Live-job bound of the submit channel (see the module doc's Channel note).
/// Matches the BlockFetcher working-set anchors: cache = max(16, T),
/// prefetch = 2·T — the writer-side hold never exceeds the depth the pending
/// FIFO + prefetch cache already account for.
const MAX_INFLIGHT_JOBS: usize = 16;

fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

/// One owned chunk to write, in output order.
struct WriteJob {
    fd: i32,
    chunk: ChunkData,
}

/// A live writer thread + its ordered job channel + captured error.
pub struct OverlapWriter {
    tx: Option<SyncSender<WriteJob>>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl OverlapWriter {
    fn spawn() -> Self {
        // Bounded: submit() applies backpressure at MAX_INFLIGHT_JOBS live
        // jobs instead of letting a lagging writer accumulate an unbounded
        // live-decoded-buffer set. The writer thread never depends on the
        // WRITER mutex, so a blocked send cannot deadlock against join().
        let (tx, rx): (SyncSender<WriteJob>, Receiver<WriteJob>) =
            std::sync::mpsc::sync_channel(MAX_INFLIGHT_JOBS);
        let handle = std::thread::Builder::new()
            .name("gzippy-out-writer".into())
            .spawn(move || -> io::Result<()> {
                // Reusable iovec scratch (one chunk's worth of parts).
                while let Ok(job) = rx.recv() {
                    let WriteJob { fd, chunk } = job;
                    let mut parts: Vec<&[u8]> = Vec::with_capacity(8);
                    chunk.append_output_iovecs(&mut parts);
                    let mut iovs = fd_vectored_write::to_io_vec(parts.iter().copied(), &[]);
                    if !iovs.is_empty() {
                        fd_vectored_write::writev_all_to_fd(fd, &mut iovs)?;
                    }
                    // Dropping `chunk` here recycles its decode buffers to the
                    // worker pool (ChunkData::Drop -> recycle_decoded_buffers),
                    // AFTER the writev completed — the iovecs were valid throughout.
                    drop(parts);
                    drop(chunk);
                }
                Ok(())
            })
            .expect("spawn output writer thread");
        OverlapWriter {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Hand an owned chunk to the writer thread for an in-order writev. The
    /// consumer has already published windows + combined the CRC; accounting
    /// is done. The chunk's buffers are released (recycled) by the writer
    /// after the write.
    fn submit(&self, fd: i32, chunk: ChunkData) {
        if let Some(tx) = &self.tx {
            // Blocks only when the writer has MAX_INFLIGHT_JOBS jobs still
            // queued (a genuinely slow sink — flow control, not the per-chunk
            // inline write this exists to overlap). A send failure means the
            // writer thread died (it captured an io::Error); the error is
            // surfaced at `join`. Drop the chunk.
            let _ = tx.send(WriteJob { fd, chunk });
            OVERLAP_WRITER_SUBMITS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Close the channel and join the writer, surfacing any write error.
    fn join(&mut self) -> io::Result<()> {
        // Drop the sender so the writer's recv loop ends.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(res) => res,
                Err(_) => Err(io::Error::other("output writer thread panicked")),
            }
        } else {
            Ok(())
        }
    }
}

// One process-global writer (single in-order output stream per run). The
// parallel-SM decode path writes exactly one output stream at a time.
static WRITER: Mutex<Option<OverlapWriter>> = Mutex::new(None);

/// Submit an owned chunk for an overlapped in-order writev. Spawns the
/// writer thread on first use. Called from `drain_one_pending`'s Linux
/// regular-fd arm ONLY when the consumer runs with pool threads (T>1) — the
/// T1 thin path and in-memory writers keep the inline write path.
pub fn submit_chunk(fd: i32, chunk: ChunkData) {
    let mut guard = WRITER.lock().unwrap();
    if guard.is_none() {
        if debug_enabled() {
            eprintln!(
                "[gzippy] overlap output writer engaged (regular-fd drain, T>1)"
            );
        }
        *guard = Some(OverlapWriter::spawn());
    }
    guard.as_ref().unwrap().submit(fd, chunk);
}

/// Drain + join the writer at end-of-stream. MUST be called before the
/// trailer CRC/ISIZE check returns success, so a write error becomes a
/// terminal Err and all bytes are on the fd. No-op if no writer was spawned.
pub fn finish() -> io::Result<()> {
    let mut w = {
        let mut guard = WRITER.lock().unwrap();
        guard.take()
    };
    let res = match w.as_mut() {
        Some(writer) => writer.join(),
        None => Ok(()),
    };
    if debug_enabled() && !w.is_none() {
        eprintln!(
            "[gzippy] overlap output writer drained: chunks={} err={:?}",
            OVERLAP_WRITER_SUBMITS.load(Ordering::Relaxed),
            res.as_ref().err()
        );
    }
    res
}
