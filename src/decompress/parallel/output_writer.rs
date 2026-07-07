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
//! OFF by default unless `GZIPPY_OVERLAP_WRITER=1`; OFF == the inline writev
//! path (byte-identical). Linux + regular-fd only (the vmsplice/pipe path is
//! unchanged — splice page-lifetime accounting needs the inline path).

#![cfg(all(unix, parallel_sm))]
// The overlap-writer machinery (`enabled`/`submit_chunk`/`OverlapWriter`/
// `WriteJob`) is reached only from the Linux-only writev branch in
// `chunk_fetcher`; on non-Linux unix it compiles but is unused. Allow dead_code
// here so the macOS lint gate stays clean without `cfg`-splitting every item.
#![allow(dead_code)]

use std::io;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::fd_vectored_write;

#[inline]
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GZIPPY_OVERLAP_WRITER").is_some())
}

/// One owned chunk to write, in output order.
struct WriteJob {
    fd: i32,
    chunk: ChunkData,
}

/// A live writer thread + its ordered job channel + captured error.
pub struct OverlapWriter {
    tx: Option<Sender<WriteJob>>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl OverlapWriter {
    fn spawn() -> Self {
        let (tx, rx): (Sender<WriteJob>, Receiver<WriteJob>) = std::sync::mpsc::channel();
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
            // A send failure means the writer thread died (it captured an
            // io::Error); the error is surfaced at `join`. Drop the chunk.
            let _ = tx.send(WriteJob { fd, chunk });
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
/// writer thread on first use.
pub fn submit_chunk(fd: i32, chunk: ChunkData) {
    let mut guard = WRITER.lock().unwrap();
    if guard.is_none() {
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
    match w.as_mut() {
        Some(writer) => writer.join(),
        None => Ok(()),
    }
}
