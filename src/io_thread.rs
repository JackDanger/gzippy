//! Write-ahead I/O threading for decompression.
//!
//! `WriteAhead` wraps any `Write + Send` and moves write syscalls to a
//! background thread. The caller sends data via a bounded channel; the
//! writer thread drains the channel sequentially. This overlaps decode
//! work with write latency (kernel copies, pipe stalls, fsync).

use std::io::{self, Write};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::{self, JoinHandle};

#[allow(dead_code)]
enum Msg {
    Data(Vec<u8>),
    Flush,
}

/// Moves writes to a background thread via a bounded channel.
///
/// The channel capacity controls how many buffers can be in flight.
/// With 1MB buffers and capacity=4, up to 4MB of decoded data can be
/// queued before the sender blocks, providing back-pressure.
#[allow(dead_code)]
pub struct WriteAhead<W: Write + Send + 'static> {
    sender: SyncSender<Msg>,
    handle: Option<JoinHandle<io::Result<W>>>,
}

#[allow(dead_code)]
impl<W: Write + Send + 'static> WriteAhead<W> {
    /// Create a new WriteAhead that writes to `writer` on a background thread.
    /// `capacity` is the number of buffers that can be in flight.
    pub fn new(writer: W, capacity: usize) -> Self {
        let (tx, rx) = sync_channel::<Msg>(capacity);

        let handle = thread::spawn(move || {
            let mut w = writer;
            for msg in rx {
                match msg {
                    Msg::Data(data) => w.write_all(&data)?,
                    Msg::Flush => w.flush()?,
                }
            }
            w.flush()?;
            Ok(w)
        });

        WriteAhead {
            sender: tx,
            handle: Some(handle),
        }
    }

    /// Send data to the writer thread. Blocks if the channel is full.
    pub fn send(&self, data: &[u8]) -> io::Result<()> {
        self.sender
            .send(Msg::Data(data.to_vec()))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread exited"))
    }

    /// Signal the writer thread to flush.
    pub fn flush(&self) -> io::Result<()> {
        self.sender
            .send(Msg::Flush)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread exited"))
    }

    /// Close the channel and wait for the writer thread to finish.
    /// Returns the underlying writer on success.
    pub fn finish(mut self) -> io::Result<W> {
        drop(self.sender.clone());
        // Drop the real sender to close the channel
        let sender = std::mem::replace(&mut self.sender, sync_channel::<Msg>(0).0);
        drop(sender);

        self.handle
            .take()
            .expect("finish called twice")
            .join()
            .map_err(|_| io::Error::other("writer thread panicked"))?
    }
}

impl<W: Write + Send + 'static> Drop for WriteAhead<W> {
    fn drop(&mut self) {
        // If finish() wasn't called, the sender drops here, closing the channel.
        // The writer thread will drain remaining messages and exit.
        // We don't join on drop to avoid blocking — the thread will clean up.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_ahead_basic() {
        let buf: Vec<u8> = Vec::new();
        let wa = WriteAhead::new(buf, 4);

        wa.send(b"hello ").unwrap();
        wa.send(b"world").unwrap();

        let result = wa.finish().unwrap();
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn test_write_ahead_empty() {
        let buf: Vec<u8> = Vec::new();
        let wa = WriteAhead::new(buf, 4);
        let result = wa.finish().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_write_ahead_large_data() {
        let buf: Vec<u8> = Vec::new();
        let wa = WriteAhead::new(buf, 4);

        let chunk = vec![0xABu8; 1024 * 1024];
        for _ in 0..10 {
            wa.send(&chunk).unwrap();
        }

        let result = wa.finish().unwrap();
        assert_eq!(result.len(), 10 * 1024 * 1024);
        assert!(result.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_write_ahead_flush() {
        let buf: Vec<u8> = Vec::new();
        let wa = WriteAhead::new(buf, 4);

        wa.send(b"data").unwrap();
        wa.flush().unwrap();
        wa.send(b"more").unwrap();

        let result = wa.finish().unwrap();
        assert_eq!(result, b"datamore");
    }

    #[test]
    fn test_write_ahead_error_propagation() {
        struct FailWriter;
        impl Write for FailWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "test error"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let wa = WriteAhead::new(FailWriter, 4);
        wa.send(b"data").unwrap(); // send succeeds (just queues)

        // finish() should propagate the write error
        let result = wa.finish();
        assert!(result.is_err());
    }

    #[test]
    fn test_write_ahead_backpressure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct SlowWriter {
            written: Arc<AtomicUsize>,
        }
        impl Write for SlowWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                std::thread::sleep(std::time::Duration::from_millis(10));
                self.written.fetch_add(buf.len(), Ordering::Relaxed);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let written = Arc::new(AtomicUsize::new(0));
        let wa = WriteAhead::new(
            SlowWriter {
                written: written.clone(),
            },
            2,
        );

        for _ in 0..5 {
            wa.send(b"x").unwrap();
        }

        let _ = wa.finish().unwrap();
        assert_eq!(written.load(Ordering::Relaxed), 5);
    }
}
