//! `compress_to_writer` promises streaming. This proves it, rather than documenting it.
//!
//! ⭐ OWNER REVIEW, 2026-08-23:
//!
//!   "The library's 'streaming' writer API buffers the entire input whenever it uses
//!    more than one thread. compress_to_writer() defaults to all CPUs, then
//!    read_to_end()s the reader before any output. This violates both the API docs
//!    and README promise for large inputs."
//!
//! The CLI already had the correct rule for the identical situation and it was never
//! generalised to the library (`compress::io`, pipe stdin: "stream directly without
//! buffering all input first. Single-threaded so output begins immediately without
//! OOM risk.").
//!
//! A doc comment cannot fail CI. This can.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Reader that records how many bytes have been handed out.
struct CountingReader {
    data: Vec<u8>,
    pos: usize,
    read_so_far: Arc<AtomicUsize>,
}

impl Read for CountingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Hand out small pieces so "how much was read before the first write" is a
        // meaningful question rather than one gulp.
        let n = buf.len().min(64 * 1024).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        self.read_so_far.store(self.pos, Ordering::SeqCst);
        Ok(n)
    }
}

/// Writer that snapshots the reader's progress at the moment of the first write.
struct SnapshotWriter {
    read_so_far: Arc<AtomicUsize>,
    read_at_first_write: Arc<AtomicUsize>,
    wrote_any: bool,
    sink: Vec<u8>,
}

impl Write for SnapshotWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if !self.wrote_any && !buf.is_empty() {
            self.wrote_any = true;
            self.read_at_first_write
                .store(self.read_so_far.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        self.sink.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Output must begin before the input has been fully consumed.
///
/// If this fails, `compress_to_writer` is buffering the whole input — the exact
/// defect the owner review named, regardless of what the doc comment says.
#[test]
fn compress_to_writer_emits_before_consuming_all_input() {
    // 8 MiB of compressible text: many blocks, so a streaming encoder has ample
    // opportunity to emit before EOF.
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(190_000);
    let total = data.len();

    let read_so_far = Arc::new(AtomicUsize::new(0));
    let read_at_first_write = Arc::new(AtomicUsize::new(usize::MAX));

    let reader = CountingReader {
        data,
        pos: 0,
        read_so_far: Arc::clone(&read_so_far),
    };
    let writer = SnapshotWriter {
        read_so_far: Arc::clone(&read_so_far),
        read_at_first_write: Arc::clone(&read_at_first_write),
        wrote_any: false,
        sink: Vec::new(),
    };

    let n = gzippy::compress_to_writer(reader, writer, 6).expect("compress_to_writer");
    assert_eq!(n, total as u64, "reported consumed length");

    let at_first = read_at_first_write.load(Ordering::SeqCst);
    assert!(at_first != usize::MAX, "no output was produced at all");
    assert!(
        at_first < total,
        "compress_to_writer read ALL {total} input bytes before emitting its first \
         output byte (read {at_first} at first write) — it is buffering, not \
         streaming, and its documentation says otherwise. The parallel encoder is \
         whole-buffer, so this function must stay single-threaded; use \
         compress_to_writer_with_threads when buffering is acceptable."
    );
}

/// Whatever it emits must still be a valid gzip stream that round-trips.
#[test]
fn streamed_output_roundtrips() {
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(190_000);
    let mut out = Vec::new();
    let n = gzippy::compress_to_writer(std::io::Cursor::new(data.clone()), &mut out, 6)
        .expect("compress_to_writer");
    assert_eq!(n, data.len() as u64);

    let back = gzippy::decompress(&out).expect("roundtrip");
    assert_eq!(back, data, "streamed output did not round-trip");
}
