//! `compress_to_writer`'s buffering contract, pinned by execution rather than
//! documentation.
//!
//! ⭐ OWNER REVIEW, 2026-08-23:
//!
//!   "The library's 'streaming' writer API buffers the entire input whenever it uses
//!    more than one thread. compress_to_writer() defaults to all CPUs, then
//!    read_to_end()s the reader before any output. This violates both the API docs
//!    and the README promise for large inputs."
//!
//! The review landed, and the fix it forced had two halves. The first half —
//! never default to all CPUs — shipped 2026-08-23: the function is
//! single-threaded. The second half — a genuinely streaming single-threaded
//! path — has never been possible since `ldx` became the production T1 parser
//! for L0-9 (whole-buffer by construction) and L10-12 has no resumable parser:
//! every level read_to_end()s, and the ~220 lines of single-pass machinery
//! that would have made the original claim true were unreachable for every
//! production level. They were deleted 2026-08-30 along with the false
//! "genuinely streaming" doc, and THIS FILE NOW PINS THE HONEST CONTRACT:
//!
//!   1. The first output byte waits for the LAST input byte — the whole-buffer
//!      behavior is asserted, so a future resumable-ldx streaming change must
//!      update this test in the same commit (it cannot drift silently in
//!      either direction).
//!   2. Whatever it emits is a valid gzip stream that round-trips.
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

/// The HONEST contract (2026-08-30): `compress_to_writer` buffers the whole
/// input — the first output byte is emitted only after the last input byte has
/// been read. This is the behavior `src/lib.rs` documents; the test exists so
/// the documentation and the code cannot drift apart. When a resumable `ldx`
/// port lands and this function genuinely streams, THIS TEST is the tripwire
/// that forces the doc, the test and the routing to change in one commit.
#[test]
fn compress_to_writer_documents_its_whole_buffer_contract() {
    // 8 MiB of compressible text: many blocks, so a streaming encoder would
    // have ample opportunity to emit before EOF — this input cannot make the
    // assertion pass by accident.
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(190_000);
    let total = data.len();

    let read_so_far = Arc::new(AtomicUsize::new(0));
    let read_at_first_write = Arc::new(AtomicUsize::new(0));

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
    assert!(
        at_first == total,
        "compress_to_writer emitted its first output byte after reading only {at_first} of \
         {total} input bytes — the documented whole-buffer contract changed. If a resumable \
         streaming path landed, update src/lib.rs AND this test in the same commit; if it \
         did not land, the routing has quietly regressed to a buffering change nobody \
         announced."
    );
}

/// Whatever it emits must still be a valid gzip stream that round-trips —
/// through our decoder AND an independent one (flate2/zlib-ng), per the
/// project's validity bar.
#[test]
fn compress_to_writer_output_roundtrips() {
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(190_000);
    let mut out = Vec::new();
    let n = gzippy::compress_to_writer(std::io::Cursor::new(data.clone()), &mut out, 6)
        .expect("compress_to_writer");
    assert_eq!(n, data.len() as u64);

    let back = gzippy::decompress(&out).expect("our-decoder roundtrip");
    assert_eq!(back, data, "our-decoder roundtrip mismatch");

    let mut d = flate2::read::GzDecoder::new(&out[..]);
    let mut independent = Vec::new();
    d.read_to_end(&mut independent)
        .expect("independent decoder accepted the stream");
    assert_eq!(independent, data, "independent-decoder roundtrip mismatch");
}
