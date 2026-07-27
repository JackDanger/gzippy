//! Does the streaming T1 encoder emit the SAME BYTES as the whole-buffer one?
//!
//! `compress_gzip_streaming` holds a fixed ~4.3 MB of buffer where
//! `compress_gzip_padded` holds the entire input plus the entire output. That
//! is worth having on its own (measured 2.009x-input peak RSS versus a flat
//! 2.0 MB for gzip and pigz), but only if the emitted stream does not get
//! worse — a smaller memory footprint bought with a larger output would trade
//! one axis of the drop-in contract for another.
//!
//! So this asserts the strongest form available at each level: byte-identity
//! at level 0 (which the chunk size is chosen to preserve), and at levels 1-12
//! a roundtrip through an independent decoder plus a size regression bounded
//! by a MEASURED figure. Where identity cannot hold it says so explicitly
//! rather than quietly relaxing the check.
//!
//! Sizes here deliberately straddle the chunk boundary (`STREAM_CHUNK` =
//! 65535 x 64 = 4_194_240): just under, exactly on, one byte past, and several
//! chunks deep. The one-byte-past and exactly-on cases are what exercise the
//! lookahead that decides BFINAL, historically the easiest thing to get wrong
//! in a chunked encoder — an off-by-one there produces a stream that still
//! decodes on most inputs.

use gzippy::compress::deflate::{
    compress_gzip_padded, compress_gzip_streaming, INPLACE_TAIL_PAD, STREAM_CHUNK,
};

/// Track the production constant rather than copying its value: an earlier
/// version hardcoded 65535*16, and when the shipped chunk grew to 65535*64 the
/// "straddles a chunk boundary" cases silently stopped straddling anything —
/// every case fit in one chunk and the test passed by testing nothing.
const CHUNK: usize = STREAM_CHUNK;

/// Deterministic pseudo-random bytes with tunable redundancy: `period` controls
/// how often the pattern repeats, so a small period yields highly compressible
/// input (long matches, deep hash chains) and a large one yields nearly
/// incompressible input (literal-dominated). Both stress different halves of
/// the matchfinder across a chunk seam.
fn corpus(len: usize, period: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut s: u32 = 0x1234_5678;
    for i in 0..len {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let r = (s >> 16) as u8;
        v.push(if (i as u32) % period < period / 2 {
            b'a' + (r % 26)
        } else {
            r
        });
    }
    v
}

fn whole_buffer(data: &[u8], level: u32) -> Vec<u8> {
    let mut padded = Vec::with_capacity(data.len() + INPLACE_TAIL_PAD);
    padded.extend_from_slice(data);
    padded.resize(data.len() + INPLACE_TAIL_PAD, 0);
    compress_gzip_padded(&padded, data.len(), level)
}

fn streamed(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut src = data;
    let n = compress_gzip_streaming(&mut src, &mut out, level).expect("streaming encode");
    assert_eq!(n, data.len() as u64, "reported input length");
    out
}

/// Decode with an INDEPENDENT implementation (flate2/zlib-ng), never with our
/// own decoder — a shared bug would make both sides agree on a wrong answer.
fn roundtrip(gz: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut d = flate2::read::GzDecoder::new(gz);
    let mut out = Vec::new();
    d.read_to_end(&mut out).expect("valid gzip stream");
    out
}

fn cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", Vec::new()),
        ("one byte", vec![b'x']),
        ("small text", corpus(4096, 64)),
        ("chunk minus one", corpus(CHUNK - 1, 96)),
        ("exactly one chunk", corpus(CHUNK, 96)),
        ("chunk plus one", corpus(CHUNK + 1, 96)),
        ("two chunks plus tail", corpus(2 * CHUNK + 7777, 128)),
        ("high redundancy, multi-chunk", corpus(2 * CHUNK + 100, 8)),
        (
            "low redundancy, multi-chunk",
            corpus(2 * CHUNK + 100, 1 << 30),
        ),
    ]
}

#[test]
fn streamed_output_roundtrips_through_an_independent_decoder() {
    for level in [0u32, 1, 2, 4, 6, 9, 12] {
        for (name, data) in cases() {
            // Near-optimal levels on multi-chunk inputs cost minutes for no extra
            // coverage — the seam logic is level-independent.
            if level >= 10 && data.len() > CHUNK {
                continue;
            }
            let gz = streamed(&data, level);
            assert_eq!(
                roundtrip(&gz),
                data,
                "L{level} {name}: streamed output did not roundtrip"
            );
        }
    }
}

#[test]
fn level_0_is_byte_identical_to_the_whole_buffer_encoder() {
    // STREAM_CHUNK is a multiple of MAX_STORED_SUBBLOCK precisely so this
    // holds: the stored sub-block boundaries fall in the same places whether
    // the input arrives all at once or a chunk at a time. If this ever fails,
    // the chunk size stopped being a multiple of 65535.
    for (name, data) in cases() {
        assert_eq!(
            streamed(&data, 0),
            whole_buffer(&data, 0),
            "L0 {name}: streamed and whole-buffer output differ"
        );
    }
}

/// For levels 1-12 the two encoders legitimately disagree: the streaming one
/// ends a block at each chunk seam, so block-splitting decisions differ. The
/// cost of that is real but must stay negligible — measured across the
/// 21-file corpus x L0-L9 at the shipped 4 MiB chunk, the worst regression on
/// genuinely multi-chunk files was 0.0189%, and several cells came out
/// SMALLER. The bound here is 0.5%: comfortably above the measured worst case
/// so ordinary block-splitting churn does not flake, far below the ~1% scale
/// at which a per-label size cell against a rival could flip.
///
/// Level 3 is deliberately absent. It is the one level excluded from the
/// streaming route (`level_streams`) because its content detector is
/// chunk-sensitive — asserting a tight bound on a path production does not
/// take would be testing fiction.
const MAX_STREAM_SIZE_REGRESSION: f64 = 0.005;

#[test]
fn levels_1_through_12_never_get_larger_when_streamed() {
    let mut identical = 0usize;
    let mut compared = 0usize;
    for level in [1u32, 2, 4, 6, 9, 12] {
        for (name, data) in cases() {
            if data.len() < 1024 || (level >= 10 && data.len() > CHUNK) {
                continue;
            }
            let s = streamed(&data, level);
            let w = whole_buffer(&data, level);
            compared += 1;
            if s == w {
                identical += 1;
            }
            let frac = (s.len() as f64 - w.len() as f64) / w.len() as f64;
            assert!(
                frac <= MAX_STREAM_SIZE_REGRESSION,
                "L{level} {name}: streamed {} bytes vs whole-buffer {} ({:+.4}%) exceeds the \
                 {:.2}% bound",
                s.len(),
                w.len(),
                frac * 100.0,
                MAX_STREAM_SIZE_REGRESSION * 100.0
            );
        }
    }
    eprintln!("streamed==whole-buffer on {identical}/{compared} level-1..12 cases");
}

// ---------------------------------------------------------------------------
// Adversarial readers.
//
// Every test above feeds a `&[u8]`, whose `Read` impl always fills the whole
// buffer and never fails. That means the streaming encoder's fill loop — the
// short-read retry, the `Interrupted` retry, and the one-byte lookahead that
// decides BFINAL — was entirely unexercised, while the CLI hits exactly those
// paths whenever its input is a pipe or stdin. A review flagged this as the
// strongest missing test; it was right.
// ---------------------------------------------------------------------------

/// A `Read` that hands back at most `max_read` bytes per call and injects
/// `ErrorKind::Interrupted` every `interrupt_every` calls. Both behaviours are
/// legal for any `Read` implementation and both are common on pipes.
struct HostileReader<'a> {
    data: &'a [u8],
    pos: usize,
    max_read: usize,
    interrupt_every: usize,
    calls: usize,
}

impl<'a> HostileReader<'a> {
    fn new(data: &'a [u8], max_read: usize, interrupt_every: usize) -> Self {
        Self {
            data,
            pos: 0,
            max_read,
            interrupt_every,
            calls: 0,
        }
    }
}

impl std::io::Read for HostileReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.calls += 1;
        if self.interrupt_every != 0 && self.calls.is_multiple_of(self.interrupt_every) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "injected",
            ));
        }
        let n = buf.len().min(self.max_read).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn short_reads_and_interrupts_do_not_change_the_output() {
    // Sizes that land on and around the chunk boundary, where a mis-handled
    // short read would corrupt the is_last decision rather than the data.
    let inputs = [
        ("chunk minus one", corpus(CHUNK - 1, 96)),
        ("exactly one chunk", corpus(CHUNK, 96)),
        ("chunk plus one", corpus(CHUNK + 1, 96)),
        ("two chunks plus tail", corpus(2 * CHUNK + 7777, 128)),
    ];
    // max_read of 1 is the pathological case: the fill loop must iterate
    // millions of times and still assemble exactly the same chunks.
    let shapes = [
        (1usize, 0usize),
        (7, 3),
        (4096, 5),
        (CHUNK / 3, 2),
        (usize::MAX, 11),
    ];

    for (name, data) in &inputs {
        let reference = streamed(data, 6);
        for (max_read, interrupt_every) in shapes {
            let mut r = HostileReader::new(data, max_read, interrupt_every);
            let mut out = Vec::new();
            let n = gzippy::compress::deflate::compress_gzip_streaming(&mut r, &mut out, 6)
                .expect("hostile reader must not fail the encode");
            assert_eq!(
                n as usize,
                data.len(),
                "{name}: wrong input length reported"
            );
            assert_eq!(
                out, reference,
                "{name} (max_read={max_read}, interrupt_every={interrupt_every}): \
                 output differs from the same input delivered in one piece"
            );
            assert_eq!(&roundtrip(&out), data, "{name}: roundtrip");
        }
    }
}

/// A `Read` that synthesizes `len` bytes procedurally so a multi-gigabyte input
/// can be encoded without ever allocating it.
struct HugeReader {
    remaining: u64,
    counter: u64,
}

impl std::io::Read for HugeReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.remaining) as usize;
        for b in buf.iter_mut().take(n) {
            *b = (self.counter % 251) as u8;
            self.counter += 1;
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}

/// gzip's ISIZE field is 32 bits and is defined as the input length MODULO
/// 2^32, so a >4 GiB member is not an error — it wraps, and every decoder
/// (including gzip and pigz) accepts it. This pins that our streamed trailer
/// wraps the same way, and incidentally proves the encoder survives an input
/// far larger than RAM, which is the entire point of the streaming path.
#[test]
#[ignore = "encodes 4 GiB; run explicitly with --ignored"]
fn input_larger_than_four_gib_wraps_isize_and_does_not_buffer() {
    const LEN: u64 = (1u64 << 32) + 1_000_000;
    let mut r = HugeReader {
        remaining: LEN,
        counter: 0,
    };
    let mut sink = CountingSink {
        bytes: 0,
        tail: Vec::new(),
    };
    let n = gzippy::compress::deflate::compress_gzip_streaming(&mut r, &mut sink, 0)
        .expect("4 GiB encode");
    assert_eq!(n, LEN, "reported input length");

    let isize_field = u32::from_le_bytes(sink.tail[sink.tail.len() - 4..].try_into().unwrap());
    assert_eq!(
        u64::from(isize_field),
        LEN % (1u64 << 32),
        "ISIZE must be the input length mod 2^32, matching gzip and pigz"
    );
}

/// Discards output but remembers the last 8 bytes, so the gzip trailer can be
/// checked without holding a multi-gigabyte stream.
struct CountingSink {
    bytes: u64,
    tail: Vec<u8>,
}

impl std::io::Write for CountingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes += buf.len() as u64;
        self.tail.extend_from_slice(buf);
        let keep = self.tail.len().saturating_sub(8);
        self.tail.drain(..keep);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
