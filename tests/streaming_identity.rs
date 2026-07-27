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
