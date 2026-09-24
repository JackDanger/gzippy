//! Property-based roundtrip tests for the ENCODER.
//!
//! The decoder has property coverage (`src/tests/inflate_proptest.rs`); the
//! encoder's riskiest code is edge GEOMETRY, which fixed fixtures under-sample:
//!   * chunk straddles in the T>1 splice path (`pipelined.rs`, 512 KiB grid),
//!   * stored-block fallback at unaligned offsets (65,535-byte sub-blocks),
//!   * multi-MiB inputs (several blocks, several stored sub-blocks),
//!   * tiny/empty inputs and the buffer tail.
//!
//! Every case drives the SAME functions the binary routes through
//! (`gzippy::compress_with_threads` -> `compress::compress_bytes` ->
//! `compress_with_pipeline`, which is what `compress::io::compress_file` calls:
//! T1 -> `encode_gzip_reader_to_writer`, T>1 -> `PipelinedGzEncoder::
//! compress_buffer_pure`), then asserts byte-exact roundtrip through TWO
//! decoders: gzippy's own, and flate2/zlib-ng as an independent oracle.
//!
//! Inputs are seeded synthetic bytes only — no corpus files (fixtures.rs rule).
//!
//! Runtime budget: the deterministic edge-size sweeps always run in full; the
//! proptest generators default to a small case count (64) overridable via
//! `PROPTEST_CASES`. `cargo test --release --test deflate_proptest` stays well
//! under ~90 s.

use proptest::prelude::*;
use std::io::Read;

// ── Geometry constants ───────────────────────────────────────────────────────

/// Mirrors `MAX_PARALLEL_BLOCK_SIZE` in `src/compress/pipelined.rs` (private).
/// `pipelined_block_size` never goes below this for multi-chunk inputs, so
/// sizes bracketing it exercise the 1-chunk/2-chunk seam transition at T>1.
const PARALLEL_CHUNK: usize = 512 * 1024;

/// DEFLATE stored-block payload maximum; level 0 splits on this grid.
const STORED_SUBBLOCK: usize = 65_535;

// ── Seeded synthetic content (no corpus, no private data) ────────────────────

struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self {
        // Never zero — xorshift's fixed point.
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// All zeros: maximal RLE, exercises stored-vs-huffman choice and long matches.
fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

/// Incompressible: every byte from a seeded xorshift (same generator family as
/// `src/fixtures.rs`). Exercises stored-block fallback and expansion headroom.
fn incompressible(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = XorShift::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        out.extend_from_slice(&rng.next().to_le_bytes());
    }
    out.truncate(n);
    out
}

/// Short-period repetition: dense overlapping matches at one distance.
fn periodic(n: usize, period: usize, seed: u64) -> Vec<u8> {
    let period = period.clamp(1, 64);
    let motif = incompressible(period, seed);
    motif.iter().copied().cycle().take(n).collect()
}

/// Text-like: words from a tiny lexicon — skewed byte histogram, match-rich,
/// wide-ish literal alphabet.
fn textish(n: usize, seed: u64) -> Vec<u8> {
    const WORDS: &[&str] = &[
        "the",
        "of",
        "and",
        "compression",
        "block",
        "stream",
        "a",
        "in",
        "to",
        "boundary",
        "huffman",
        "match",
        "literal",
        "seam",
        "chunk",
    ];
    let mut rng = XorShift::new(seed);
    let mut out = Vec::with_capacity(n + 16);
    while out.len() < n {
        out.extend_from_slice(WORDS[(rng.next() as usize) % WORDS.len()].as_bytes());
        out.push(if rng.next().is_multiple_of(13) {
            b'\n'
        } else {
            b' '
        });
    }
    out.truncate(n);
    out
}

/// A repeated motif whose occurrences straddle `boundary`: one copy ends just
/// after it, earlier copies sit well before it, so a long match must reach
/// across the chunk edge (splice-path geometry the T>1 grid cuts at).
fn motif_straddle(n: usize, boundary: usize, seed: u64) -> Vec<u8> {
    let mut out = textish(n, seed ^ 0x9E37_79B9_7F4A_7C15);
    let motif = incompressible(199, seed ^ 0x5851_F42D_4C95_7F2D);
    if n > motif.len() {
        // Copies leading up to and across the boundary, plus a few earlier so
        // the matchfinder has history on the left side of the seam.
        let mut positions = vec![
            boundary.saturating_sub(motif.len() * 2),
            boundary.saturating_sub(motif.len()),
            boundary.saturating_sub(motif.len() / 2), // straddles the edge
            boundary,
            boundary + 3,
        ];
        positions.push(boundary.saturating_sub(PARALLEL_CHUNK / 2));
        for pos in positions {
            if pos + motif.len() <= n {
                out[pos..pos + motif.len()].copy_from_slice(&motif);
            }
        }
    }
    out
}

// ── The property: compress -> decode twice -> byte-identical ─────────────────

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(
        a.iter()
            .zip(b.iter())
            .position(|(x, y)| x != y)
            .unwrap_or_else(|| a.len().min(b.len())),
    )
}

/// Compress `data` at (`level`, `threads`) through the production routing
/// table, then assert byte-exact roundtrip through gzippy's own decoder AND
/// through flate2/zlib-ng as an independent oracle.
fn assert_roundtrip(data: &[u8], level: u8, threads: usize, label: &str) {
    let gz = gzippy::compress_with_threads(data, level, threads).unwrap_or_else(|e| {
        panic!(
            "compress failed [{label}] len={} L{level} T{threads}: {e}",
            data.len()
        )
    });

    // Our own decoder.
    let ours = gzippy::decompress(&gz).unwrap_or_else(|e| {
        panic!(
            "gzippy decode failed [{label}] len={} L{level} T{threads}: {e}",
            data.len()
        )
    });
    if let Some(i) = first_diff(&ours, data) {
        panic!(
            "gzippy roundtrip MISMATCH [{label}] len={} L{level} T{threads}: \
             decoded len={} first-diff at byte {}",
            data.len(),
            ours.len(),
            i
        );
    }

    // Independent oracle: flate2 (zlib-ng). MultiGzDecoder also covers any
    // multi-member framing, so a framing change fails loudly here, not silently.
    let mut oracle = Vec::with_capacity(data.len());
    flate2::read::MultiGzDecoder::new(&gz[..])
        .read_to_end(&mut oracle)
        .unwrap_or_else(|e| {
            panic!(
                "flate2 oracle decode failed [{label}] len={} L{level} T{threads}: {e} \
                 (output is not independently-readable gzip)",
                data.len()
            )
        });
    if let Some(i) = first_diff(&oracle, data) {
        panic!(
            "flate2 oracle roundtrip MISMATCH [{label}] len={} L{level} T{threads}: \
             decoded len={} first-diff at byte {}",
            data.len(),
            oracle.len(),
            i
        );
    }
}

// ── Deterministic edge-geometry sweeps (always run, no sampling) ─────────────

const THREADS: [usize; 2] = [1, 4];

/// Empty and tiny inputs at EVERY level 0..=9, both thread counts. The empty
/// member, the 1-byte member, and sub-minimum-match inputs are all distinct
/// header/trailer geometries.
#[test]
fn tiny_inputs_all_levels() {
    for n in [0usize, 1, 2, 3, 4, 8, 63, 64] {
        for level in 0u8..=9 {
            for &threads in &THREADS {
                assert_roundtrip(
                    &incompressible(n, 0xA1 + n as u64),
                    level,
                    threads,
                    "tiny-inc",
                );
                assert_roundtrip(&zeros(n), level, threads, "tiny-zeros");
            }
        }
    }
}

/// Sizes bracketing the stored-sub-block grid (65,535). Level 0 must split
/// exactly here; higher levels may choose stored blocks for incompressible
/// input at unaligned bit offsets.
#[test]
fn stored_subblock_boundary() {
    for n in [
        STORED_SUBBLOCK - 1,
        STORED_SUBBLOCK,
        STORED_SUBBLOCK + 1,
        2 * STORED_SUBBLOCK + 1,
    ] {
        for level in [0u8, 1, 6, 9] {
            for &threads in &THREADS {
                assert_roundtrip(&incompressible(n, 0xB2), level, threads, "stored-inc");
                assert_roundtrip(&zeros(n), level, threads, "stored-zeros");
                assert_roundtrip(&periodic(n, 7, 0xB3), level, threads, "stored-period7");
            }
        }
    }
}

/// Sizes bracketing the T>1 parallel chunk (512 KiB): chunk-1, chunk, chunk+1,
/// and chunk*2+7 (two full chunks plus a 7-byte tail chunk). This is the seam
/// splice path plus the short-tail-chunk geometry.
#[test]
fn parallel_chunk_boundary() {
    for n in [
        PARALLEL_CHUNK - 1,
        PARALLEL_CHUNK,
        PARALLEL_CHUNK + 1,
        PARALLEL_CHUNK * 2 + 7,
    ] {
        for level in [0u8, 1, 6, 9] {
            for &threads in &THREADS {
                assert_roundtrip(&zeros(n), level, threads, "chunk-zeros");
                assert_roundtrip(&incompressible(n, 0xC4), level, threads, "chunk-inc");
                assert_roundtrip(&textish(n, 0xC5), level, threads, "chunk-text");
            }
        }
    }
}

/// A long repeated motif whose copies straddle the 512 KiB chunk edge: the
/// matchfinder on the right side of a T>1 seam wants a match whose source is
/// on the left side. Also placed at the second seam of a 2-chunk+tail input.
#[test]
fn long_match_straddles_chunk_boundary() {
    for (n, boundary) in [
        (PARALLEL_CHUNK * 2 + 7, PARALLEL_CHUNK),
        (PARALLEL_CHUNK * 2 + 7, PARALLEL_CHUNK * 2),
        (PARALLEL_CHUNK + 4096, PARALLEL_CHUNK),
    ] {
        let data = motif_straddle(n, boundary, 0xD6);
        for level in [1u8, 6, 9] {
            for &threads in &THREADS {
                assert_roundtrip(&data, level, threads, "motif-straddle");
            }
        }
    }
}

/// Multi-megabyte sizes around ~4.2 MiB (65,535 * 64): several blocks at
/// every level, several stored sub-blocks at L0 — the multi-MB boundary class
/// the old streaming chunk grid used to define.
#[test]
fn multi_megabyte_boundary_t1() {
    const MB4: usize = 4_194_240;
    for n in [MB4 - 1, MB4, MB4 + 1] {
        for level in [0u8, 1, 6] {
            assert_roundtrip(&zeros(n), level, 1, "mb-zeros");
            assert_roundtrip(&textish(n, 0xE7), level, 1, "mb-text");
        }
    }
    // One T>1 pass over the same boundary: 4 MiB at T4 is a full multi-chunk
    // grid with a 1-byte logical tail.
    assert_roundtrip(&textish(MB4 + 1, 0xE8), 6, 4, "mb-text-t4");
}

// ── Proptest: randomized shapes x sizes x level x threads ────────────────────

fn cases() -> u32 {
    // Default SMALL so the release suite stays under budget; the deterministic
    // sweeps above always run in full. Override: PROPTEST_CASES=1024.
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

/// Sizes biased toward edge geometry, not uniform noise.
fn edge_size() -> impl Strategy<Value = usize> {
    prop_oneof![
        2 => 0usize..=4,
        3 => 5usize..=64,
        3 => 65usize..=4096,
        3 => (STORED_SUBBLOCK - 64)..=(STORED_SUBBLOCK + 64),
        2 => (2 * STORED_SUBBLOCK - 32)..=(2 * STORED_SUBBLOCK + 32),
        3 => (PARALLEL_CHUNK - 64)..=(PARALLEL_CHUNK + 64),
    ]
}

#[derive(Debug, Clone)]
enum Shape {
    Zeros,
    Incompressible(u64),
    Periodic(usize, u64),
    Textish(u64),
    MotifStraddle(u64),
}

fn shape() -> impl Strategy<Value = Shape> {
    prop_oneof![
        1 => Just(Shape::Zeros),
        2 => any::<u64>().prop_map(Shape::Incompressible),
        2 => (1usize..=32, any::<u64>()).prop_map(|(p, s)| Shape::Periodic(p, s)),
        2 => any::<u64>().prop_map(Shape::Textish),
        2 => any::<u64>().prop_map(Shape::MotifStraddle),
    ]
}

fn materialize(shape: &Shape, n: usize) -> (Vec<u8>, &'static str) {
    match shape {
        Shape::Zeros => (zeros(n), "zeros"),
        Shape::Incompressible(s) => (incompressible(n, *s), "incompressible"),
        Shape::Periodic(p, s) => (periodic(n, *p, *s), "periodic"),
        Shape::Textish(s) => (textish(n, *s), "textish"),
        // Straddle whichever grid line the size actually crosses; fall back to
        // the midpoint for small inputs so the motif still lands in-bounds.
        Shape::MotifStraddle(s) => {
            let boundary = if n > PARALLEL_CHUNK {
                PARALLEL_CHUNK
            } else if n > STORED_SUBBLOCK {
                STORED_SUBBLOCK
            } else {
                n / 2
            };
            (motif_straddle(n, boundary, *s), "motif-straddle")
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(),
        ..ProptestConfig::default()
    })]

    /// Arbitrary shaped input x level 0..=9 x threads {1,4}: compress through
    /// the production routing table, decode through gzippy AND flate2,
    /// byte-identical to the input.
    #[test]
    fn roundtrip_shaped(
        sh in shape(),
        n in edge_size(),
        level in 0u8..=9,
        threads in prop_oneof![Just(1usize), Just(4usize)],
    ) {
        let (data, label) = materialize(&sh, n);
        assert_roundtrip(&data, level, threads, label);
    }

    /// Genuinely arbitrary bytes (proptest-generated, shrinkable) for the
    /// small-input range where uniform noise is affordable and shrinking is
    /// most useful.
    #[test]
    fn roundtrip_arbitrary_bytes(
        data in proptest::collection::vec(any::<u8>(), 0..2048),
        level in 0u8..=9,
        threads in prop_oneof![Just(1usize), Just(4usize)],
    ) {
        assert_roundtrip(&data, level, threads, "arbitrary");
    }
}
