//! Production T1 gzip must have the same DEFLATE structure as libdeflate.
//!
//! `ldx` is the in-tree byte-exact port of libdeflate's raw DEFLATE encoder.
//! Comparing the production gzip body against it catches a routing fallback,
//! a small-input shortcut, or a header metadata regression without requiring a
//! vendor executable in every test environment.
//!
//! The L1/L6/L7 measured exceptions (`level_uses_ldx`) route to the legacy
//! encoder instead; for those levels this file pins metadata + independent
//! roundtrip, and `tests/one_encode_only.rs` pins the routing itself.

use gzippy::compress::{
    deflate::encode_gzip_bytes_to_vec, ldx::compress_for_diff, pipelined::PipelinedGzEncoder,
};
use gzippy::fixtures;

fn expected_header(level: u32) -> [u8; 10] {
    let xfl = match level {
        1 => 4,
        8..=u32::MAX => 2,
        _ => 0,
    };
    [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, xfl, 0xff]
}

#[test]
fn t1_gzip_matches_the_libdeflate_port_at_every_vendor_level() {
    let mut inputs = vec![
        ("empty", Vec::new()),
        ("one-byte", vec![0xA5]),
        ("short", b"libdeflate structure parity".to_vec()),
    ];
    inputs.extend(
        fixtures::NAMES
            .iter()
            .map(|&name| (name, fixtures::generate(name))),
    );

    for (name, input) in inputs {
        for level in 1..=9 {
            let gzip = encode_gzip_bytes_to_vec(&input, level);

            assert_eq!(
                &gzip[..10],
                &expected_header(level),
                "{name} L{level}: gzip metadata differs from libdeflate"
            );
            assert_eq!(
                u32::from_le_bytes(gzip[gzip.len() - 8..gzip.len() - 4].try_into().unwrap()),
                crc32fast::hash(&input),
                "{name} L{level}: CRC32"
            );
            assert_eq!(
                u32::from_le_bytes(gzip[gzip.len() - 4..].try_into().unwrap()),
                input.len() as u32,
                "{name} L{level}: ISIZE"
            );

            if matches!(level, 1 | 6 | 7) {
                // Documented L1/L6/L7 exception: these levels route to the legacy
                // encoder (measured size wins, see `level_uses_ldx`), so their body
                // is NOT the port's. Pin what still holds: valid gzip that
                // round-trips through an independent decoder. The routing itself
                // is pinned by `tests/one_encode_only.rs` (encoder-entry census).
                assert_eq!(
                    independent_roundtrip(&gzip),
                    input,
                    "{name} L{level}: exception-level output failed independent roundtrip"
                );
                continue;
            }

            let raw = compress_for_diff(level, &input)
                .unwrap_or_else(|| panic!("{name} L{level}: ldx level missing"));
            assert_eq!(
                &gzip[10..gzip.len() - 8],
                raw.as_slice(),
                "{name} L{level}: production DEFLATE body differs from libdeflate"
            );
        }
    }
}

/// Decode with an INDEPENDENT implementation (flate2/zlib-ng), never with our
/// own decoder — a shared bug would make both sides agree on a wrong answer.
fn independent_roundtrip(gz: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut d = flate2::read::GzDecoder::new(gz);
    let mut out = Vec::new();
    d.read_to_end(&mut out)
        .expect("independent decoder accepted the stream");
    out
}

#[test]
fn every_thread_count_emits_valid_deterministic_gzip() {
    // STEP-2 bar (CLAUDE.md): the ONLY correctness bar at every thread count is
    // VALID GZIP — a byte-exact roundtrip through our decoder plus one
    // independent decoder. T>1 may emit DIFFERENT BYTES than T1; this test
    // used to assert byte-identity with the whole-stream route, and that rule
    // was retracted three separate times (the branch's own T>1 pipeline commit
    // measured ~5% smaller L9 output at T4, which the old pin made impossible).
    // What survives here: validity at every requested thread count, and
    // run-to-run determinism at a FIXED (level, threads) — the size board
    // compares bytes, so a flaky thread count would make every verdict noise.
    let input = fixtures::generate("text");
    for level in 1..=12 {
        // Exercise ordinary, non-power-of-two, high, and extreme requested
        // counts so a future routing change cannot accidentally break validity
        // or determinism at one of them.
        for threads in [1, 2, 3, 4, 8, 16, 31, 64, 256, usize::MAX] {
            let first = {
                let mut encoder = PipelinedGzEncoder::new(level, threads);
                encoder.set_minimal_gzip_header(true);
                let mut out = Vec::new();
                encoder.compress_buffer_pure(&input, &mut out).unwrap();
                out
            };
            let ours = gzippy::decompress_with_threads(&first, 1).unwrap_or_else(|e| {
                panic!("L{level} T{threads}: our decoder rejected its own output: {e}")
            });
            assert_eq!(ours, input, "L{level} T{threads}: our-decoder roundtrip");
            assert_eq!(
                independent_roundtrip(&first),
                input,
                "L{level} T{threads}: independent-decoder roundtrip"
            );

            let second = {
                let mut encoder = PipelinedGzEncoder::new(level, threads);
                encoder.set_minimal_gzip_header(true);
                let mut out = Vec::new();
                encoder.compress_buffer_pure(&input, &mut out).unwrap();
                out
            };
            assert_eq!(
                second, first,
                "L{level} T{threads}: run-to-run output changed (non-deterministic)"
            );
        }
    }
}
