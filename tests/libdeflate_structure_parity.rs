//! Production T1 gzip must have the same DEFLATE structure as libdeflate.
//!
//! `ldx` is the in-tree byte-exact port of libdeflate's raw DEFLATE encoder.
//! Comparing the production gzip body against it catches a routing fallback,
//! a small-input shortcut, or a header metadata regression without requiring a
//! vendor executable in every test environment.

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
            let raw = compress_for_diff(level, &input)
                .unwrap_or_else(|| panic!("{name} L{level}: ldx level missing"));

            assert_eq!(
                &gzip[..10],
                &expected_header(level),
                "{name} L{level}: gzip metadata differs from libdeflate"
            );
            assert_eq!(
                &gzip[10..gzip.len() - 8],
                raw.as_slice(),
                "{name} L{level}: production DEFLATE body differs from libdeflate"
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
        }
    }
}

#[test]
fn every_thread_count_preserves_the_whole_stream_structure() {
    let input = fixtures::generate("text");
    for level in 1..=12 {
        let expected = encode_gzip_bytes_to_vec(&input, level);
        for threads in [2, 4, 16] {
            let mut encoder = PipelinedGzEncoder::new(level, threads);
            encoder.set_minimal_gzip_header(true);
            let mut actual = Vec::new();
            encoder.compress_buffer_pure(&input, &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "L{level} T{threads}: parallel entry point changed whole-stream structure"
            );
        }
    }
}
