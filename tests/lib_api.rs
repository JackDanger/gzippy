//! Integration tests for gzippy's public library API.
//!
//! Each test exercises a specific path in gzippy's routing table so regressions
//! in individual backends surface as clearly-named failures.

use gzippy::{DecodePath, GzippyError};
use std::io::Write;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_text(n: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog\n"
        .iter()
        .cycle()
        .take(n)
        .copied()
        .collect()
}

fn make_incompressible(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 6364136223846793005 + 1442695040888963407) >> 56) as u8)
        .collect()
}

fn gzip_encode_with_flate2(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn multi_member_gzip(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for part in parts {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(part).unwrap();
        out.extend(enc.finish().unwrap());
    }
    out
}

// ── round-trip: compression levels 1–9, single-threaded ──────────────────────

#[test]
fn round_trip_levels_1_through_9_single_thread() {
    let data = make_text(64 * 1024);
    for level in 1u8..=9 {
        let compressed = gzippy::compress_with_threads(&data, level, 1)
            .unwrap_or_else(|e| panic!("compress L{level} failed: {e}"));
        let decompressed = gzippy::decompress(&compressed)
            .unwrap_or_else(|e| panic!("decompress L{level} failed: {e}"));
        assert_eq!(decompressed, data, "round-trip mismatch at L{level}");
    }
}

// ── round-trip: multi-threaded (parallel paths) ───────────────────────────────

#[test]
fn round_trip_parallel_low_levels() {
    // T>1 L1-5 → ParallelGzEncoder (gzippy "GZ" multi-block)
    let data = make_text(512 * 1024);
    for level in [1u8, 3, 5] {
        let compressed = gzippy::compress_with_threads(&data, level, 4)
            .unwrap_or_else(|e| panic!("compress T4 L{level} failed: {e}"));
        let decompressed = gzippy::decompress_with_threads(&compressed, 4)
            .unwrap_or_else(|e| panic!("decompress T4 L{level} failed: {e}"));
        assert_eq!(
            decompressed, data,
            "parallel round-trip mismatch at L{level}"
        );
    }
}

#[test]
fn round_trip_parallel_high_levels() {
    // T>1 L6-9 → PipelinedGzEncoder (single-member, gzip-compatible)
    let data = make_text(256 * 1024);
    for level in [6u8, 9] {
        let compressed = gzippy::compress_with_threads(&data, level, 4)
            .unwrap_or_else(|e| panic!("compress T4 L{level} failed: {e}"));
        let decompressed = gzippy::decompress(&compressed)
            .unwrap_or_else(|e| panic!("decompress T4 L{level} failed: {e}"));
        assert_eq!(
            decompressed, data,
            "pipelined round-trip mismatch at L{level}"
        );
    }
}

// ── decompress_to_writer ──────────────────────────────────────────────────────

#[test]
fn decompress_to_writer_matches_vec_api() {
    let data = make_text(32 * 1024);
    let compressed = gzippy::compress(&data, 6).unwrap();

    let via_vec = gzippy::decompress(&compressed).unwrap();

    let mut via_writer = Vec::new();
    let bytes = gzippy::decompress_to_writer(&compressed, &mut via_writer).unwrap();

    assert_eq!(via_vec, via_writer);
    assert_eq!(bytes, data.len() as u64);
}

// ── decompress_bytes with explicit thread counts ──────────────────────────────

#[test]
fn decompress_single_thread_matches_multi_thread() {
    let data = make_text(128 * 1024);
    let compressed = gzip_encode_with_flate2(&data, 6);

    let t1 = gzippy::decompress_with_threads(&compressed, 1).unwrap();
    let t4 = gzippy::decompress_with_threads(&compressed, 4).unwrap();
    assert_eq!(t1, t4);
    assert_eq!(t1, data);
}

// ── multi-member gzip ─────────────────────────────────────────────────────────

#[test]
fn decompress_multi_member_stream() {
    let parts: &[&[u8]] = &[
        &make_text(50_000),
        &make_incompressible(50_000),
        &make_text(50_000),
    ];
    let compressed = multi_member_gzip(parts);

    // Ensure it's actually seen as multi-member
    let path = gzippy::classify(&compressed, 1);
    assert_eq!(
        path,
        DecodePath::MultiMemberSeq,
        "expected multi-member classification"
    );

    let mut expected = Vec::new();
    for p in parts {
        expected.extend_from_slice(p);
    }

    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, expected);
}

#[test]
fn decompress_multi_member_parallel() {
    let parts: &[&[u8]] = &[&make_text(100_000), &make_text(100_000)];
    let compressed = multi_member_gzip(parts);

    let path = gzippy::classify(&compressed, 4);
    assert_eq!(path, DecodePath::MultiMemberPar);

    let mut expected = Vec::new();
    for p in parts {
        expected.extend_from_slice(p);
    }

    let decompressed = gzippy::decompress_with_threads(&compressed, 4).unwrap();
    assert_eq!(decompressed, expected);
}

// ── classify (routing inspection) ────────────────────────────────────────────

#[test]
fn classify_single_member_t1() {
    let compressed = gzip_encode_with_flate2(&make_text(4096), 6);
    let path = gzippy::classify(&compressed, 1);
    // On x86_64 with ISA-L: IsalSingle. Elsewhere: LibdeflateSingle.
    assert!(
        matches!(path, DecodePath::IsalSingle | DecodePath::LibdeflateSingle),
        "unexpected path {path:?} for T1 single-member"
    );
}

#[test]
fn classify_single_member_t4() {
    let compressed = gzip_encode_with_flate2(&make_text(4096), 6);
    let path = gzippy::classify(&compressed, 4);
    // Still single-member (no multi-block markers) even with T4
    assert!(
        matches!(path, DecodePath::IsalSingle | DecodePath::LibdeflateSingle),
        "unexpected path {path:?} for T4 single-member"
    );
}

#[test]
fn classify_multi_member_t1_vs_t4() {
    let compressed = multi_member_gzip(&[&make_text(10_000), &make_text(10_000)]);
    assert_eq!(gzippy::classify(&compressed, 1), DecodePath::MultiMemberSeq);
    assert_eq!(gzippy::classify(&compressed, 4), DecodePath::MultiMemberPar);
}

#[test]
fn classify_gzippy_parallel_format() {
    // Parallel L1-5 with T>1 produces the gzippy "GZ" multi-block format.
    let data = make_text(512 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 3, 4).unwrap();
    let path = gzippy::classify(&compressed, 4);
    assert_eq!(
        path,
        DecodePath::GzippyParallel,
        "T>1 L3 output should be GzippyParallel format"
    );
}

// ── edge cases ────────────────────────────────────────────────────────────────

#[test]
fn empty_input_round_trip() {
    let compressed = gzippy::compress(b"", 6).unwrap();
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert!(decompressed.is_empty());
}

#[test]
fn single_byte_round_trip() {
    let data = b"x";
    let compressed = gzippy::compress(data, 6).unwrap();
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed.as_slice(), data);
}

#[test]
fn incompressible_data_round_trips() {
    let data = make_incompressible(128 * 1024);
    let compressed = gzippy::compress(&data, 1).unwrap();
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn level_clamping_does_not_panic() {
    // Level 0 and 12 are valid; these should not error.
    let data = make_text(1024);
    gzippy::compress_with_threads(&data, 0, 1).unwrap();
    gzippy::compress_with_threads(&data, 12, 1).unwrap();
}

// ── interoperability ──────────────────────────────────────────────────────────

#[test]
fn gzippy_output_decompresses_with_flate2() {
    // Single-threaded output must be a valid standard gzip stream.
    let data = make_text(32 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 6, 1).unwrap();

    let mut dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn flate2_output_decompresses_with_gzippy() {
    // gzippy must handle standard gzip input produced by any tool.
    let data = make_text(32 * 1024);
    let compressed = gzip_encode_with_flate2(&data, 6);
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

// ── error handling ────────────────────────────────────────────────────────────

#[test]
fn decompress_garbage_returns_ok_empty() {
    // Non-gzip input (no 0x1f 0x8b header) returns 0 bytes rather than an error —
    // consistent with how the CLI handles non-gzip stdin.
    let result = gzippy::decompress(b"this is not gzip data");
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn decompress_truncated_gzip_errors() {
    // A valid-looking header but truncated body should be an error.
    let mut compressed = gzip_encode_with_flate2(&make_text(1024), 6);
    let trunc_len = compressed.len() / 2;
    compressed.truncate(trunc_len);
    let result = gzippy::decompress(&compressed);
    assert!(
        matches!(
            result,
            Err(GzippyError::Decompression(_))
                | Err(GzippyError::Io(_))
                | Err(GzippyError::InvalidArgument(_))
        ),
        "expected decompression error, got {result:?}"
    );
}
