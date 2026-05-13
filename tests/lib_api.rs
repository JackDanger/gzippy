//! Integration tests for gzippy's public library API.
//!
//! Each test pins a specific path in gzippy's routing table so regressions
//! in individual backends surface as clearly-named failures.

use gzippy::{DecodePath, GzippyError};
use std::io::{Read, Write};

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
        .map(|i| ((i as u64 * 6364136223846793005 + 1442695040888963407) >> 56) as u8)
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

// ── single-threaded round-trip: all standard levels ──────────────────────────

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

// ── parallel round-trips ──────────────────────────────────────────────────────

#[test]
fn round_trip_parallel_low_levels() {
    // T>1 L1–5 → ParallelGzEncoder (gzippy "GZ" multi-block)
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
    // T>1 L6–9 → PipelinedGzEncoder (single-member, gzip-compatible)
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

// ── compress_to_writer / compress_to_writer_with_threads ─────────────────────

#[test]
fn compress_to_writer_round_trip() {
    let data = make_text(128 * 1024);
    let reader = std::io::Cursor::new(&data);

    let mut compressed = Vec::new();
    let consumed = gzippy::compress_to_writer(reader, &mut compressed, 6).unwrap();

    assert_eq!(
        consumed,
        data.len() as u64,
        "consumed bytes should match input length"
    );
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn compress_to_writer_with_threads_t1_produces_standard_gzip() {
    // T=1 output must be a standard gzip stream (decompressible by flate2).
    let data = make_text(64 * 1024);
    let reader = std::io::Cursor::new(&data);

    let mut compressed = Vec::new();
    gzippy::compress_to_writer_with_threads(reader, &mut compressed, 6, 1).unwrap();

    let mut dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn compress_to_writer_with_threads_parallel_round_trip() {
    let data = make_text(512 * 1024);
    let reader = std::io::Cursor::new(&data);

    let mut compressed = Vec::new();
    gzippy::compress_to_writer_with_threads(reader, &mut compressed, 3, 4).unwrap();

    // Parallel L3 → GzippyParallel format; decompressible by gzippy.
    assert_eq!(gzippy::classify(&compressed, 4), DecodePath::GzippyParallel);
    let decompressed = gzippy::decompress_with_threads(&compressed, 4).unwrap();
    assert_eq!(decompressed, data);
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

#[test]
fn decompress_to_writer_with_threads_t1_matches_vec_api() {
    let data = make_text(64 * 1024);
    let compressed = gzip_encode_with_flate2(&data, 6);

    let via_vec = gzippy::decompress_with_threads(&compressed, 1).unwrap();

    let mut via_writer = Vec::new();
    let bytes = gzippy::decompress_to_writer_with_threads(&compressed, &mut via_writer, 1).unwrap();

    assert_eq!(via_vec, via_writer);
    assert_eq!(bytes, data.len() as u64);
}

// ── thread count parity: Vec API matches writer API ───────────────────────────

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
    // ISA-L on x86_64, LibdeflateSingle elsewhere.
    assert!(
        matches!(path, DecodePath::IsalSingle | DecodePath::LibdeflateSingle),
        "unexpected path {path:?} for T1 single-member"
    );
}

#[test]
fn classify_single_member_t4() {
    let compressed = gzip_encode_with_flate2(&make_text(4096), 6);
    let path = gzippy::classify(&compressed, 4);
    // Still single-member (no multi-block markers) even with T4.
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
    let data = make_text(512 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 3, 4).unwrap();
    let path = gzippy::classify(&compressed, 4);
    assert_eq!(
        path,
        DecodePath::GzippyParallel,
        "T>1 L3 should produce gzippy GZ multi-block format"
    );
}

// ── format interop ────────────────────────────────────────────────────────────

#[test]
fn single_thread_output_decompresses_with_standard_tools() {
    // T=1 always produces a standard gzip stream.
    let data = make_text(32 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 6, 1).unwrap();

    let mut dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn pipelined_parallel_high_level_is_standard_gzip() {
    // T>1 L6–9 → PipelinedGzEncoder → single-member → standard tools can read it.
    let data = make_text(128 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 9, 4).unwrap();

    let mut dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn parallel_low_level_gz_format_not_readable_by_standard_tools() {
    // T>1 L0–5 → ParallelGzEncoder → gzippy "GZ" multi-block format.
    // Standard tools see the BGZF-style header and fail or produce wrong output.
    let data = make_text(512 * 1024);
    let compressed = gzippy::compress_with_threads(&data, 3, 4).unwrap();

    // Confirm it is the gzippy format.
    assert_eq!(gzippy::classify(&compressed, 1), DecodePath::GzippyParallel);

    // flate2 GzDecoder will not produce the correct output for this format.
    let mut dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    let _ = dec.read_to_end(&mut out); // may error or produce partial output
    assert_ne!(
        out, data,
        "gzippy GZ format should NOT be readable by standard flate2 GzDecoder"
    );
}

#[test]
fn standard_gzip_from_any_tool_decompresses_correctly() {
    let data = make_text(32 * 1024);
    let compressed = gzip_encode_with_flate2(&data, 6);
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
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
    let compressed = gzippy::compress_with_threads(&data, 1, 1).unwrap();
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn level_clamping_does_not_panic() {
    let data = make_text(1024);
    // Levels 0 and 12 are the extremes of the valid range.
    gzippy::compress_with_threads(&data, 0, 1).unwrap();
    gzippy::compress_with_threads(&data, 12, 1).unwrap();
}

#[test]
fn compress_to_writer_empty_input() {
    let mut compressed = Vec::new();
    let consumed =
        gzippy::compress_to_writer(std::io::Cursor::new(b""), &mut compressed, 6).unwrap();
    assert_eq!(consumed, 0);
    let decompressed = gzippy::decompress(&compressed).unwrap();
    assert!(decompressed.is_empty());
}

// ── error handling ────────────────────────────────────────────────────────────

#[test]
fn decompress_non_gzip_returns_ok_empty() {
    // Non-gzip magic → Ok(empty) for all decompress variants.
    let garbage = b"this is not gzip data";
    assert_eq!(gzippy::decompress(garbage).unwrap(), b"");
    assert_eq!(gzippy::decompress_with_threads(garbage, 1).unwrap(), b"");

    let mut out = Vec::new();
    assert_eq!(gzippy::decompress_to_writer(garbage, &mut out).unwrap(), 0);
    assert_eq!(
        gzippy::decompress_to_writer_with_threads(garbage, &mut out, 1).unwrap(),
        0
    );
}

#[test]
fn decompress_truncated_gzip_errors() {
    let mut compressed = gzip_encode_with_flate2(&make_text(1024), 6);
    compressed.truncate(compressed.len() / 2);
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

#[test]
fn compress_raw_and_decompress_raw_roundtrip() {
    let data = make_text(64 * 1024);
    for level in [1u8, 6, 9] {
        let compressed = gzippy::compress_raw(&data, level).unwrap();
        let decompressed = gzippy::decompress_raw(&compressed).unwrap();
        assert_eq!(
            decompressed, data,
            "raw deflate roundtrip failed at level {level}"
        );
    }
}

#[test]
fn compress_raw_output_is_not_gzip() {
    // A gzip decoder must reject raw-deflate output (no gzip header present).
    let data = make_text(1024);
    let compressed = gzippy::compress_raw(&data, 6).unwrap();
    let mut gz_dec = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    assert!(
        gz_dec.read_to_end(&mut out).is_err(),
        "compress_raw output should be rejected by a gzip decoder"
    );
}

#[test]
fn decompress_raw_bad_data_errors() {
    let result = gzippy::decompress_raw(b"not valid deflate!!!!!");
    assert!(
        result.is_err(),
        "expected error on invalid raw deflate input"
    );
}

// ── Deflate64 (ZIP method 9) ──────────────────────────────────────────────────

// Build a stored Deflate64 block (BTYPE=00, same format as DEFLATE stored).
fn make_deflate64_stored(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 65535);
    let len = data.len() as u16;
    let mut out = vec![0x01u8]; // BFINAL=1, BTYPE=00
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn decompress_deflate64_stored_block_round_trip() {
    let data = b"hello, deflate64 world!";
    let compressed = make_deflate64_stored(data);
    let decompressed = gzippy::decompress_deflate64(&compressed).unwrap();
    assert_eq!(decompressed.as_slice(), data.as_slice());
}

#[test]
fn decompress_deflate64_to_writer_returns_byte_count() {
    let data = b"hello, deflate64 writer variant!";
    let compressed = make_deflate64_stored(data);
    let mut out = Vec::new();
    let n = gzippy::decompress_deflate64_to_writer(&compressed, &mut out).unwrap();
    assert_eq!(n, data.len() as u64);
    assert_eq!(out.as_slice(), data.as_slice());
}

#[test]
fn decompress_deflate64_invalid_input_errors() {
    let result = gzippy::decompress_deflate64(b"not valid deflate64!!!");
    assert!(result.is_err(), "expected error on invalid Deflate64 input");
}

#[test]
fn compress_raw_interops_with_flate2() {
    let data = make_text(32 * 1024);

    // gzippy compress, flate2 decompress
    let compressed = gzippy::compress_raw(&data, 6).unwrap();
    let mut dec = flate2::read::DeflateDecoder::new(compressed.as_slice());
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    assert_eq!(out, data);

    // flate2 compress, gzippy decompress
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
    enc.write_all(&data).unwrap();
    let flate2_compressed = enc.finish().unwrap();
    let decompressed = gzippy::decompress_raw(&flate2_compressed).unwrap();
    assert_eq!(decompressed, data);
}
