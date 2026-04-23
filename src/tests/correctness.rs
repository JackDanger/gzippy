//! Comprehensive correctness tests — from tiny to full-size.
//!
//! Philosophy: every algorithm is correct for its designated inputs.
//! There is no "fallback" — just routing decisions. If an algorithm
//! is routed to, it MUST produce correct output.
//!
//! Layers:
//!   Layer 0: Data generators (oracle)
//!   Layer 1: Gzip format parsing (header, trailer, ISIZE, CRC)
//!   Layer 2: Format detection (BGZF, multi-member, single-member)
//!   Layer 3: Individual decoders (libdeflate, consume_first, marker_decode)
//!   Layer 4: Routing correctness (router chooses right path, path succeeds)
//!   Layer 5: Cross-validation (all decoders agree on same input)
//!   Layer 6: Thread-count independence (T1..T8 produce identical output)
//!   Layer 7: Performance bounds (throughput within expected range)

#[cfg(test)]
mod tests {
    use std::io::Write;

    // =========================================================================
    // Layer 0: Data generators
    // =========================================================================

    fn make_zeros(size: usize) -> Vec<u8> {
        vec![0u8; size]
    }

    fn make_single_byte(size: usize, byte: u8) -> Vec<u8> {
        vec![byte; size]
    }

    fn make_sequential(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    fn make_random_seeded(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
        }
        data
    }

    fn make_mixed(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xcafebabe;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnop\n",
        ];
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 2 {
                data.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[((rng >> 24) as usize) % phrases.len()];
                let remaining = size - data.len();
                data.extend_from_slice(&phrase[..remaining.min(phrase.len())]);
            }
        }
        data.truncate(size);
        data
    }

    /// Two-symbol alphabet — triggers pathological Huffman trees
    fn make_binary(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xabcd1234;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push(if (rng >> 32).is_multiple_of(2) {
                b'A'
            } else {
                b'B'
            });
        }
        data
    }

    fn compress_single_member(data: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    fn compress_multi_member(data: &[u8]) -> Vec<u8> {
        let chunk_size = 256 * 1024;
        let mut output = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(chunk).unwrap();
            output.extend_from_slice(&enc.finish().unwrap());
        }
        output
    }

    fn compress_bgzf(data: &[u8]) -> Vec<u8> {
        use crate::compress::parallel::{compress_single_member, GzipHeaderInfo};
        let mut output = Vec::new();
        let header = GzipHeaderInfo::default();
        compress_single_member(&mut output, data, 1, &header).unwrap();
        output
    }

    fn decompress_reference(gz_data: &[u8]) -> Vec<u8> {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;
        let mut decoder = MultiGzDecoder::new(gz_data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).unwrap();
        output
    }

    fn get_deflate_data(gz: &[u8]) -> &[u8] {
        let header = crate::experiments::marker_decode::skip_gzip_header(gz).unwrap();
        &gz[header..gz.len() - 8]
    }

    // =========================================================================
    // Layer 1: Gzip format parsing
    // =========================================================================

    #[test]
    fn test_header_too_short() {
        assert!(crate::experiments::marker_decode::skip_gzip_header(&[0x1f, 0x8b]).is_err());
        assert!(crate::experiments::marker_decode::skip_gzip_header(&[]).is_err());
        assert!(crate::experiments::marker_decode::skip_gzip_header(&[0x1f]).is_err());
    }

    #[test]
    fn test_header_bad_magic() {
        let mut data = compress_single_member(b"hello");
        data[0] = 0x00;
        assert!(crate::experiments::marker_decode::skip_gzip_header(&data).is_err());
    }

    #[test]
    fn test_header_bad_method() {
        let mut data = compress_single_member(b"hello");
        data[2] = 0x09; // method must be 8 (deflate)
        assert!(crate::experiments::marker_decode::skip_gzip_header(&data).is_err());
    }

    #[test]
    fn test_header_minimal_size() {
        let data = compress_single_member(b"hello world");
        let header_size = crate::experiments::marker_decode::skip_gzip_header(&data).unwrap();
        assert!(header_size >= 10, "minimal header must be >= 10 bytes");
        assert!(header_size < data.len(), "header can't be the whole file");
    }

    #[test]
    fn test_header_fextra() {
        let bgzf = compress_bgzf(b"test data for fextra");
        let header_size = crate::experiments::marker_decode::skip_gzip_header(&bgzf).unwrap();
        assert!(
            header_size > 10,
            "BGZF header with FEXTRA should be > 10 bytes, got {}",
            header_size
        );
    }

    #[test]
    fn test_header_fname() {
        let mut header = vec![0x1f, 0x8b, 0x08];
        header.push(0x08); // FNAME flag
        header.extend_from_slice(&[0, 0, 0, 0]); // MTIME
        header.push(0); // XFL
        header.push(0xFF); // OS
        header.extend_from_slice(b"test.txt\0");
        header.extend_from_slice(&[0x03, 0x00]); // empty stored block
        header.extend_from_slice(&[0, 0, 0, 0]); // CRC32
        header.extend_from_slice(&[0, 0, 0, 0]); // ISIZE

        let size = crate::experiments::marker_decode::skip_gzip_header(&header).unwrap();
        assert_eq!(size, 10 + 9, "10 base + 'test.txt\\0' = 19");
    }

    #[test]
    fn test_header_fcomment() {
        let mut header = vec![0x1f, 0x8b, 0x08];
        header.push(0x10); // FCOMMENT flag
        header.extend_from_slice(&[0, 0, 0, 0, 0, 0xFF]); // MTIME + XFL + OS
        header.extend_from_slice(b"a comment\0");
        header.extend_from_slice(&[0x03, 0x00, 0, 0, 0, 0, 0, 0, 0, 0]);
        let size = crate::experiments::marker_decode::skip_gzip_header(&header).unwrap();
        assert_eq!(size, 10 + 10, "10 base + 'a comment\\0' = 20");
    }

    #[test]
    fn test_isize_correct_for_various_sizes() {
        for &size in &[0usize, 1, 100, 10_000, 100_000, 1_000_000] {
            let data = make_mixed(size);
            let gz = compress_single_member(&data);
            let isize_val = u32::from_le_bytes([
                gz[gz.len() - 4],
                gz[gz.len() - 3],
                gz[gz.len() - 2],
                gz[gz.len() - 1],
            ]);
            assert_eq!(
                isize_val as usize, size,
                "ISIZE must equal {} for size {}",
                size, size
            );
        }
    }

    #[test]
    fn test_crc32_in_trailer() {
        let data = b"hello world! testing crc32 validation.";
        let gz = compress_single_member(data);
        let stored_crc = u32::from_le_bytes([
            gz[gz.len() - 8],
            gz[gz.len() - 7],
            gz[gz.len() - 6],
            gz[gz.len() - 5],
        ]);
        let computed_crc = crc32fast::hash(data);
        assert_eq!(stored_crc, computed_crc);
    }

    #[test]
    fn test_crc32_zeros() {
        let data = make_zeros(1000);
        let gz = compress_single_member(&data);
        let stored_crc = u32::from_le_bytes([
            gz[gz.len() - 8],
            gz[gz.len() - 7],
            gz[gz.len() - 6],
            gz[gz.len() - 5],
        ]);
        let computed_crc = crc32fast::hash(&data);
        assert_eq!(stored_crc, computed_crc);
    }

    // =========================================================================
    // Layer 2: Format detection
    // =========================================================================

    #[test]
    fn test_detect_bgzf_positive() {
        let bgzf = compress_bgzf(&make_mixed(512 * 1024));
        assert!(crate::decompress::format::has_bgzf_markers(&bgzf));
    }

    #[test]
    fn test_detect_bgzf_negative_single() {
        let single = compress_single_member(&make_mixed(512 * 1024));
        assert!(!crate::decompress::format::has_bgzf_markers(&single));
    }

    #[test]
    fn test_detect_bgzf_negative_multi() {
        let multi = compress_multi_member(&make_mixed(512 * 1024));
        assert!(!crate::decompress::format::has_bgzf_markers(&multi));
    }

    #[test]
    fn test_detect_bgzf_too_short() {
        assert!(!crate::decompress::format::has_bgzf_markers(&[0x1f, 0x8b]));
        assert!(!crate::decompress::format::has_bgzf_markers(&[]));
    }

    #[test]
    fn test_detect_multi_member_positive() {
        let multi = compress_multi_member(&make_mixed(2 * 1024 * 1024));
        assert!(crate::decompress::format::is_likely_multi_member(&multi));
    }

    #[test]
    fn test_detect_multi_member_negative_single() {
        let single = compress_single_member(&make_mixed(2 * 1024 * 1024));
        assert!(!crate::decompress::format::is_likely_multi_member(&single));
    }

    #[test]
    fn test_detect_format_mutual_exclusion() {
        let data = make_mixed(1024 * 1024);
        let single = compress_single_member(&data);
        let multi = compress_multi_member(&data);
        let bgzf = compress_bgzf(&data);

        assert!(!crate::decompress::format::has_bgzf_markers(&single));
        assert!(!crate::decompress::format::is_likely_multi_member(&single));

        assert!(!crate::decompress::format::has_bgzf_markers(&multi));
        assert!(crate::decompress::format::is_likely_multi_member(&multi));

        assert!(crate::decompress::format::has_bgzf_markers(&bgzf));
    }

    // =========================================================================
    // Layer 3: Individual decoder correctness — libdeflate
    // =========================================================================

    #[test]
    fn test_libdeflate_empty() {
        let gz = compress_single_member(b"");
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_libdeflate_single_byte() {
        for b in [0u8, 1, 127, 128, 254, 255] {
            let gz = compress_single_member(&[b]);
            let mut out = Vec::new();
            crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
            assert_eq!(out, vec![b], "byte {}", b);
        }
    }

    #[test]
    fn test_libdeflate_zeros_1m() {
        let data = make_zeros(1_000_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_libdeflate_random_1m() {
        let data = make_random_seeded(1_000_000, 42);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_libdeflate_mixed_2m() {
        let data = make_mixed(2_000_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // =========================================================================
    // Layer 3: Individual decoder correctness — consume_first
    // =========================================================================

    #[test]
    fn test_consume_first_empty() {
        let gz = compress_single_member(b"");
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_consume_first_single_byte() {
        for b in [0u8, 1, 127, 255] {
            let gz = compress_single_member(&[b]);
            let deflate = get_deflate_data(&gz);
            let mut out = vec![0u8; 65536];
            let size =
                crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                    .unwrap();
            assert_eq!(&out[..size], &[b], "byte {}", b);
        }
    }

    #[test]
    fn test_consume_first_zeros_500k() {
        let data = make_zeros(500_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    #[test]
    fn test_consume_first_sequential_500k() {
        let data = make_sequential(500_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    #[test]
    fn test_consume_first_random_500k() {
        let data = make_random_seeded(500_000, 12345);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    #[test]
    fn test_consume_first_mixed_2m() {
        let data = make_mixed(2_000_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    #[test]
    fn test_consume_first_binary_500k() {
        let data = make_binary(500_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    // =========================================================================
    // Layer 3: Individual decoder correctness — inflate_into_pub
    // =========================================================================

    #[test]
    fn test_inflate_into_pub_zeros_500k() {
        let data = make_zeros(500_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size = crate::decompress::bgzf::inflate_into_pub(deflate, &mut out).unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    #[test]
    fn test_inflate_into_pub_mixed_2m() {
        let data = make_mixed(2_000_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let size = crate::decompress::bgzf::inflate_into_pub(deflate, &mut out).unwrap();
        assert_eq!(&out[..size], &data[..]);
    }

    // =========================================================================
    // Layer 3: Individual decoder correctness — marker_decode
    //
    // The marker decoder is tested at multiple granularities:
    //   a) Output length correctness (does it produce the right number of bytes?)
    //   b) Output content correctness (are the bytes right?)
    //   c) Scaling (does it work at 1B, 1KB, 1MB?)
    //   d) Data patterns (zeros, sequential, random, binary, RLE)
    //   e) Marker resolution (split + rejoin matches sequential)
    // =========================================================================

    /// Helper: decode deflate data through marker_decode, return output bytes
    fn marker_decode_full(deflate: &[u8]) -> Vec<u8> {
        let mut decoder = crate::experiments::marker_decode::MarkerDecoder::new(deflate, 0);
        match decoder.decode_until(usize::MAX) {
            Ok(_) => {}
            Err(e) => {
                panic!(
                    "marker_decode failed after {} output bytes: {}",
                    decoder.output().len(),
                    e
                );
            }
        }
        decoder.output().iter().map(|&v| v as u8).collect()
    }

    #[test]
    fn test_marker_decode_empty() {
        let gz = compress_single_member(b"");
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 0, "empty input must produce empty output");
    }

    #[test]
    fn test_marker_decode_single_bytes() {
        for b in [0u8, 1, 127, 128, 254, 255] {
            let gz = compress_single_member(&[b]);
            let deflate = get_deflate_data(&gz);
            let out = marker_decode_full(deflate);
            assert_eq!(out, vec![b], "single byte {} failed", b);
        }
    }

    #[test]
    fn test_marker_decode_short_strings() {
        for s in [b"hello" as &[u8], b"a", b"ab", b"abc", b"\x00\x00\x00"] {
            let gz = compress_single_member(s);
            let deflate = get_deflate_data(&gz);
            let out = marker_decode_full(deflate);
            assert_eq!(out, s, "short string {:?} failed", s);
        }
    }

    // Mixed data tests at various sizes. Sizes 50-1024 currently
    // trigger marker_decode bugs (excess output, "Invalid distance code").
    // The 10KB+ sizes work because flate2's compressor produces different
    // deflate block structures at larger sizes.

    // The marker_decode bug for mixed data depends on deflate block structure,
    // not data size per se. The make_mixed generator at certain sizes produces
    // deflate streams where the bit-by-bit reader overshoots past EOB.
    // Test at multiple sizes to find which work and which don't.

    #[test]
    fn test_marker_decode_mixed_100kb() {
        let data = make_mixed(100_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), data.len(), "length mismatch at 100KB");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_mixed_500k() {
        let data = make_mixed(500_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), data.len(), "length mismatch at 500KB");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_sequential_100k() {
        let data = make_sequential(100_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), data.len(), "length mismatch");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_random_100k() {
        let data = make_random_seeded(100_000, 42);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), data.len(), "length mismatch");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_binary_100k() {
        let data = make_binary(100_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), data.len(), "length mismatch");
        assert_eq!(out, data);
    }

    // -- RLE / zeros: the pathological case --

    #[test]
    fn test_marker_decode_zeros_10() {
        let data = make_zeros(10);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 10, "zeros 10B: length");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_zeros_258() {
        let data = make_zeros(258);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 258, "zeros 258B: length (one max-length match)");
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_zeros_1000() {
        let data = make_zeros(1000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 1000, "zeros 1KB: length");
        assert_eq!(out, data);
    }

    // =========================================================================
    // marker_decode: individual size + pattern tests
    //
    // Fix: match libdeflate/rapidgzip behavior — after BFINAL block
    // returns Ok(true), DO NOT attempt to read more blocks.
    // =========================================================================

    #[test]
    fn test_marker_decode_zeros_10k() {
        let data = make_zeros(10_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 10_000);
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_zeros_100k() {
        let data = make_zeros(100_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 100_000);
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_zeros_1m() {
        let data = make_zeros(1_000_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 1_000_000, "got {}", out.len());
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_rle_a_1m() {
        let data = make_single_byte(1_000_000, b'A');
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 1_000_000, "got {}", out.len());
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_rle_ff_1m() {
        let data = make_single_byte(1_000_000, 0xFF);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(out.len(), 1_000_000, "got {}", out.len());
        assert_eq!(out, data);
    }

    #[test]
    fn test_marker_decode_mixed_small_sizes() {
        for &size in &[10, 20, 30, 40, 50, 100, 200, 500, 1024, 2000, 5000] {
            let data = make_mixed(size);
            let gz = compress_single_member(&data);
            let deflate = get_deflate_data(&gz);
            let mut decoder = crate::experiments::marker_decode::MarkerDecoder::new(deflate, 0);
            match decoder.decode_until(usize::MAX) {
                Ok(_) => {
                    let out: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
                    assert_eq!(
                        out.len(),
                        size,
                        "mixed {} bytes: expected {} got {}",
                        size,
                        size,
                        out.len()
                    );
                    assert_eq!(out, data, "mixed {} bytes: content mismatch", size);
                }
                Err(e) => {
                    panic!(
                        "mixed {} bytes: decode error after {} output bytes: {}",
                        size,
                        decoder.output().len(),
                        e
                    );
                }
            }
        }
    }

    /// Find the size at which marker_decode errors on mixed data.
    /// Diagnostic — always passes, prints result.
    #[test]
    fn test_marker_decode_mixed_find_error_threshold() {
        let mut last_ok = 0;
        for &size in &[
            10, 50, 100, 200, 500, 800, 1000, 1024, 2000, 5000, 10_000, 50_000,
        ] {
            let data = make_mixed(size);
            let gz = compress_single_member(&data);
            let deflate = get_deflate_data(&gz);
            let mut decoder = crate::experiments::marker_decode::MarkerDecoder::new(deflate, 0);
            match decoder.decode_until(usize::MAX) {
                Ok(_) => {
                    let out: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
                    if out == data {
                        last_ok = size;
                    } else {
                        eprintln!("CONTENT MISMATCH at size {}: got {} bytes", size, out.len());
                        return;
                    }
                }
                Err(e) => {
                    eprintln!(
                        "DECODE ERROR at size {}: {} (output so far: {} bytes), last_ok={}",
                        size,
                        e,
                        decoder.output().len(),
                        last_ok
                    );
                    return;
                }
            }
        }
        eprintln!("marker_decode mixed: OK up to {}", last_ok);
    }

    /// Verify marker_decode handles all mixed sizes correctly (regression test).
    #[test]
    fn test_marker_decode_mixed_all_sizes_correct() {
        for &size in &[10, 20, 30, 40, 50, 100, 500, 1024, 5000, 10_000] {
            let data = make_mixed(size);
            let gz = compress_single_member(&data);
            let deflate = get_deflate_data(&gz);
            let out = marker_decode_full(deflate);
            assert_eq!(out.len(), size, "mixed {}: length mismatch", size);
            assert_eq!(out, data, "mixed {}: content mismatch", size);
        }
    }

    /// Verify marker_decode handles all zeros sizes (up to 1M) correctly (regression test).
    #[test]
    fn test_marker_decode_zeros_all_sizes_correct() {
        for &size in &[
            1, 10, 100, 258, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000, 1_000_000,
        ] {
            let data = make_zeros(size);
            let gz = compress_single_member(&data);
            let deflate = get_deflate_data(&gz);
            let out = marker_decode_full(deflate);
            assert_eq!(out.len(), size, "zeros {}: length mismatch", size);
            assert_eq!(out, data, "zeros {}: content mismatch", size);
        }
    }

    // -- No-markers invariant --

    #[test]
    fn test_marker_decode_no_markers_from_beginning() {
        for data in [
            make_mixed(100_000),
            make_zeros(100_000),
            make_sequential(100_000),
            make_random_seeded(100_000, 99),
        ] {
            let gz = compress_single_member(&data);
            let deflate = get_deflate_data(&gz);
            let mut decoder = crate::experiments::marker_decode::MarkerDecoder::new(deflate, 0);
            decoder.decode_until(usize::MAX).unwrap();
            assert_eq!(
                decoder.marker_count(),
                0,
                "decoding from bit 0 must produce zero markers"
            );
        }
    }

    // -- Marker resolution --

    #[test]
    fn test_marker_decode_split_and_rejoin() {
        let data = make_mixed(2_000_000);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);

        let scan =
            crate::decompress::scan_inflate::scan_deflate_fast(deflate, 256 * 1024, 0).unwrap();
        if scan.checkpoints.is_empty() {
            eprintln!("no checkpoints, skipping");
            return;
        }

        let cp = &scan.checkpoints[0];
        let split_bit = cp.input_byte_pos * 8 - (cp.bitsleft as u8) as usize;

        // Chunk 0
        let mut dec0 = crate::experiments::marker_decode::MarkerDecoder::new(deflate, 0);
        dec0.decode_until_bit(usize::MAX, split_bit).unwrap();
        assert_eq!(dec0.marker_count(), 0);
        let chunk0: Vec<u8> = dec0.output().iter().map(|&v| v as u8).collect();

        // Chunk 1
        let start_byte = split_bit / 8;
        let rel_bit = split_bit % 8;
        let mut dec1 =
            crate::experiments::marker_decode::MarkerDecoder::new(&deflate[start_byte..], rel_bit);
        dec1.decode_until(usize::MAX).unwrap();

        // Window from chunk 0's last 32KB
        let ws = crate::experiments::marker_decode::WINDOW_SIZE;
        let window_start = chunk0.len().saturating_sub(ws);
        let window = &chunk0[window_start..];

        // Resolve
        let mut chunk1_data = dec1.output().to_vec();
        crate::experiments::marker_decode::replace_markers(&mut chunk1_data, window);
        let chunk1: Vec<u8> = chunk1_data.iter().map(|&v| v as u8).collect();

        let mut combined = chunk0;
        combined.extend_from_slice(&chunk1);

        assert_eq!(combined.len(), scan.total_output_size, "length mismatch");
        assert_eq!(combined, data, "split+rejoin must match original");
    }

    // =========================================================================
    // Layer 3: Sequential multi-member decoder
    // =========================================================================

    #[test]
    fn test_sequential_multi_member_mixed() {
        let data = make_mixed(2_000_000);
        let multi = compress_multi_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_multi_member_sequential(&multi, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_sequential_multi_member_zeros() {
        let data = make_zeros(500_000);
        let multi = compress_multi_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_multi_member_sequential(&multi, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // =========================================================================
    // Layer 4: Routing — each route must succeed for its format
    // =========================================================================

    #[test]
    fn test_bgzf_route_t1() {
        let data = make_mixed(2 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        assert!(crate::decompress::format::has_bgzf_markers(&bgzf));
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 1).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_bgzf_route_t4() {
        let data = make_mixed(2 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_bgzf_route_t8() {
        let data = make_mixed(4 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 8).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_bgzf_route_zeros() {
        let data = make_zeros(1_000_000);
        let bgzf = compress_bgzf(&data);
        assert!(crate::decompress::format::has_bgzf_markers(&bgzf));
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_bgzf_route_small() {
        let data = b"small bgzf test";
        let bgzf = compress_bgzf(data);
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 1).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_multi_member_parallel_route() {
        let data = make_mixed(2 * 1024 * 1024);
        let multi = compress_multi_member(&data);
        assert!(crate::decompress::format::is_likely_multi_member(&multi));
        let output =
            crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(&multi, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_single_member_route() {
        let data = make_mixed(2 * 1024 * 1024);
        let single = compress_single_member(&data);
        assert!(!crate::decompress::format::has_bgzf_markers(&single));
        assert!(!crate::decompress::format::is_likely_multi_member(&single));
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&single, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // =========================================================================
    // Layer 5: Cross-validation — all production decoders agree
    // =========================================================================

    fn cross_validate_production(data: &[u8], label: &str) {
        let gz = compress_single_member(data);
        let deflate = get_deflate_data(&gz);

        let ref_output = decompress_reference(&gz);
        assert_eq!(ref_output, data, "{}: flate2 reference", label);

        // libdeflate (production gzip path)
        let mut ld_out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut ld_out).unwrap();
        assert_eq!(ld_out, data, "{}: libdeflate", label);

        // consume_first (experimental pure-Rust)
        let mut cf_out = vec![0u8; data.len() + 65536];
        let cf_size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut cf_out)
                .unwrap();
        assert_eq!(&cf_out[..cf_size], data, "{}: consume_first", label);

        // inflate_into_pub (production BGZF inflate)
        let mut ip_out = vec![0u8; data.len() + 65536];
        let ip_size = crate::decompress::bgzf::inflate_into_pub(deflate, &mut ip_out).unwrap();
        assert_eq!(&ip_out[..ip_size], data, "{}: inflate_into_pub", label);
    }

    #[test]
    fn test_production_decoders_agree_zeros() {
        cross_validate_production(&make_zeros(1_000_000), "zeros");
    }

    #[test]
    fn test_production_decoders_agree_sequential() {
        cross_validate_production(&make_sequential(1_000_000), "sequential");
    }

    #[test]
    fn test_production_decoders_agree_random() {
        cross_validate_production(&make_random_seeded(1_000_000, 77), "random");
    }

    #[test]
    fn test_production_decoders_agree_mixed() {
        cross_validate_production(&make_mixed(2_000_000), "mixed");
    }

    #[test]
    fn test_production_decoders_agree_binary() {
        cross_validate_production(&make_binary(500_000), "binary");
    }

    #[test]
    fn test_production_decoders_agree_rle() {
        cross_validate_production(&make_single_byte(1_000_000, b'Z'), "rle-Z");
    }

    // Marker decoder separately — known to have a zeros bug
    fn cross_validate_marker_decode(data: &[u8], label: &str) {
        let gz = compress_single_member(data);
        let deflate = get_deflate_data(&gz);
        let out = marker_decode_full(deflate);
        assert_eq!(
            out.len(),
            data.len(),
            "{}: marker_decode length: got {} expected {}",
            label,
            out.len(),
            data.len()
        );
        assert_eq!(out, data, "{}: marker_decode content", label);
    }

    #[test]
    fn test_marker_decode_agrees_sequential() {
        cross_validate_marker_decode(&make_sequential(1_000_000), "sequential");
    }

    #[test]
    fn test_marker_decode_agrees_random() {
        cross_validate_marker_decode(&make_random_seeded(1_000_000, 77), "random");
    }

    #[test]
    fn test_marker_decode_agrees_mixed() {
        cross_validate_marker_decode(&make_mixed(2_000_000), "mixed");
    }

    #[test]
    fn test_marker_decode_agrees_binary() {
        cross_validate_marker_decode(&make_binary(500_000), "binary");
    }

    #[test]
    fn test_marker_decode_agrees_zeros() {
        cross_validate_marker_decode(&make_zeros(1_000_000), "zeros");
    }

    #[test]
    fn test_marker_decode_agrees_rle() {
        cross_validate_marker_decode(&make_single_byte(1_000_000, b'A'), "rle-A");
    }

    // =========================================================================
    // Layer 6: Thread-count independence
    // =========================================================================

    #[test]
    fn test_bgzf_t1_through_t8() {
        let data = make_mixed(4 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        for t in 1..=8 {
            let output =
                crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, t).unwrap();
            assert_eq!(output, data, "BGZF T{} differs", t);
        }
    }

    #[test]
    fn test_multi_member_t1_through_t8() {
        let data = make_mixed(4 * 1024 * 1024);
        let multi = compress_multi_member(&data);

        let mut t1 = Vec::new();
        crate::decompress::decompress_multi_member_sequential(&multi, &mut t1).unwrap();
        assert_eq!(t1, data, "T1 sequential");

        for t in 2..=8 {
            let output =
                crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(&multi, t)
                    .unwrap();
            assert_eq!(output, data, "multi-member T{}", t);
        }
    }

    #[test]
    fn test_bgzf_zeros_t1_through_t4() {
        let data = make_zeros(2 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        for t in 1..=4 {
            let output =
                crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, t).unwrap();
            assert_eq!(output, data, "BGZF zeros T{}", t);
        }
    }

    // =========================================================================
    // Layer 7: Performance bounds (sanity — catch degenerate regressions)
    // =========================================================================

    #[test]
    fn test_perf_bgzf_t1_not_degenerate() {
        let data = make_mixed(4 * 1024 * 1024);
        let bgzf = compress_bgzf(&data);
        let t = std::time::Instant::now();
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 1).unwrap();
        let mbps = output.len() as f64 / t.elapsed().as_secs_f64() / 1e6;
        assert_eq!(output, data);
        assert!(mbps > 50.0, "BGZF T1: {:.0} MB/s too slow", mbps);
    }

    #[test]
    fn test_perf_single_member_not_degenerate() {
        let data = make_mixed(4 * 1024 * 1024);
        let single = compress_single_member(&data);
        let t = std::time::Instant::now();
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&single, &mut out).unwrap();
        let mbps = out.len() as f64 / t.elapsed().as_secs_f64() / 1e6;
        assert_eq!(out, data);
        assert!(mbps > 50.0, "single-member: {:.0} MB/s too slow", mbps);
    }

    #[test]
    fn test_perf_consume_first_not_degenerate() {
        let data = make_mixed(4 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let deflate = get_deflate_data(&gz);
        let mut out = vec![0u8; data.len() + 65536];
        let t = std::time::Instant::now();
        let size =
            crate::experiments::consume_first_decode::inflate_consume_first(deflate, &mut out)
                .unwrap();
        let mbps = size as f64 / t.elapsed().as_secs_f64() / 1e6;
        assert_eq!(&out[..size], &data[..]);
        assert!(mbps > 50.0, "consume_first: {:.0} MB/s too slow", mbps);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_max_distance_match() {
        let mut data = Vec::with_capacity(40_000);
        let pattern = b"ABCDEFGHIJKLMNOP";
        data.extend_from_slice(pattern);
        data.extend_from_slice(&vec![0u8; 32768 - pattern.len()]);
        data.extend_from_slice(pattern);
        let gz = compress_single_member(&data);
        let ref_out = decompress_reference(&gz);
        assert_eq!(ref_out, data);
    }

    #[test]
    fn test_rle_10m() {
        let data = vec![b'A'; 10_000_000];
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_incompressible_random() {
        let data = make_random_seeded(100_000, 999);
        let gz = compress_single_member(&data);
        assert!(gz.len() > data.len() * 95 / 100);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_bgzf_exact_block_boundary() {
        let data = vec![b'X'; 65535];
        let bgzf = compress_bgzf(&data);
        let output = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&bgzf, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_multi_member_single_chunk() {
        let data = make_mixed(100_000);
        let multi = compress_multi_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_multi_member_sequential(&multi, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // =========================================================================
    // Corruption detection
    // =========================================================================

    #[test]
    fn test_corrupted_deflate_detected() {
        let data = make_mixed(100_000);
        let mut gz = compress_single_member(&data);
        let mid = gz.len() / 2;
        gz[mid] ^= 0xFF;
        let mut out = Vec::new();
        let result = crate::decompress::decompress_single_member_libdeflate(&gz, &mut out);
        assert!(result.is_err() || out != data);
    }

    #[test]
    fn test_truncated_gzip_detected() {
        let data = make_mixed(100_000);
        let gz = compress_single_member(&data);
        let truncated = &gz[..gz.len() - 100];
        let mut out = Vec::new();
        let result = crate::decompress::decompress_single_member_libdeflate(truncated, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_crc_detected_by_reference() {
        let data = make_mixed(10_000);
        let mut gz = compress_single_member(&data);
        // Flip a bit in the CRC32 field
        let idx = gz.len() - 5;
        gz[idx] ^= 0x01;
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(&gz[..]);
        let mut out = Vec::new();
        let result = decoder.read_to_end(&mut out);
        assert!(result.is_err(), "CRC corruption must be detected");
    }

    // =========================================================================
    // Tier 1: Decompression routing tests
    // =========================================================================

    #[test]
    fn test_routing_bgzf_through_decompress_gzip_libdeflate() {
        let data = make_mixed(500_000);
        let bgzf = compress_bgzf(&data);
        assert!(
            crate::decompress::format::has_bgzf_markers(&bgzf),
            "BGZF data must be detected as BGZF"
        );
        let mut output = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(&bgzf, &mut output, 4).unwrap();
        assert_eq!(output, data, "BGZF through router must match original");
    }

    #[test]
    fn test_routing_multi_member_through_decompress_gzip_libdeflate() {
        let data = make_mixed(500_000);
        let multi = compress_multi_member(&data);
        assert!(
            !crate::decompress::format::has_bgzf_markers(&multi),
            "multi-member must not be detected as BGZF"
        );
        assert!(
            crate::decompress::format::is_likely_multi_member(&multi),
            "multi-member must be detected as multi-member"
        );
        let mut output = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(&multi, &mut output, 4).unwrap();
        assert_eq!(
            output, data,
            "multi-member through router must match original"
        );
    }

    #[test]
    fn test_routing_single_member_through_decompress_gzip_libdeflate() {
        let data = make_mixed(500_000);
        let single = compress_single_member(&data);
        assert!(
            !crate::decompress::format::has_bgzf_markers(&single),
            "single-member must not be detected as BGZF"
        );
        assert!(
            !crate::decompress::format::is_likely_multi_member(&single),
            "single-member must not be detected as multi-member"
        );
        let mut output = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(&single, &mut output, 4).unwrap();
        assert_eq!(
            output, data,
            "single-member through router must match original"
        );
    }

    #[test]
    fn test_routing_bgzf_t1_through_router() {
        let data = make_mixed(200_000);
        let bgzf = compress_bgzf(&data);
        let mut output = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(&bgzf, &mut output, 1).unwrap();
        assert_eq!(output, data, "BGZF T1 through router must match");
    }

    #[test]
    fn test_routing_multi_member_t1_through_router() {
        let data = make_mixed(500_000);
        let multi = compress_multi_member(&data);
        let mut output = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(&multi, &mut output, 1).unwrap();
        assert_eq!(output, data, "multi-member T1 through router must match");
    }

    #[test]
    fn test_routing_decompress_gzip_to_vec_bgzf() {
        let data = make_mixed(500_000);
        let bgzf = compress_bgzf(&data);
        let output = crate::decompress::decompress_gzip_to_vec(&bgzf, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_routing_decompress_gzip_to_vec_single() {
        let data = make_mixed(500_000);
        let single = compress_single_member(&data);
        let output = crate::decompress::decompress_gzip_to_vec(&single, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_routing_decompress_gzip_to_vec_multi() {
        let data = make_mixed(500_000);
        let multi = compress_multi_member(&data);
        let output = crate::decompress::decompress_gzip_to_vec(&multi, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_decompress_single_member_routes_correctly() {
        let data = make_mixed(200_000);
        let single = compress_single_member(&data);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&single, &mut output, 1).unwrap();
        assert_eq!(
            output, data,
            "decompress_single_member must produce correct output"
        );
    }

    #[test]
    fn test_decompress_single_member_multithread_still_correct() {
        let data = make_mixed(200_000);
        let single = compress_single_member(&data);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&single, &mut output, 4).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_all_three_formats_through_router_agree() {
        let data = make_mixed(500_000);
        let bgzf = compress_bgzf(&data);
        let multi = compress_multi_member(&data);
        let single = compress_single_member(&data);

        let out_bgzf = crate::decompress::decompress_gzip_to_vec(&bgzf, 4).unwrap();
        let out_multi = crate::decompress::decompress_gzip_to_vec(&multi, 4).unwrap();
        let out_single = crate::decompress::decompress_gzip_to_vec(&single, 4).unwrap();

        assert_eq!(out_bgzf, data, "BGZF output mismatch");
        assert_eq!(out_multi, data, "multi-member output mismatch");
        assert_eq!(out_single, data, "single-member output mismatch");
    }

    // =========================================================================
    // Tier 2: Format detection edge cases
    // =========================================================================

    // --- is_likely_multi_member edge cases ---

    #[test]
    fn test_multi_member_false_positive_embedded_magic() {
        // Craft data that contains 1f 8b 08 inside a single-member gzip stream.
        // The pattern should appear inside the compressed payload.
        let mut data = Vec::with_capacity(200_000);
        for _ in 0..100 {
            data.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00]);
            data.extend_from_slice(&make_sequential(2000));
        }
        let gz = compress_single_member(&data);
        assert!(
            !crate::decompress::format::is_likely_multi_member(&gz),
            "single-member with embedded magic must not be detected as multi-member"
        );
    }

    #[test]
    fn test_multi_member_rejects_zero_isize_trailer() {
        // Build a fake multi-member where the "preceding ISIZE" is 0.
        // is_likely_multi_member checks preceding_isize > 0.
        let data = make_mixed(100_000);
        let mut member1 = compress_single_member(&data[..50_000]);
        // Zero out the ISIZE field in the trailer (last 4 bytes)
        let len = member1.len();
        member1[len - 4] = 0;
        member1[len - 3] = 0;
        member1[len - 2] = 0;
        member1[len - 1] = 0;

        let member2 = compress_single_member(&data[50_000..]);
        let mut combined = member1;
        combined.extend_from_slice(&member2);

        assert!(
            !crate::decompress::format::is_likely_multi_member(&combined),
            "multi-member with zero ISIZE trailer must be rejected"
        );
    }

    #[test]
    fn test_multi_member_rejects_reserved_flags() {
        let data = make_mixed(100_000);
        let member1 = compress_single_member(&data[..50_000]);
        let mut member2 = compress_single_member(&data[50_000..]);
        // Set reserved flag bits (0xE0) in the second member's flags byte
        member2[3] |= 0x20;

        let mut combined = member1;
        combined.extend_from_slice(&member2);

        assert!(
            !crate::decompress::format::is_likely_multi_member(&combined),
            "multi-member with reserved flags must be rejected"
        );
    }

    #[test]
    fn test_multi_member_rejects_bad_xfl() {
        let data = make_mixed(100_000);
        let member1 = compress_single_member(&data[..50_000]);
        let mut member2 = compress_single_member(&data[50_000..]);
        // Set XFL to an invalid value (byte 8)
        member2[8] = 0x07;

        let mut combined = member1;
        combined.extend_from_slice(&member2);

        assert!(
            !crate::decompress::format::is_likely_multi_member(&combined),
            "multi-member with bad XFL byte must be rejected"
        );
    }

    #[test]
    fn test_multi_member_rejects_bad_os() {
        let data = make_mixed(100_000);
        let member1 = compress_single_member(&data[..50_000]);
        let mut member2 = compress_single_member(&data[50_000..]);
        // Set OS to an invalid value (byte 9); valid is 0-13 and 255
        member2[9] = 0x80;

        let mut combined = member1;
        combined.extend_from_slice(&member2);

        assert!(
            !crate::decompress::format::is_likely_multi_member(&combined),
            "multi-member with bad OS byte must be rejected"
        );
    }

    #[test]
    fn test_multi_member_accepts_valid_multi() {
        let data = make_mixed(600_000);
        let multi = compress_multi_member(&data);
        assert!(
            crate::decompress::format::is_likely_multi_member(&multi),
            "valid multi-member must be detected"
        );
    }

    #[test]
    fn test_multi_member_rejects_too_short() {
        assert!(!crate::decompress::format::is_likely_multi_member(&[
            0x1f, 0x8b, 0x08
        ]));
        assert!(!crate::decompress::format::is_likely_multi_member(&[]));
        assert!(!crate::decompress::format::is_likely_multi_member(
            &[0u8; 35]
        ));
    }

    // --- has_bgzf_markers edge cases ---

    #[test]
    fn test_bgzf_markers_non_rz_fextra() {
        // Build a gzip header with FEXTRA but a non-matching subfield ID ("XX")
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x04, // flags: FEXTRA set
            0x00, 0x00, 0x00, 0x00, // mtime
            0x00, // xfl
            0xFF, // os
        ];
        let xlen: u16 = 8;
        header.extend_from_slice(&xlen.to_le_bytes());
        // Subfield: "XX" with 4 bytes of data
        header.extend_from_slice(b"XX");
        header.extend_from_slice(&4u16.to_le_bytes());
        header.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        assert!(
            !crate::decompress::format::has_bgzf_markers(&header),
            "FEXTRA with non-RZ subfield must not be detected as BGZF"
        );
    }

    #[test]
    fn test_bgzf_markers_multiple_subfields_with_rz() {
        // Build header with two subfields: "XX" (4 bytes) + "GZ" (2 bytes)
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x04, // flags: FEXTRA set
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        // "XX" subfield: 4 bytes data + "GZ" subfield: 2 bytes data
        // Total XLEN = 4+4 + 4+2 = 14
        let xlen: u16 = 14;
        header.extend_from_slice(&xlen.to_le_bytes());
        // First subfield: "XX" with 4 bytes
        header.extend_from_slice(b"XX");
        header.extend_from_slice(&4u16.to_le_bytes());
        header.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        // Second subfield: "GZ" with 2 bytes
        header.extend_from_slice(b"GZ");
        header.extend_from_slice(&2u16.to_le_bytes());
        header.extend_from_slice(&[0x00, 0x00]);

        assert!(
            crate::decompress::format::has_bgzf_markers(&header),
            "FEXTRA with GZ subfield after another subfield must be detected"
        );
    }

    #[test]
    fn test_bgzf_markers_xlen_too_short() {
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x04, // flags: FEXTRA
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        let xlen: u16 = 3; // too short for any subfield (need 4 for id + len)
        header.extend_from_slice(&xlen.to_le_bytes());
        header.extend_from_slice(&[0x00, 0x00, 0x00]); // 3 bytes of garbage

        assert!(
            !crate::decompress::format::has_bgzf_markers(&header),
            "FEXTRA with xlen too short must not match"
        );
    }

    #[test]
    fn test_bgzf_markers_no_fextra_flag() {
        let header = vec![
            0x1f, 0x8b, 0x08, 0x00, // flags: no FEXTRA
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert!(!crate::decompress::format::has_bgzf_markers(&header));
    }

    #[test]
    fn test_bgzf_markers_too_short_input() {
        assert!(!crate::decompress::format::has_bgzf_markers(&[]));
        assert!(!crate::decompress::format::has_bgzf_markers(&[0x1f, 0x8b]));
        assert!(!crate::decompress::format::has_bgzf_markers(&[0x1f; 15]));
    }

    // --- parse_gzip_header_size edge cases ---

    #[test]
    fn test_parse_header_size_minimal() {
        let gz = compress_single_member(b"hello");
        let size = crate::decompress::format::parse_gzip_header_size(&gz);
        assert_eq!(size, Some(10), "minimal gzip header is 10 bytes");
    }

    #[test]
    fn test_parse_header_size_fhcrc() {
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x02, // flags: FHCRC
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, // FHCRC adds 2 bytes
            0x00, 0x00,
        ];
        // Pad to make it look reasonable
        header.extend_from_slice(&[0u8; 20]);

        let size = crate::decompress::format::parse_gzip_header_size(&header);
        assert_eq!(size, Some(12), "FHCRC adds 2 bytes to header");
    }

    #[test]
    fn test_parse_header_size_fname() {
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x08, // flags: FNAME
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        header.extend_from_slice(b"test.txt\0");
        header.extend_from_slice(&[0u8; 20]);

        let size = crate::decompress::format::parse_gzip_header_size(&header);
        assert_eq!(size, Some(10 + 9), "FNAME 'test.txt\\0' adds 9 bytes");
    }

    #[test]
    fn test_parse_header_size_fcomment() {
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x10, // flags: FCOMMENT
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        header.extend_from_slice(b"a comment\0");
        header.extend_from_slice(&[0u8; 20]);

        let size = crate::decompress::format::parse_gzip_header_size(&header);
        assert_eq!(size, Some(10 + 10), "FCOMMENT 'a comment\\0' adds 10 bytes");
    }

    #[test]
    fn test_parse_header_size_fextra() {
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x04, // flags: FEXTRA
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        let xlen: u16 = 6;
        header.extend_from_slice(&xlen.to_le_bytes());
        header.extend_from_slice(&[0u8; 6]); // 6 bytes of extra data
        header.extend_from_slice(&[0u8; 20]);

        let size = crate::decompress::format::parse_gzip_header_size(&header);
        assert_eq!(size, Some(10 + 2 + 6), "FEXTRA with xlen=6 adds 8 bytes");
    }

    #[test]
    fn test_parse_header_size_all_flags() {
        // FEXTRA + FNAME + FCOMMENT + FHCRC = 0x04 | 0x08 | 0x10 | 0x02 = 0x1E
        let mut header = vec![
            0x1f, 0x8b, 0x08, 0x1E, // all optional flags
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        ];
        // FEXTRA: xlen=4, 4 bytes data
        header.extend_from_slice(&4u16.to_le_bytes());
        header.extend_from_slice(&[0xAA; 4]);
        // FNAME: "f.gz\0"
        header.extend_from_slice(b"f.gz\0");
        // FCOMMENT: "c\0"
        header.extend_from_slice(b"c\0");
        // FHCRC: 2 bytes
        header.extend_from_slice(&[0x00, 0x00]);
        header.extend_from_slice(&[0u8; 20]);

        let size = crate::decompress::format::parse_gzip_header_size(&header);
        // 10 base + 2 xlen + 4 extra + 5 fname + 2 fcomment + 2 fhcrc = 25
        assert_eq!(size, Some(25));
    }

    #[test]
    fn test_parse_header_size_truncated_fextra() {
        // FEXTRA flag set but not enough data for xlen
        let header = vec![
            0x1f, 0x8b, 0x08, 0x04, // FEXTRA
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, // Only 1 byte, need 2 for xlen
            0x06,
        ];
        assert_eq!(
            crate::decompress::format::parse_gzip_header_size(&header),
            None
        );
    }

    #[test]
    fn test_parse_header_size_bad_magic() {
        let header = vec![0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF];
        assert_eq!(
            crate::decompress::format::parse_gzip_header_size(&header),
            None
        );
    }

    #[test]
    fn test_parse_header_size_too_short() {
        assert_eq!(
            crate::decompress::format::parse_gzip_header_size(&[0x1f, 0x8b]),
            None
        );
        assert_eq!(crate::decompress::format::parse_gzip_header_size(&[]), None);
    }

    // --- Non-gzip silent empty output ---

    #[test]
    fn test_non_gzip_through_to_vec_returns_empty() {
        let non_gzip = b"this is not gzip data at all";
        let result = crate::decompress::decompress_gzip_to_vec(non_gzip, 4).unwrap();
        assert!(result.is_empty(), "non-gzip data must produce empty output");
    }

    #[test]
    fn test_non_gzip_through_libdeflate_returns_zero() {
        let non_gzip = b"this is not gzip data at all";
        let mut output = Vec::new();
        let bytes =
            crate::decompress::decompress_gzip_libdeflate(non_gzip, &mut output, 4).unwrap();
        assert_eq!(bytes, 0, "non-gzip must return 0 bytes written");
        assert!(output.is_empty());
    }

    #[test]
    fn test_single_byte_input_returns_empty() {
        let result = crate::decompress::decompress_gzip_to_vec(&[0x1f], 1).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let result = crate::decompress::decompress_gzip_to_vec(&[], 1).unwrap();
        assert!(result.is_empty());
    }

    // --- read_gzip_isize ---

    #[test]
    fn test_read_gzip_isize_correct() {
        let data = make_mixed(100_000);
        let gz = compress_single_member(&data);
        let isize_val = crate::decompress::format::read_gzip_isize(&gz).unwrap();
        assert_eq!(
            isize_val as usize,
            data.len(),
            "ISIZE must match original data length"
        );
    }

    #[test]
    fn test_read_gzip_isize_too_short() {
        assert!(crate::decompress::format::read_gzip_isize(&[0u8; 17]).is_none());
        assert!(crate::decompress::format::read_gzip_isize(&[]).is_none());
    }

    #[test]
    fn test_read_gzip_isize_empty_original() {
        let gz = compress_single_member(&[]);
        let isize_val = crate::decompress::format::read_gzip_isize(&gz).unwrap();
        assert_eq!(isize_val, 0, "ISIZE of empty data must be 0");
    }

    // --- InsufficientSpace retry ---

    #[test]
    fn test_decompress_with_wrong_isize_still_works() {
        // Corrupt the ISIZE trailer to be tiny (1), forcing the buffer retry loop
        let data = make_mixed(100_000);
        let mut gz = compress_single_member(&data);
        let len = gz.len();
        // Set ISIZE to 1 (way too small)
        gz[len - 4] = 1;
        gz[len - 3] = 0;
        gz[len - 2] = 0;
        gz[len - 1] = 0;
        // libdeflate checks CRC + ISIZE in the trailer, so this will fail.
        // But decompress_single_member_libdeflate_pub uses gzip_decompress_ex
        // which verifies the trailer. A wrong ISIZE = bad data.
        let mut out = Vec::new();
        let result = crate::decompress::decompress_single_member_libdeflate(&gz, &mut out);
        // Either error (CRC/ISIZE mismatch) or the data is wrong
        assert!(result.is_err() || out != data);
    }

    #[test]
    fn test_decompress_with_zero_isize_trailer() {
        // ISIZE = 0 but data is non-empty → will cause mismatch
        let data = make_mixed(10_000);
        let mut gz = compress_single_member(&data);
        let len = gz.len();
        gz[len - 4] = 0;
        gz[len - 3] = 0;
        gz[len - 2] = 0;
        gz[len - 1] = 0;
        let mut out = Vec::new();
        let result = crate::decompress::decompress_single_member_libdeflate(&gz, &mut out);
        assert!(result.is_err() || out != data);
    }

    // =========================================================================
    // ISA-L error path tests (architecture-independent)
    // =========================================================================

    #[test]
    fn test_decompress_single_member_corrupt_errors() {
        let data = make_mixed(100_000);
        let mut gz = compress_single_member(&data);
        let mid = gz.len() / 2;
        gz[mid] ^= 0xFF;
        let mut out = Vec::new();
        let result = crate::decompress::decompress_single_member(&gz, &mut out, 1);
        // Must error or produce wrong output — never silently succeed with wrong data
        assert!(
            result.is_err() || out != data,
            "corrupt data through decompress_single_member must not silently produce correct output"
        );
    }

    #[test]
    fn test_decompress_single_member_valid_zeros() {
        let data = make_zeros(200_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_single_member_valid_sequential() {
        let data = make_sequential(200_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_single_member_valid_random() {
        let data = make_random_seeded(200_000, 42);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_single_member_valid_mixed() {
        let data = make_mixed(200_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_single_member_valid_binary() {
        let data = make_binary(200_000);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_decompress_single_member_valid_single_byte() {
        let data = make_single_byte(200_000, 0xAA);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 1).unwrap();
        assert_eq!(out, data);
    }

    // =========================================================================
    // Gap 1: End-to-end routing through parallel single-member path (>=4MB)
    // =========================================================================

    #[test]
    fn test_routing_single_member_4mb_parallel_correct() {
        let data = make_mixed(5 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 4).unwrap();
        assert_eq!(out, data, "parallel single-member 5MB must match original");
    }

    #[test]
    fn test_routing_single_member_8mb_parallel_through_router() {
        let data = make_mixed(8 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let result = crate::decompress::decompress_gzip_to_vec(&gz, 4).unwrap();
        assert_eq!(
            result, data,
            "8MB single-member through full router must match"
        );
    }

    #[test]
    fn test_routing_single_member_parallel_t2_correct() {
        let data = make_mixed(5 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&gz, &mut out, 2).unwrap();
        assert_eq!(
            out, data,
            "parallel at T2 must still produce correct output"
        );
    }

    #[test]
    fn test_routing_single_member_parallel_thread_independence_large() {
        let data = make_mixed(6 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let mut results = Vec::new();
        for threads in [2, 4, 8] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&gz, &mut out, threads).unwrap();
            results.push(out);
        }
        for (i, r) in results.iter().enumerate().skip(1) {
            assert_eq!(
                results[0],
                *r,
                "T{} output must match T{} output",
                [2, 4, 8][i],
                [2, 4, 8][0]
            );
        }
        assert_eq!(results[0], data);
    }

    #[test]
    fn test_routing_single_member_zeros_8mb_parallel() {
        let data = make_zeros(8 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let result = crate::decompress::decompress_gzip_to_vec(&gz, 4).unwrap();
        assert_eq!(result, data, "8MB zeros through parallel must match");
    }

    // =========================================================================
    // Streaming decompress tests (zlib-ng path for arm64 large files)
    // =========================================================================

    #[test]
    fn test_streaming_decompress_roundtrip_mixed() {
        let data = make_mixed(2 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut out).unwrap();
        assert_eq!(out, data, "streaming decompress must match original");
    }

    #[test]
    fn test_streaming_decompress_roundtrip_zeros() {
        let data = make_zeros(5 * 1024 * 1024);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut out).unwrap();
        assert_eq!(out, data, "streaming zeros must match");
    }

    #[test]
    fn test_streaming_decompress_roundtrip_random() {
        let data = make_random_seeded(1024 * 1024, 0xdeadbeef);
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut out).unwrap();
        assert_eq!(out, data, "streaming random data must match");
    }

    #[test]
    fn test_streaming_matches_libdeflate() {
        let data = make_mixed(3 * 1024 * 1024);
        let gz = compress_single_member(&data);

        let mut streaming_out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut streaming_out).unwrap();

        let mut libdeflate_out = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(&gz, &mut libdeflate_out).unwrap();

        assert_eq!(
            streaming_out, libdeflate_out,
            "streaming and libdeflate must produce byte-identical output"
        );
    }

    #[test]
    fn test_streaming_decompress_small_file() {
        let data = b"hello world streaming test".to_vec();
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_streaming_decompress_empty() {
        let data = Vec::new();
        let gz = compress_single_member(&data);
        let mut out = Vec::new();
        crate::decompress::decompress_single_member_streaming(&gz, &mut out).unwrap();
        assert!(out.is_empty());
    }

    // =========================================================================
    // Block size tuning tests (L1/L2 with many threads)
    // =========================================================================

    #[test]
    fn test_block_size_capped_at_256k_with_many_threads() {
        let size = crate::compress::parallel::get_optimal_block_size(1, 100 * 1024 * 1024, 16);
        assert!(
            size <= 256 * 1024,
            "L1 with 16 threads should use <= 256KB blocks, got {}KB",
            size / 1024
        );
    }

    #[test]
    fn test_block_size_allows_large_with_few_threads() {
        let size = crate::compress::parallel::get_optimal_block_size(1, 100 * 1024 * 1024, 4);
        assert!(
            size > 256 * 1024,
            "L1 with 4 threads should allow blocks > 256KB, got {}KB",
            size / 1024
        );
    }

    #[test]
    fn test_block_size_threshold_at_8_threads() {
        let below = crate::compress::parallel::get_optimal_block_size(1, 100 * 1024 * 1024, 7);
        let at = crate::compress::parallel::get_optimal_block_size(1, 100 * 1024 * 1024, 8);
        assert!(
            below >= at,
            "threshold should change at 8 threads: 7t={}KB, 8t={}KB",
            below / 1024,
            at / 1024
        );
    }

    #[test]
    fn test_block_size_l2_also_capped() {
        let size = crate::compress::parallel::get_optimal_block_size(2, 50 * 1024 * 1024, 14);
        assert!(
            size <= 256 * 1024,
            "L2 with 14 threads should use <= 256KB blocks, got {}KB",
            size / 1024
        );
    }

    #[test]
    fn test_block_size_l6_unchanged() {
        let size = crate::compress::parallel::get_optimal_block_size(6, 100 * 1024 * 1024, 16);
        assert_eq!(
            size,
            64 * 1024,
            "L6 block size should be 64KB regardless of threads"
        );
    }
}
