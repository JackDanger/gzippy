//! Decompression Routing Oracle
//!
//! Verifies that every file type (BGZF, multi-member, single-member, zlib)
//! is correctly detected, routed to the right decompression path, and produces
//! byte-identical output regardless of thread count.
//!
//! Layer 0: FileOracle — creates known test files of each type
//! Layer 1: Format Detection — verify has_bgzf_markers, is_likely_multi_member
//! Layer 2: Path Correctness — each path produces valid output
//! Layer 3: Output Identity — same output regardless of path or thread count

#[cfg(test)]
mod tests {
    use crate::decompression::has_bgzf_markers;
    use crate::parallel_compress::{compress_single_member, GzipHeaderInfo};
    use std::io::Write;

    // =========================================================================
    // Layer 0: FileOracle — create known test files
    // =========================================================================

    struct FileOracle {
        original: Vec<u8>,
        single_member_gz: Vec<u8>,
        multi_member_gz: Vec<u8>,
        bgzf_gz: Vec<u8>,
    }

    impl FileOracle {
        fn new(size: usize) -> Self {
            let original = make_test_data(size);

            let single_member_gz = compress_single_member_gzip(&original);
            let multi_member_gz = compress_multi_member_gzip(&original);
            let bgzf_gz = compress_bgzf_gzip(&original);

            Self {
                original,
                single_member_gz,
                multi_member_gz,
                bgzf_gz,
            }
        }
    }

    fn make_test_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xcafebabe;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOP\n",
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

    /// Single-member gzip (what `gzip` produces).
    fn compress_single_member_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Multi-member gzip (what `pigz` produces — concatenated gzip members).
    fn compress_multi_member_gzip(data: &[u8]) -> Vec<u8> {
        let chunk_size = 256 * 1024;
        let mut output = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(chunk).unwrap();
            output.extend_from_slice(&encoder.finish().unwrap());
        }
        output
    }

    /// BGZF gzip (what gzippy produces — blocks with FEXTRA markers).
    fn compress_bgzf_gzip(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let header = GzipHeaderInfo::default();
        compress_single_member(&mut output, data, 1, &header).unwrap();
        output
    }

    /// Decompress using flate2 as a reference implementation.
    fn decompress_reference(gz_data: &[u8]) -> Vec<u8> {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;
        let mut decoder = MultiGzDecoder::new(gz_data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).unwrap();
        output
    }

    // =========================================================================
    // Layer 0 tests: Oracle self-consistency
    // =========================================================================

    #[test]
    fn test_file_oracle_roundtrip() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let from_single = decompress_reference(&oracle.single_member_gz);
        let from_multi = decompress_reference(&oracle.multi_member_gz);
        let from_bgzf = decompress_reference(&oracle.bgzf_gz);

        assert_eq!(
            from_single, oracle.original,
            "single-member roundtrip failed"
        );
        assert_eq!(from_multi, oracle.original, "multi-member roundtrip failed");
        assert_eq!(from_bgzf, oracle.original, "bgzf roundtrip failed");

        eprintln!(
            "oracle: {}B original, single={}B, multi={}B, bgzf={}B",
            oracle.original.len(),
            oracle.single_member_gz.len(),
            oracle.multi_member_gz.len(),
            oracle.bgzf_gz.len()
        );
    }

    // =========================================================================
    // Layer 1: Format Detection
    // =========================================================================

    #[test]
    fn test_detect_bgzf() {
        let oracle = FileOracle::new(512 * 1024);

        assert!(
            has_bgzf_markers(&oracle.bgzf_gz),
            "BGZF data should be detected as BGZF"
        );
        assert!(
            !has_bgzf_markers(&oracle.single_member_gz),
            "single-member should NOT be detected as BGZF"
        );
        // Multi-member from flate2 won't have BGZF markers
        assert!(
            !has_bgzf_markers(&oracle.multi_member_gz),
            "multi-member (flate2) should NOT be detected as BGZF"
        );
    }

    #[test]
    fn test_detect_multi_member() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let multi = crate::decompression::is_likely_multi_member_pub(&oracle.multi_member_gz);
        let single = crate::decompression::is_likely_multi_member_pub(&oracle.single_member_gz);
        let bgzf = crate::decompression::is_likely_multi_member_pub(&oracle.bgzf_gz);

        eprintln!("detection: multi={} single={} bgzf={}", multi, single, bgzf);

        assert!(multi, "multi-member should be detected as multi-member");
        assert!(
            !single,
            "single-member should NOT be detected as multi-member"
        );
    }

    // =========================================================================
    // Layer 2: Path Correctness — each decompression path produces valid output
    // =========================================================================

    #[test]
    fn test_bgzf_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // T1
        let output_t1 = crate::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 1).unwrap();
        assert_eq!(
            output_t1, oracle.original,
            "BGZF T1 output doesn't match original"
        );

        // T4
        let output_t4 = crate::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
        assert_eq!(
            output_t4, oracle.original,
            "BGZF T4 output doesn't match original"
        );

        eprintln!("BGZF path: T1 and T4 produce identical correct output");
    }

    #[test]
    fn test_multi_member_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // Parallel path
        let output_par =
            crate::bgzf::decompress_multi_member_parallel_to_vec(&oracle.multi_member_gz, 4)
                .unwrap();
        assert_eq!(
            output_par, oracle.original,
            "multi-member parallel output doesn't match original"
        );

        // Sequential path
        let mut output_seq = Vec::new();
        crate::decompression::decompress_multi_member_sequential_pub(
            &oracle.multi_member_gz,
            &mut output_seq,
        )
        .unwrap();
        assert_eq!(
            output_seq, oracle.original,
            "multi-member sequential output doesn't match original"
        );

        eprintln!("multi-member path: parallel and sequential produce identical correct output");
    }

    #[test]
    fn test_single_member_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // Sequential fallback
        let mut output = Vec::new();
        crate::decompression::decompress_single_member_libdeflate_pub(
            &oracle.single_member_gz,
            &mut output,
        )
        .unwrap();
        assert_eq!(
            output, oracle.original,
            "single-member libdeflate output doesn't match original"
        );

        eprintln!("single-member path: libdeflate produces correct output");
    }

    // =========================================================================
    // Layer 3: Output Identity — same output regardless of thread count
    // =========================================================================

    #[test]
    fn test_bgzf_thread_independence() {
        let oracle = FileOracle::new(4 * 1024 * 1024);

        let outputs: Vec<Vec<u8>> = (1..=8)
            .map(|t| crate::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, t).unwrap())
            .collect();

        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(
                output,
                &oracle.original,
                "BGZF T{} output differs from original",
                i + 1
            );
        }

        // Verify all outputs are identical to each other
        for i in 1..outputs.len() {
            assert_eq!(
                outputs[0],
                outputs[i],
                "BGZF T1 and T{} produce different output",
                i + 1
            );
        }

        eprintln!("BGZF: T1-T8 all produce identical output");
    }

    #[test]
    fn test_cross_format_output_identity() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let from_bgzf = crate::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
        let from_multi =
            crate::bgzf::decompress_multi_member_parallel_to_vec(&oracle.multi_member_gz, 4)
                .unwrap();

        let mut from_single = Vec::new();
        crate::decompression::decompress_single_member_libdeflate_pub(
            &oracle.single_member_gz,
            &mut from_single,
        )
        .unwrap();

        assert_eq!(
            from_bgzf, oracle.original,
            "BGZF decompressed differs from original"
        );
        assert_eq!(
            from_multi, oracle.original,
            "multi-member decompressed differs from original"
        );
        assert_eq!(
            from_single, oracle.original,
            "single-member decompressed differs from original"
        );

        eprintln!("all three formats produce byte-identical output from the same input");
    }
}
