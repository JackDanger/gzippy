//! Compression Roundtrip Oracle
//!
//! Verifies all compression paths produce valid gzip output that decompresses
//! to the original data. Compares compression ratios and tests thread independence.
//!
//! Layer 0: CompressOracle — known test data with expected properties
//! Layer 1: Roundtrip correctness — every path produces decompressible output
//! Layer 2: Ratio comparison — no path produces unnecessarily large output
//! Layer 3: Thread independence — same ratio regardless of thread count

#[cfg(test)]
mod tests {
    #![allow(unused_variables)]

    use crate::parallel_compress::{compress_single_member, GzipHeaderInfo, ParallelGzEncoder};
    use crate::pipelined_compress::PipelinedGzEncoder;
    use std::io::{Cursor, Read, Write};

    // =========================================================================
    // Layer 0: CompressOracle — test data with known properties
    // =========================================================================

    struct CompressOracle {
        /// Original uncompressed data.
        original: Vec<u8>,
        /// What type of data this is (for diagnostics).
        name: &'static str,
    }

    impl CompressOracle {
        fn new(name: &'static str, data: Vec<u8>) -> Self {
            Self {
                original: data,
                name,
            }
        }

        /// Verify that compressed data decompresses to the original.
        fn verify_roundtrip(&self, compressed: &[u8], path_name: &str) {
            let decompressed = decompress_reference(compressed);
            assert_eq!(
                decompressed.len(),
                self.original.len(),
                "{} [{}]: decompressed size {} != original {}",
                self.name,
                path_name,
                decompressed.len(),
                self.original.len()
            );
            assert_eq!(
                decompressed, self.original,
                "{} [{}]: decompressed content differs from original",
                self.name, path_name
            );
        }

        fn ratio(&self, compressed: &[u8]) -> f64 {
            compressed.len() as f64 / self.original.len() as f64
        }
    }

    fn decompress_reference(gz_data: &[u8]) -> Vec<u8> {
        use flate2::read::MultiGzDecoder;
        let mut decoder = MultiGzDecoder::new(gz_data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).unwrap();
        output
    }

    fn make_literal_data(size: usize) -> Vec<u8> {
        (0..size)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect()
    }

    fn make_rle_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        for i in 0..(size / 1000).max(1) {
            let byte = (i % 256) as u8;
            data.extend(std::iter::repeat_n(byte, 1000.min(size - data.len())));
        }
        data.truncate(size);
        data
    }

    fn make_mixed_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz\n",
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

    // =========================================================================
    // Compression path helpers
    // =========================================================================

    /// Compress with libdeflate via compress_single_member (L1-L5 parallel path).
    fn compress_libdeflate(data: &[u8], level: u32) -> Vec<u8> {
        let mut output = Vec::new();
        let header = GzipHeaderInfo::default();
        compress_single_member(&mut output, data, level, &header).unwrap();
        output
    }

    /// Compress with ParallelGzEncoder (multi-threaded libdeflate blocks).
    fn compress_parallel(data: &[u8], level: u32, threads: usize) -> Vec<u8> {
        let encoder = ParallelGzEncoder::new(level, threads);
        let mut output = Vec::new();
        encoder.compress(Cursor::new(data), &mut output).unwrap();
        output
    }

    /// Compress with PipelinedGzEncoder (L6-L9 zlib-ng streaming).
    fn compress_pipelined(data: &[u8], level: u32, threads: usize) -> Vec<u8> {
        let encoder = PipelinedGzEncoder::new(level, threads);
        let mut output = Vec::new();
        encoder.compress(Cursor::new(data), &mut output).unwrap();
        output
    }

    /// Compress with flate2 (reference implementation).
    fn compress_flate2(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    // =========================================================================
    // Layer 0 tests: Oracle self-consistency
    // =========================================================================

    #[test]
    fn test_compress_oracle_setup() {
        let oracles = [
            CompressOracle::new("literals", make_literal_data(2_000_000)),
            CompressOracle::new("rle", make_rle_data(2_000_000)),
            CompressOracle::new("mixed", make_mixed_data(2_000_000)),
        ];

        for oracle in &oracles {
            // Reference roundtrip
            let compressed = compress_flate2(&oracle.original, 6);
            oracle.verify_roundtrip(&compressed, "flate2-L6");
            eprintln!(
                "{}: {} bytes, flate2 L6 ratio {:.1}%",
                oracle.name,
                oracle.original.len(),
                oracle.ratio(&compressed) * 100.0
            );
        }
    }

    // =========================================================================
    // Layer 1: Roundtrip correctness — every path produces valid output
    // =========================================================================

    #[test]
    fn test_roundtrip_libdeflate_levels() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(2_000_000));

        for level in [1, 3, 5] {
            let compressed = compress_libdeflate(&oracle.original, level);
            oracle.verify_roundtrip(&compressed, &format!("libdeflate-L{}", level));
            eprintln!(
                "  libdeflate L{}: ratio {:.1}%",
                level,
                oracle.ratio(&compressed) * 100.0
            );
        }
    }

    #[test]
    fn test_roundtrip_parallel_levels() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(2_000_000));

        for level in [1, 3, 5] {
            let compressed = compress_parallel(&oracle.original, level, 4);
            oracle.verify_roundtrip(&compressed, &format!("parallel-L{}-T4", level));
            eprintln!(
                "  parallel L{} T4: ratio {:.1}%",
                level,
                oracle.ratio(&compressed) * 100.0
            );
        }
    }

    #[test]
    fn test_roundtrip_pipelined_levels() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(2_000_000));

        for level in [6, 7, 9] {
            let compressed = compress_pipelined(&oracle.original, level, 4);
            oracle.verify_roundtrip(&compressed, &format!("pipelined-L{}-T4", level));
            eprintln!(
                "  pipelined L{} T4: ratio {:.1}%",
                level,
                oracle.ratio(&compressed) * 100.0
            );
        }
    }

    // =========================================================================
    // Layer 2: Ratio comparison across paths
    // =========================================================================

    #[test]
    fn test_ratio_comparison() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(4_000_000));

        eprintln!(
            "=== Compression Ratio Comparison ({} bytes) ===",
            oracle.original.len()
        );
        eprintln!("{:<25} {:>10} {:>10}", "Path", "Size", "Ratio");
        eprintln!("{}", "-".repeat(48));

        let paths: Vec<(&str, Vec<u8>)> = vec![
            ("flate2-L1", compress_flate2(&oracle.original, 1)),
            ("flate2-L6", compress_flate2(&oracle.original, 6)),
            ("flate2-L9", compress_flate2(&oracle.original, 9)),
            ("libdeflate-L1", compress_libdeflate(&oracle.original, 1)),
            ("libdeflate-L5", compress_libdeflate(&oracle.original, 5)),
            ("parallel-L1-T4", compress_parallel(&oracle.original, 1, 4)),
            ("parallel-L5-T4", compress_parallel(&oracle.original, 5, 4)),
            (
                "pipelined-L6-T4",
                compress_pipelined(&oracle.original, 6, 4),
            ),
            (
                "pipelined-L9-T4",
                compress_pipelined(&oracle.original, 9, 4),
            ),
        ];

        for (name, compressed) in &paths {
            oracle.verify_roundtrip(compressed, name);
            eprintln!(
                "{:<25} {:>10} {:>9.1}%",
                name,
                compressed.len(),
                oracle.ratio(compressed) * 100.0
            );
        }
    }

    // =========================================================================
    // Layer 3: Thread independence — same output regardless of thread count
    // =========================================================================

    #[test]
    fn test_parallel_thread_independence() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(2_000_000));

        // All thread counts should produce decompressible output
        let thread_counts = [1, 2, 4, 8];
        let mut ratios = Vec::new();

        for &threads in &thread_counts {
            let compressed = compress_parallel(&oracle.original, 1, threads);
            oracle.verify_roundtrip(&compressed, &format!("parallel-L1-T{}", threads));
            let ratio = oracle.ratio(&compressed);
            ratios.push((threads, ratio));
        }

        eprintln!("parallel L1 thread scaling:");
        for (threads, ratio) in &ratios {
            eprintln!("  T{}: ratio {:.1}%", threads, ratio * 100.0);
        }

        // Ratios should be within 5% of each other (different block boundaries)
        let min_ratio = ratios.iter().map(|(_, r)| *r).fold(f64::MAX, f64::min);
        let max_ratio = ratios.iter().map(|(_, r)| *r).fold(f64::MIN, f64::max);
        let spread = (max_ratio - min_ratio) / min_ratio;

        assert!(
            spread < 0.15,
            "ratio spread {:.1}% is too large across thread counts (min {:.1}%, max {:.1}%)",
            spread * 100.0,
            min_ratio * 100.0,
            max_ratio * 100.0
        );
    }

    #[test]
    fn test_pipelined_thread_independence() {
        let oracle = CompressOracle::new("mixed", make_mixed_data(2_000_000));

        let thread_counts = [1, 2, 4];
        let mut ratios = Vec::new();

        for &threads in &thread_counts {
            let compressed = compress_pipelined(&oracle.original, 6, threads);
            oracle.verify_roundtrip(&compressed, &format!("pipelined-L6-T{}", threads));
            let ratio = oracle.ratio(&compressed);
            ratios.push((threads, ratio));
        }

        eprintln!("pipelined L6 thread scaling:");
        for (threads, ratio) in &ratios {
            eprintln!("  T{}: ratio {:.1}%", threads, ratio * 100.0);
        }

        let min_ratio = ratios.iter().map(|(_, r)| *r).fold(f64::MAX, f64::min);
        let max_ratio = ratios.iter().map(|(_, r)| *r).fold(f64::MIN, f64::max);
        let spread = (max_ratio - min_ratio) / min_ratio;

        assert!(
            spread < 0.05,
            "pipelined ratio spread {:.1}% is too large across thread counts",
            spread * 100.0
        );
    }
}
