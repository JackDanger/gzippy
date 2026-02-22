//! Inflate Performance Oracle
//!
//! Block-by-block comparison of consume_first_decode (pure Rust) vs libdeflate
//! (C FFI) to identify exactly which block types and data patterns cause the
//! performance gap.
//!
//! Layer 0: InflateOracle — extract individual deflate blocks with metadata
//! Layer 1: Correctness — verify both implementations produce identical output
//! Layer 2: Per-block timing — time each block with both implementations
//! Layer 3: Pattern analysis — categorize blocks and correlate with speed gaps

#[cfg(test)]
mod tests {
    #![allow(unused_variables)]

    use crate::consume_first_decode::{inflate_consume_first, Bits};
    use crate::scan_inflate::scan_deflate_fast;
    use std::io::Write;
    use std::time::Instant;

    // =========================================================================
    // Layer 0: InflateOracle — block-level ground truth
    // =========================================================================

    #[derive(Clone)]
    #[allow(dead_code)]
    struct BlockInfo {
        index: usize,
        btype: u8,
        bfinal: bool,
        start_byte: usize,
        start_bit_offset: u8,
        output_size: usize,
        output_offset: usize,
    }

    struct InflateOracle {
        deflate_data: Vec<u8>,
        expected_output: Vec<u8>,
        blocks: Vec<BlockInfo>,
    }

    impl InflateOracle {
        fn from_gzip(gzip_data: &[u8]) -> Self {
            let header_size =
                crate::marker_decode::skip_gzip_header(gzip_data).expect("valid gzip header");
            let deflate_data = &gzip_data[header_size..gzip_data.len() - 8];

            // Scan to get block boundaries
            let scan = scan_deflate_fast(deflate_data, 1, 0).expect("scan should succeed");

            // Parse blocks to get per-block metadata
            let blocks = parse_blocks(deflate_data, scan.total_output_size);

            // Full decode for expected output
            let mut output = vec![0u8; scan.total_output_size + 65536];
            let size = inflate_consume_first(deflate_data, &mut output).expect("inflate failed");
            output.truncate(size);

            Self {
                deflate_data: deflate_data.to_vec(),
                expected_output: output,
                blocks,
            }
        }

        fn from_raw(data: &[u8]) -> Self {
            let gz = compress_gzip(data);
            Self::from_gzip(&gz)
        }

        fn block_type_name(btype: u8) -> &'static str {
            match btype {
                0 => "stored",
                1 => "fixed",
                2 => "dynamic",
                _ => "unknown",
            }
        }

        fn summary(&self) {
            let mut type_counts = [0usize; 3];
            let mut type_bytes = [0usize; 3];
            for b in &self.blocks {
                if (b.btype as usize) < 3 {
                    type_counts[b.btype as usize] += 1;
                    type_bytes[b.btype as usize] += b.output_size;
                }
            }
            eprintln!(
                "inflate oracle: {} blocks, {} bytes output",
                self.blocks.len(),
                self.expected_output.len()
            );
            for t in 0..3 {
                if type_counts[t] > 0 {
                    eprintln!(
                        "  {}: {} blocks, {} bytes ({:.1}%)",
                        Self::block_type_name(t as u8),
                        type_counts[t],
                        type_bytes[t],
                        type_bytes[t] as f64 / self.expected_output.len() as f64 * 100.0
                    );
                }
            }
        }
    }

    fn compress_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Parse a deflate stream to extract per-block metadata.
    fn parse_blocks(deflate_data: &[u8], total_output: usize) -> Vec<BlockInfo> {
        let mut blocks = Vec::new();
        let mut bits = Bits::new(deflate_data);
        let mut output = vec![0u8; total_output + 65536];
        let mut out_pos = 0usize;
        let mut index = 0;

        loop {
            let block_start_byte = bits.pos;
            let block_start_bits = bits.bitsleft as u8;

            if bits.available() < 3 {
                bits.refill();
            }

            let header = bits.peek();
            let bfinal = (header & 1) != 0;
            let btype = ((header >> 1) & 3) as u8;
            bits.consume(3);

            let prev_pos = out_pos;
            match btype {
                0 => {
                    out_pos = crate::consume_first_decode::decode_stored_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap()
                }
                1 => {
                    out_pos = crate::consume_first_decode::decode_fixed_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap()
                }
                2 => {
                    out_pos = crate::consume_first_decode::decode_dynamic_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap()
                }
                _ => break,
            }

            // Compute start bit position from saved state
            let real_bitsleft = (block_start_bits as usize) & 0x3F;
            let start_bit_pos = block_start_byte * 8 - real_bitsleft;

            blocks.push(BlockInfo {
                index,
                btype,
                bfinal,
                start_byte: start_bit_pos / 8,
                start_bit_offset: (start_bit_pos % 8) as u8,
                output_size: out_pos - prev_pos,
                output_offset: prev_pos,
            });

            index += 1;

            if bfinal {
                break;
            }
        }

        blocks
    }

    fn make_test_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOP\n",
            b"how vexingly quick daft zebras jump. ",
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
    // Layer 0 tests: Oracle self-consistency
    // =========================================================================

    #[test]
    fn test_inflate_oracle_synthetic() {
        let data = make_test_data(4 * 1024 * 1024);
        let oracle = InflateOracle::from_raw(&data);
        oracle.summary();
        assert_eq!(oracle.expected_output, data);
        assert!(!oracle.blocks.is_empty());
    }

    #[test]
    fn test_inflate_oracle_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };
        let oracle = InflateOracle::from_gzip(&gz);
        oracle.summary();
        assert!(!oracle.blocks.is_empty());
    }

    // =========================================================================
    // Layer 1: Correctness — both paths produce identical output
    // =========================================================================

    #[test]
    fn test_consume_first_matches_libdeflate() {
        let data = make_test_data(4 * 1024 * 1024);
        let oracle = InflateOracle::from_raw(&data);

        // consume_first
        let mut cf_output = vec![0u8; oracle.expected_output.len() + 65536];
        let cf_size =
            inflate_consume_first(&oracle.deflate_data, &mut cf_output).expect("cf inflate");
        cf_output.truncate(cf_size);

        // libdeflate
        let mut ld_output = vec![0u8; oracle.expected_output.len() + 65536];
        let ld_size = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut ld_output)
            .expect("ld inflate");
        ld_output.truncate(ld_size);

        assert_eq!(
            cf_size, ld_size,
            "output size mismatch: cf={} ld={}",
            cf_size, ld_size
        );
        assert_eq!(cf_output, ld_output, "output content mismatch");
        assert_eq!(cf_output, oracle.expected_output, "doesn't match expected");

        eprintln!(
            "correctness: consume_first and libdeflate produce identical {} byte output",
            cf_size
        );
    }

    #[test]
    fn test_consume_first_matches_libdeflate_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };
        let oracle = InflateOracle::from_gzip(&gz);

        let mut cf_output = vec![0u8; oracle.expected_output.len() + 65536];
        let cf_size =
            inflate_consume_first(&oracle.deflate_data, &mut cf_output).expect("cf inflate");

        let mut ld_output = vec![0u8; oracle.expected_output.len() + 65536];
        let ld_size = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut ld_output)
            .expect("ld inflate");

        assert_eq!(cf_size, ld_size, "silesia output size mismatch");
        assert_eq!(
            &cf_output[..cf_size],
            &ld_output[..ld_size],
            "silesia output content mismatch"
        );
        eprintln!(
            "silesia correctness: both produce identical {} byte output",
            cf_size
        );
    }

    // =========================================================================
    // Layer 2: Performance comparison — full stream timing
    // =========================================================================

    #[test]
    fn test_inflate_perf_comparison() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = InflateOracle::from_raw(&data);

        let trials = 10;

        // Warm up
        let mut output = vec![0u8; oracle.expected_output.len() + 65536];
        let _ = inflate_consume_first(&oracle.deflate_data, &mut output);
        let _ = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut output);

        // Time consume_first
        let start = Instant::now();
        for _ in 0..trials {
            let _ = inflate_consume_first(&oracle.deflate_data, &mut output);
        }
        let cf_elapsed = start.elapsed();

        // Time libdeflate
        let start = Instant::now();
        for _ in 0..trials {
            let _ = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut output);
        }
        let ld_elapsed = start.elapsed();

        let cf_mbps =
            (oracle.expected_output.len() as f64 * trials as f64) / cf_elapsed.as_secs_f64() / 1e6;
        let ld_mbps =
            (oracle.expected_output.len() as f64 * trials as f64) / ld_elapsed.as_secs_f64() / 1e6;
        let ratio = cf_mbps / ld_mbps * 100.0;

        eprintln!(
            "=== Inflate Performance (synthetic {} bytes) ===",
            oracle.expected_output.len()
        );
        eprintln!("  consume_first: {:.0} MB/s", cf_mbps);
        eprintln!("  libdeflate:    {:.0} MB/s", ld_mbps);
        eprintln!("  ratio:         {:.1}% of libdeflate", ratio);
    }

    #[test]
    fn test_inflate_perf_comparison_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };
        let oracle = InflateOracle::from_gzip(&gz);

        let trials = 5;
        let mut output = vec![0u8; oracle.expected_output.len() + 65536];

        // Warm up
        let _ = inflate_consume_first(&oracle.deflate_data, &mut output);
        let _ = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut output);

        let start = Instant::now();
        for _ in 0..trials {
            let _ = inflate_consume_first(&oracle.deflate_data, &mut output);
        }
        let cf_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..trials {
            let _ = crate::bgzf::inflate_into_pub(&oracle.deflate_data, &mut output);
        }
        let ld_elapsed = start.elapsed();

        let cf_mbps =
            (oracle.expected_output.len() as f64 * trials as f64) / cf_elapsed.as_secs_f64() / 1e6;
        let ld_mbps =
            (oracle.expected_output.len() as f64 * trials as f64) / ld_elapsed.as_secs_f64() / 1e6;
        let ratio = cf_mbps / ld_mbps * 100.0;

        eprintln!(
            "=== Inflate Performance (silesia {} bytes) ===",
            oracle.expected_output.len()
        );
        eprintln!("  consume_first: {:.0} MB/s", cf_mbps);
        eprintln!("  libdeflate:    {:.0} MB/s", ld_mbps);
        eprintln!("  ratio:         {:.1}% of libdeflate", ratio);
        oracle.summary();
    }

    // =========================================================================
    // Layer 2.5: Per-block timing — isolate which blocks cause the gap
    // =========================================================================

    #[test]
    fn test_per_block_timing_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };
        let oracle = InflateOracle::from_gzip(&gz);

        // Group blocks into ~1MB regions and time each region with both decoders.
        // We can't time individual deflate blocks independently (no reset between blocks),
        // but we CAN time the full stream and correlate with block metadata.

        // Approach: decode the full stream with consume_first in per-block mode,
        // timing each block separately.
        struct BlockTiming {
            index: usize,
            btype: u8,
            output_size: usize,
            cf_ns: u64,
        }

        let mut timings = Vec::with_capacity(oracle.blocks.len());
        let mut output = vec![0u8; oracle.expected_output.len() + 65536];
        let mut bits = Bits::new(&oracle.deflate_data);
        let mut out_pos = 0usize;

        for block in &oracle.blocks {
            if bits.available() < 3 {
                bits.refill();
            }
            let header = bits.peek();
            let _bfinal = (header & 1) != 0;
            let btype = ((header >> 1) & 3) as u8;
            bits.consume(3);

            let start = Instant::now();

            match btype {
                0 => {
                    out_pos = crate::consume_first_decode::decode_stored_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap();
                }
                1 => {
                    out_pos = crate::consume_first_decode::decode_fixed_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap();
                }
                2 => {
                    out_pos = crate::consume_first_decode::decode_dynamic_pub(
                        &mut bits,
                        &mut output,
                        out_pos,
                    )
                    .unwrap();
                }
                _ => break,
            }

            let elapsed = start.elapsed();
            timings.push(BlockTiming {
                index: block.index,
                btype,
                output_size: out_pos - block.output_offset,
                cf_ns: elapsed.as_nanos() as u64,
            });
        }

        // Aggregate by block type
        let mut type_total_ns = [0u64; 3];
        let mut type_total_bytes = [0u64; 3];
        let mut type_count = [0u32; 3];
        let mut slowest_blocks: Vec<(usize, u8, f64, usize)> = Vec::new();

        for t in &timings {
            if (t.btype as usize) < 3 {
                type_total_ns[t.btype as usize] += t.cf_ns;
                type_total_bytes[t.btype as usize] += t.output_size as u64;
                type_count[t.btype as usize] += 1;
            }
            if t.output_size > 0 {
                let mbps = t.output_size as f64 / (t.cf_ns as f64 / 1e9) / 1e6;
                slowest_blocks.push((t.index, t.btype, mbps, t.output_size));
            }
        }

        eprintln!("=== Per-Block Timing (silesia, consume_first) ===");
        eprintln!(
            "{:<10} {:>6} {:>12} {:>10} {:>10}",
            "Type", "Count", "Output MB", "Time ms", "MB/s"
        );
        eprintln!("{}", "-".repeat(55));
        for t in 0..3 {
            if type_count[t] > 0 {
                let mb = type_total_bytes[t] as f64 / 1e6;
                let ms = type_total_ns[t] as f64 / 1e6;
                let mbps = type_total_bytes[t] as f64 / (type_total_ns[t] as f64 / 1e9) / 1e6;
                eprintln!(
                    "{:<10} {:>6} {:>12.2} {:>10.2} {:>10.0}",
                    InflateOracle::block_type_name(t as u8),
                    type_count[t],
                    mb,
                    ms,
                    mbps
                );
            }
        }

        // Show slowest 10 blocks
        slowest_blocks.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        eprintln!("\n--- 10 Slowest Blocks ---");
        eprintln!(
            "{:<8} {:>8} {:>10} {:>10}",
            "Block#", "Type", "Size KB", "MB/s"
        );
        for (idx, btype, mbps, size) in slowest_blocks.iter().take(10) {
            eprintln!(
                "{:<8} {:>8} {:>10.1} {:>10.0}",
                idx,
                InflateOracle::block_type_name(*btype),
                *size as f64 / 1024.0,
                mbps
            );
        }

        // Show fastest 5 blocks for comparison
        eprintln!("\n--- 5 Fastest Blocks ---");
        for (idx, btype, mbps, size) in slowest_blocks.iter().rev().take(5) {
            eprintln!(
                "{:<8} {:>8} {:>10.1} {:>10.0}",
                idx,
                InflateOracle::block_type_name(*btype),
                *size as f64 / 1024.0,
                mbps
            );
        }
    }

    // =========================================================================
    // Layer 3: Pattern analysis — which block types cause the gap
    // =========================================================================

    #[test]
    fn test_inflate_pattern_analysis() {
        type PatternFn = fn() -> Vec<u8>;
        let patterns: &[(&str, PatternFn)] = &[
            ("literals", || {
                // Mostly unique bytes — tests literal decoding
                (0..4_000_000u32)
                    .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
                    .collect()
            }),
            ("rle", || {
                // Long runs of the same byte — tests distance=1 copy
                let mut data = Vec::with_capacity(4_000_000);
                for _ in 0..100 {
                    let byte = (data.len() % 256) as u8;
                    data.extend(std::iter::repeat_n(byte, 40_000));
                }
                data
            }),
            ("short_matches", || {
                // Repeating short patterns — tests small distance copies
                let pattern = b"abcdefgh12345678";
                let mut data = Vec::with_capacity(4_000_000);
                while data.len() < 4_000_000 {
                    data.extend_from_slice(pattern);
                }
                data.truncate(4_000_000);
                data
            }),
            ("mixed", || {
                // Mix of literals and matches (realistic data)
                make_test_data(4_000_000)
            }),
        ];

        eprintln!("=== Inflate Pattern Analysis ===");
        eprintln!(
            "{:<15} {:>10} {:>10} {:>8}",
            "Pattern", "CF MB/s", "LD MB/s", "Ratio"
        );
        eprintln!("{}", "-".repeat(48));

        for (name, gen) in patterns {
            let data = gen();
            let gz = compress_gzip(&data);
            let header_size = crate::marker_decode::skip_gzip_header(&gz).expect("valid header");
            let deflate = &gz[header_size..gz.len() - 8];

            let mut output = vec![0u8; data.len() + 65536];
            let trials = 10;

            // Warm up
            let _ = inflate_consume_first(deflate, &mut output);
            let _ = crate::bgzf::inflate_into_pub(deflate, &mut output);

            let start = Instant::now();
            for _ in 0..trials {
                let _ = inflate_consume_first(deflate, &mut output);
            }
            let cf_elapsed = start.elapsed();

            let start = Instant::now();
            for _ in 0..trials {
                let _ = crate::bgzf::inflate_into_pub(deflate, &mut output);
            }
            let ld_elapsed = start.elapsed();

            let cf_mbps = (data.len() as f64 * trials as f64) / cf_elapsed.as_secs_f64() / 1e6;
            let ld_mbps = (data.len() as f64 * trials as f64) / ld_elapsed.as_secs_f64() / 1e6;
            let ratio = cf_mbps / ld_mbps * 100.0;

            eprintln!(
                "{:<15} {:>10.0} {:>10.0} {:>7.1}%",
                name, cf_mbps, ld_mbps, ratio
            );
        }
    }
}
