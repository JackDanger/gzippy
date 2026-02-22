//! Layered test harness for rapidgzip-style parallel decompression pipeline.
//!
//! Uses `scan_deflate_fast` as a ground-truth oracle to verify each pipeline
//! component independently before composing them together.
//!
//! Layer 0: DeflateOracle — ground truth (checkpoints + expected output)
//! Layer 1: MarkerDecoder — decode from known boundaries with/without windows
//! Layer 2: Block search — find real boundaries near guesses
//! Layer 3: Chunk decoder — search + decode composition
//! Layer 4: Pipeline — orchestration, threading, output assembly

#[cfg(test)]
mod tests {
    #![allow(unused_variables)]
    use crate::block_finder::BlockFinder;
    use crate::marker_decode::{MarkerDecoder, MARKER_BASE, WINDOW_SIZE};
    use crate::scan_inflate::{scan_deflate_fast, ScanCheckpoint};
    use std::collections::HashSet;
    use std::io::Write;

    // =========================================================================
    // Layer 0: DeflateOracle
    // =========================================================================

    /// Ground truth for a deflate stream. Built from scan_deflate_fast (for
    /// checkpoints with exact block boundaries and windows) plus a full
    /// sequential decode (for expected output bytes).
    struct DeflateOracle {
        deflate_data: Vec<u8>,
        expected_output: Vec<u8>,
        checkpoints: Vec<ScanCheckpoint>,
        total_output_size: usize,
    }

    impl DeflateOracle {
        /// Build oracle from raw gzip data.
        fn from_gzip(gzip_data: &[u8]) -> Self {
            let header_size =
                crate::marker_decode::skip_gzip_header(gzip_data).expect("valid gzip header");
            let deflate_data = &gzip_data[header_size..gzip_data.len() - 8];

            // Get checkpoints via scan pass (every 256KB for dense coverage)
            let scan = scan_deflate_fast(deflate_data, 256 * 1024, 0)
                .expect("scan_deflate_fast should succeed");

            // Get expected output via full decode
            let mut output = vec![0u8; scan.total_output_size + 65536];
            let actual_size =
                crate::consume_first_decode::inflate_consume_first(deflate_data, &mut output)
                    .expect("inflate should succeed");
            output.truncate(actual_size);

            assert_eq!(
                actual_size, scan.total_output_size,
                "scan and inflate disagree on output size"
            );

            Self {
                deflate_data: deflate_data.to_vec(),
                expected_output: output,
                checkpoints: scan.checkpoints,
                total_output_size: scan.total_output_size,
            }
        }

        /// Build oracle from raw uncompressed data (compresses it first).
        fn from_raw(data: &[u8]) -> Self {
            let gzip_data = compress_to_gzip(data);
            Self::from_gzip(&gzip_data)
        }

        /// Expected output bytes for a given range.
        fn expected_bytes(&self, offset: usize, len: usize) -> &[u8] {
            &self.expected_output[offset..offset + len]
        }

        /// Set of all known block boundary bit positions.
        fn known_bit_positions(&self) -> HashSet<usize> {
            self.checkpoints.iter().map(checkpoint_bit_pos).collect()
        }

        /// Output range for a given checkpoint pair (start_idx, end_idx).
        /// end_idx can be checkpoints.len() to mean "to the end".
        fn chunk_expected(&self, start_idx: usize, end_idx: usize) -> (usize, usize, &[u8]) {
            let start_offset = self.checkpoints[start_idx].output_offset;
            let end_offset = if end_idx < self.checkpoints.len() {
                self.checkpoints[end_idx].output_offset
            } else {
                self.total_output_size
            };
            (
                start_offset,
                end_offset,
                &self.expected_output[start_offset..end_offset],
            )
        }
    }

    /// Extract the actual bit position from a checkpoint.
    /// The Bits reader uses libdeflate's pattern where only the low 8 bits
    /// of bitsleft are meaningful (high bits may be garbage from wrapping_sub).
    fn checkpoint_bit_pos(cp: &ScanCheckpoint) -> usize {
        let real_bitsleft = (cp.bitsleft as u8) as usize;
        cp.input_byte_pos * 8 - real_bitsleft
    }

    fn compress_to_gzip(data: &[u8]) -> Vec<u8> {
        // Use default compression (level 6) — produces many small deflate blocks,
        // giving us dense block boundaries for testing.
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Generate semi-random data with enough structure for deflate to produce
    /// many dynamic Huffman blocks.
    fn make_test_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOP\n",
            b"how vexingly quick daft zebras jump. ",
            b"the five boxing wizards jump quickly. ",
        ];
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 2 {
                // ~40% random bytes
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
    fn test_oracle_from_synthetic() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        assert_eq!(oracle.expected_output, data);
        assert!(!oracle.checkpoints.is_empty(), "should have checkpoints");
        eprintln!(
            "oracle: {} checkpoints for {} bytes ({} deflate bytes)",
            oracle.checkpoints.len(),
            oracle.total_output_size,
            oracle.deflate_data.len()
        );

        // Checkpoints should be monotonically increasing in output offset
        for i in 1..oracle.checkpoints.len() {
            assert!(
                oracle.checkpoints[i].output_offset > oracle.checkpoints[i - 1].output_offset,
                "checkpoint {} output_offset not increasing",
                i
            );
        }

        // Each checkpoint window should be 32KB (or less for early checkpoints)
        for (i, cp) in oracle.checkpoints.iter().enumerate() {
            assert!(
                cp.window.len() <= WINDOW_SIZE,
                "checkpoint {} window too large: {}",
                i,
                cp.window.len()
            );
            if cp.output_offset >= WINDOW_SIZE {
                assert_eq!(
                    cp.window.len(),
                    WINDOW_SIZE,
                    "checkpoint {} should have full 32KB window at offset {}",
                    i,
                    cp.output_offset
                );
            }
        }
    }

    #[test]
    fn test_oracle_checkpoint_windows_match_output() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        for (i, cp) in oracle.checkpoints.iter().enumerate() {
            if cp.output_offset < WINDOW_SIZE {
                continue;
            }
            let window_start = cp.output_offset - WINDOW_SIZE;
            let expected_window = &oracle.expected_output[window_start..cp.output_offset];
            assert_eq!(
                cp.window, expected_window,
                "checkpoint {} window doesn't match expected output at offset {}",
                i, cp.output_offset
            );
        }
    }

    #[test]
    fn test_oracle_from_silesia() {
        let gzip_data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };
        let oracle = DeflateOracle::from_gzip(&gzip_data);
        eprintln!(
            "silesia oracle: {} checkpoints, {} bytes output, {} bytes deflate",
            oracle.checkpoints.len(),
            oracle.total_output_size,
            oracle.deflate_data.len()
        );
        assert!(oracle.checkpoints.len() > 10);
        assert_eq!(oracle.total_output_size, oracle.expected_output.len());
    }

    // =========================================================================
    // Layer 1: MarkerDecoder with known-good boundaries
    // =========================================================================

    /// Decode chunk 0 (from start, no markers needed) and verify output.
    #[test]
    fn test_layer1_chunk0_no_markers() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);
        assert!(!oracle.checkpoints.is_empty());

        let cp0 = &oracle.checkpoints[0];

        // Decode from bit 0 with empty window (chunk 0 has no prior context)
        let mut decoder = MarkerDecoder::with_window(&oracle.deflate_data, 0, &[]);
        decoder
            .decode_until(cp0.output_offset)
            .expect("decode should succeed");

        let output = decoder.to_bytes(&[]);
        let expected = oracle.expected_bytes(0, cp0.output_offset);

        assert_eq!(
            output.len(),
            expected.len(),
            "chunk 0 output size mismatch: got {}, expected {}",
            output.len(),
            expected.len()
        );
        assert_eq!(output, expected, "chunk 0 output content mismatch");
        assert!(
            !decoder.has_markers(),
            "chunk 0 should have no markers (has window)"
        );
    }

    /// Decode a middle chunk WITH the correct window (no markers).
    #[test]
    fn test_layer1_middle_chunk_with_window() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            eprintln!("not enough checkpoints, skipping");
            return;
        }

        // Pick a middle checkpoint
        let idx = oracle.checkpoints.len() / 2;
        let cp = &oracle.checkpoints[idx];

        // Compute the bit position to start decoding from
        let start_bit = checkpoint_bit_pos(cp);

        // Determine how much output to decode
        let (_, end_offset, expected) = oracle.chunk_expected(idx, idx + 1);
        let chunk_size = end_offset - cp.output_offset;

        // Decode with the oracle's window
        let mut decoder = MarkerDecoder::with_window(&oracle.deflate_data, start_bit, &cp.window);
        decoder
            .decode_until(chunk_size)
            .expect("decode should succeed");

        let output = decoder.to_bytes(&cp.window);

        assert_eq!(
            output.len(),
            expected.len(),
            "middle chunk output size mismatch: got {}, expected {} (checkpoint {})",
            output.len(),
            expected.len(),
            idx
        );
        // Compare first divergence point for better error messages
        if output != expected {
            for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
                if a != b {
                    panic!(
                        "middle chunk diverges at byte {}: got {:#x}, expected {:#x} (checkpoint {})",
                        i, a, b, idx
                    );
                }
            }
        }
    }

    /// Decode a middle chunk WITHOUT window (marker mode), then resolve markers.
    #[test]
    fn test_layer1_middle_chunk_markers_then_resolve() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            eprintln!("not enough checkpoints, skipping");
            return;
        }

        let idx = oracle.checkpoints.len() / 2;
        let cp = &oracle.checkpoints[idx];
        let start_bit = checkpoint_bit_pos(cp);
        let (_, end_offset, expected) = oracle.chunk_expected(idx, idx + 1);
        let chunk_size = end_offset - cp.output_offset;

        // Decode WITHOUT window (marker mode)
        let mut decoder = MarkerDecoder::new(&oracle.deflate_data, start_bit);
        decoder
            .decode_until(chunk_size)
            .expect("decode should succeed");

        // Should have markers since back-references can't resolve
        let raw_output = decoder.output();
        let markers: Vec<_> = raw_output.iter().filter(|&&v| v >= MARKER_BASE).collect();
        eprintln!(
            "middle chunk: {} bytes output, {} markers ({:.1}%)",
            raw_output.len(),
            markers.len(),
            markers.len() as f64 / raw_output.len() as f64 * 100.0
        );

        // Resolve markers with the correct window
        let resolved = decoder.to_bytes(&cp.window);

        assert_eq!(
            resolved.len(),
            expected.len(),
            "resolved output size mismatch"
        );
        assert_eq!(resolved, expected, "resolved output content mismatch");
    }

    /// Verify marker resolution across ALL checkpoints (exhaustive).
    #[test]
    fn test_layer1_all_checkpoints_with_window() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        let mut failures = 0;
        for idx in 0..oracle.checkpoints.len() {
            let cp = &oracle.checkpoints[idx];
            let start_bit = checkpoint_bit_pos(cp);
            let (_, end_offset, expected) = oracle.chunk_expected(idx, idx + 1);
            let chunk_size = end_offset - cp.output_offset;

            if chunk_size == 0 {
                continue;
            }

            let mut decoder =
                MarkerDecoder::with_window(&oracle.deflate_data, start_bit, &cp.window);
            match decoder.decode_until(chunk_size) {
                Ok(_) => {
                    let output = decoder.to_bytes(&cp.window);
                    if output.len() != expected.len() || output != expected {
                        eprintln!(
                            "FAIL checkpoint {}: output {} bytes, expected {} bytes",
                            idx,
                            output.len(),
                            expected.len()
                        );
                        failures += 1;
                    }
                }
                Err(e) => {
                    eprintln!("FAIL checkpoint {}: decode error: {}", idx, e);
                    failures += 1;
                }
            }
        }

        eprintln!(
            "layer 1 exhaustive: {}/{} checkpoints passed",
            oracle.checkpoints.len() - failures,
            oracle.checkpoints.len()
        );
        assert_eq!(failures, 0, "{} checkpoints failed", failures);
    }

    // =========================================================================
    // Layer 2: Block search accuracy
    // =========================================================================

    /// Measure block finder precision against oracle-known boundaries.
    #[test]
    fn test_layer2_block_finder_precision() {
        let data = make_test_data(4 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        let known_positions = oracle.known_bit_positions();
        eprintln!(
            "oracle has {} known block boundaries",
            known_positions.len()
        );

        let finder = BlockFinder::new(&oracle.deflate_data);
        let total_bits = oracle.deflate_data.len() * 8;
        let candidates = finder.find_blocks(0, total_bits);

        let true_positives = candidates
            .iter()
            .filter(|c| known_positions.contains(&c.bit_offset))
            .count();
        let false_positives = candidates.len() - true_positives;

        eprintln!(
            "block finder: {} candidates, {} true positives, {} false positives",
            candidates.len(),
            true_positives,
            false_positives
        );
        eprintln!(
            "precision: {:.1}%, recall: {:.1}%",
            if candidates.is_empty() {
                0.0
            } else {
                true_positives as f64 / candidates.len() as f64 * 100.0
            },
            if known_positions.is_empty() {
                0.0
            } else {
                true_positives as f64 / known_positions.len() as f64 * 100.0
            }
        );

        // We don't assert precision > X% here — this is a diagnostic test.
        // The important thing is we KNOW the numbers.
    }

    /// Test that block finder candidates near oracle boundaries can actually
    /// decode successfully (trial decode validation).
    #[test]
    fn test_layer2_trial_decode_at_known_boundaries() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        let mut successes = 0;
        let mut failures = 0;

        for (idx, cp) in oracle.checkpoints.iter().enumerate() {
            let start_bit = checkpoint_bit_pos(cp);

            // Try to decode a small amount from this known-good position
            let mut decoder = MarkerDecoder::new(&oracle.deflate_data, start_bit);
            match decoder.decode_until(4096) {
                Ok(_) if decoder.output().len() >= 1024 => {
                    successes += 1;
                }
                Ok(_) => {
                    eprintln!(
                        "checkpoint {}: decoded only {} bytes (bit {})",
                        idx,
                        decoder.output().len(),
                        start_bit
                    );
                    failures += 1;
                }
                Err(e) => {
                    eprintln!(
                        "checkpoint {}: decode failed at bit {}: {}",
                        idx, start_bit, e
                    );
                    failures += 1;
                }
            }
        }

        eprintln!(
            "trial decode: {}/{} oracle boundaries decode successfully",
            successes,
            oracle.checkpoints.len()
        );
        // All oracle boundaries should be decodable
        assert_eq!(
            failures, 0,
            "{} oracle boundaries failed trial decode",
            failures
        );
    }

    /// Test searching near a known boundary: given a guess offset by up to
    /// 8KB from a real boundary, can we find the real boundary?
    #[test]
    fn test_layer2_search_near_known_boundary() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            eprintln!("not enough checkpoints, skipping");
            return;
        }

        let known_positions = oracle.known_bit_positions();
        let finder = BlockFinder::new(&oracle.deflate_data);

        let mut found = 0;
        let mut missed = 0;

        for cp in &oracle.checkpoints {
            let real_bit = checkpoint_bit_pos(cp);

            // Search in a 16KB window around the real position
            let search_start = real_bit.saturating_sub(8 * 1024 * 8);
            let search_end = (real_bit + 8 * 1024 * 8).min(oracle.deflate_data.len() * 8);

            let candidates = finder.find_blocks(search_start, search_end);

            // Check if any candidate is exactly the real boundary
            let hit = candidates.iter().any(|c| c.bit_offset == real_bit);

            if hit {
                found += 1;
            } else {
                missed += 1;
                eprintln!(
                    "missed boundary at bit {} ({} candidates in range)",
                    real_bit,
                    candidates.len()
                );
            }
        }

        eprintln!(
            "search near boundary: found {}/{} ({:.1}%)",
            found,
            oracle.checkpoints.len(),
            found as f64 / oracle.checkpoints.len() as f64 * 100.0
        );
    }

    // =========================================================================
    // Layer 3: Chunk decoder (search + decode composition)
    // =========================================================================

    /// The chunk decoder function: given a guess position, search for a valid
    /// block start nearby and decode from it.
    struct ChunkResult {
        output: Vec<u16>,
        final_window: Vec<u8>,
        start_bit: usize,
        has_markers: bool,
        marker_count: usize,
    }

    /// Search for a valid block boundary near `guess_byte`, then decode.
    /// If `window` is provided, decodes with that window (no markers).
    /// If `window` is None, decodes in marker mode.
    /// Returns None if no valid block boundary found.
    fn decode_chunk(
        deflate_data: &[u8],
        guess_byte: usize,
        end_byte: usize,
        window: Option<&[u8]>,
    ) -> Option<ChunkResult> {
        let total_bits = deflate_data.len() * 8;

        // Search range: 16KB around the guess (128K bits)
        let guess_bit = guess_byte * 8;
        let search_start = guess_bit.saturating_sub(8 * 1024 * 8);
        let search_end = (guess_bit + 8 * 1024 * 8).min(total_bits);

        let finder = BlockFinder::new(deflate_data);
        let candidates = finder.find_blocks(search_start, search_end);

        // Sort by distance from guess
        let mut sorted: Vec<_> = candidates
            .iter()
            .map(|c| {
                let dist = c.bit_offset.abs_diff(guess_bit);
                (dist, c.bit_offset)
            })
            .collect();
        sorted.sort();

        // Try each candidate, closest first
        for &(_, bit_offset) in &sorted {
            let mut decoder = if let Some(win) = window {
                MarkerDecoder::with_window(deflate_data, bit_offset, win)
            } else {
                MarkerDecoder::new(deflate_data, bit_offset)
            };

            let max_output = (end_byte - guess_byte) * 4; // generous estimate
            match decoder.decode_until(max_output) {
                Ok(_) if decoder.output().len() >= 1024 => {
                    return Some(ChunkResult {
                        output: decoder.output().to_vec(),
                        final_window: decoder.final_window(),
                        start_bit: bit_offset,
                        has_markers: decoder.has_markers(),
                        marker_count: if decoder.has_markers() {
                            decoder
                                .output()
                                .iter()
                                .filter(|&&v| v >= MARKER_BASE)
                                .count()
                        } else {
                            0
                        },
                    });
                }
                _ => continue,
            }
        }

        None
    }

    /// Decode from an exact oracle checkpoint with its window. Should produce
    /// byte-perfect output with no markers.
    #[test]
    fn test_layer3_decode_with_oracle_window() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            return;
        }

        let idx = oracle.checkpoints.len() / 2;
        let cp = &oracle.checkpoints[idx];
        let next_byte = if idx + 1 < oracle.checkpoints.len() {
            oracle.checkpoints[idx + 1].input_byte_pos
        } else {
            oracle.deflate_data.len()
        };

        let result = decode_chunk(
            &oracle.deflate_data,
            cp.input_byte_pos,
            next_byte,
            Some(&cp.window),
        );

        let result = result.expect("decode from oracle checkpoint should succeed");
        assert!(
            !result.has_markers,
            "should have no markers with correct window"
        );

        let (_, end_offset, expected) = oracle.chunk_expected(idx, idx + 1);
        let output_bytes: Vec<u8> = result.output.iter().map(|&v| v as u8).collect();
        let cmp_len = expected.len().min(output_bytes.len());
        assert_eq!(
            &output_bytes[..cmp_len],
            &expected[..cmp_len],
            "output doesn't match expected"
        );
    }

    /// Decode from an oracle checkpoint in marker mode, then resolve.
    #[test]
    fn test_layer3_decode_marker_mode_then_resolve() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            return;
        }

        let idx = oracle.checkpoints.len() / 2;
        let cp = &oracle.checkpoints[idx];
        let next_byte = if idx + 1 < oracle.checkpoints.len() {
            oracle.checkpoints[idx + 1].input_byte_pos
        } else {
            oracle.deflate_data.len()
        };

        let result = decode_chunk(
            &oracle.deflate_data,
            cp.input_byte_pos,
            next_byte,
            None, // marker mode
        );

        let result = result.expect("decode should succeed");
        eprintln!(
            "marker mode: {} output values, {} markers ({:.1}%)",
            result.output.len(),
            result.marker_count,
            result.marker_count as f64 / result.output.len() as f64 * 100.0
        );

        // Resolve markers with the correct window
        let resolved: Vec<u8> = result
            .output
            .iter()
            .map(|&v| {
                if v <= 255 {
                    v as u8
                } else {
                    let offset = (v - MARKER_BASE) as usize;
                    if offset < cp.window.len() {
                        cp.window[cp.window.len() - 1 - offset]
                    } else {
                        0
                    }
                }
            })
            .collect();

        let (_, _, expected) = oracle.chunk_expected(idx, idx + 1);
        let cmp_len = expected.len().min(resolved.len());
        assert_eq!(
            &resolved[..cmp_len],
            &expected[..cmp_len],
            "resolved output doesn't match expected"
        );
    }

    /// Decode with a guess that's offset from the real boundary.
    #[test]
    fn test_layer3_decode_with_offset_guess() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            return;
        }

        let idx = oracle.checkpoints.len() / 2;
        let cp = &oracle.checkpoints[idx];
        let next_byte = if idx + 1 < oracle.checkpoints.len() {
            oracle.checkpoints[idx + 1].input_byte_pos
        } else {
            oracle.deflate_data.len()
        };

        // Offset the guess by 2KB
        let guess_byte = cp.input_byte_pos + 2048;
        if guess_byte >= next_byte {
            eprintln!("guess would exceed chunk range, skipping");
            return;
        }

        let result = decode_chunk(
            &oracle.deflate_data,
            guess_byte,
            next_byte,
            Some(&cp.window),
        );

        match result {
            Some(r) => {
                eprintln!(
                    "offset guess found boundary at bit {} (real was byte {}), decoded {} values",
                    r.start_bit,
                    cp.input_byte_pos,
                    r.output.len()
                );
            }
            None => {
                eprintln!(
                    "offset guess at byte {} found no boundary (real was byte {})",
                    guess_byte, cp.input_byte_pos
                );
            }
        }
    }

    // =========================================================================
    // Layer 4: Pipeline orchestration
    // =========================================================================

    /// A block guess for the pipeline. If `exact_bit` is set, the decoder
    /// uses that bit position directly without searching.
    struct BlockGuess {
        guess_byte: usize,
        end_byte: usize,
        exact_bit: Option<usize>,
        window: Option<Vec<u8>>,
        /// Maximum output bytes for this chunk (None = 16MB default).
        max_output: Option<usize>,
    }

    /// A source of block guesses for the pipeline.
    trait BlockSource: Send {
        fn next_block(&mut self) -> Option<BlockGuess>;
    }

    /// Oracle source: returns known-good checkpoint positions with exact bit
    /// offsets and windows. Tests pipeline machinery without block-finding.
    struct OracleBlockSource {
        guesses: Vec<BlockGuess>,
        idx: usize,
    }

    impl OracleBlockSource {
        fn from_oracle(oracle: &DeflateOracle) -> Self {
            let mut guesses = Vec::new();
            for i in 0..oracle.checkpoints.len() {
                let cp = &oracle.checkpoints[i];
                let end = if i + 1 < oracle.checkpoints.len() {
                    oracle.checkpoints[i + 1].input_byte_pos
                } else {
                    oracle.deflate_data.len()
                };
                let chunk_output_size = if i + 1 < oracle.checkpoints.len() {
                    oracle.checkpoints[i + 1].output_offset - cp.output_offset
                } else {
                    oracle.total_output_size - cp.output_offset
                };
                guesses.push(BlockGuess {
                    guess_byte: cp.input_byte_pos,
                    end_byte: end,
                    exact_bit: Some(checkpoint_bit_pos(cp)),
                    window: Some(cp.window.clone()),
                    max_output: Some(chunk_output_size),
                });
            }
            Self { guesses, idx: 0 }
        }
    }

    impl BlockSource for OracleBlockSource {
        fn next_block(&mut self) -> Option<BlockGuess> {
            if self.idx < self.guesses.len() {
                let g = &self.guesses[self.idx];
                self.idx += 1;
                Some(BlockGuess {
                    guess_byte: g.guess_byte,
                    end_byte: g.end_byte,
                    exact_bit: g.exact_bit,
                    window: g.window.clone(),
                    max_output: g.max_output,
                })
            } else {
                None
            }
        }
    }

    /// Spacing source: returns evenly-spaced byte positions (production mode).
    /// No exact bit offset or window — decoder must search and use markers.
    struct SpacingBlockSource {
        spacing: usize,
        data_len: usize,
        current: usize,
    }

    impl SpacingBlockSource {
        fn new(data_len: usize, num_chunks: usize) -> Self {
            let spacing = data_len / num_chunks;
            Self {
                spacing,
                data_len,
                current: 0,
            }
        }
    }

    impl BlockSource for SpacingBlockSource {
        fn next_block(&mut self) -> Option<BlockGuess> {
            if self.current >= self.data_len {
                return None;
            }
            let start = self.current;
            let end = (start + self.spacing).min(self.data_len);
            self.current = end;
            Some(BlockGuess {
                guess_byte: start,
                end_byte: end,
                exact_bit: None,
                window: None,
                max_output: None,
            })
        }
    }

    /// Adversarial source: returns positions guaranteed NOT to be block boundaries.
    struct AdversarialBlockSource {
        guesses: Vec<BlockGuess>,
        idx: usize,
    }

    impl AdversarialBlockSource {
        fn new(oracle: &DeflateOracle, num_chunks: usize) -> Self {
            let known = oracle.known_bit_positions();
            let spacing = oracle.deflate_data.len() / num_chunks;
            let mut guesses = Vec::new();
            for i in 0..num_chunks {
                let byte_pos = i * spacing + spacing / 3;
                let bit_pos = byte_pos * 8;
                if !known.contains(&bit_pos) {
                    let end = ((i + 1) * spacing).min(oracle.deflate_data.len());
                    guesses.push(BlockGuess {
                        guess_byte: byte_pos,
                        end_byte: end,
                        exact_bit: None,
                        window: None,
                        max_output: None,
                    });
                }
            }
            Self { guesses, idx: 0 }
        }
    }

    impl BlockSource for AdversarialBlockSource {
        fn next_block(&mut self) -> Option<BlockGuess> {
            if self.idx < self.guesses.len() {
                let result = self.idx;
                self.idx += 1;
                Some(BlockGuess {
                    guess_byte: self.guesses[result].guess_byte,
                    end_byte: self.guesses[result].end_byte,
                    exact_bit: None,
                    window: None,
                    max_output: None,
                })
            } else {
                None
            }
        }
    }

    /// Decode a chunk from an exact bit position (no searching).
    fn decode_chunk_exact(
        deflate_data: &[u8],
        start_bit: usize,
        window: Option<&[u8]>,
        max_output: usize,
    ) -> Option<ChunkResult> {
        let mut decoder = if let Some(win) = window {
            MarkerDecoder::with_window(deflate_data, start_bit, win)
        } else {
            MarkerDecoder::new(deflate_data, start_bit)
        };

        match decoder.decode_until(max_output) {
            Ok(_) => Some(ChunkResult {
                output: decoder.output().to_vec(),
                final_window: decoder.final_window(),
                start_bit,
                has_markers: decoder.has_markers(),
                marker_count: if decoder.has_markers() {
                    decoder
                        .output()
                        .iter()
                        .filter(|&&v| v >= MARKER_BASE)
                        .count()
                } else {
                    0
                },
            }),
            Err(_) => None,
        }
    }

    /// Run the pipeline: sequential orchestrator assigns chunks from a
    /// BlockSource, decodes them (searching for real boundaries if needed),
    /// propagates windows, resolves markers, assembles output.
    fn run_pipeline(
        deflate_data: &[u8],
        first_chunk_end: usize,
        source: &mut dyn BlockSource,
        _num_threads: usize,
    ) -> Option<Vec<u8>> {
        // Phase 1: Decode chunk 0 (always from bit 0, with empty window)
        let mut decoder0 = MarkerDecoder::with_window(deflate_data, 0, &[]);
        match decoder0.decode_until(first_chunk_end) {
            Ok(_) => {}
            Err(_) => return None,
        }
        let chunk0_bytes: Vec<u8> = decoder0.output().iter().map(|&v| v as u8).collect();
        let chunk0_window = decoder0.final_window();

        // Phase 2: Collect chunk specs
        let mut chunk_guesses: Vec<BlockGuess> = Vec::new();
        while let Some(guess) = source.next_block() {
            chunk_guesses.push(guess);
        }

        if chunk_guesses.is_empty() {
            return Some(chunk0_bytes);
        }

        // Phase 3: Decode chunks in parallel (marker mode unless window provided)
        let chunk_results: Vec<Option<ChunkResult>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk_guesses
                .iter()
                .map(|guess| {
                    let data = deflate_data;
                    let exact_bit = guess.exact_bit;
                    let start = guess.guess_byte;
                    let end = guess.end_byte;
                    let window = guess.window.as_deref();
                    let max_out = guess.max_output.unwrap_or(16 * 1024 * 1024);
                    s.spawn(move || {
                        if let Some(bit) = exact_bit {
                            decode_chunk_exact(data, bit, window, max_out)
                        } else {
                            decode_chunk(data, start, end, window)
                        }
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let success_count = chunk_results.iter().filter(|r| r.is_some()).count();
        if success_count == 0 {
            return Some(chunk0_bytes);
        }

        // Phase 4: Sequential window propagation + marker resolution
        let mut final_output = chunk0_bytes;
        let mut prev_window = chunk0_window;

        for (i, result) in chunk_results.iter().enumerate() {
            match result {
                Some(chunk) => {
                    let resolved: Vec<u8> = chunk
                        .output
                        .iter()
                        .map(|&v| {
                            if v <= 255 {
                                v as u8
                            } else {
                                let offset = (v - MARKER_BASE) as usize;
                                if offset < prev_window.len() {
                                    prev_window[prev_window.len() - 1 - offset]
                                } else {
                                    0
                                }
                            }
                        })
                        .collect();
                    final_output.extend_from_slice(&resolved);
                    prev_window = chunk.final_window.clone();
                }
                None => {
                    eprintln!("pipeline: chunk {} failed, output may be truncated", i);
                    break;
                }
            }
        }

        Some(final_output)
    }

    /// Pipeline with oracle block source (perfect boundaries + windows).
    /// Tests threading, ordering, and window propagation — not block finding.
    #[test]
    fn test_layer4_pipeline_oracle_source() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        if oracle.checkpoints.len() < 3 {
            eprintln!("not enough checkpoints, skipping");
            return;
        }

        // Chunk 0 decodes from start to first checkpoint
        let first_chunk_end = oracle.checkpoints[0].output_offset;

        // Oracle source provides exact bit positions + windows for all checkpoints
        let mut source = OracleBlockSource::from_oracle(&oracle);

        let output = run_pipeline(&oracle.deflate_data, first_chunk_end, &mut source, 4);

        let output = output.expect("oracle pipeline should produce output");
        eprintln!(
            "oracle pipeline: {} bytes output, expected {}",
            output.len(),
            oracle.total_output_size
        );

        // With oracle source (exact positions + windows), output should be
        // byte-identical to expected.
        assert_eq!(
            output.len(),
            oracle.total_output_size,
            "oracle pipeline output size mismatch"
        );
        assert_eq!(
            output, oracle.expected_output,
            "oracle pipeline output content mismatch"
        );
    }

    /// Pipeline with spacing source (production-like evenly-spaced guesses).
    #[test]
    fn test_layer4_pipeline_spacing_source() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        let first_chunk_end = if !oracle.checkpoints.is_empty() {
            oracle.checkpoints[0].output_offset
        } else {
            oracle.total_output_size
        };

        let num_chunks = 4;
        let mut source = SpacingBlockSource::new(oracle.deflate_data.len(), num_chunks);
        // Skip the first chunk range (handled by chunk 0)
        source.current = source.spacing;

        let output = run_pipeline(&oracle.deflate_data, first_chunk_end, &mut source, 4);

        match output {
            Some(out) => {
                eprintln!(
                    "spacing pipeline: {} bytes output, expected {}",
                    out.len(),
                    oracle.total_output_size
                );
                // Verify chunk 0 is correct
                let cmp_len = first_chunk_end.min(out.len());
                assert_eq!(
                    &out[..cmp_len],
                    &oracle.expected_output[..cmp_len],
                    "chunk 0 from spacing pipeline doesn't match"
                );
            }
            None => {
                eprintln!("spacing pipeline returned None (no valid blocks found — expected for some data)");
            }
        }
    }

    /// Pipeline with adversarial source (positions that aren't block boundaries).
    /// Should either find nearby real boundaries or gracefully produce partial output.
    #[test]
    fn test_layer4_pipeline_adversarial_source() {
        let data = make_test_data(8 * 1024 * 1024);
        let oracle = DeflateOracle::from_raw(&data);

        let first_chunk_end = if !oracle.checkpoints.is_empty() {
            oracle.checkpoints[0].output_offset
        } else {
            oracle.total_output_size
        };

        let mut source = AdversarialBlockSource::new(&oracle, 4);

        let output = run_pipeline(&oracle.deflate_data, first_chunk_end, &mut source, 4);

        match output {
            Some(out) => {
                eprintln!(
                    "adversarial pipeline: {} bytes output (expected {})",
                    out.len(),
                    oracle.total_output_size
                );
                // Chunk 0 should still be correct
                let cmp_len = first_chunk_end.min(out.len());
                assert_eq!(
                    &out[..cmp_len],
                    &oracle.expected_output[..cmp_len],
                    "chunk 0 should be correct even with adversarial source"
                );
            }
            None => {
                eprintln!("adversarial pipeline returned None (expected)");
            }
        }
    }
}
