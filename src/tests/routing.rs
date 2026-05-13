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
    use crate::compress::parallel::{compress_single_member, GzipHeaderInfo};
    use crate::decompress::format::has_bgzf_markers;
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

        let multi = crate::decompress::format::is_likely_multi_member(&oracle.multi_member_gz);
        let single = crate::decompress::format::is_likely_multi_member(&oracle.single_member_gz);
        let bgzf = crate::decompress::format::is_likely_multi_member(&oracle.bgzf_gz);

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
        let output_t1 =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 1).unwrap();
        assert_eq!(
            output_t1, oracle.original,
            "BGZF T1 output doesn't match original"
        );

        // T4
        let output_t4 =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
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
        let output_par = crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(
            &oracle.multi_member_gz,
            4,
        )
        .unwrap();
        assert_eq!(
            output_par, oracle.original,
            "multi-member parallel output doesn't match original"
        );

        // Sequential path
        let mut output_seq = Vec::new();
        crate::decompress::decompress_multi_member_sequential(
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
        crate::decompress::decompress_single_member_libdeflate(
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
    // Production routing: a single-member input large enough to clear the
    // 10 MiB parallel-path gate must produce correct output via
    // `decompress_single_member`. On x86_64 this exercises the parallel ISA-L
    // `inflatePrime` path wired in at v0.3.0; on arm64 it exercises libdeflate.
    // Either way, bytes must match.
    // =========================================================================

    #[test]
    fn test_single_member_routing_multithread() {
        // Use mostly-random data so it doesn't compress past the 10 MiB gate.
        // Target: ~14 MiB compressed.
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "test input must exceed parallel-path 10 MiB gate (got {} bytes)",
            compressed.len()
        );

        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(
            output, original,
            "decompress_single_member(T=4) output mismatch — production routing broken"
        );
    }

    // =========================================================================
    // Deletion-trap killer — the routing-assertion test from the v0.6 marker
    // decoder premortem.
    //
    // Every prior marker-based attempt in this codebase has been deleted
    // during cleanup because it lived outside the production CLI path. The
    // only thing that prevents the next cleanup from doing the same is a
    // test that fails when production routing *silently falls back* away
    // from the marker pipeline. Output-equivalence tests don't catch that —
    // gzippy will still produce correct output via the ISA-L sequential
    // fallback even if the marker pipeline is gone.
    //
    // `MARKER_PIPELINE_RUNS` is a process-global counter incremented on
    // every successful run of `parallel::single_member::decompress_parallel`.
    // We snapshot it around a real CLI-shaped decode and assert it moved.
    // On platforms where the parallel path is correctly gated off (arm64,
    // non-ISA-L builds) the test is a no-op.
    //
    // See `docs/marker-decoder-plan.md` for the full rationale.
    // =========================================================================
    #[test]
    fn test_marker_pipeline_actually_runs_on_x86_64_isal() {
        use std::sync::atomic::Ordering;

        // Serialize against any other test that calls `decompress_parallel`
        // concurrently — under `cargo test`'s default parallel execution,
        // another increment between `before` and `after` would mask a real
        // silent-fallback regression with a false-positive pass. (Copilot
        // review on PR #94.)
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let before = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(
            output, original,
            "output bytes wrong — but routing might also be broken; check the next assertion"
        );
        let after = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);

        // Only assert the counter moved on platforms where the parallel
        // marker pipeline is the intended path. Elsewhere the routing
        // correctly steers to libdeflate/zlib-ng and `after == before`.
        #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
        assert!(
            after > before,
            "MARKER_PIPELINE_RUNS did not increment ({before} -> {after}); \
             routing fell back silently. This is the failure mode that has \
             caused every prior marker-decoder to be deleted as 'dead code.' \
             Check that `decompress_single_member`'s parallel gate is \
             reachable (ISA-L available, num_threads > 1, data > 10 MiB)."
        );
        #[cfg(not(all(target_arch = "x86_64", feature = "isal-compression")))]
        let _ = (before, after); // suppress unused-vars on non-target platforms
    }

    // =========================================================================
    // Performance regression guard — the CI gap that masked v0.3.0.
    //
    // The single-member parallel path must not be SLOWER than the single-thread
    // ISA-L baseline on the same input. v0.3.0–v0.5.0 had a buggy speculation
    // design that re-decoded the entire stream sequentially in phase 2; the
    // parallel path ran at ~1.75× the elapsed time of pure sequential.
    //
    // The v0.5.1 redesign is correct but does 2N total compute work (phase 1
    // empty-dict decode + phase 2 re-decode with speculative window). On a
    // machine with < 4 physical cores, `decompress_single_member`'s routing
    // gate skips the parallel path entirely — sequential ISA-L wins outright
    // there — so on such hardware this test compares sequential against
    // sequential and trivially passes (ratio ~1.0). The interesting assertion
    // fires on ≥4-core machines where the parallel path is taken; there the
    // expected ratio is well below 1.0 (parallel faster than sequential).
    //
    // Threshold: parallel must complete in ≤ 1.5× sequential elapsed. v0.3.0
    // crossed 1.75× — well above this; the 1.5× ceiling catches that class
    // while leaving CI/noise headroom on ≥4-core developer machines.
    // =========================================================================
    #[test]
    fn test_single_member_parallel_not_slower_than_sequential() {
        // Same fixture shape as the routing-correctness test above. Clears the
        // 10 MiB parallel-path gate.
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        // Helper: best of 3 runs, to dampen noise.
        let bench = |threads: usize| -> std::time::Duration {
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let mut sink = Vec::with_capacity(original.len());
                let t = std::time::Instant::now();
                crate::decompress::decompress_single_member(&compressed, &mut sink, threads)
                    .expect("decompress");
                let elapsed = t.elapsed();
                assert_eq!(
                    sink.len(),
                    original.len(),
                    "T={threads} produced wrong byte count"
                );
                best = best.min(elapsed);
            }
            best
        };

        let seq = bench(1);
        let par = bench(4);

        let ratio = par.as_secs_f64() / seq.as_secs_f64().max(1e-9);
        let seq_mbps = (original.len() as f64) / seq.as_secs_f64() / 1e6;
        let par_mbps = (original.len() as f64) / par.as_secs_f64() / 1e6;
        let physical = num_cpus::get_physical();
        eprintln!(
            "single-member ({physical} physical cores): \
             sequential={seq_mbps:.0} MB/s  parallel(T=4)={par_mbps:.0} MB/s  ratio={ratio:.2}"
        );

        assert!(
            ratio < 1.5,
            "parallel single-member must not be > 1.5× slower than sequential: \
             par={par:?} seq={seq:?} ratio={ratio:.2} physical_cores={physical}"
        );
    }

    /// 60% random / 40% short repetition — compresses to ~60% of original.
    /// Sized to clear the 10 MiB parallel-path gate without being huge.
    fn make_low_entropy_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xfeedface;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    // =========================================================================
    // Layer 3: Output Identity — same output regardless of thread count
    // =========================================================================

    #[test]
    fn test_bgzf_thread_independence() {
        let oracle = FileOracle::new(4 * 1024 * 1024);

        let outputs: Vec<Vec<u8>> = (1..=8)
            .map(|t| {
                crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, t)
                    .unwrap()
            })
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

        let from_bgzf =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
        let from_multi = crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(
            &oracle.multi_member_gz,
            4,
        )
        .unwrap();

        let mut from_single = Vec::new();
        crate::decompress::decompress_single_member_libdeflate(
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

    // =========================================================================
    // Layer 1b: classify_gzip routing table assertions
    //
    // These tests are the canonical check that the routing table in
    // decompression.rs::classify_gzip matches the actual file formats.
    // If routing changes, update CLAUDE.md first, then fix these tests.
    // =========================================================================

    #[test]
    fn test_classify_gzippy_parallel() {
        let oracle = FileOracle::new(512 * 1024);
        // gzippy-parallel output ("GZ" FEXTRA) → GzippyParallel regardless of threads
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.bgzf_gz, 1),
            crate::decompress::DecodePath::GzippyParallel,
            "gzippy-parallel T1 should classify as GzippyParallel"
        );
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.bgzf_gz, 4),
            crate::decompress::DecodePath::GzippyParallel,
            "gzippy-parallel T4 should classify as GzippyParallel"
        );
    }

    #[test]
    fn test_classify_multi_member() {
        let oracle = FileOracle::new(2 * 1024 * 1024);
        // T1 multi-member → sequential
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.multi_member_gz, 1),
            crate::decompress::DecodePath::MultiMemberSeq,
            "multi-member T1 should classify as MultiMemberSeq"
        );
        // T4 multi-member → parallel
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.multi_member_gz, 4),
            crate::decompress::DecodePath::MultiMemberPar,
            "multi-member T4 should classify as MultiMemberPar"
        );
    }

    #[test]
    fn test_classify_single_member() {
        use crate::decompress::{classify_gzip, DecodePath};
        let oracle = FileOracle::new(512 * 1024);
        let path = classify_gzip(&oracle.single_member_gz, 4);
        // Single-member must route to one of the three single-member paths.
        // The exact path depends on whether ISA-L is available on this machine.
        assert!(
            matches!(
                path,
                DecodePath::IsalSingle | DecodePath::StreamingSingle | DecodePath::LibdeflateSingle
            ),
            "single-member should classify as a single-member path, got {:?}",
            path
        );
    }
}
