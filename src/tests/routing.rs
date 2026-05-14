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
    // The v0.6 marker pipeline has no per-physical-core routing gate (the
    // earlier core floor was dropped — see `docs/marker-decoder-plan.md`).
    // On ≥4-physical-core hardware parallel comfortably beats sequential
    // (tight assertion: ratio < 1.5 catches v0.3.0-class 1.75× regression).
    // On <4 physical cores (e.g. 2-core CI runners) parallel-at-T=4 pays
    // Amdahl tax that sequential T=1 doesn't, so ratios in the 1.5–2.0×
    // range are structural — but we still assert ratio < 3.0 there so a
    // *catastrophic* regression (5×+) on small-core hardware doesn't slip
    // through (Opus advisor feedback on PR #97: removing the assertion
    // entirely lost regression protection on the most common CI class).
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

        // Two-tier threshold (Opus advisor feedback on PR #97): on
        // ≥4 physical cores the bar is tight (1.5×) — anything looser
        // wouldn't catch the v0.3.0-class 1.75× algorithmic regression.
        // On <4 physical cores keep a *relaxed* bar (3.0×) instead of
        // disabling the assertion entirely, so a catastrophic regression
        // on 2-core hardware (e.g. parallel path becomes 5× slower than
        // sequential) is still caught. The structural 1.5–2.0× tax on
        // 2-core CI is below 3.0×.
        let threshold = if physical >= 4 { 1.5 } else { 3.0 };
        assert!(
            ratio < threshold,
            "parallel single-member must not be > {threshold:.1}× slower than sequential \
             on {physical}-physical-core hardware: \
             par={par:?} seq={seq:?} ratio={ratio:.2}"
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

    /// Mixed-entropy data designed to push zlib L1 into emitting many short
    /// blocks (a mix of BTYPE=00/01/10). Specifically: 70% random bytes
    /// (uncompressible — short stored or stored-mixed blocks),
    /// 30% short repeats from a small phrase set (compresses well — short
    /// dynamic / fixed-Huffman blocks). At L1 with this entropy mix, zlib
    /// emits many small blocks rather than one giant one; the resulting
    /// compressed size is large enough (~12 MiB at 24 MiB original) to
    /// clear the 10 MiB parallel gate.
    ///
    /// This is the failure class the cross-chunk consistency correction
    /// sweep addresses: phase 1a's speculative pick lands near a block
    /// boundary, but not always *at* one — without correction the
    /// pipeline silently fell back to libdeflate.
    fn make_btype01_heavy_data(size: usize) -> Vec<u8> {
        let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xb0bd1ec0de;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < 70 {
                // Random byte.
                data.push((rng >> 16) as u8);
            } else {
                // Short phrase repetition.
                let phrase = phrases[(rng as usize) % phrases.len()];
                let to_take = phrase.len().min(size - data.len());
                data.extend_from_slice(&phrase[..to_take]);
            }
        }
        data.truncate(size);
        data
    }

    /// gzip-encode at level 1 (fastest / most fixed-Huffman emissions).
    fn compress_single_member_gzip_l1(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Companion to `test_marker_pipeline_actually_runs_on_x86_64_isal` —
    /// same shape (decompress_single_member at T=4, counter snapshot), but
    /// against a fixture engineered to maximize fixed-Huffman (BTYPE=01)
    /// block density. BlockFinder excludes BTYPE=01 candidates by design,
    /// so phase 1a's speculative starts on this fixture often land on
    /// non-boundary positions. The cross-chunk consistency correction in
    /// `single_member::phase1c_resolve_consistency` (Opus advisor approach
    /// #2, refined) handles this via the induction "chunk N's decoded
    /// end_bit is always a real block boundary; correct chunk N+1's
    /// start to chunks[N].end_bit and re-decode chunk N+1." No candidate
    /// lists, no strictness ramp, no top-K — just forward propagation.
    ///
    /// Second deletion-trap killer (BTYPE=01-heavy companion to the
    /// BTYPE=00/10 routing test). On `cfg(target_arch = "x86_64",
    /// feature = "isal-compression")` it asserts the pipeline ran
    /// end-to-end. On arm64 the parallel path is gated off and the
    /// counter check is a no-op.
    #[test]
    fn test_marker_pipeline_runs_on_btype01_heavy_input() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_btype01_heavy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip_l1(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let before_runs = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let before_retries =
            crate::decompress::parallel::single_member::MARKER_PIPELINE_RETRY_ITERATIONS
                .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original, "byte-perfect output");
        let after_runs = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let after_retries =
            crate::decompress::parallel::single_member::MARKER_PIPELINE_RETRY_ITERATIONS
                .load(Ordering::Relaxed);

        #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
        {
            assert!(
                after_runs > before_runs,
                "MARKER_PIPELINE_RUNS did not increment ({before_runs} -> {after_runs}) on \
                 BTYPE=01-heavy fixture — BlockFinder enhancement regressed or \
                 never landed. This is the second deletion-trap killer; the \
                 first one (low-entropy fixture) passes because BlockFinder \
                 finds BTYPE=10 candidates there. The point of THIS test is \
                 that real-world data sometimes has only BTYPE=01 boundaries \
                 in a chunk's search range."
            );
            // Sibling killer for G5 (Opus advisor PR #97 review): without
            // this assertion, the test passes even when phase 1a happens
            // to land on a real boundary by luck — i.e. when G5 is a
            // no-op. The whole point of the BTYPE=01-heavy fixture is to
            // *force* misaligned speculative starts so G5's chain-decode
            // correction sweep has to fire. If a future change makes
            // phase 1a always succeed on this fixture, the cross-chunk
            // correction code path goes untested. This counter says "yes,
            // at least one chunk needed correction" and locks in coverage.
            assert!(
                after_retries > before_retries,
                "MARKER_PIPELINE_RETRY_ITERATIONS did not increment \
                 ({before_retries} -> {after_retries}) on BTYPE=01-heavy fixture — \
                 G5 cross-chunk correction code path is not being exercised \
                 by the very fixture designed to force it. Either (a) phase 1a \
                 got lucky on every chunk (regenerate fixture with more BTYPE=01 \
                 density), or (b) phase 1c was bypassed by a routing change."
            );
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "isal-compression")))]
        let _ = (before_runs, after_runs, before_retries, after_retries);
    }

    // =========================================================================
    // Optimization counter assertions — lock in rapidgzip-style optimizations.
    //
    // The marker pipeline can SILENTLY degrade in ways that still produce
    // byte-perfect output but trash the perf budget. Output-equivalence
    // tests don't catch them; only counter-snapshot tests do.
    //
    // The structural optimizations we mirror from rapidgzip
    // (`vendor/rapidgzip/.../GzipChunk.hpp:413-657`) are:
    //
    //   1. Per-chunk cleanData → ISA-L handoff. Each worker decodes a
    //      ~32 KB bootstrap with the marker decoder, then hands off to
    //      ISA-L with `isal_inflate_set_dict`. Without the handoff, a
    //      worker decodes the entire chunk at pure-Rust speed
    //      (~50 MB/s/thread) instead of ISA-L speed (~163 MB/s/thread).
    //      Test: `test_isal_handoff_fires_on_every_chunk_healthy_data`.
    //
    //   2. Bootstrap is bounded (~32 KB per chunk, NOT the full chunk).
    //      `BOOTSTRAP_OUTPUT_BYTES / num_workers` should be 32-128 KB
    //      typically. A chunk where the bootstrap accumulates MB of
    //      output means markers are propagating too aggressively through
    //      chunk-local copies, or the exit condition is broken.
    //      Test: `test_bootstrap_bounded_to_clean_window_size`.
    //
    //   3. ISA-L produces the bulk of output bytes. On healthy data,
    //      `ISAL_OUTPUT_BYTES / total_output_bytes` should be ≥ 0.99.
    //      If it's lower, workers are falling through to
    //      `marker_finish_after_bootstrap` (slow path).
    //      Test: `test_isal_produces_at_least_99_percent_of_output`.
    //
    //   4. Phase 1a boundary search is sub-quadratic. `BOUNDARY_VALIDATIONS`
    //      grows linearly with `num_chunks`, not quadratically with
    //      `SEARCH_RADIUS * num_chunks`. The tier-2 byte-aligned brute
    //      force was removed because it issued 16384 validation calls
    //      per failed chunk; we now cap at BlockFinder's candidate count.
    //      Test: `test_phase1a_validation_count_stays_bounded`.
    //
    // All four tests share the `MARKER_PIPELINE_TEST_LOCK` mutex so
    // their counter snapshots don't race with each other or with the
    // existing routing-killer tests. They are gated to
    // `cfg(target_arch = "x86_64", feature = "isal-compression")`
    // because the optimizations themselves only fire on that target;
    // arm64 routes to the libdeflate single-thread path which doesn't
    // exercise this code.
    // =========================================================================

    #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
    #[test]
    fn test_isal_handoff_fires_on_every_chunk_healthy_data() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(compressed.len() > 10 * 1024 * 1024);

        let before = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        let after = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();
        assert_eq!(output, original, "byte-perfect output");

        let delta = after.delta(&before);

        // On 24 MiB healthy low-entropy data at T=4, BlockFinder finds
        // boundaries for all 4 chunks. Each chunk's bootstrap should
        // exit on the first block boundary past 32 KB clean tail, then
        // hand off to ISA-L. The LAST chunk might decode less than
        // 32 KB of bootstrap output and fall through (no handoff), so
        // we assert ≥ 3 (the first 3 chunks definitely hand off).
        assert!(
            delta.handoff_fired >= 3,
            "expected ≥3 of 4 chunks to take the bootstrap→ISA-L handoff path, \
             got {} — bootstrap may be running for the whole chunk instead of \
             exiting at 32 KB clean tail. counters delta: {delta:?}",
            delta.handoff_fired
        );
        assert_eq!(
            delta.slow_path_used, 0,
            "no worker should hit marker_finish_after_bootstrap on healthy data; \
             {} did — ISA-L was unexpectedly rejecting the speculative input \
             slice OR the bootstrap clean_window was malformed. delta: {delta:?}",
            delta.slow_path_used
        );
        // Opus advisor flagged that the existing tests didn't assert
        // on these counters even though "healthy data" implies zero
        // retries and zero silent boundary-misses. Without these, a
        // regression where phase 1c re-decodes every chunk (because
        // ISA-L silently disagrees with BlockFinder on healthy data)
        // would pass the handoff_fired check but ruin perf.
        assert_eq!(
            delta.retry_iterations, 0,
            "phase 1c retry fired {} times on healthy data — phase 1a or \
             ISA-L is producing inconsistent end_bits. delta: {delta:?}",
            delta.retry_iterations
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
    #[test]
    fn test_bootstrap_bounded_to_clean_window_size() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let before = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        let after = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();

        let delta = after.delta(&before);

        // Per-chunk bootstrap output budget: in the absence of marker
        // propagation pathology, bootstrap exits within 1-3 deflate
        // blocks of accumulating 32 KB clean tail. At flate2 default
        // L6 on low-entropy data, blocks are ~32-64 KB. The bound:
        // 32 KB (one block of buildup) + 256 KB (one outlier block
        // post-threshold) = 288 KB per chunk × 4 = 1.15 MB total.
        //
        // We use 2 MB as the alarm threshold — slack for compressor
        // variability but tight enough that "bootstrap decodes the
        // whole chunk" (which would be ~6 MB output / chunk × 4 = 24 MB)
        // immediately fails this test.
        //
        // If this test ever red-lines, the most likely cause is
        // `emit_match`'s chunk-local copy spreading markers forward
        // through the trailing 32 KB indefinitely. The fix is in
        // `decode_chunk_bootstrap`'s exit condition — see
        // `fast_marker_inflate.rs:651-660`.
        let bootstrap_kb = delta.bootstrap_output_bytes / 1024;
        let total_output_kb = (output.len() / 1024) as u64;
        assert!(
            delta.bootstrap_output_bytes < 2 * 1024 * 1024,
            "bootstrap output {bootstrap_kb} KiB exceeds 2 MiB across 4 chunks — \
             markers are propagating through chunk-local copies and never \
             clearing the trailing 32 KB. delta: {delta:?}, output: {total_output_kb} KiB"
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
    #[test]
    fn test_isal_produces_bulk_of_output() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let before = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        let after = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();

        let delta = after.delta(&before);
        let total_output = output.len() as u64;

        // ISA-L should produce the bulk of output bytes; the
        // bootstrap markers + slow-path fallback should be a small
        // share. This is the structural reason gzippy can match
        // rapidgzip's per-thread throughput — most of the work
        // happens in ISA-L's hand-tuned asm inner loop, not the
        // pure-Rust marker decoder.
        //
        // Threshold is 85% (was initially 99% — too tight). For a
        // 24 MiB fixture at T=4, each chunk is 6 MiB. The bootstrap
        // exits at the first block boundary after 32 KB of clean
        // tail; with L6 deflate's ~64-256 KB block sizes on mixed
        // entropy data, the bootstrap can decode 200-500 KB before
        // exiting (5-8% of a 6 MiB chunk). On Silesia-class inputs
        // (120 MB chunks at T=4) the same absolute bootstrap is
        // 0.3-0.5% so the ISA-L share lands at 99.5%+. We don't
        // run a Silesia-sized fixture in unit tests; the 85% bar
        // here catches the catastrophic regression ("bootstrap
        // runs for the whole chunk → ISA-L share ~ 0%") on the
        // 24 MiB fixture without false-positiving on the legitimate
        // small-chunk overhead.
        //
        // CI measured 92.71% on this fixture pre-relax (delta:
        // bootstrap=1.8 MB, isal=23.3 MB, total=24 MB at T=4).
        let isal_share = (delta.isal_output_bytes as f64) / (total_output as f64);
        assert!(
            isal_share >= 0.85,
            "ISA-L produced only {:.2}% of output ({} of {} bytes) — \
             bootstrap is running for too much of each chunk, OR the \
             slow path is firing. delta: {delta:?}",
            isal_share * 100.0,
            delta.isal_output_bytes,
            total_output
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "isal-compression"))]
    #[test]
    fn test_phase1a_validation_count_stays_bounded() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let before = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        let after = crate::decompress::parallel::single_member::OptimizationCounters::snapshot();

        let delta = after.delta(&before);

        // BlockFinder emits a bounded number of candidates per chunk
        // (a few per 32 KB block × a 512 KiB search radius = at most
        // a few hundred). With early-exit on first valid candidate,
        // the actual call count per chunk is typically 1-3 — the
        // first BlockFinder candidate after the chunk anchor usually
        // validates.
        //
        // 1000 total validations across 4 chunks is the upper bound
        // for healthy data. Higher means BlockFinder is emitting too
        // many candidates OR our validators are rejecting real
        // boundaries (e.g., the lowered MIN_CAP broke validation).
        //
        // The pre-`5a68ad9` design with tier-2 brute force could
        // issue 16384+ calls on a failed chunk; this test would have
        // caught that as a regression on the failed-chunk path.
        assert!(
            delta.boundary_validations < 1000,
            "phase 1a issued {} try_decode_at calls across 4 chunks — \
             expected < 1000 on healthy data. BlockFinder may be \
             emitting too many candidates, OR validation is rejecting \
             real boundaries forcing more probes. delta: {delta:?}",
            delta.boundary_validations
        );
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
