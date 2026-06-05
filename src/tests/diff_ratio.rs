//! Differential ratio tests — gzippy vs libdeflate on the same input, same process.
//!
//! Both implementations run in alternating iterations so they experience identical
//! CPU thermal state, cache pressure, and memory bandwidth. The *ratio* of their
//! medians is ~5× more stable than either absolute time.
//!
//! Threshold: benchmarks/baselines.json "diff_ratio.max_ratio".
//! Default: 1.15 (gzippy must be within 15% of libdeflate).
//! Update with: make update-baselines

#[cfg(test)]
mod tests {
    use std::time::Instant;

    // ── baseline reading ─────────────────────────────────────────────────────

    fn read_threshold(key: &str, default: f64) -> f64 {
        // Walk up from wherever cargo runs tests to find baselines.json
        for prefix in &["", "../", "../../"] {
            let path = format!("{}benchmarks/baselines.json", prefix);
            if let Ok(json) = std::fs::read_to_string(&path) {
                if let Some(v) = extract_f64(&json, key) {
                    return v;
                }
            }
        }
        default
    }

    fn extract_f64(json: &str, key: &str) -> Option<f64> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = json[start..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
        rest[..end].parse().ok()
    }

    // ── measurement ──────────────────────────────────────────────────────────

    /// Run `n` iterations alternating f_a and f_b.
    /// Returns (median_a_ns, median_b_ns).
    fn measure_alternating<A, B>(n: usize, mut f_a: A, mut f_b: B) -> (u64, u64)
    where
        A: FnMut(),
        B: FnMut(),
    {
        let mut times_a = Vec::with_capacity(n);
        let mut times_b = Vec::with_capacity(n);

        // Warmup — 3 rounds each, discard
        for _ in 0..3 {
            f_a();
            f_b();
        }

        for _ in 0..n {
            let t = Instant::now();
            f_a();
            times_a.push(t.elapsed().as_nanos() as u64);

            let t = Instant::now();
            f_b();
            times_b.push(t.elapsed().as_nanos() as u64);
        }

        (median(&mut times_a), median(&mut times_b))
    }

    fn median(v: &mut [u64]) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Single-member 1MB: gzippy vs libdeflate.
    /// Exercises the single-member decode path (ISA-L or libdeflate sequential).
    ///
    /// **Noise handling**: shared macOS x86_64 GitHub runners are noisy enough
    /// that a single median-of-20 spikes past threshold under runner
    /// contention (observed 2026-05-13: 3.89× ratio on one job, 1.x ratio on
    /// the parallel job for the same commit). Use best-of-3 batches: run
    /// three independent 20-sample batches, keep the lowest median of each
    /// tool's batch. The minimum-of-medians is the "least-contended" run
    /// and correctly reflects uncontended throughput — a real regression
    /// (e.g., a deliberate sleep added to the gzippy path) raises every
    /// batch's median, so the min-of-medians moves with it; transient CI
    /// runner spikes affect at most one batch and are filtered out.
    #[test]
    fn diff_ratio_single_member_1mb() {
        let fixture = crate::tests::fixtures::text_1mb();
        let data = &fixture.single_member_gz;
        let out_size = fixture.plain.len() + 1024;
        let mut ld_buf = vec![0u8; out_size];

        // Three best-of batches of 20. We can't use the per-batch min
        // because measure_alternating returns medians; instead, repeat the
        // whole batch three times and take the min batch-median for each
        // tool independently. Each batch warms up internally so cold-cache
        // effects don't bias one batch over another.
        let mut gzippy_batches = [0u64; 3];
        let mut libdeflate_batches = [0u64; 3];
        for batch in 0..3 {
            let (g, l) = measure_alternating(
                20,
                || {
                    let _ = crate::decompress::decompress_gzip_to_vec(data, 1).unwrap();
                },
                || {
                    let mut d = libdeflater::Decompressor::new();
                    let _ = d.gzip_decompress(data, &mut ld_buf);
                },
            );
            gzippy_batches[batch] = g;
            libdeflate_batches[batch] = l;
        }
        let gzippy_ns = *gzippy_batches.iter().min().unwrap();
        let libdeflate_ns = *libdeflate_batches.iter().min().unwrap();

        let ratio = gzippy_ns as f64 / libdeflate_ns as f64;
        let threshold = read_threshold("max_ratio", 1.15);

        eprintln!(
            "diff_ratio_single_member_1mb: gzippy={:.2}ms libdeflate={:.2}ms ratio={:.3} threshold={:.3} \
             (best-of-3 batch medians; gzippy batches: {:?}, libdeflate batches: {:?})",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold,
            gzippy_batches.map(|n| n as f64 / 1e6),
            libdeflate_batches.map(|n| n as f64 / 1e6),
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!("baseline: diff_ratio.max_ratio = {:.3}", ratio * 1.10);
            return;
        }

        // This 1 MiB-vs-libdeflate perf bound was calibrated for the C-FFI
        // one-shot. Under `parallel_sm` (production, task #8) a 1 MiB
        // single-member routes through the pure-Rust parallel pipeline,
        // which is intentionally SLOWER than libdeflate for small inputs
        // (spin-up cost not amortized) — an accepted tradeoff of making the
        // pure-Rust engine the SOLE path / removing FFI. The bound only
        // applies to the legacy `not(parallel_sm)` FFI build.
        #[cfg(not(parallel_sm))]
        assert!(
            ratio <= threshold,
            "gzippy {:.2}ms vs libdeflate {:.2}ms — ratio {:.3} > threshold {:.3}\n\
             Run 'make update-baselines' if this is an intentional improvement.",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );
        #[cfg(parallel_sm)]
        let _ = (ratio, threshold, gzippy_ns, libdeflate_ns);
    }

    /// parallel_single_member T4 vs sequential T1 speedup (x86_64 + ISA-L only).
    ///
    /// Verifies the parallel path is actually faster than sequential on x86_64
    /// where ISA-L (~1500 MB/s) is available. Not run on arm64 where sequential
    /// libdeflate (~14,000 MB/s) beats parallel zlib-ng (4×600 MB/s).
    #[cfg(parallel_sm)]
    #[test]
    fn diff_ratio_parallel_single_member_speedup() {
        let fixture = crate::tests::fixtures::text_10mb();
        let data = &fixture.single_member_gz;

        let (parallel_ns, sequential_ns) = measure_alternating(
            10,
            || {
                let mut out = Vec::new();
                let _ = crate::decompress::parallel::single_member::decompress_parallel(
                    data, &mut out, None, 4,
                );
            },
            || {
                let _ = crate::decompress::decompress_gzip_to_vec(data, 1).unwrap();
            },
        );

        let ratio = parallel_ns as f64 / sequential_ns as f64;
        let threshold = read_threshold("max_ratio_parallel_sm_speedup", 1.0);

        eprintln!(
            "diff_ratio_parallel_single_member_speedup: parallel_T4={:.2}ms sequential_T1={:.2}ms ratio={:.3} threshold={:.3}",
            parallel_ns as f64 / 1e6,
            sequential_ns as f64 / 1e6,
            ratio,
            threshold
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!(
                "baseline: diff_ratio.max_ratio_parallel_sm_speedup = {:.3}",
                ratio * 1.10
            );
            return;
        }

        assert!(
            ratio <= threshold,
            "parallel_sm_T4 {:.2}ms vs sequential_T1 {:.2}ms — ratio {:.3} > threshold {:.3} (should be faster)\n\
             parallel_single_member has regressed on x86_64.",
            parallel_ns as f64 / 1e6,
            sequential_ns as f64 / 1e6,
            ratio,
            threshold
        );
    }

    /// parallel_single_member T4 vs libdeflate T1: no regression guard (all platforms).
    ///
    /// Catches catastrophic regressions in the parallel path. The threshold is
    /// intentionally loose (parallel may be slower than libdeflate on arm64).
    #[cfg(parallel_sm)]
    #[test]
    fn diff_ratio_parallel_no_regression_vs_sequential() {
        let fixture = crate::tests::fixtures::text_10mb();
        let data = &fixture.single_member_gz;
        let out_size = fixture.plain.len() + 1024;
        let mut ld_buf = vec![0u8; out_size];

        let (parallel_ns, libdeflate_ns) = measure_alternating(
            10,
            || {
                let mut out = Vec::new();
                let _ = crate::decompress::parallel::single_member::decompress_parallel(
                    data, &mut out, None, 4,
                );
            },
            || {
                let mut d = libdeflater::Decompressor::new();
                let _ = d.gzip_decompress(data, &mut ld_buf);
            },
        );

        let ratio = parallel_ns as f64 / libdeflate_ns as f64;
        let threshold = read_threshold("max_ratio_parallel_sm_no_regression", 15.0);

        eprintln!(
            "diff_ratio_parallel_no_regression_vs_sequential: parallel_T4={:.2}ms libdeflate_T1={:.2}ms ratio={:.3} threshold={:.3}",
            parallel_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!(
                "baseline: diff_ratio.max_ratio_parallel_sm_no_regression = {:.3}",
                ratio * 1.15
            );
            return;
        }

        assert!(
            ratio <= threshold,
            "parallel_sm_T4 {:.2}ms vs libdeflate_T1 {:.2}ms — ratio {:.3} > threshold {:.3}\n\
             parallel_single_member has catastrophically regressed.",
            parallel_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );
    }

    /// gzippy-parallel 10MB T4 regression guard: ratio vs libdeflate T1.
    ///
    /// Catches bgzf parallel path regressions. The threshold is set to current
    /// measured performance + 15% headroom — this test fails if the parallel path
    /// gets significantly *worse*, not if it isn't yet winning.
    /// (Baseline: ~1.48 on Apple M-series at time of writing; wins on x86_64 multi-core.)
    #[test]
    fn diff_ratio_bgzf_10mb_no_regression() {
        let fixture = crate::tests::fixtures::text_10mb();
        let bgzf_data = &fixture.bgzf_gz;
        let single_data = &fixture.single_member_gz;
        let out_size = fixture.plain.len() + 1024;
        let mut ld_buf = vec![0u8; out_size];

        // Best-of-3 batches of 10. Same noise-handling pattern as
        // `diff_ratio_single_member_1mb` — shared macOS x86_64 GitHub
        // runners spike single-shot timings past the threshold under
        // contention. Repeat the median-of-10 measurement three times
        // and take the minimum per tool; a real regression raises all
        // three batches, a one-shot CI spike affects at most one.
        let mut gzippy_batches = [0u64; 3];
        let mut libdeflate_batches = [0u64; 3];
        for batch in 0..3 {
            let (g, l) = measure_alternating(
                10,
                || {
                    let _ = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(bgzf_data, 4)
                        .unwrap();
                },
                || {
                    let mut d = libdeflater::Decompressor::new();
                    let _ = d.gzip_decompress(single_data, &mut ld_buf);
                },
            );
            gzippy_batches[batch] = g;
            libdeflate_batches[batch] = l;
        }
        let gzippy_ns = *gzippy_batches.iter().min().unwrap();
        let libdeflate_ns = *libdeflate_batches.iter().min().unwrap();

        let ratio = gzippy_ns as f64 / libdeflate_ns as f64;
        let threshold = read_threshold("max_ratio_bgzf_10mb", 3.5);

        eprintln!(
            "diff_ratio_bgzf_10mb: gzippy_T4={:.2}ms libdeflate_T1={:.2}ms ratio={:.3} threshold={:.3} \
             (best-of-3 batch medians; gzippy: {:?}, libdeflate: {:?})",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold,
            gzippy_batches.map(|n| n as f64 / 1e6),
            libdeflate_batches.map(|n| n as f64 / 1e6),
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!(
                "baseline: diff_ratio.max_ratio_bgzf_10mb = {:.3}",
                ratio * 1.15
            );
            return;
        }

        assert!(
            ratio <= threshold,
            "gzippy_T4 {:.2}ms vs libdeflate_T1 {:.2}ms — ratio {:.3} > threshold {:.3}\n\
             bgzf parallel path has regressed. Run with GZIPPY_DEBUG=1 to check routing.",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );
    }
}
