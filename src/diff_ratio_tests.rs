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

    fn median(v: &mut Vec<u64>) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Single-member 1MB: gzippy vs libdeflate.
    /// Exercises the single-member decode path (ISA-L or libdeflate sequential).
    #[test]
    fn diff_ratio_single_member_1mb() {
        let fixture = crate::test_fixtures::text_1mb();
        let data = &fixture.single_member_gz;
        let out_size = fixture.plain.len() + 1024;
        let mut ld_buf = vec![0u8; out_size];

        let (gzippy_ns, libdeflate_ns) = measure_alternating(
            20,
            || {
                let _ = crate::decompression::decompress_gzip_to_vec_pub(data, 1).unwrap();
            },
            || {
                let mut d = libdeflater::Decompressor::new();
                let _ = d.gzip_decompress(data, &mut ld_buf);
            },
        );

        let ratio = gzippy_ns as f64 / libdeflate_ns as f64;
        let threshold = read_threshold("max_ratio", 1.15);

        eprintln!(
            "diff_ratio_single_member_1mb: gzippy={:.2}ms libdeflate={:.2}ms ratio={:.3} threshold={:.3}",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!("baseline: diff_ratio.max_ratio = {:.3}", ratio * 1.10);
            return;
        }

        assert!(
            ratio <= threshold,
            "gzippy {:.2}ms vs libdeflate {:.2}ms — ratio {:.3} > threshold {:.3}\n\
             Run 'make update-baselines' if this is an intentional improvement.",
            gzippy_ns as f64 / 1e6,
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
        let fixture = crate::test_fixtures::text_10mb();
        let bgzf_data = &fixture.bgzf_gz;
        let single_data = &fixture.single_member_gz;
        let out_size = fixture.plain.len() + 1024;
        let mut ld_buf = vec![0u8; out_size];

        let (gzippy_ns, libdeflate_ns) = measure_alternating(
            10,
            || {
                let _ = crate::bgzf::decompress_bgzf_parallel_to_vec(bgzf_data, 4).unwrap();
            },
            || {
                let mut d = libdeflater::Decompressor::new();
                let _ = d.gzip_decompress(single_data, &mut ld_buf);
            },
        );

        let ratio = gzippy_ns as f64 / libdeflate_ns as f64;
        let threshold = read_threshold("max_ratio_bgzf_10mb", 2.0);

        eprintln!(
            "diff_ratio_bgzf_10mb: gzippy_T4={:.2}ms libdeflate_T1={:.2}ms ratio={:.3} threshold={:.3}",
            gzippy_ns as f64 / 1e6,
            libdeflate_ns as f64 / 1e6,
            ratio,
            threshold
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!("baseline: diff_ratio.max_ratio_bgzf_10mb = {:.3}", ratio * 1.15);
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
