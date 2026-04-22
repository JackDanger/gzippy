//! Allocation budget tests.
//!
//! Asserts that hot decompression paths don't acquire more heap allocations
//! than a committed baseline. Any PR that adds a `Vec::new()`, `.collect()`,
//! `String::from()`, etc. to a hot path will fail here with a specific count.
//!
//! Baseline: benchmarks/baselines.json "alloc_budget.*"
//! Update with: RECORD_BASELINES=1 cargo test --release alloc_budget -- --nocapture
//!
//! Note: counts include reallocs (Vec growth). The budget has 20% headroom
//! over the measured baseline so minor allocator differences don't flake.

#[cfg(test)]
mod tests {
    use crate::alloc_counter::CountingAllocator;

    // ── baseline reading ─────────────────────────────────────────────────────

    fn read_baseline_u64(key: &str, default: u64) -> u64 {
        for prefix in &["", "../", "../../"] {
            let path = format!("{}benchmarks/baselines.json", prefix);
            if let Ok(json) = std::fs::read_to_string(&path) {
                if let Some(v) = extract_u64(&json, key) {
                    return v;
                }
            }
        }
        default
    }

    fn extract_u64(json: &str, key: &str) -> Option<u64> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = json[start..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit())?;
        rest[..end].parse().ok()
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Single-member 1MB: count allocations on the default decompression path.
    /// This exercises ISA-L (x86_64) or libdeflate (arm64).
    #[test]
    fn alloc_budget_single_member_1mb() {
        let fixture = crate::test_fixtures::text_1mb();

        // Warmup — ensures lazy statics are initialized before we count
        let _ = crate::decompression::decompress_gzip_to_vec_pub(&fixture.single_member_gz, 1);

        CountingAllocator::reset();
        let _ = crate::decompression::decompress_gzip_to_vec_pub(&fixture.single_member_gz, 1)
            .unwrap();
        let count = CountingAllocator::count();
        let bytes = CountingAllocator::bytes();

        let baseline = read_baseline_u64("decompress_1mb_max_allocs", 999999);

        eprintln!(
            "alloc_budget_single_member_1mb: {} allocs, {} bytes (baseline <= {})",
            count, bytes, baseline
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!("baseline: alloc_budget.decompress_1mb_max_allocs = {}", (count as f64 * 1.20) as u64);
            return;
        }

        assert!(
            count <= baseline,
            "alloc budget exceeded: {} allocations > baseline {}\n\
             Check for new Vec/String/collect() on the single-member hot path.",
            count,
            baseline
        );
    }

    /// gzippy-parallel 10MB T4: count allocations on the bgzf parallel path.
    #[test]
    fn alloc_budget_bgzf_10mb() {
        let fixture = crate::test_fixtures::text_10mb();

        // Warmup
        let _ = crate::bgzf::decompress_bgzf_parallel_to_vec(&fixture.bgzf_gz, 4);

        CountingAllocator::reset();
        let _ = crate::bgzf::decompress_bgzf_parallel_to_vec(&fixture.bgzf_gz, 4).unwrap();
        let count = CountingAllocator::count();
        let bytes = CountingAllocator::bytes();

        let baseline = read_baseline_u64("decompress_bgzf_10mb_max_allocs", 999999);

        eprintln!(
            "alloc_budget_bgzf_10mb: {} allocs, {} bytes (baseline <= {})",
            count, bytes, baseline
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            println!("baseline: alloc_budget.decompress_bgzf_10mb_max_allocs = {}", (count as f64 * 1.20) as u64);
            return;
        }

        assert!(
            count <= baseline,
            "alloc budget exceeded: {} allocations > baseline {}\n\
             Check for new Vec/String/collect() on the bgzf parallel hot path.",
            count,
            baseline
        );
    }
}
