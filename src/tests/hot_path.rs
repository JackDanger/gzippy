//! Hot-path hit-rate assertions for the bgzf decoder.
//!
//! Asserts that the multi-symbol fast path fires at an acceptable rate on
//! representative data. A regression where the multi-sym gate flips off
//! (e.g., a code-length table change) will appear here immediately.
//!
//! Baseline: benchmarks/baselines.json "hit_rates.*"
//! Update with: RECORD_BASELINES=1 cargo test --release hot_path -- --nocapture
//!
//! Counter source: crate::decompress::bgzf::hot_counters (active in test builds).

#[cfg(test)]
mod tests {
    use crate::decompress::bgzf::hot_counters;

    // ── baseline reading ─────────────────────────────────────────────────────

    fn read_baseline_f64(key: &str, default: f64) -> f64 {
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

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Multi-sym fast path rate on compressible text (bgzf format, 10MB).
    ///
    /// Text data produces short Huffman codes (many 3–5 bit), which triggers
    /// the multi-sym optimization. If this rate drops significantly, we've
    /// broken or bypassed the fast path for the most common real-world input.
    #[test]
    fn hot_path_multi_sym_rate_bgzf_text_10mb() {
        let fixture = crate::tests::fixtures::text_10mb();

        // Warmup
        let _ = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&fixture.bgzf_gz, 4);

        hot_counters::reset();
        let _ = crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&fixture.bgzf_gz, 4).unwrap();
        let (dynamic, multi_sym, standard) = hot_counters::snapshot();

        if dynamic == 0 {
            // No dynamic blocks (e.g., all stored or fixed) — nothing to assert
            eprintln!("hot_path_multi_sym_rate: no dynamic blocks in fixture, skipping");
            return;
        }

        let rate = multi_sym as f64 / dynamic as f64;
        let min_rate = read_baseline_f64("multi_sym_min", 0.0);

        eprintln!(
            "hot_path_multi_sym_rate_bgzf_text_10mb: {}/{} blocks multi-sym ({:.1}%), {}/{} standard ({:.1}%), threshold >= {:.1}%",
            multi_sym, dynamic, rate * 100.0,
            standard, dynamic, standard as f64 / dynamic as f64 * 100.0,
            min_rate * 100.0
        );

        if std::env::var("RECORD_BASELINES").is_ok() {
            // Record observed rate with 10% downward margin
            println!("baseline: hit_rates.multi_sym_min = {:.3}", (rate * 0.90).max(0.0));
            return;
        }

        assert!(
            rate >= min_rate,
            "multi-sym rate {:.1}% < baseline {:.1}%\n\
             The bgzf fast path is being bypassed for {} of {} dynamic blocks.\n\
             Check decode_dynamic_into() in bgzf.rs — the use_multi_sym gate may have changed.",
            rate * 100.0,
            min_rate * 100.0,
            standard,
            dynamic
        );
    }
}
