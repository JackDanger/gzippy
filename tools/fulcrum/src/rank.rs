//! Fusion + ranking: combine the causal (Coz), critical-path, and
//! mechanistic layers over the four candidate regions into ONE ranked
//! lever list — the FULCRUM deliverable.
//!
//! Coz elasticity is the PRIMARY key (it directly answers ∂wall/∂speed).
//! The critical-path on-path share is a corroborating signal (a region
//! with high elasticity AND high on-path time is a confident lever; high
//! elasticity but ~0 on-path time is suspicious and flagged). The mechanism
//! string tells you HOW to attack the top lever.

use crate::coz::CozProfile;
use crate::critpath::CritPath;
use crate::mech::Mech;
use std::collections::BTreeMap;

/// One fused lever row.
pub struct Lever {
    pub region: String,
    /// Coz wall-elasticity (∂program-speedup / ∂region-speedup), the lever score.
    pub elasticity: f64,
    pub elasticity_lo: f64,
    pub elasticity_hi: f64,
    /// Fraction of the critical path attributed to this region (corroboration).
    pub on_path_fraction: f64,
    pub on_critical_path: bool,
    pub mechanism: String,
    pub coz_samples: f64,
    pub note: String,
}

/// The function-name substrings that back each region (shared with mech).
pub fn region_funcs(region: &str) -> Vec<&'static str> {
    match region {
        "bootstrap" => vec!["bootstrap_with_deflate_block", "deflate_block", "Block"],
        "bulk_inflate" => vec![
            "decode_block",
            "decode_chunk_isal",
            "decode_chunk_pure_bulk",
            "isal",
        ],
        "absorb" => vec!["absorb_isal_tail", "append_markered", "memmove", "memcpy"],
        "scan" => vec![
            "scan",
            "block_finder",
            "BlockFinder",
            "find_next",
            "RawBlockFinder",
        ],
        _ => vec![],
    }
}

/// Map a critpath label back to a region (the critpath uses worker-span
/// names like `blocked-on:worker.bootstrap`, `worker.isal_stream_inflate`).
fn label_region(label: &str) -> Option<&'static str> {
    let l = label;
    if l.contains("bootstrap") {
        Some("bootstrap")
    } else if l.contains("isal_stream_inflate") || l.contains("pure_bulk") || l.contains("bulk") {
        Some("bulk_inflate")
    } else if l.contains("scan") || l.contains("block_finder") {
        Some("scan")
    } else if l.contains("absorb") || l.contains("append") {
        Some("absorb")
    } else {
        None
    }
}

/// On-path fraction per region, summed from the critical-path entries.
fn on_path_by_region(crit: &CritPath) -> BTreeMap<String, f64> {
    let mut m: BTreeMap<String, f64> = BTreeMap::new();
    for e in &crit.entries {
        if let Some(r) = label_region(&e.label) {
            *m.entry(r.to_string()).or_default() += e.fraction;
        }
    }
    m
}

/// Produce the ranked lever list. `coz` may be None (critpath-only mode),
/// in which case ranking falls back to on-path fraction.
pub fn rank(coz: Option<&CozProfile>, crit: &CritPath, mech: Option<&Mech>) -> Vec<Lever> {
    let on_path = on_path_by_region(crit);
    let regions = ["bootstrap", "bulk_inflate", "absorb", "scan"];
    let mut levers = Vec::new();

    for region in regions {
        // The lever score is the PEAK-line elasticity (the single highest-
        // confidence line you'd optimize), NOT the weighted median — the
        // median can be masked to ~0 by a high-sample near-zero line in the
        // same region (observed: bootstrap's deflate_block.rs:1168 @18k
        // samples ≈0 masks :1170 @2.7k samples +0.36). The median is kept as
        // the CI-context band.
        let (elasticity, lo, hi, samples) = coz
            .and_then(|c| c.region_curves.get(region))
            .map(|rc| {
                let (peak, _peak_n) = rc.peak_line_elasticity();
                let (_med, lo, hi) = rc.elasticity_ci();
                (peak, lo, hi, rc.samples)
            })
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN, 0.0));

        let opf = *on_path.get(region).unwrap_or(&0.0);
        let on_cp = opf > 0.02;

        let mechanism = mech
            .map(|m| m.region_mechanism(&region_funcs(region)))
            .unwrap_or_else(|| "(no perf capture)".into());

        // Coherence flags: the most valuable cross-check.
        let mut note = String::new();
        if !elasticity.is_nan() {
            if elasticity.abs() < 0.03 && opf < 0.05 {
                note = "confirmed non-lever (≈0 elasticity AND off critical path)".into();
            } else if elasticity > 0.05 && opf < 0.02 {
                note = "HIGH elasticity but ~0 on-path — suspicious, verify".into();
            } else if elasticity > 0.05 && on_cp {
                note = "CONFIRMED lever (positive elasticity AND on critical path)".into();
            }
        }

        levers.push(Lever {
            region: region.to_string(),
            elasticity,
            elasticity_lo: lo,
            elasticity_hi: hi,
            on_path_fraction: opf,
            on_critical_path: on_cp,
            mechanism,
            coz_samples: samples,
            note,
        });
    }

    // Rank: by Coz elasticity if present (NaN sorts last), else by on-path.
    levers.sort_by(|a, b| {
        let ka = if a.elasticity.is_nan() {
            a.on_path_fraction
        } else {
            a.elasticity
        };
        let kb = if b.elasticity.is_nan() {
            b.on_path_fraction
        } else {
            b.elasticity
        };
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    levers
}

/// Render the ranked lever list as a human report block.
pub fn render(levers: &[Lever]) -> String {
    let mut s = String::new();
    s.push_str("\n================  FULCRUM — RANKED LEVER LIST  ================\n");
    s.push_str("(elasticity = ∂program-speedup / ∂region-speedup, Coz virtual speedup)\n\n");
    s.push_str(&format!(
        "  {:<14} {:>12} {:>16} {:>8} {:>5}  {}\n",
        "region", "elasticity", "95% CI", "on-path", "CP?", "mechanism / note"
    ));
    s.push_str(&format!("  {}\n", "-".repeat(96)));
    for (i, l) in levers.iter().enumerate() {
        let el = if l.elasticity.is_nan() {
            "    n/a".to_string()
        } else {
            format!("{:+.3}", l.elasticity)
        };
        let ci = if l.elasticity.is_nan() {
            "        --".to_string()
        } else {
            format!("[{:+.3},{:+.3}]", l.elasticity_lo, l.elasticity_hi)
        };
        s.push_str(&format!(
            "  {}{:<12} {:>12} {:>16} {:>7.1}% {:>5}  {}\n",
            if i == 0 { "▶ " } else { "  " },
            l.region,
            el,
            ci,
            l.on_path_fraction * 100.0,
            if l.on_critical_path { "yes" } else { "no" },
            l.mechanism,
        ));
        if !l.note.is_empty() {
            s.push_str(&format!("  {:<14} └─ {}\n", "", l.note));
        }
    }
    s.push_str(&format!("  {}\n", "-".repeat(96)));
    if let Some(top) = levers
        .iter()
        .find(|l| !l.elasticity.is_nan() || l.on_path_fraction > 0.0)
    {
        s.push_str(&format!(
            "\n  NEXT LEVER → {} (highest wall-elasticity). {}\n",
            top.region, top.mechanism
        ));
    }
    s
}
