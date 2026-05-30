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
    /// Coz PEAK-line wall-elasticity (∂program-speedup / ∂region-speedup).
    pub elasticity: f64,
    pub elasticity_lo: f64,
    pub elasticity_hi: f64,
    /// Fraction of the critical path attributed to this region.
    pub on_path_fraction: f64,
    pub on_critical_path: bool,
    /// THE lever score = elasticity × on_path_fraction — the EXPECTED wall
    /// move if you fully speed this region. Two regions can have similar
    /// elasticity but vastly different on-path share (bulk +0.47 @5% vs
    /// bootstrap +0.36 @91%); this product is what actually ranks them, and
    /// is exactly why FULCRUM fuses the causal AND critical-path layers
    /// rather than trusting elasticity alone (the CPU-sum-lie trap in
    /// reverse: a high elasticity off the critical path is a small lever).
    pub lever_score: f64,
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

        // THE lever score: expected wall move = elasticity × on-path share.
        // A NaN elasticity (no coz signal) falls back to on-path alone (a
        // region that gates the wall is a lever even if coz couldn't sample
        // it), so a heavily on-path region never ranks below an off-path one
        // just for lack of coz experiments.
        let lever_score = if elasticity.is_nan() {
            opf
        } else {
            (elasticity.max(0.0)) * opf
        };

        let mechanism = mech
            .map(|m| m.region_mechanism(&region_funcs(region)))
            .unwrap_or_else(|| "(no perf capture)".into());

        // Coherence flags: the most valuable cross-check.
        let mut note = String::new();
        if !elasticity.is_nan() {
            if elasticity.abs() < 0.03 && opf < 0.05 {
                note = "confirmed non-lever (≈0 elasticity AND off critical path)".into();
            } else if elasticity > 0.05 && opf < 0.02 {
                note = "high elasticity but ~0 on-path — small lever (off critical path)".into();
            } else if elasticity > 0.05 && on_cp {
                note = format!(
                    "CONFIRMED lever — expected wall-move {:.0}% (elasticity {:+.2} × {:.0}% on-path)",
                    lever_score * 100.0,
                    elasticity,
                    opf * 100.0
                );
            }
        } else if opf > 0.10 {
            note =
                "gates the wall (on critical path) but coz built no experiment — re-probe".into();
        }

        levers.push(Lever {
            region: region.to_string(),
            elasticity,
            elasticity_lo: lo,
            elasticity_hi: hi,
            on_path_fraction: opf,
            on_critical_path: on_cp,
            lever_score,
            mechanism,
            coz_samples: samples,
            note,
        });
    }

    // Rank by the FUSED lever score (elasticity × on-path), descending. This
    // is the fix that makes bootstrap (+0.36 @91% on-path) out-rank
    // bulk_inflate (+0.47 @5% on-path) despite the lower elasticity — the
    // critical-path layer disambiguates similar elasticities.
    levers.sort_by(|a, b| {
        b.lever_score
            .partial_cmp(&a.lever_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    levers
}

/// Render the ranked lever list as a human report block.
pub fn render(levers: &[Lever]) -> String {
    let mut s = String::new();
    s.push_str("\n================  FULCRUM — RANKED LEVER LIST  ================\n");
    s.push_str(
        "lever-score = wall-elasticity × on-critical-path share = EXPECTED wall move.\n\
         (elasticity = ∂program-speedup/∂region-speedup, Coz; on-path from the\n\
          consumer-anchored critical path. Ranking by the PRODUCT, not elasticity\n\
          alone — a high elasticity off the critical path is a small lever.)\n\n",
    );
    s.push_str(&format!(
        "  {:<13} {:>11} {:>10} {:>8} {:>5}  {}\n",
        "region", "lever-score", "elasticity", "on-path", "CP?", "mechanism"
    ));
    s.push_str(&format!("  {}\n", "-".repeat(98)));
    for (i, l) in levers.iter().enumerate() {
        let el = if l.elasticity.is_nan() {
            "   n/a".to_string()
        } else {
            format!("{:+.3}", l.elasticity)
        };
        let score = format!("{:.3}", l.lever_score);
        // Trim the mechanism's verbose funcs[...] list for the table; the
        // full mechanism stays available in the Lever struct.
        let mech_short = l
            .mechanism
            .split(" | funcs[")
            .next()
            .unwrap_or(&l.mechanism);
        s.push_str(&format!(
            "  {}{:<11} {:>11} {:>10} {:>7.1}% {:>5}  {}\n",
            if i == 0 { "▶ " } else { "  " },
            l.region,
            score,
            el,
            l.on_path_fraction * 100.0,
            if l.on_critical_path { "yes" } else { "no" },
            mech_short,
        ));
        if !l.note.is_empty() {
            s.push_str(&format!("  {:<13} └─ {}\n", "", l.note));
        }
    }
    s.push_str(&format!("  {}\n", "-".repeat(98)));
    if let Some(top) = levers.first() {
        s.push_str(&format!(
            "\n  NEXT LEVER → {}  (lever-score {:.3} = elasticity {} × {:.0}% on-path)\n   {}\n",
            top.region,
            top.lever_score,
            if top.elasticity.is_nan() {
                "n/a".to_string()
            } else {
                format!("{:+.2}", top.elasticity)
            },
            top.on_path_fraction * 100.0,
            top.mechanism
                .split(" | funcs[")
                .next()
                .unwrap_or(&top.mechanism),
        ));
    }
    s
}
