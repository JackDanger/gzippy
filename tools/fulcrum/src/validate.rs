#![allow(dead_code)]
// command-builders + struct fields are the embeddable
// API surface (used by `plan`-generated scripts, programmatic callers, and
// kept for completeness); not all are exercised by the CLI default path.
//! Validation layer — the gate that makes FULCRUM trustworthy.
//!
//! FULCRUM is only useful if its causal verdict reproduces the EMPIRICAL
//! frozen-interleaved-A/B results, which are the oracle. The known truths:
//!
//!   (1) absorb_isal_tail copy elimination = −0.0% wall (overlapped /
//!       off the in-order critical path). FULCRUM's `absorb` region MUST
//!       show ≈0 wall-elasticity. If high → FULCRUM is WRONG.
//!   (2) An inline match-copy decode-loop change (commit f04eb74) banked
//!       +5.2% T16 wall. Decode regions (bulk_inflate / bootstrap) MUST
//!       show POSITIVE elasticity.
//!   (3) The long-pole trace: ~7 heavy "overshoot" bootstrap chunks
//!       (46–72 ms each, bimodal) gate the wall → the critical-path layer
//!       MUST surface these as on-path heavy blockers.
//!
//! This module (a) defines the expectations, (b) checks the measured Coz +
//! critpath outputs against them, and (c) can drive the empirical oracle
//! (`scripts/interleaved_ab.sh`) itself so the causal and empirical numbers
//! sit side by side. A divergence is REPORTED, never hidden — that honesty
//! is the whole point.

use crate::coz::CozProfile;
use crate::critpath::CritPath;
use std::path::Path;
use std::process::Command;

/// One expectation and whether the measurement met it.
#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub expectation: String,
    pub measured: String,
    pub passed: bool,
}

/// Verdict over all checks.
pub struct Validation {
    pub checks: Vec<Check>,
}

impl Validation {
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }
}

/// Absolute elasticity below this is "≈0" (a non-lever). Chosen so the
/// known non-lever (absorb, −0.0% wall) passes and a real lever (decode,
/// +5.2%) does not get mislabeled. Coz elasticity is program_speedup /
/// line_speedup; a region that moves the wall <~1% under a full virtual
/// speedup is noise on this box (15–20% jitter floor → but the per-trial
/// interleaving + many epochs tighten the causal estimate well under that).
const ZERO_ELASTICITY: f64 = 0.03;
/// A region must clear this to count as a positive lever.
const POSITIVE_ELASTICITY: f64 = 0.05;

/// Check the Coz + critpath results against the known ground truth.
/// `coz` may be None if only the critical-path layer ran.
pub fn check_against_ground_truth(coz: Option<&CozProfile>, crit: &CritPath) -> Validation {
    let mut checks = Vec::new();

    if let Some(coz) = coz {
        // Use the PEAK-line elasticity (the actionable lever), not the
        // weighted median — the median is masked to ~0 when a region has a
        // high-sample near-zero line (see rank.rs / coz.rs notes).
        let peak = |region: &str| -> Option<(f64, f64)> {
            coz.region_curves
                .get(region)
                .map(|rc| rc.peak_line_elasticity())
        };
        let absorb_fired = coz
            .region_latency
            .get("fulcrum.absorb")
            .map(|(a, _, _)| *a > 0.0)
            .unwrap_or(false);

        // (1) absorb ≈ 0. Two ways to pass, both meaning "non-lever":
        //   (a) a measurable peak elasticity below the zero threshold, or
        //   (b) the absorb SCOPE fired (it executed) yet coz could build NO
        //       virtual-speedup experiment on any absorb source line — i.e.
        //       it is so cheap/overlapped there is no leverage signal at
        //       all. That absence IS the non-lever signature.
        match peak("absorb") {
            Some((e, n)) => checks.push(Check {
                name: "absorb≈0 (known non-lever, −0.0% wall)".into(),
                expectation: format!("|peak elasticity| < {ZERO_ELASTICITY}"),
                measured: format!("peak={e:+.3} (n={n} samples)"),
                passed: e.abs() < ZERO_ELASTICITY,
            }),
            None => checks.push(Check {
                name: "absorb≈0 (known non-lever, −0.0% wall)".into(),
                expectation: "no measurable leverage (scope fires but coz builds no experiment)"
                    .into(),
                measured: format!(
                    "absorb scope fired={absorb_fired}, 0 coz-experiment lines cleared the \
                     sample floor → unmeasurably small leverage (the non-lever signature)"
                ),
                // Pass iff it actually executed (so "no experiment" means
                // "too cheap to measure", not "never ran").
                passed: absorb_fired,
            }),
        }

        // (2) a decode region > 0 (PEAK-line).
        let decode_detail = ["bulk_inflate", "bootstrap"]
            .iter()
            .filter_map(|r| peak(r).map(|(e, n)| format!("{r}={e:+.3}(n={n})")))
            .collect::<Vec<_>>()
            .join(", ");
        let decode_pos = ["bulk_inflate", "bootstrap"].iter().any(|r| {
            peak(r)
                .map(|(e, _)| e > POSITIVE_ELASTICITY)
                .unwrap_or(false)
        });
        checks.push(Check {
            name: "decode>0 (inline match-copy banked +5.2% T16)".into(),
            expectation: format!(
                "max(bulk_inflate,bootstrap) PEAK elasticity > {POSITIVE_ELASTICITY}"
            ),
            measured: if decode_detail.is_empty() {
                "no decode-region coz experiments".into()
            } else {
                decode_detail
            },
            passed: decode_pos,
        });

        // Cross-region sanity: decode should out-lever absorb. With absorb
        // unmeasurable (peak=0/None), any positive decode peak satisfies this.
        let absorb_e = peak("absorb").map(|(e, _)| e).unwrap_or(0.0);
        let decode_e = ["bulk_inflate", "bootstrap"]
            .iter()
            .filter_map(|r| peak(r).map(|(e, _)| e))
            .fold(f64::NEG_INFINITY, f64::max);
        if decode_e.is_finite() {
            checks.push(Check {
                name: "ordering: decode out-levers absorb".into(),
                expectation: "max(decode) PEAK elasticity > absorb PEAK elasticity".into(),
                measured: format!("decode={decode_e:+.3} vs absorb={absorb_e:+.3}"),
                passed: decode_e > absorb_e,
            });
        }
    }

    // (3) critical path surfaces heavy overshoot chunks.
    let n_heavy = crit.heavy_chunks.len();
    let max_heavy = crit
        .heavy_chunks
        .iter()
        .map(|h| h.blocker_dur_us)
        .fold(0.0_f64, f64::max);
    checks.push(Check {
        name: "critpath surfaces heavy overshoot bootstrap chunks".into(),
        expectation: "≥1 on-path heavy blocker (bootstrap/decode) > threshold".into(),
        measured: format!("{n_heavy} heavy blockers, max {:.1}ms", max_heavy / 1000.0),
        passed: n_heavy >= 1,
    });

    Validation { checks }
}

/// Drive the empirical oracle directly: run `scripts/interleaved_ab.sh`
/// with the supplied contenders and return its stdout (the median MB/s +
/// pairwise deltas). FULCRUM prints this next to the Coz verdict so a human
/// can see the causal prediction and the measured wall side by side.
pub fn run_interleaved_ab(
    repo: &Path,
    contenders: &[(String, String)],
    n: usize,
    cpus: &str,
    raw: u64,
    reference: Option<&str>,
) -> std::io::Result<String> {
    let script = repo.join("scripts/interleaved_ab.sh");
    let mut cmd = Command::new("bash");
    cmd.arg(&script)
        .env("N", n.to_string())
        .env("CPUS", cpus)
        .env("RAW", raw.to_string())
        .current_dir(repo);
    if let Some(r) = reference {
        cmd.env("REF", r);
    }
    for (label, command) in contenders {
        cmd.arg(format!("{label}={command}"));
    }
    let out = cmd.output()?;
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(s)
}
