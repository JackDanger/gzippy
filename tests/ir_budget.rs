//! Instruction-budget ratchets: the wall axis's deterministic proxy.
//!
//! Wall time cannot be asserted in a unit test (noise, load, arch), but
//! instructions-per-byte under cachegrind is repeatable to a fraction of a
//! percent — and this campaign CALIBRATED the proxy before trusting it
//! (docs in the pin file): cutting instructions 21% moved the paired wall
//! 5.4% at T1, while cutting data writes 27% at flat instructions moved it
//! ZERO. Ir/B is the budget worth holding; store counts are not.
//!
//! Budgets are pinned in tests/fingerprints/ir_budget.tsv with provenance.
//! The ratchet only TIGHTENS: an optimization re-pins lower in its own PR; a
//! regression fails here with the measured number. Tolerance covers
//! cachegrind's small env sensitivity, not real regressions.
//!
//! Runs wherever valgrind exists (trainer, solvency, Linux CI); elsewhere it
//! states that it measured nothing and passes — silence is not certification.

use gzippy::fixtures;
use std::process::Command;

const TOLERANCE: f64 = 1.03;

fn cachegrind_ir(bin: &str, args: &[&str]) -> Option<u64> {
    let out = Command::new("valgrind")
        .args([
            "--tool=cachegrind",
            "--cache-sim=no",
            "--cachegrind-out-file=/dev/null",
        ])
        .arg(bin)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()?;
    let err = String::from_utf8_lossy(&out.stderr);
    err.lines()
        .find(|l| l.contains("I refs:"))
        .and_then(|l| {
            l.split("I refs:")
                .nth(1)?
                .trim()
                .replace(',', "")
                .parse()
                .ok()
        })
}

#[test]
fn instruction_budgets_hold() {
    if Command::new("valgrind").arg("--version").output().is_err() {
        eprintln!("ir_budget: valgrind not on this host — MEASURED NOTHING (not a pass of the budgets)");
        return;
    }
    let pins = std::fs::read_to_string("tests/fingerprints/ir_budget.tsv")
        .expect("tests/fingerprints/ir_budget.tsv missing — generate rows with tests/ir_budget.rs::print_current (cargo test --release print_current -- --ignored --nocapture)");
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for line in pins.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (fixture, level, budget_ir_per_b): (&str, u32, f64) =
            (c[0], c[1].parse().unwrap(), c[2].parse().unwrap());
        let data = fixtures::generate(fixture);
        let path = dir.path().join(fixture);
        std::fs::write(&path, &data).unwrap();
        let Some(ir) = cachegrind_ir(
            bin,
            &[&format!("-{level}"), "-p", "1", "-c", path.to_str().unwrap()],
        ) else {
            failures.push(format!("{fixture} L{level}: cachegrind produced no I-refs line"));
            continue;
        };
        let ir_per_b = ir as f64 / data.len() as f64;
        checked += 1;
        if ir_per_b > budget_ir_per_b * TOLERANCE {
            failures.push(format!(
                "{fixture} L{level}: {ir_per_b:.2} Ir/B exceeds budget {budget_ir_per_b:.2} (+{:.1}% past tolerance) — \
                 an instruction regression on the calibrated wall proxy; find the mechanism or revert",
                (ir_per_b / budget_ir_per_b - 1.0) * 100.0
            ));
        } else if ir_per_b < budget_ir_per_b / TOLERANCE {
            eprintln!(
                "ir_budget: {fixture} L{level} improved to {ir_per_b:.2} Ir/B (budget {budget_ir_per_b:.2}) — ratchet it down in this PR"
            );
        }
    }
    assert!(checked > 0, "budgets file had no rows");
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n"));
}

/// Prints current Ir/B for every fixture cell — the pin generator.
///   cargo test --release --test ir_budget print_current -- --ignored --nocapture
#[test]
#[ignore]
fn print_current() {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    println!("fixture\tlevel\tir_per_b");
    for &fixture in fixtures::NAMES {
        let data = fixtures::generate(fixture);
        let path = dir.path().join(fixture);
        std::fs::write(&path, &data).unwrap();
        for level in [1u32, 6] {
            if let Some(ir) = cachegrind_ir(
                bin,
                &[&format!("-{level}"), "-p", "1", "-c", path.to_str().unwrap()],
            ) {
                println!("{fixture}\t{level}\t{:.2}", ir as f64 / data.len() as f64);
            }
        }
    }
}
