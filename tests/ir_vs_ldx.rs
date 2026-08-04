//! The per-label CPU GOAL, as a test: beat the libdeflate algorithm and every
//! rival binary on instructions per byte, at the level the user typed.
//!
//! tests/ir_budget.rs ratchets us against OUR OWN past. This file ratchets us
//! against the COMPETITION, on the calibrated proxy (see ir_budget.tsv's pin
//! header: -21% Ir moved paired T1 wall 5.4%; -27% data writes moved it 0).
//!
//! Two legs, one goal sheet each:
//!
//! 1. **ours vs ldx** (tests/fingerprints/ir_vs_ldx.tsv): the in-tree `ldx`
//!    module is an exact port of libdeflate v1.23, byte-identical at L0-L9, so
//!    it is the libdeflate ALGORITHM runnable in-process on any box. Both
//!    engines run through the same runner binary (examples/ir_runner.rs) under
//!    cachegrind in the same test invocation, so the ratio is same-host,
//!    same-build, same-startup — host drift cancels. Every pinned margin > 1.0
//!    is a NAMED OPEN GAP (we spend more Ir than the algorithm we ported);
//!    margins <= 1.0 are held wins. The ratchet only tightens.
//!
//! 2. **ours vs rival binaries** (tests/fingerprints/rivals_ir.tsv): gzip,
//!    pigz -p1, libdeflate-gzip, igzip, measured end-to-end under cachegrind
//!    ON TRAINER (the box with all four). Ours is re-measured here, on this
//!    host, via the shipped CLI (`-{level} -p 1 -c`); rival rows whose binary
//!    or version is absent locally are skipped with a notice — a skip is not a
//!    pass.
//!
//! Runs wherever valgrind exists; elsewhere it states that it measured nothing
//! and passes — silence is not certification.

use gzippy::fixtures;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Levels the goal sheet covers: the two shipped fast levels, the default, and
/// the deep end. Never generalise a margin across levels not in this set.
const LEVELS: &[u32] = &[1, 2, 6, 9];

/// Tolerance on the ours/ldx RATIO. Both arms run on the same host in the same
/// test invocation, so environment drift cancels; this covers cachegrind's
/// residual run-to-run jitter only.
const RATIO_TOLERANCE: f64 = 1.02;

/// Tolerance on ours-vs-rival-pin. The rival side is a trainer pin; the ours
/// side is measured here, possibly under a different rustc, so cross-build
/// drift of a few percent is real and must not fire the goal alarm.
const RIVAL_TOLERANCE: f64 = 1.05;

const LDX_PINS: &str = "tests/fingerprints/ir_vs_ldx.tsv";
const RIVAL_PINS: &str = "tests/fingerprints/rivals_ir.tsv";

fn have_valgrind() -> bool {
    Command::new("valgrind").arg("--version").output().is_ok()
}

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
    err.lines().find(|l| l.contains("I refs:")).and_then(|l| {
        l.split("I refs:")
            .nth(1)?
            .trim()
            .replace(',', "")
            .parse()
            .ok()
    })
}

/// Locate (building if needed) the shared runner binary next to the gzippy bin.
fn runner_bin() -> String {
    let gz = PathBuf::from(env!("CARGO_BIN_EXE_gzippy"));
    let dir = gz.parent().expect("bin has a parent dir");
    let runner = dir.join("examples").join("ir_runner");
    if !runner.exists() {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let mut cmd = Command::new(cargo);
        cmd.args(["build", "--example", "ir_runner"]);
        if gz.to_string_lossy().contains("/release/") {
            cmd.arg("--release");
        }
        let status = cmd
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("spawn cargo build --example ir_runner");
        assert!(status.success(), "building examples/ir_runner failed");
    }
    runner.to_string_lossy().into_owned()
}

fn materialize_fixtures(dir: &std::path::Path) -> HashMap<String, (PathBuf, usize)> {
    let mut m = HashMap::new();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.join(name);
        std::fs::write(&path, &data).unwrap();
        m.insert(name.to_string(), (path, data.len()));
    }
    m
}

/// First line of `<rival> --version` (libdeflate-gzip uses -V), or None if the
/// binary is not on this box.
fn local_rival_version(name: &str) -> Option<String> {
    let (bin, vflag) = match name {
        "libdeflate" => ("libdeflate-gzip", "-V"),
        b => (b, "--version"),
    };
    let o = Command::new(bin).arg(vflag).output().ok()?;
    let s = String::from_utf8_lossy(if o.stdout.is_empty() {
        &o.stderr
    } else {
        &o.stdout
    })
    .lines()
    .next()?
    .trim()
    .to_string();
    Some(s)
}

/// The rival's T1 compress command for `-{level} -c <file>` (stdout is nulled
/// by cachegrind_ir). Mirrors examples/fingerprint_tool.rs::rival_commands.
fn rival_command(name: &str, level: u32, file: &str) -> Option<(String, Vec<String>)> {
    let lvl = format!("-{level}");
    match name {
        "gzip" => Some(("gzip".into(), vec![lvl, "-c".into(), file.into()])),
        "pigz" => Some((
            "pigz".into(),
            vec![lvl, "-p".into(), "1".into(), "-c".into(), file.into()],
        )),
        "libdeflate" => Some((
            "libdeflate-gzip".into(),
            vec![lvl, "-c".into(), file.into()],
        )),
        // igzip only has levels 0-3.
        "igzip" if level <= 3 => Some((
            "igzip".into(),
            vec![lvl, "-T".into(), "1".into(), "-c".into(), file.into()],
        )),
        "igzip" => None,
        other => panic!("unknown rival '{other}'"),
    }
}

struct RivalPinFile {
    /// rival name -> version line the pins were taken at.
    versions: HashMap<String, String>,
    /// (fixture, level, rival) -> (rival_ir_per_b, margin).
    rows: Vec<(String, u32, String, f64, f64)>,
}

fn parse_rival_pins(text: &str) -> RivalPinFile {
    let mut versions = HashMap::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#   ") {
            if let Some((name, ver)) = rest.split_once(": ") {
                versions.insert(name.trim().to_string(), ver.trim().to_string());
            }
            continue;
        }
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        rows.push((
            c[0].to_string(),
            c[1].parse().unwrap(),
            c[2].to_string(),
            c[3].parse().unwrap(),
            // column 4 is ours_ir_per_b at pin time (provenance, not asserted)
            c[5].parse().unwrap(),
        ));
    }
    RivalPinFile { versions, rows }
}

#[test]
fn ir_goal_margins_hold() {
    if !have_valgrind() {
        eprintln!(
            "ir_vs_ldx: valgrind not on this host — MEASURED NOTHING (not a pass of the goals)"
        );
        return;
    }
    let ldx_pins = std::fs::read_to_string(LDX_PINS).unwrap_or_else(|_| {
        panic!("{LDX_PINS} missing — seed it with UPDATE_IR_VS_LDX=1 cargo test --release --test ir_vs_ldx update_ldx_pins -- --ignored --nocapture")
    });
    let rival_pins = std::fs::read_to_string(RIVAL_PINS).unwrap_or_else(|_| {
        panic!("{RIVAL_PINS} missing — seed it ON TRAINER with UPDATE_RIVALS_IR=1 cargo test --release --test ir_vs_ldx update_rival_pins -- --ignored --nocapture")
    });
    let runner = runner_bin();
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let files = materialize_fixtures(dir.path());
    let mut failures = Vec::new();
    let mut checked = 0;

    // ── Leg 1: ours vs ldx, same runner, same host, same invocation ─────────
    // Also caches the shipped-CLI Ir/B per cell for leg 2.
    let mut cli_ir_per_b: HashMap<(String, u32), f64> = HashMap::new();
    for line in ldx_pins.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (fixture, level, margin): (&str, u32, f64) =
            (c[0], c[1].parse().unwrap(), c[4].parse().unwrap());
        let (path, len) = &files[fixture];
        let path = path.to_str().unwrap();
        let lvl = level.to_string();
        let (Some(ours), Some(ldx)) = (
            cachegrind_ir(&runner, &["ours", &lvl, path]),
            cachegrind_ir(&runner, &["ldx", &lvl, path]),
        ) else {
            failures.push(format!(
                "{fixture} L{level}: cachegrind produced no I-refs line"
            ));
            continue;
        };
        let ratio = ours as f64 / ldx as f64;
        checked += 1;
        if ratio > margin * RATIO_TOLERANCE {
            if margin > 1.0 {
                failures.push(format!(
                    "{fixture} L{level}: ours {ratio:.3}x ldx, past even the pinned OPEN-GOAL margin {margin:.3} — \
                     the goal is <=1.0 and this pin marks how far we already were; do not raise it, close it"
                ));
            } else {
                failures.push(format!(
                    "{fixture} L{level}: ours {ratio:.3}x ldx exceeds held margin {margin:.3} — \
                     a regression against the libdeflate algorithm itself; find the mechanism or revert"
                ));
            }
        } else if ratio < margin / RATIO_TOLERANCE {
            eprintln!(
                "ir_vs_ldx: {fixture} L{level} improved to {ratio:.3}x ldx (margin {margin:.3}) — tighten the pin in this PR{}",
                if margin > 1.0 && ratio <= 1.0 { "; an OPEN GOAL just closed" } else { "" }
            );
        }
        if let Some(ir) = cachegrind_ir(bin, &[&format!("-{level}"), "-p", "1", "-c", path]) {
            cli_ir_per_b.insert((fixture.to_string(), level), ir as f64 / *len as f64);
        }
    }

    // ── Leg 2: shipped CLI vs rival binary pins (trainer-measured) ──────────
    let rivals = parse_rival_pins(&rival_pins);
    let mut usable: HashMap<String, bool> = HashMap::new();
    for (rival, pinned_ver) in &rivals.versions {
        let ok = match local_rival_version(rival) {
            Some(v) if &v == pinned_ver => true,
            Some(v) => {
                eprintln!(
                    "ir_vs_ldx: rival '{rival}' is '{v}' here but pinned at '{pinned_ver}' — skipping its rows (a skip is not a pass)"
                );
                false
            }
            None => {
                eprintln!(
                    "ir_vs_ldx: rival '{rival}' not on this host — skipping its rows (a skip is not a pass)"
                );
                false
            }
        };
        usable.insert(rival.clone(), ok);
    }
    for (fixture, level, rival, rival_pin, margin) in &rivals.rows {
        if !usable.get(rival).copied().unwrap_or(false) {
            continue;
        }
        let Some(ours) = cli_ir_per_b.get(&(fixture.clone(), *level)) else {
            failures.push(format!(
                "{fixture} L{level}: no shipped-CLI Ir measurement to grade vs {rival}"
            ));
            continue;
        };
        let ratio = ours / rival_pin;
        checked += 1;
        if ratio > margin * RIVAL_TOLERANCE {
            if *margin > 1.0 {
                failures.push(format!(
                    "{fixture} L{level} vs {rival}: ours {ratio:.3}x their {rival_pin:.2} Ir/B, past the pinned OPEN-GOAL margin {margin:.3} — \
                     the per-label goal is <=1.0; do not raise this pin, close it"
                ));
            } else {
                failures.push(format!(
                    "{fixture} L{level} vs {rival}: ours {ratio:.3}x their {rival_pin:.2} Ir/B exceeds held margin {margin:.3} — \
                     a regression against a rival we used to beat; find the mechanism or revert"
                ));
            }
        } else if ratio < margin / RIVAL_TOLERANCE {
            eprintln!(
                "ir_vs_ldx: {fixture} L{level} vs {rival} improved to {ratio:.3}x (margin {margin:.3}) — tighten the pin in this PR{}",
                if *margin > 1.0 && ratio <= 1.0 { "; an OPEN GOAL just closed" } else { "" }
            );
        }
    }
    assert!(checked > 0, "pin files had no gradeable rows");
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n"));
}

fn pin_header_provenance() -> String {
    let sh = |cmd: &str, args: &[&str]| {
        Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "?".into())
    };
    format!(
        "# Pinned on host '{}' , gzippy @ {}, {}.\n",
        sh("uname", &["-n"]),
        sh("git", &["rev-parse", "--short", "HEAD"]),
        sh("date", &["-u", "+%Y-%m-%d"])
    )
}

/// Round a margin UP so the pin can never be tighter than the measurement.
fn ceil3(x: f64) -> f64 {
    (x * 1000.0).ceil() / 1000.0
}

/// Regenerates tests/fingerprints/ir_vs_ldx.tsv. Run where valgrind exists:
///   UPDATE_IR_VS_LDX=1 cargo test --release --test ir_vs_ldx update_ldx_pins -- --ignored --nocapture
#[test]
#[ignore]
fn update_ldx_pins() {
    if std::env::var("UPDATE_IR_VS_LDX").is_err() {
        eprintln!("set UPDATE_IR_VS_LDX=1 to (re)write {LDX_PINS}");
        return;
    }
    assert!(have_valgrind(), "update_ldx_pins needs valgrind");
    let runner = runner_bin();
    let dir = tempfile::tempdir().unwrap();
    let files = materialize_fixtures(dir.path());
    let mut out = String::new();
    out.push_str(
        "# THE GOAL SHEET, leg 1: ours vs the libdeflate ALGORITHM (the in-tree ldx\n\
         # port of libdeflate v1.23, byte-identical L0-L9), as Ir/B under cachegrind.\n\
         # Both engines run through examples/ir_runner.rs on the same host in the same\n\
         # test invocation, so the RATIO is what is pinned and host drift cancels.\n\
         # ldx runs raw DEFLATE (no gzip framing/crc) — a FLOOR for libdeflate's cost,\n\
         # so every margin here is slightly HARDER than the true goal, never easier.\n\
         # margin > 1.0 = NAMED OPEN PERF GAP (goal is <=1.0): do not raise it, close it.\n\
         # margin <= 1.0 = held win; the ratchet only TIGHTENS, in the improving PR.\n\
         # ours_ir_per_b / ldx_ir_per_b are the seed-time absolutes (provenance, not asserted).\n\
         # Regenerate: UPDATE_IR_VS_LDX=1 cargo test --release --test ir_vs_ldx update_ldx_pins -- --ignored --nocapture\n",
    );
    out.push_str(&pin_header_provenance());
    out.push_str("fixture\tlevel\tours_ir_per_b\tldx_ir_per_b\tmargin\n");
    for &fixture in fixtures::NAMES {
        let (path, len) = &files[fixture];
        let path = path.to_str().unwrap();
        for &level in LEVELS {
            let lvl = level.to_string();
            let ours = cachegrind_ir(&runner, &["ours", &lvl, path]).expect("ours run");
            let ldx = cachegrind_ir(&runner, &["ldx", &lvl, path]).expect("ldx run");
            let (o, l) = (ours as f64 / *len as f64, ldx as f64 / *len as f64);
            let margin = ceil3(o / l);
            println!("{fixture}\tL{level}\tours {o:.2} Ir/B\tldx {l:.2} Ir/B\tmargin {margin:.3}");
            out.push_str(&format!(
                "{fixture}\t{level}\t{o:.2}\t{l:.2}\t{margin:.3}\n"
            ));
        }
    }
    std::fs::write(LDX_PINS, out).unwrap();
    eprintln!("wrote {LDX_PINS}");
}

/// Regenerates tests/fingerprints/rivals_ir.tsv. ONLY ON TRAINER — the box with
/// all four rival binaries; it refuses to write a partial goal sheet.
///   UPDATE_RIVALS_IR=1 cargo test --release --test ir_vs_ldx update_rival_pins -- --ignored --nocapture
#[test]
#[ignore]
fn update_rival_pins() {
    if std::env::var("UPDATE_RIVALS_IR").is_err() {
        eprintln!("set UPDATE_RIVALS_IR=1 to (re)write {RIVAL_PINS}");
        return;
    }
    assert!(have_valgrind(), "update_rival_pins needs valgrind");
    let rivals = ["gzip", "pigz", "libdeflate", "igzip"];
    let versions: Vec<(String, String)> = rivals
        .iter()
        .map(|r| {
            let v = local_rival_version(r).unwrap_or_else(|| {
                panic!("rival '{r}' missing on this box — rival pins regenerate ONLY on trainer, which has all four")
            });
            (r.to_string(), v)
        })
        .collect();
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let files = materialize_fixtures(dir.path());
    let mut out = String::new();
    out.push_str(
        "# THE GOAL SHEET, leg 2: ours vs the rival BINARIES end-to-end (Ir per input\n\
         # byte under cachegrind, T1, frozen fixtures). REGENERATE ONLY ON TRAINER —\n\
         # the box with all four rivals; a partial regen is refused by the tool.\n\
         # margin = pinned allowance for ours_ir_per_b / rival_ir_per_b, per label.\n\
         # margin > 1.0 = NAMED OPEN PERF GAP (goal is <=1.0): do not raise it, close it.\n\
         # margin <= 1.0 = held win; the ratchet only TIGHTENS, in the improving PR.\n\
         # ours_ir_per_b is the seed-time shipped-CLI cost (provenance, not asserted).\n\
         # The test skips a rival's rows where its binary/version differs locally.\n\
         # Regenerate: UPDATE_RIVALS_IR=1 cargo test --release --test ir_vs_ldx update_rival_pins -- --ignored --nocapture\n\
         # A pin is only comparable at these rival versions:\n",
    );
    for (r, v) in &versions {
        out.push_str(&format!("#   {r}: {v}\n"));
    }
    out.push_str(&pin_header_provenance());
    out.push_str("fixture\tlevel\trival\trival_ir_per_b\tours_ir_per_b\tmargin\n");
    for &fixture in fixtures::NAMES {
        let (path, len) = &files[fixture];
        let path = path.to_str().unwrap();
        for &level in LEVELS {
            let ours = cachegrind_ir(bin, &[&format!("-{level}"), "-p", "1", "-c", path])
                .expect("ours CLI run");
            let o = ours as f64 / *len as f64;
            for rival in &rivals {
                let Some((rbin, rargs)) = rival_command(rival, level, path) else {
                    continue; // level not offered per-label (igzip > 3)
                };
                let rargs: Vec<&str> = rargs.iter().map(|s| s.as_str()).collect();
                let rir = cachegrind_ir(&rbin, &rargs)
                    .unwrap_or_else(|| panic!("{rival} L{level} {fixture}: no I-refs line"));
                let r = rir as f64 / *len as f64;
                let margin = ceil3(o / r);
                println!(
                    "{fixture}\tL{level}\t{rival}\ttheirs {r:.2} Ir/B\tours {o:.2} Ir/B\tmargin {margin:.3}"
                );
                out.push_str(&format!(
                    "{fixture}\t{level}\t{rival}\t{r:.2}\t{o:.2}\t{margin:.3}\n"
                ));
            }
        }
    }
    std::fs::write(RIVAL_PINS, out).unwrap();
    eprintln!("wrote {RIVAL_PINS}");
}
