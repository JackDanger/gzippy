//! surface_probe — walk the response surface, name the cliffs.
//!
//! Usage:  surface_probe [--gzippy <path>] [--out <tsv>] [--len <bytes>]
//!                       [--keep <dir>] [--only <point-id>]
//!
//! `--keep` writes each generated point into <dir> instead of deleting it, and
//! `--only` restricts the walk to one point id. A cliff is only useful if its
//! coordinates can be handed to the next instrument (`divergence_accounting`,
//! `fulcrum why`), and that needs the bytes on disk.
//!
//! For each of the ~60 declared points in `gzippy::fixtures::surface_points()`
//! (axes: literal entropy, repeat period, match-length profile, alphabet size,
//! record structure) this probe:
//!   1. generates the point (default 1 MiB),
//!   2. compresses it with OUR SHIPPED BINARY (default: the `gzippy` next to
//!      this example under target/release) at L1/L6/L9, T1,
//!   3. roundtrips our output through `gzip -dc` and VOIDS the run on any
//!      mismatch — a corrupt-but-smaller output can never score,
//!   4. compresses with libdeflate-gzip and gzip at the same level,
//!   5. emits one TSV row per (point, level) with sizes and ratios.
//!
//! Then it flags CLIFFS: pairs of points adjacent along exactly one axis where
//! ratio-vs-rival crosses 1.0 or jumps by more than 2 points. Each cliff is a
//! generalization boundary with its content coordinates named — the failure
//! modes a NEW archive type would hit. The surface is a MEASUREMENT, not a
//! ratchet: nothing pins the ratios; tests pin only the generator bytes.

use gzippy::fixtures::{surface_generate, surface_points, SurfaceParams};
use std::collections::BTreeMap;
use std::process::{Command, Stdio};

const LEVELS: [u32; 3] = [1, 6, 9];
const JUMP: f64 = 0.02;

fn run_compress(cmd: &mut Command, what: &str) -> Vec<u8> {
    let out = cmd
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|e| panic!("spawn {what}: {e}"));
    assert!(out.status.success(), "{what} failed: {:?}", out.status);
    assert!(!out.stdout.is_empty(), "{what} produced no bytes");
    out.stdout
}

fn order0_entropy(data: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for &b in data {
        hist[b as usize] += 1;
    }
    let n = data.len() as f64;
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Axis accessors for adjacency: (name, value-as-i64 extractor).
type Axis = (&'static str, fn(&SurfaceParams) -> i64);
const AXES: [Axis; 5] = [
    ("entropy", |p| p.entropy_bits as i64),
    ("period", |p| p.period as i64),
    ("match_profile", |p| p.long_matches as i64),
    ("alphabet", |p| p.alphabet as i64),
    ("records", |p| p.records as i64),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let mut gzippy_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut len: usize = 1 << 20;
    let mut keep_dir: Option<String> = None;
    let mut only: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gzippy" => gzippy_path = Some(args.next().expect("--gzippy PATH")),
            "--out" => out_path = Some(args.next().expect("--out FILE")),
            "--len" => len = args.next().expect("--len N").parse().expect("--len N"),
            "--keep" => keep_dir = Some(args.next().expect("--keep DIR")),
            "--only" => only = Some(args.next().expect("--only POINT_ID")),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    // Default subject: the release gzippy sitting beside this example —
    // measure the binary that ships, not a library path that might not route
    // the same way.
    let gz = gzippy_path.unwrap_or_else(|| {
        let exe = std::env::current_exe().expect("current_exe");
        let p = exe
            .parent()
            .and_then(|d| d.parent())
            .map(|d| d.join("gzippy"))
            .expect("locate sibling gzippy");
        assert!(
            p.is_file(),
            "no gzippy at {} — build with `cargo build --release` or pass --gzippy",
            p.display()
        );
        p.to_string_lossy().into_owned()
    });
    eprintln!("surface_probe: subject={gz} len={len}");

    let tmp = std::env::temp_dir().join(format!("surface_probe_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("tmpdir");

    let mut pts = surface_points();
    if let Some(id) = &only {
        pts.retain(|p| &p.id() == id);
        assert!(!pts.is_empty(), "--only {id} matched no declared point");
    }
    if let Some(d) = &keep_dir {
        std::fs::create_dir_all(d).expect("keep dir");
    }
    // rows[(point_idx, level)] -> (ours, libdeflate, gzip)
    let mut rows: BTreeMap<(usize, u32), (usize, usize, usize)> = BTreeMap::new();
    let mut tsv = String::new();
    tsv.push_str(
        "point\tentropy_target\tentropy_measured\tperiod\tmatch_profile\talphabet\trecords\t\
         level\tours\tlibdeflate\tgzip\tratio_vs_libdeflate\tratio_vs_gzip\n",
    );
    for (i, p) in pts.iter().enumerate() {
        let data = surface_generate(p, len);
        let h0 = order0_entropy(&data);
        let input = match &keep_dir {
            Some(d) => std::path::Path::new(d).join(p.id()),
            None => tmp.join(p.id()),
        };
        std::fs::write(&input, &data).expect("write point");
        for level in LEVELS {
            let ours = run_compress(
                Command::new(&gz).args([
                    format!("-{level}"),
                    "-p".into(),
                    "1".into(),
                    "-c".into(),
                    input.to_string_lossy().into_owned(),
                ]),
                "gzippy",
            );
            // Roundtrip through an independent decoder or the row is VOID.
            let cpath = tmp.join(format!("{}.l{level}.gz", p.id()));
            std::fs::write(&cpath, &ours).expect("write compressed");
            let rt = run_compress(
                Command::new("gzip").args(["-dc", &cpath.to_string_lossy()]),
                "gzip -dc roundtrip",
            );
            assert!(
                rt == data,
                "ROUNDTRIP FAILED: {} L{level} — output is not valid gzip of the input; \
                 the whole surface is void",
                p.id()
            );
            std::fs::remove_file(&cpath).ok();
            let ld = run_compress(
                Command::new("libdeflate-gzip").args([
                    format!("-{level}"),
                    "-c".into(),
                    input.to_string_lossy().into_owned(),
                ]),
                "libdeflate-gzip",
            );
            let gzr = run_compress(
                Command::new("gzip").args([
                    format!("-{level}"),
                    "-c".into(),
                    input.to_string_lossy().into_owned(),
                ]),
                "gzip",
            );
            rows.insert((i, level), (ours.len(), ld.len(), gzr.len()));
            tsv.push_str(&format!(
                "{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\n",
                p.id(),
                p.entropy_bits,
                h0,
                p.period,
                if p.long_matches { "long" } else { "short" },
                p.alphabet,
                p.records as u8,
                level,
                ours.len(),
                ld.len(),
                gzr.len(),
                ours.len() as f64 / ld.len() as f64,
                ours.len() as f64 / gzr.len() as f64,
            ));
        }
        if keep_dir.is_none() {
            std::fs::remove_file(&input).ok();
        }
        eprint!("\rsurface_probe: {}/{} points", i + 1, pts.len());
    }
    eprintln!();
    std::fs::remove_dir_all(&tmp).ok();

    match &out_path {
        Some(f) => std::fs::write(f, &tsv).expect("write tsv"),
        None => print!("{tsv}"),
    }

    // ── Cliff detection ────────────────────────────────────────────────────
    // Adjacency: fix four axes, sort the points sharing them by the fifth,
    // pair consecutive values. Works uniformly across the sub-grids.
    //
    // WHAT COUNTS AS A CLIFF. The first version flagged any adjacent pair whose
    // ratio moved >2 points and produced 290 "cliffs" on 60 points — a list
    // that long is a shrug, not a finding, and most of it was ratio drift
    // between two comfortable WINS (0.98 -> 0.92), which is not a
    // generalization boundary at all. A boundary is where the VERDICT is at
    // stake, so a pair qualifies only when:
    //   CROSS      — the verdict flips: one side <= 1.0, the other > 1.0.
    //   NEAR-JUMP  — |dr| > 2 points AND some endpoint is within `NEAR` of the
    //                verdict line, i.e. the next axis step could flip it.
    // Everything else is reported only as a count, not as a coordinate.
    const NEAR: f64 = 0.02;
    let mut crossings: Vec<String> = Vec::new();
    let mut near_jumps: Vec<String> = Vec::new();
    let mut far_jumps = 0u32;
    let mut axis_counts: BTreeMap<(&str, &str), u32> = BTreeMap::new();
    for (axis_i, (axis_name, get)) in AXES.iter().enumerate() {
        let mut groups: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();
        for (i, p) in pts.iter().enumerate() {
            let key: Vec<i64> = AXES
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != axis_i)
                .map(|(_, (_, g))| g(p))
                .collect();
            groups.entry(key).or_default().push(i);
        }
        for members in groups.values() {
            let mut sorted = members.clone();
            sorted.sort_by_key(|&i| get(&pts[i]));
            for w in sorted.windows(2) {
                let (a, b) = (w[0], w[1]);
                for level in LEVELS {
                    let ra = rows[&(a, level)];
                    let rb = rows[&(b, level)];
                    for (rival, sa, sb) in [
                        ("libdeflate", (ra.0, ra.1), (rb.0, rb.1)),
                        ("gzip", (ra.0, ra.2), (rb.0, rb.2)),
                    ] {
                        let r_a = sa.0 as f64 / sa.1 as f64;
                        let r_b = sb.0 as f64 / sb.1 as f64;
                        let crosses = (r_a - 1.0) * (r_b - 1.0) < 0.0;
                        let jumps = (r_a - r_b).abs() > JUMP;
                        let near = r_a.max(r_b) > 1.0 - NEAR;
                        if !crosses && !jumps {
                            continue;
                        }
                        if !crosses && !near {
                            far_jumps += 1;
                            continue;
                        }
                        let kind = if crosses { "CROSS" } else { "NEAR-JUMP" };
                        let line = format!(
                            "CLIFF\t{kind}\trival={rival}\tL{level}\taxis={axis_name}\t\
                             from={} r={r_a:.4}\tto={} r={r_b:.4}\td={:+.4}",
                            pts[a].id(),
                            pts[b].id(),
                            r_b - r_a,
                        );
                        *axis_counts.entry((*axis_name, rival)).or_default() += 1;
                        if crosses {
                            crossings.push(line);
                        } else {
                            near_jumps.push(line);
                        }
                    }
                }
            }
        }
    }
    println!(
        "# CLIFFS: {} verdict CROSSings, {} NEAR-JUMPs (within {:.0} pt of 1.0); \
         {far_jumps} large moves away from the line are counted only.",
        crossings.len(),
        near_jumps.len(),
        NEAR * 100.0,
    );
    for l in crossings.iter().chain(near_jumps.iter()) {
        println!("{l}");
    }
    println!("# BY AXIS x RIVAL (cliffs at the verdict line)");
    for ((axis, rival), n) in &axis_counts {
        println!("#   {axis:<14} {rival:<11} {n}");
    }
    eprintln!(
        "surface_probe: {} cliff(s) at the verdict line across {} points x {:?}",
        crossings.len() + near_jumps.len(),
        pts.len(),
        LEVELS
    );
}
