//! Probe 1 (final-lap lever #364 investigation, agent-16): the L9/T>1
//! config-cost matrix.
//!
//! Times the EXACT production chunk entry (`encode_deflate_splice_chunk_to_sink`)
//! over one 1.8 MB chunk + 32 KiB prefix dictionary, for the variants that
//! discriminate the pigz-L9/T4-loss hypotheses:
//!
//! 1. production: `parallel=true, level=9`  → `params_parallel(9)` ≡
//!    `params_parallel(11)` with the trunk's L9 retune: near-optimal, depth
//!    400-alias knobs, **passes 2** (level.rs `max_optim_passes = 2`).
//! 2. T1 engine:  `parallel=false, level=9` → `params(9)` (Lazy2, depth 600,
//!    nice 258 — the ldx-shaped parse the T1 cells ship).
//! 3. control:    `parallel=true, level=11` (L11 default knobs, **passes 4** —
//!    since the retune this NO LONGER aliases variant 1: wall(1)/wall(3)
//!    now prices the passes 2 vs 4 delta).
//! 4. control:    `parallel=false, level=11`.
//!
//! READS: best-of-k wall via std::time::Instant + output bytes (the size
//! ledger side; the near-opt upgrade was *paid for* in size).
//!
//! DISCRIMINATION (docs/board/sprint-2026-09-25.md):
//!   wall(1)/wall(2) >= 2.5  → the near-opt upgrade is the multiplier
//!                             (agent-16 predicted 3.5-4.5x from 61f0f01d)
//!   wall(1) ~ wall(2)       → REFUTED; escalate to scheduler phase accounting
//!   wall(1)/wall(3)         → the passes-2 retune's own chunk-level price.
//!
//! Measurement only (`#[ignore]`), never a regression gate.

use std::time::Instant;

const CHUNK: usize = 1_800_000;
const DICT: usize = 32 * 1024;

fn build_corpus() -> Vec<u8> {
    if let Ok(path) = std::env::var("PROBE_CORPUS") {
        // real-corporus mode: the file must be at least DICT + CHUNK + 4096
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }

    // non-periodic deterministic corpus: interleave the four frozen fixtures
    let mut base = Vec::new();
    for name in gzippy::fixtures::NAMES {
        base.extend(gzippy::fixtures::generate(name));
    }
    let mut cursor = [0usize; 4];
    let target = CHUNK + DICT + 4096;
    while base.len() < target {
        for (i, name) in gzippy::fixtures::NAMES.iter().enumerate() {
            let piece = gzippy::fixtures::generate(name);
            let start = cursor[i].min(piece.len());
            let take = (piece.len() / 8)
                .max(4096)
                .min(piece.len().saturating_sub(start));
            base.extend_from_slice(&piece[start..start + take]);
            cursor[i] = (start + take) % piece.len().max(1);
        }
    }
    base
}

fn run(
    label: &str,
    body: &[u8],
    dict: &[u8],
    level: u8,
    parallel: bool,
    input_total_len: usize,
    runs: usize,
) -> (f64, usize) {
    let mut times = Vec::with_capacity(runs);
    let mut bytes = 0usize;
    for _ in 0..2 {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
    }
    for _ in 0..runs {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let t = Instant::now();
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
        times.push(t.elapsed().as_secs_f64());
        bytes = out.len();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    println!("{label}: best {best:.4}s bytes {bytes}");
    (best, bytes)
}

#[ignore]
fn matrix_for(level: u8, parallel: bool, label: &'static str, runs: usize) {
    let data = build_corpus();
    let body = &data[DICT..DICT + CHUNK];
    let dict = &data[..DICT];
    run(label, body, dict, level, parallel, 200 * 1024 * 1024, runs);
}

#[test]
#[ignore]
fn prod_l9t4_depth400() {
    matrix_for(9, true, "1 production (near-opt@L11, depth400)", 5);
}

#[test]
#[ignore]
fn t1_engine_lazy2_depth600() {
    matrix_for(9, false, "2 T1 engine (Lazy2 depth600)", 5);
}

#[test]
#[ignore]
fn l11_alias_check() {
    matrix_for(11, true, "3 control (L11 parallel=true alias)", 5);
}

#[test]
#[ignore]
fn depth_split_d100() {
    matrix_for(9, true, "4 nearoptimal:100:150 (depth100 ladder)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p1() {
    matrix_for(9, true, "5 nearoptimal:400:150:1 (depth400, passes1)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p4() {
    matrix_for(
        9,
        true,
        "6 nearoptimal:400:150:4 (L11 default, passes 4)",
        3,
    );
}
