//! Probe 1 (final-lap lever #364 investigation, agent-16): the L9/T>1
//! config-cost matrix.
//!
//! Times the EXACT production chunk entry (`encode_deflate_splice_chunk_to_sink`)
//! over one 1.8 MB chunk + 32 KiB prefix dictionary, for the variants that
//! discriminate the pigz-L9/T4-loss hypotheses:
//!
//! 1. production: `parallel=true, level=9`  → `params_parallel(9)` ≡
//!    `params_parallel(11)` (near-optimal, depth 400, passes 4).
//! 2. T1 engine:  `parallel=false, level=9` → `params(9)` (Lazy2, depth 600,
//!    nice 258 — the ldx-shaped parse the T1 cells ship).
//! 3. control:    `parallel=true, level=11` (the params alias; must match variant 1).
//! 4. control:    `parallel=false, level=11`.
//!
//! READS: best-of-k wall via std::time::Instant + output bytes (the size
//! ledger side; the near-opt upgrade was *paid for* in size).
//!
//! DISCRIMINATION (docs/board/sprint-2026-09-25.md):
//!   wall(1)/wall(2) >= 2.5  → the near-opt upgrade is the multiplier
//!                             (agent-16 predicted 3.5-4.5x from 61f0f01d)
//!   wall(1) ~ wall(2)       → REFUTED; escalate to scheduler phase accounting
//! The wall(1)/wall(3) control confirms the params alias.
//!
//! Measurement only (`#[ignore]`), never a regression gate.

use std::time::Instant;

const CHUNK: usize = 1_800_000;
const DICT: usize = 32 * 1024;

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

#[test]
#[ignore = "measurement probe — run with --ignored --nocapture"]
fn l9_t4_chunk_cost_matrix() {
    // a ~1.9 MB deterministic corpus (the frozen text fixture, extended
    // cyclically to the production chunk size it carries at silesia.tar/T4)
    let mut data = gzippy::fixtures::generate("text");
    if data.is_empty() {
        data = vec![0x21u8; CHUNK + DICT + 64];
    }
    while data.len() < CHUNK + DICT + 4096 {
        let take = (CHUNK + DICT + 4096 - data.len()).min(data.len());
        data.extend_from_slice(&data.clone()[..take]);
    }
    assert!(
        data.len() >= CHUNK + DICT,
        "corpus too small even after extension"
    );
    let body = &data[DICT..DICT + CHUNK];
    let dict = &data[..DICT];
    let input_total_len = 200 * 1024 * 1024; // production-shaped silesia.tar size

    let (t_prod, b_prod) = run(
        "1 production (L9 parallel=true = near-opt@L11, depth400)",
        body,
        dict,
        9,
        true,
        input_total_len,
        5,
    );
    let (t_t1, b_t1) = run(
        "2 T1 engine  (L9 parallel=false = Lazy2 depth600)",
        body,
        dict,
        9,
        false,
        input_total_len,
        5,
    );
    let (t_c, b_c) = run(
        "3 control    (L11 parallel=true, the params_parallel alias)",
        body,
        dict,
        11,
        true,
        input_total_len,
        5,
    );
    let (t_c2, b_c2) = run(
        "4 control    (L11 parallel=false)",
        body,
        dict,
        11,
        false,
        input_total_len,
        3,
    );

    println!("=== matrix ===");
    println!(
        "wall(production)/wall(T1 engine): {:.3}  (>=2.5 CONFIRMS the near-opt upgrade as the multiplier)",
        t_prod / t_t1
    );
    println!(
        "bytes: production {b_prod}  T1-engine {b_t1}  delta = {}",
        b_prod as i64 - b_t1 as i64
    );
    println!(
        "controls: t(11,true) {:.4}s vs t(9,true) {:.4}s (alias check ~1.0 expected)",
        t_c, t_prod
    );
    println!(
        "bytes(11,true) {b_c} == bytes(9,true) {b_prod}: {}",
        b_c == b_prod
    );
    let _ = (t_c2, b_c2);
}
