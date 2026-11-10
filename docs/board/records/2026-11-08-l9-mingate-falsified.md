# 2026-11-08 — L9-T>1 near-opt pass-2 gate (min_improvement_to_continue) — FALSIFIED as a wall lever

Probe branch `lever/l9-mingate` (left bare at origin/main; binaries sha-stamped `f5afff2f`(main) / `b80a92ea`(mi32) / `4b4207aa`(mi64); corpus recipe-verified 16,085,866 B).

Question: at the shipped L9-T>1 (near-opt passes 2 + nice 96), is raising `min_improvement_to_continue` 16 → 32/64 a wall lever? The hypothesis assumed the pre-flight gate (`near_optimal.rs:395`) can skip the pass-2 DP.

**CODE REASON (why it cannot):** the pass loop runs `find_min_cost_path` + `compute_true_cost` at the TOP of each iteration (`near_optimal.rs:393-394`) and checks the improvement gate AFTER the new cost is computed. With `max_optim_passes = 2` both passes' DP always execute; the knob only (a) decides whether the survived pass-2 path seeds the next block and (b) marks `best_true_cost` stale for the only-lits/static comparison. No DP work is skippable via this knob.

**Measured (stamped corpus, interleaved, three sessions, quiet box load 5.7-8.7, self-bands ≤±1.25%):**
- -p4 main/mi32/mi64 = 380.9 / 378.2 / 375.8 ms → ratios 0.9857 / 0.9853 (tie band).
- -p2 = 633.9 / 638.3 / 634.5 ms → 1.0053 / 1.0014 (noise).
- Sizes: main 3,740,910 (exact anchor) / mi32 +20 B / mi64 +200 B; tabular:L9:T4 = 232,057 all arms, stream byte-identical (`2f6b84f7…`) — the ledger cell cannot flip on this knob (fixture has no blocks in the [16, mi) improvement band... and mi32 is byte-inert on all 34 members; mi64 flips 13 member shas by ±1-15 B, all in band).
- Roundtrips 105 + 6 streams, zero failures.

**VERDICT: FALSIFIED.** The only remaining L9-T>1 wall reducer is engine work inside `find_min_cost_path`/the fill (per-pass cost-sheet reuse, candidate-cache slimming) — its own lever. Keeping shipped `min_improvement_to_continue = 16`.
