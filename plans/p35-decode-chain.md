# P3.5 — decode-chain restructuring in the contig clean fast loop (FROZEN, banked)

Date: 2026-06-11. Branch `engine/p35-decode-chain`, base `dd2c21b6`.
Mandate: plans/orchestrator-status.md "REMOVAL ORACLES MERGED" — the Huffman-
decode/bit-read DEPENDENCY CHAIN owns the DECODE ceiling (642.6 ms = 50.9% of
native T1 silesia wall, conservative lower bound; STORE side exhausted at
≤94 ms; BMI2 instructions already emitted — the cost is the DEPENDENCES).
Four pre-registered candidates; per-candidate masked frozen T1 A/B (≥7, ran 9).

## The ladder (each separately committed; all byte-exact by construction)

1. **c1 `8f93526b` — NEXT-SYMBOL PRELOAD before the backref copy (SHIP).**
   libdeflate decompress_template.h:555-572 shape: in the backref arm, the
   threshold refill + next-litlen LUT load now issue BEFORE
   `emit_backref_contig`, so the copy's latency overlaps the next symbol's
   dependent table load; the iteration ends with `continue 'fast`. Refill +
   decode read only input-side state (`lb`, `lut_litlen`); the copy +
   sparsity bookkeeping touch only output/disjoint `self` fields — pure
   instruction-scheduling reorder.
2. **c4 `1bfe0af3` — backstop-free litlen decode at prefilled sites (TIE,
   KEPT).** `LutLitLenCode::decode_prefilled` = identical lookup minus the
   `available()<32` load+branch+refill backstop; used ONLY at the three
   preload sites immediately after a refill. Proof on the method doc:
   post-refill `available()<32` ⟹ input exhausted ⟹ backstop is a no-op in
   every case. Kept on TIE per the layering rule
   ([[feedback_layer_dont_revert_whole_system]]).
3. **c2 `5b85ffc5` — fused litlen→dist lookahead (SHIP).** The dist
   short-LUT load issues at the moment the litlen packet resolves to a lone
   non-literal (post-consume `bitbuf` final), BEFORE the EOB/MAX/length
   branch tree, carried as `Option<DistEntry>` (Copy u32) to the backref
   arm. Same index, same table, no consumes between the sites ⇒ identical
   value; a wasted load on EOB reads always-valid table memory.
4. **c3 — branchless entry dispatch: NO-RESTRUCTURE (gated by measurement,
   as pre-registered).** perf COUNTING mode works in the LXC (sampling
   stays blocked). T1 silesia, forced-SM, cpu_core PMU:
   total 940.0M branches / 17.18M misses (1.83%); the NODECODE
   record/replay oracle (decode chain fully removed, 2798/2798 hits) still
   shows 13.89M misses on 527M branches (2.63%). Net decode-chain mispredict
   share ≤ 3.3M of 17.2M — and the replay arm's own lit-vs-backref op
   dispatch re-creates the same irregular pattern, so the true
   entry-dispatch share is smaller still. Mechanism: the mispredicts live on
   the store/copy + pipeline side (they survive complete decode removal),
   and a computed dispatch trades branch misses for equally-irregular
   indirect-target misses. ≤3.3M × ~17 cyc ≈ ≤56M cyc ≈ low-tens-of-ms
   upper bound — not the lever. Also: across the c0→c142 ladder misses are
   FLAT (17.11M → 17.18M) while instructions fell 4.0% — the wins are chain
   scheduling, not branch behavior, confirming dispatch was never the cost.

## Frozen T1 silesia ladder (bench-lock acquired, no_turbo=1 readback,
## interleaved 4-arm best-of-9, taskset -c 0, sink /dev/null)

| arm  | median ms (spread)  | step Δ | verdict |
|------|---------------------|--------|---------|
| c0   | 1262 (1261–1263)    | —      | base == banked 1263.0 (same binary sha `68a963b7…`) |
| c1   | 1221 (1219–1240)    | **−41 ms (−3.2%)** | SHIP |
| c14  | 1219 (1215–1239)    | −2 ms  | TIE (within wobble class), kept |
| c142 | 1183 (1182–1218)    | **−36 ms (−3.0%)** | SHIP |

Cumulative **1262 → 1183 = −79 ms (−6.3%)**, ~12% of the 642.6 ms decode
ceiling consumed. Guest binaries: bin-c0 `68a963b7…` (byte-identical rebuild
of dd2c21b6 == prior session's bin-oracle-native — toolchain reproducible),
bin-c1 `b75c2bce…`, bin-c14 `6a42a207…`, bin-c142 `84323684…`.

## Instruction/branch deltas (perf counting, forced-SM T1 silesia, cpu_core)

| arm  | instructions | branches | branch-misses |
|------|--------------|----------|---------------|
| c0   | 4920.8M      | 985.2M   | 17.11M (1.74%) |
| c1   | 4848.7M      | 967.4M   | 17.08M |
| c14  | 4884.9M (+36M vs c1 — inline-site duplication; wall TIE) | 965.4M | 17.08M |
| c142 | 4724.6M      | 940.0M   | 17.18M |

## contig_prof before/after (T1 silesia, GZIPPY_CONTIG_PROF=1; iteration and
## byte counts IDENTICAL c0 vs c142 — an independent byte-exactness witness)

| class   | iters      | c0 cyc/iter | c142 cyc/iter | Δ |
|---------|------------|-------------|---------------|----|
| lit1    | 3,303,758  | 23.7        | 20.2          | −14.8% |
| litpack | 1,227,631  | 25.6        | 21.9          | −14.5% |
| litchn  | 6,319,731  | 23.1        | 20.3          | −12.1% |
| backref | 17,971,348 | 20.3        | 18.1          | −10.8% |
| classed total | —    | 620.2M cyc  | 546.8M cyc    | **−11.8%** |

Whole-loop improvement across ALL classes — consistent with the perturbation
finding that the T1 gap is whole-loop, not class-concentrated.

## Correctness gauntlet

- Per-commit local Rosetta (x86-64-v2) differentials: contig+dist_table 14/14,
  silesia 20/20 (c1); 34/34 (c4, c2).
- fmt per commit; clippy default-features 0 warnings; pure-rust-inflate x86
  target: 46 warnings == base 46, 8 `not_unsafe_ptr_arg_deref` errors == base
  8 (pre-existing at dd2c21b6, none added).
- Guest native lib suite: 949 passed / 0 failed / 12 ignored.
- Guest integration tests: see commit/status note.
- sha grid {silesia, model, bignasa, storedmix} × T{1,4,8,16}, bin-c142
  gzippy-native, forced-SM: see status note (run below).
- Untimed sha pre-check: all 4 binaries × silesia × T{1,8}: 8/8 OK.

## Final frozen 3-cell (c0 vs c142): see orchestrator-status entry.

## Honesty ledger / anomalies (verbatim)

- Local laptop data volume was FULL (222 MiB free) at session start — the
  known full-disk trap; caught BEFORE the first release build by `df` (freed
  ~4 GB of regenerable old-worktree target dirs).
- First two laptop perf-stat runs on the guest omitted
  `GZIPPY_FORCE_PARALLEL_SM=1`; discarded and re-run with it (numbers above
  are the forced-SM runs; the unforced numbers were within ~0.3% anyway).
- c14 instruction count is HIGHER than c1 (+36M) despite removing a branch —
  decode_prefilled duplicates the lookup body at three inline sites; wall
  TIE'd and the layered c142 still lands −160M vs c14. Honest wobble note.
- c142 rep1 = 1218 ms (warm-up outlier; reps 2–9 sit 1182–1185). Median used.
- rsync to the guest reported `cannot delete non-empty directory:
  vendor/rapidgzip/...` — pre-existing guest-side build artifacts
  (build-trace) intentionally left in place; not part of the build.
