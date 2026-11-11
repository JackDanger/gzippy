# Wave B — wall census on the merged main (@ 13c32862, PRs #457+#458) — 2026-11-11

Branch `records/wave-B-wall-census` (committed from worktree A = origin/main tree).
**Every wall number here is HYPOTHESIS-TIER**: unfrozen laptop, co-loaded box (a
parallel wave agent ran its own census work from `.review-tmp/wC-bytes` during
parts of this run — load1 ranged 5→50 across the session), sub-5% deltas are
local TIES. The frozen box adjudicates anything under 5% with us.

## Setup and provenance

- Arm **A = MAIN**: worktree `.review-tmp/wC-walls/wt-main` @ `origin/main` =
  `13c328628aac8d7de4682b9f06a38a4683b960cb` (merge of #457 near-opt DP removal +
  NEON len-scan, and #458 routing L9–12 at threads ≤ 2 to chunked Lazy2-600).
  Binary `gzA` sha256 `fe8bdab354dec553…`
- Arm **B = BASE**: worktree `.review-tmp/wC-walls/wt-base` @ `7e0d08cdaf384f5d429b7930633d6e9b26237524`
  (merge of PR #456 — main immediately before #457/#458, both arm trees verified with
  their own `git rev-parse`). Binary `gzB` sha256 `f5afff2f2c91ac5c…`
- Both arms: `cargo build --release` from separate `--target-dir`s (the global
  target-dir trap), submodules linked via `scripts/dev/worktree-link.sh`. Build
  times 49.5 s / 49.7 s.
- Machine: Mac16,7, Apple M4 Pro, 14 CPUs, macOS
  (`sysctl hw.ncpu = 14`). All loads recorded per rep in the raw log.
- Corpus (all sha-stamped in the raw log):
  - `in16m.bin` = 16,085,866 B, sha `1010fa2f787d0a92…` — byte-identical to the
    wave-9 stamped corpus (`cat corpus/silesia_blocks/*.bin ×2 +
    test_data/text-1MB.txt ×2`), so the wave-9 receipt numbers are directly
    comparable. The `.review-tmp/in16m.bin` artifact from 2026-11-07 has the same
    sha.
  - `logs-card26m.bin` = first 26,214,400 B of `benchmark_data/logs.txt` (log class)
  - `binary26m.bin` = first 26,214,400 B of `benchmark_data/software.archive` (binary class)
  - `text26m.bin` = `test_data/text-1MB.txt` ×25 = 26,214,400 B (text class; synthetic repetition)
- Method: per cell, warm run per arm, then 8 reps of interleaved A,B,B,A
  (16 samples/arm; rerun pass: 14 reps = 28/arm), median wall, `binary -n -LMEVEL -pN -c INPUT > /dev/null`.
  Medians absorb co-runner spikes; the raw log keeps every sample (`[raw]` lines) plus
  per-rep load1/5/15. Output to file + Py-file `results.json` per cell; sha256 output stamp
  once per cell (deterministic instrument); streams whose bytes CHANGED roundtrip
  through python3's `gzip` module (independent decoder) before being reported.
- Driver scripts: `.review-tmp/wC-walls/{census.py,rerun.py,aacal.py,startup_probe.py,render2.py}`.
  Raw log: `out/raw-timings.log.txt` (committed here as `raw-timings.log.txt`, plus
  `results.json` + `rerun.json` + the A/A + startup logs).

## A/A calibration (same binary, both arms)

Quantifies today's noise floor before reading any A/B delta:

| cell | A med | B med | same-binary delta |
|---|---|---|---|
| L2 -p2 (load ~7.3) | 47.8 | 49.0 | −2.44% |
| L4 -p1 (load ~7.5) | 68.8 | 67.9 | +1.33% |
| L9 -p2 (load 7.5→12.6) | 660.0 | 660.8 | −0.13% |
| L12 -p2 (load 12.6→13.5) | 1246.3 | 1236.0 | +0.83% |

One single-sample outlier in the A/A run: B=1003.6 ms against ~660 ms cell median
(one +52% spike, load 12.6). Same-binary medians therefore move ±2.5% even with
interleaving, individual reps can spike >50% under contention, and any census
delta below ±5% is a TIE.

## Part 1 — Primary matrix: 12 levels × {T1, T2, T4} on in16m

| cell | MAIN ms | BASE ms | MAIN vs BASE | bytes | load1 med | verdict tier |
|---|---|---|---|---|---|---|
| in16m L01 -p1 | 75.8 | 74.7 | +1.45% | identical | 21 | TIE/hyp |
| in16m L01 -p2 | 45.7 | 47.3 | −3.49% | identical | 27 | TIE/hyp |
| in16m L01 -p4 | 35.4 | 34.8 | +1.71% | identical | 27 | TIE/hyp |
| in16m L02 -p1 | 67.4 | 67.2 | +0.26% | identical | 27 | TIE/hyp |
| in16m L02 -p2 | 59.8 | 58.1 | +2.94% | identical | 35 | TIE/hyp |
| in16m L02 -p4 | 45.5 | 42.8 | +6.17% | identical | 35 | +5% → RERUN → +1.42% TIE/hyp |
| in16m L03 -p1 | 76.8 | 79.3 | −3.12% | identical | 41 | TIE/hyp |
| in16m L03 -p2 | 69.6 | 67.7 | +2.71% | identical | 41 | TIE/hyp |
| in16m L03 -p4 | 53.6 | 52.5 | +2.20% | identical | 47 | TIE/hyp |
| in16m L04 -p1 | 75.2 | 76.1 | −1.23% | identical | 47 | TIE/hyp |
| in16m L04 -p2 | 74.2 | 84.1 | −11.74% | identical | 53 | −11.7% → RERUN → −0.17% TIE/hyp |
| in16m L04 -p4 | 65.9 | 62.7 | +5.04% | identical | 53 | +5% → RERUN → +0.03% TIE/hyp |
| in16m L05 -p1 | 81.8 | 85.5 | −4.37% | identical | 53 | TIE/hyp |
| in16m L05 -p2 | 74.1 | 73.7 | +0.57% | identical | 50 | TIE/hyp |
| in16m L05 -p4 | 50.6 | 51.0 | −0.64% | identical | 50 | TIE/hyp |
| in16m L06 -p1 | 108.0 | 108.1 | −0.01% | identical | 50 | TIE/hyp |
| in16m L06 -p2 | 92.5 | 91.2 | +1.44% | identical | 47 | TIE/hyp |
| in16m L06 -p4 | 62.9 | 62.2 | +1.12% | identical | 44 | TIE/hyp |
| in16m L07 -p1 | 138.8 | 141.2 | −1.76% | identical | 44 | TIE/hyp |
| in16m L07 -p2 | 100.1 | 99.7 | +0.47% | identical | 44 | TIE/hyp |
| in16m L07 -p4 | 61.7 | 62.5 | −1.33% | identical | 44 | TIE/hyp |
| in16m L08 -p1 | 176.2 | 180.2 | −2.21% | identical | 44 | TIE/hyp |
| in16m L08 -p2 | 107.2 | 105.6 | +1.49% | identical | 37 | TIE/hyp |
| in16m L08 -p4 | 71.3 | 71.8 | −0.72% | identical | 37 | TIE/hyp |
| in16m L09 -p1 | 228.0 | 232.8 | −2.05% | identical | 37 | TIE/hyp (route untouched at T1) |
| in16m L09 -p2 | 113.1 | 638.8 | −82.29% | DIFFER (MAIN +2.03%) | 38 | flip (roundtrip OK) |
| in16m L09 -p4 | 368.9 | 379.2 | −2.71% | identical | 36 | −2.4% rerun (TIE/hyp, direction = claim) |
| in16m L10 -p1 | 1072.7 | 1085.1 | −1.15% | identical | 37 | TIE/hyp |
| in16m L10 -p2 | 114.8 | 604.1 | −81.00% | DIFFER (MAIN +1.86%) | 36 | flip (roundtrip OK) |
| in16m L10 -p4 | 345.8 | 352.5 | −1.91% | identical | 36 | TIE/hyp |
| in16m L11 -p1 | 1420.5 | 1579.3 | −10.06% | identical | 35 | −9.5% rerun — real (Lever A/B class) |
| in16m L11 -p2 | 111.8 | 870.0 | −87.14% | DIFFER (MAIN +2.29%) | 36 | flip (roundtrip OK) |
| in16m L11 -p4 | 471.3 | 505.4 | −6.75% | identical | 36 | −7.2% rerun — real (Lever A/B class) |
| in16m L12 -p1 | 1905.2 | 2321.1 | −17.92% | identical | 37 | −16.5% rerun — real (Lever A/B class) |
| in16m L12 -p2 | 111.7 | 1176.0 | −90.50% | DIFFER (MAIN +2.39%) | 36 | flip (roundtrip OK) |
| in16m L12 -p4 | 580.5 | 671.1 | −13.50% | identical | 36 | −14.0% rerun — real (Lever A/B class) |

## Part 2 — Rerun pass (14 reps, 28 samples/arm) for the outside-band cells

All reruns land outside the read peaks (load1 med 10–15); they replace the
8-rep first-pass number for the verdict:

| cell | MAIN ms | BASE ms | MAIN vs BASE | bytes |
|---|---|---|---|---|
| in16m L12 -p1 [rerun] | 1993.9 | 2387.7 | −16.49% | identical |
| in16m L11 -p1 [rerun] | 1459.4 | 1612.4 | −9.49% | identical |
| in16m L12 -p4 [rerun] | 589.6 | 685.9 | −14.04% | identical |
| in16m L11 -p4 [rerun] | 495.5 | 534.0 | −7.21% | identical |
| in16m L09 -p4 [rerun] | 364.3 | 373.3 | −2.41% | identical |
| in16m L04 -p4 [rerun] | 40.7 | 40.7 | +0.03% | identical |
| in16m L02 -p4 [rerun] | 30.4 | 29.9 | +1.42% | identical |
| in16m L04 -p2 [rerun] | 61.2 | 61.3 | −0.17% | identical |
| logs-card26m L12 -p4 [rerun] | 781.8 | 1566.0 | −50.08% | identical |
| logs-card26m L11 -p4 [rerun] | 291.2 | 376.9 | −22.75% | identical |
| logs-card26m L09 -p4 [rerun] | 222.8 | 281.9 | −20.95% | identical |
| logs-card26m L10 -p4 [rerun] | 160.5 | 182.1 | −11.86% | identical |
| logs-card26m L06 -p4 [rerun] | 24.1 | 24.2 | −0.45% | identical |
| binary26m L12 -p4 [rerun] | 724.2 | 1304.8 | −44.50% | identical |
| binary26m L11 -p4 [rerun] | 577.7 | 957.0 | −39.64% | identical |
| binary26m L09 -p4 [rerun] | 214.8 | 217.7 | −1.36% | identical |
| text26m L09 -p4 [rerun] | 479.4 | 482.1 | −0.56% | identical |
| text26m L01 -p4 [rerun] | 47.2 | 47.1 | +0.21% | identical |
| text26m L04 -p2 [rerun] | 179.0 | 179.0 | −0.01% | identical |

## Part 3 — Member-class cells (13–14 cells per class, 8 reps)

`logs-card26m` (log class), `binary26m` (binary), `text26m` (text). Byte column:
every -p2 L9–12 cell differs (the #458 flip's deliberate size-for-wall trade —
the wave-9 receipt adjudicated those sizes against the vendor bar, not against
near-opt); every other cell is byte-identical.

### logs-card26m

| cell | MAIN ms | BASE ms | MAIN vs BASE | bytes |
|---|---|---|---|---|
| L09 -p4 | 215.6 → **222.8 ⌁ rerun** | 272.5 → **281.9** | **−20.95%** | identical |
| L10 -p4 | 160.0 → **160.5 ⌁ rerun** | 181.4 → **182.1** | **−11.86%** | identical |
| L11 -p4 | 281.9 → **291.2 ⌁ rerun** | 365.1 → **376.9** | **−22.75%** | identical |
| L12 -p4 | 767.0 → **781.8 ⌁ rerun** | 1542.8 → **1566.0** | **−50.08%** | identical |
| L09 -p2 | 76.2 | 506.5 | −84.96% | DIFFER (MAIN +1.60%) |
| L10 -p2 | 76.5 | 332.8 | −77.02% | DIFFER (MAIN −11.68%) |
| L11 -p2 | 76.7 | 677.5 | −88.68% | DIFFER (MAIN +1.71%) |
| L12 -p2 | 77.0 | 2860.3 | −97.31% | DIFFER (MAIN +1.71%) |
| L08 -p1 | 105.7 | 105.8 | −0.03% | identical |
| L08 -p2 | 60.0 | 60.0 | −0.15% | identical |
| L08 -p4 | 33.7 | 33.9 | −0.60% | identical |
| L01 -p4 | 12.5 | 12.6 | −0.73% | identical |
| L04 -p2 | 25.0 | 25.1 | −0.24% | identical |
| L06 -p4 | 26.9 → **24.1 ⌁ rerun** | 25.7 → **24.2** | **−0.45%** | identical |

### binary26m

| cell | MAIN ms | BASE ms | MAIN vs BASE | bytes |
|---|---|---|---|---|
| L09 -p4 | 215.9 → **214.8 ⌁ rerun** | 217.6 → **217.7** | **−1.36%** | identical |
| L10 -p4 | 205.1 | 198.6 | +3.30% | identical |
| L11 -p4 | 579.1 → **577.7 ⌁ rerun** | 959.7 → **957.0** | **−39.64%** | identical |
| L12 -p4 | 817.5 → **724.2 ⌁ rerun** | 1511.9 → **1304.8** | **−44.50%** | identical |
| L09 -p2 | 71.1 | 408.5 | −82.60% | DIFFER (MAIN −13.65%) |
| L10 -p2 | 70.0 | 343.8 | −79.62% | DIFFER (MAIN −13.65%) |
| L11 -p2 | 70.0 | 1791.5 | −96.09% | DIFFER (MAIN −0.71%) |
| L12 -p2 | 70.5 | 2533.0 | −97.22% | DIFFER (MAIN +2.45%) |
| L08 -p1 | 93.0 | 92.8 | +0.20% | identical |
| L08 -p2 | 52.6 | 52.9 | −0.67% | identical |
| L08 -p4 | 34.9 | 35.1 | −0.61% | identical |
| L01 -p4 | 15.2 | 15.8 | −3.54% | identical |
| L04 -p2 | 22.5 | 22.5 | −0.31% | identical |
| L06 -p4 | 13.9 | 13.8 | +0.89% | identical |

### text26m

(Synthetic corpus: text-1MB ×25. Big repetitive file; near-opt is the slow class
at -p1..-p4, so the flip band reads differently — see below.)

| cell | MAIN ms | BASE ms | MAIN vs BASE | bytes |
|---|---|---|---|---|
| L09 -p4 | 564.3 → **479.4 ⌁ rerun** | 526.0 → **482.1** | **−0.56%** | identical |
| L10 -p4 | 484.7 | 494.2 | −1.93% | identical |
| L11 -p4 | 589.8 | 601.0 | −1.88% | identical |
| L12 -p4 | 1002.9 | 1013.5 | −1.05% | identical |
| L09 -p2 | 305.8 | 900.2 | −66.03% | DIFFER (MAIN +3.30%) |
| L10 -p2 | 303.3 | 895.7 | −66.14% | DIFFER (MAIN +3.30%) |
| L11 -p2 | 307.8 | 1114.6 | −72.38% | DIFFER (MAIN +3.35%) |
| L12 -p2 | 308.0 | 1335.9 | −76.94% | DIFFER (MAIN +3.36%) |
| L08 -p1 | 540.7 | 541.9 | −0.22% | identical |
| L08 -p2 | 305.5 | 303.7 | +0.60% | identical |
| L08 -p4 | 173.2 | 165.3 | +4.79% | identical |
| L01 -p4 | 64.8 → **47.2 ⌁ rerun** | 59.2 → **47.1** | **+0.21%** | identical |
| L04 -p2 | 265.6 → **179.0 ⌁ rerun** | 221.7 → **179.0** | **−0.01%** | identical |
| L06 -p4 | 150.3 | 156.9 | −4.22% | identical |

## Part 4 — The named hypotheses, verdict by verdict

1. **L9/L11 at -p4, both arms (lever A/B claim −3..−4%)** — direction CONFIRMED, magnitude varies by class and coordinate, hypothesis-tier:
   - in16m: L9 −2.41% (rerun), L11 −7.21% (rerun). The claim's −3..−4% sits at/inside the L9 value; L11 measures beyond it.
   - on the member classes the effect is much larger than claimed: logs L9 −20.95%, L11 −22.75%; binary L9 −1.36% / L11 −39.64% / L12 −44.5%; text unchanged in the tie band. **These excesses are a new finding to carry** — mechanism is the Lever-A DP re-run removal, whose payoff scales with the DP share of the near-opt cost (higher effort levels, more passes), and with the content's match density (binary strong, text weak). Hypothesis-tier: the frozen box confirms; the local work here was suppressing the co-runner hypothesis (see reruns).
2. **L9–12 at -p2** — collapse CONFIRMED everywhere: in16m medians 111.7–114.8 ms vs BASE 604–1176 ms (x0.095–0.19, i.e. MAIN is 5.3–10.5x faster; the wave-9 receipt class x0.129 = 866–986 → 112–116 reproduces: MAIN's absolute medians sit exactly in the receipt band, BASE straddles it under load ~12–21), logs −77..−97%, binary −80..−97%, text −66..−77%. Sizes: the chunked-Lazy2-T2 streams differ from the BASE near-opt streams by +1.6…+3.4% on logs/in16m/text and −0.7…−13.7% on binary/logs (content-dependent: on the binary/logs classes the flip arm's stream is actually SMALLER; on text it pays ~+3.3%); every changed stream roundtrips through the independent python3-gzip decoder, and the per-member vendor bar is the wave-9 card's own (31/31 at-or-below `min(gzip -9, pigz -9)`, aggregate −2.4%) — not re-adjudicated here.
3. **L9–12 at -p4/T4 = UNCHANGED** — the *bytes* are unchanged everywhere (below), but the *wall* is NOT: the near-opt engine cells move −0.6%…−50% (idx: lever A/B acts inside the shared near-opt DP). So hypothesis 3 is REFUTED FOR WALL on this census at every T4 near-opt coordinate except in16m L9/L10 and binary L9 (inside ties); byte-identity is the part that holds. PR #457's own −3..−4% claim is the in16m L9/L11 -p4 slice of a bigger real effect.
   The byte half of the claim: PR #457's receipts quote the census — `in16m -9 -p2/-p4 = 327b083588efca21…, size 3,740,910 unchanged; tabular:L9:T4 = 232,057 untouched; 34/34 members byte-identical; 2/2 fingerprint_suite; 4/4 thread_byte_parity` — and this census re-asserts it directly on today's binaries: output sha256 IDENTICAL MAIN vs BASE at every non-flip cell, including in16m L9 -p4 = exactly **3,740,910 B** (the #448 anchor) and L12 -p4 = 3,727,885 B.
4. **L8 / L1–L7 any T unchanged (route untouched)** — all 21 such cells land inside ±5% on first pass; the two read peaks (in16m L2 -p4 +6.17%, L4 -p4 +5.04%) and all five outside-band member cells collapsed to ties at 14 reps in a quieter window (largest residual +1.42%, in16m L2 -p4). L6 -p4 logs: −0.45% rerun. **No route-untouched cell moved beyond noise. NO REGRESSION anywhere on the census.**

## Blocker verdict

**NO BLOCKERS.** The workflow demanded: any cell where MAIN is slower than BASE by >+5% must be rerun with more reps before reporting. Five such cells occurred in the 8-rep first pass (in16m L4-p4 +5.04%, in16m L2-p4 +6.17%, text26m L9-p4 +7.28%, text26m L1-p4 +9.37%, text26m L4-p2 +19.79%), plus logs-card26m L6-p4 (+4.95%) was rerun anyway as borderline. **All six reran at 14 reps (28 samples/arm) in a lower-load window (load1 med 9–15 vs 20–53 during the first pass) and all collapsed into ties**, largest residual +1.42% (in16m L2 -p4) — establishing that the first-pass outliers were co-runner contention (the byte-census wave agent `wC-bytes` was concurrently running its own compressions; load1 ranged up to 50 during this census), NOT #457/#458 regressions.

## Startup-cost footnote (task item 4)

1-byte input at `-9 -p4` (the `1byte:L9:T4` pin coordinate), interleaved 200-spawn
blocks (5 blocks, `median` of per-block per-spawn ms):

- Load ~33 window (byte-census co-runner active): A 4.786 vs B 4.126 ms/spawn,
  B/A 0.862, but block medians swung B/A 0.81⌁1.44 — unresolvable under contention.
- Quieter window (load1 ~6): A(MAIN) 2.767 vs B(BASE) 2.756 ms/spawn, B/A 0.996;
  block B/A band 0.996–1.019.

Verdict: the #457 mechanism pin moved `huffman_make_code_calls` 16→14 at
`1byte:L9:T4` (pin regen commit `b467653a`), which is sub-µs class; the
process-granularity wall probe cannot resolve it and measures a TIE today.
No startup wall regression.

## Materials (in this record dir)

- `README.md` — this file (tables + verdicts)
- `raw-timings.log.txt` — the append-only per-sample log (every rep with per-rep load, every cell byte-size + sha pass, roundtrip checks; 78 census cells + 19 rerun cells)
- `results.json` (8-rep primary + member cells), `rerun.json` (14-rep pass)
- `aa-calibration.log`, `startup-probe.log`, `provenance.txt`, `tables.md` (raw markdown tables)

| file | exact contents |
|---|---|
| `raw-timings.log.txt` | wall census raw samples + rerun pass + byte phase (78 cells × 8 reps × 4 samples interleaved + 19 cells × 14 reps × 4) |
| `results.json` / `rerun.json` | structured result rows per cell (medians, raw sample arrays, sizes, shas) |
| `aa-calibration.log.txt` | same-binary A/A receipts (4 cells, 6 reps) |
| `startup-probe.log.txt` | 1-byte -9 -p4 per-spawn bursts (5 blocks) |
| `provenance.txt` | binary shas + git positions |
| `tables.md` | generator's rendered tables (same content as above) |
