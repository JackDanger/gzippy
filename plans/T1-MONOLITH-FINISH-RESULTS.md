# T1-MONOLITH-FINISH — RESULTS (gated, Intel-only, NOT-YET-LAW)

Branch `t1-monolith-finish` @43d2ae21 (off origin/kernel-converge-A @39acc213).
Box: neurotic Intel i7-13700T LXC (un-freezable → `taskset -c 4` P-core pin +
interleaved randomized best-of-N + freq-stability watch; turbo on, gov=powersave;
cyc/byte is the frequency-invariant signal). /dev/null both arms. sha==zcat
verified every arm/corpus. Comparator `/usr/bin/igzip` 2.31.1 (ISA-L C/asm).
Build: `cargo build --release --no-default-features --features gzippy-native`,
`RUSTFLAGS=-C target-cpu=native`, grep-verified on the box checkout.

## VERDICT: PARTIAL (fulcrum optgate REFUSED the wall win — INSTRUCTION-ONLY)
The T1-MONOLITH-STREAMING path is byte-exact, fault-storm-free, sheds the
per-chunk scaffold INSTRUCTIONS (RESOLVED), and never regresses — but it does
NOT reach igzip parity and is NOT a cyc/byte wall win. The residual gz→igzip gap
is kernel cycle-efficiency (cyc/byte / IPC), NOT the per-chunk scaffold. Shipped
OPT-IN (`GZIPPY_STREAM_MONOLITH=1`); production T1 default stays thin-T1.

## fulcrum optgate (base=thin-T1, after=streaming-monolith, rg=igzip; N=16)
All four corpora: **⊘ WALL-WIN REFUSED [INSTRUCTION-ONLY]**.

| corpus | instr/byte base→after (Δ, spread) | cyc/byte base→after (Δ, spread) | gz/igzip gap closed |
|--------|-----------------------------------|----------------------------------|---------------------|
| silesia  | 13.417→13.374 (Δ+0.044, sp 0.005 RESOLVED) | 5.272→5.246 (Δ+0.026 < sp 0.085 UNRESOLVED) | 1.210→1.204 (2.9%) |
| nasa     | 4.450→4.427 (Δ+0.023, sp 0.004 RESOLVED)   | 2.065→2.051 (Δ+0.014 < sp 0.192 UNRESOLVED) | 1.310→1.301 (2.8%) |
| monorepo | 8.590→8.561 (Δ+0.030, sp 0.009 RESOLVED)   | 3.910→3.879 (Δ+0.031 < sp 0.395 UNRESOLVED) | 1.304→1.294 (3.4%) |
| squishy  | 14.770→14.722 (Δ+0.048, sp 0.004 RESOLVED) | 5.311→5.277 (Δ+0.034 < sp 0.050 UNRESOLVED) | 1.159→1.152 (4.6%) |

Interpretation (the optgate output, not prose inference): the streaming monolith
DOES remove instructions (the per-chunk scaffold), instr/byte improves above
spread on every corpus — but those instructions were cycle-cheap, so cyc/byte
does not move beyond spread ("the memcpy lesson"). Only 2.8–4.6% of the gz→igzip
cyc/byte gap closes. No clean-path regression.

## Wall (best-of-12, taskset cpu4, /dev/null, sha==zcat OK) + faults
| corpus | mono/igzip | thin/igzip | mono/thin | mono faults | thin faults | igzip faults |
|--------|-----------|-----------|-----------|-------------|-------------|--------------|
| silesia  | 1.199 | 1.207 | 0.994 | 3568 | 4367 | 610 |
| nasa     | 1.288 | 1.309 | 0.984 | 2840 | 3524 | 611 |
| monorepo | 1.295 | 1.321 | 0.980 | 2817 | 2987 | 608 |
| squishy  | 1.158 | 1.163 | 0.996 | 5178 | 6060 | 612 |

A/A |mono−mono2| = 0–3 ms ≪ Δ. mono is wall-faster than thin on all 4 (0.4–2.0%)
but every Δ is within spread → wall TIE vs thin (optgate-consistent).

NO FAULT-STORM (the prior full-ISIZE monolith was FALSIFIED at ~68,950 faults):
streaming monolith faults 2817–5178 are BELOW the thin-T1 baseline and ~100×
below the old monolith — the resident-pool streaming fixed the page-fault storm.

## vs PRE-REGISTERED FALSIFIER (plans/T1-MONOLITH-FINISH-FALSIFIER.md)
- byte-exact: **YES** (sha==zcat all 4 corpora, both arms; differential
  flate2+libdeflate in-commit).
- mono/igzip → ≤1.01: **NO** (1.16–1.30; 2.8–4.6% of cyc/byte gap closed).
- scaffold cyc collapsed: **PARTIAL** — instr/byte RESOLVED-shed, cyc/byte NOT.
- no fault-storm: **YES** (faults < thin, ~100× below old monolith).
- no T>1 regression: **YES** (strictly T1-gated; T>1 path untouched).
⇒ **PARTIAL**. Residual region (fulcrum-localized) = cyc/byte kernel
cycle-efficiency / IPC (igzip IPC 2.6–2.9 vs gz 2.2–2.8), NOT the per-chunk
scaffold. The next lever is the inner-kernel cycle cost, not architecture.

## Gates
Gate-0 PASS (sha==zcat all reps; A/A ≪ Δ; /dev/null both; non-inert routing —
MONOLITH_STREAM banner + counter; differential same-commit). Gate-1 PASS (N=12
wall / N=16 perf interleaved; cyc/byte freq-invariant). Gate-2 = fulcrum optgate
(the wall-win verdict): REFUSED INSTRUCTION-ONLY on all 4. Gate-4 PASS (path
banner, HEAD 43d2ae21). **Gate-3 OWED**: Intel-only → NOT-YET-LAW; AMD/aarch64
replication owed before any LAW claim.

## DECISION
Kept OPT-IN (`GZIPPY_STREAM_MONOLITH=1`), production T1 default = thin-T1
(byte-identical to before this cycle). Rationale: FULCRUM (the sole oracle)
REFUSED the wall win; a production default change must ride a gated WALL win. The
implementation is byte-exact, fault-storm-free, instruction-lighter, and
structurally closer to igzip — preserved for promotion if the cyc/byte residual
is later closed. Artifacts: plans/t1-monolith-finish-data/ (samples.csv,
optgate-*.json, build_optgate.py, RESULTS-raw.log); harness
scripts/bench/_t1_monolith_finish_guest.sh.
