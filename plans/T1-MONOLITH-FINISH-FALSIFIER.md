# T1-MONOLITH-FINISH — PRE-REGISTERED FALSIFIER

> **OUTCOME (2026-06-22, gated Intel, NOT-YET-LAW): PARTIAL.** byte-exact + no
> fault-storm + no regression + instr/byte RESOLVED-shed, but fulcrum optgate
> REFUSED the wall win (INSTRUCTION-ONLY) on all 4 corpora — cyc/byte did NOT
> improve beyond spread, only 2.8–4.6% of the gz→igzip gap closed, mono/igzip
> still 1.16–1.30. Residual = kernel cyc/byte, NOT the scaffold. Shipped OPT-IN
> (`GZIPPY_STREAM_MONOLITH=1`); production T1 default stays thin-T1. Full numbers:
> plans/T1-MONOLITH-FINISH-RESULTS.md.


Committed BEFORE building (CLAUDE.md Measurement PROTOCOL; pre-register hypothesis +
falsifier). Branch `t1-monolith-finish` off `origin/kernel-converge-A` @39acc213.

## HYPOTHESIS (unvalidated until gated)
The gated x86-T1 gzippy-native-vs-igzip gap (+21–30%, F-6ce077591bb5 / F-8bd982118f3d)
is the per-chunk **DRIVER SCAFFOLD** (the pure-Rust decode KERNEL is acquitted —
parity-or-faster than ISA-L). A **T1-MONOLITH-STREAMING** path that decodes the whole
single-member deflate body in ONE continuous serial pass (one `Block`, one marker ctx,
one `set_initial_window`; no per-chunk ChunkData lifecycle / window clone-reseed /
block-finder / WindowMap / threadpool / boundary recording) **while STREAMING output
through the small fixed-size resident pool buffer** (flush-all-but-trailing-32KiB +
memmove history to front + continue — NOT a full-ISIZE buffer) will shed the scaffold
cyc WITHOUT re-introducing the prior monolith's fault-storm, reaching igzip-T1 parity.

The prior monolith (project_x86_t1_monolith_2026_06_21) FAILED only because its single
full-ISIZE buffer (212 MB silesia) first-touched the whole output (~4× igzip faults).
The resident-pool streaming flush fixes exactly that.

## PRE-REGISTERED THRESHOLDS (frozen)
Measured via `fulcrum optgate` (base thin-T1 vs T1-monolith-streaming vs igzip),
Intel neurotic LXC, `taskset -c 4` P-core pin + freq-stability gate + interleaved
best-of-N≥9, /dev/null both arms, sha==zcat each rep, igzip self-test ≈1.0.
Corpora: silesia, nasa, monorepo, squishy.

- **FINISHED (WIN)** iff ALL hold:
  1. byte-exact: sha256(gz-native-T1-monolith output) == sha256(zcat) on every corpus;
     real-corpus differential (flate2 + libdeflate, multi-member + stored + multiple
     sizes) passes in the SAME commit.
  2. gz-native-T1-monolith / igzip drops from the ~1.21–1.30 baseline toward **≤ 1.01**
     (igzip parity) on the corpora (Δ ≫ inter-run spread).
  3. DRIVER_SCAFFOLD cyc collapsed (fulcrum region/excess) AND **no fault-storm**:
     minor-faults / RSS stay at or below the thin-T1 resident-pool baseline (NOT the
     prior monolith's ~4× inflation).
  4. NO T>1 regression (path is strictly T==1-gated; T4/T8 byte-identical & within noise).

- **PARTIAL** iff a real gated drop that does NOT reach ≤1.01: report the new ratio per
  corpus + re-run fulcrum to localize the residual region. Keep iff byte-exact + no
  fault-storm + no T>1 regression + a Δ≫spread improvement over the thin-T1 baseline.

- **FALSIFIED (REVERT)** iff ANY: not byte-exact on any corpus; fault-storm returns
  (faults/RSS inflated vs thin-T1 baseline); regresses past the thin-T1 baseline on any
  cell; or T>1 changes.

## SCOPE / GATES
Gate-0: sha==zcat + differential same-commit + A/A ≪ Δ + /dev/null both + igzip
self-test ≈1.0 + non-inert routing (MONOLITH_T1_RUNS / new streaming counter > 0,
GZIPPY_DEBUG path). Gate-1: N≥9 interleaved freq-stable, Δ vs spread. Gate-2: fulcrum
optgate is the wall-win verdict (refuses instr-only / loaded-window / byte-mismatch /
sub-spread / single-arch). Gate-4: GZIPPY_DEBUG path assertion + HEAD sha.
**Gate-3 OWED**: Intel-only → NOT-YET-LAW; AMD/aarch64 replication owed for LAW.
