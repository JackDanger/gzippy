# T1 Non-Resumable Kernel (NRK) — gated cross-arch VERDICT

**2026-06-23. User-directed heroic build: a T1-dedicated non-resumable decode
kernel (`run_contig_t1_nrk`, env `GZIPPY_T1_NRK=1`) to test whether shedding the
resumable contract converts the ~2 instr/B T1-vs-igzip surplus to a cyc/B win.**

## What was built (byte-exact, cursor-reviewed)
`run_contig_t1_nrk` (asm_kernel.rs) = a fork of `run_contig` that DELETES the
per-iteration p0/d0 un-consume anchor + the `85:` re-read + `EXIT_RECLASS`, and
exits EOB CONSUMED (igzip `_04` shape, `EXIT_NRK_EOB`). Dispatched in
`decode_clean_into_contig` (T1-only, native-only, opt-in); `run_contig` (resumable)
UNTOUCHED for T>1 + the marker path. cursor-agent reviewed the asm fork: no
correctness bugs (register liveness, EOB cnt==1/consumed cursor, subtable inline,
boundary all PASS). Byte-exact: sha==zcat on silesia/nasa/monorepo, BOTH arches.

## Gated A/B (same binary, env toggle; N=15 paired-interleaved, /dev/null, single-core)
| arch | corpus | Δinstr/B | Δcyc/B (paired) | verdict |
|------|--------|----------|-----------------|---------|
| Intel | silesia  | -4.07% | -0.72% [-1.65,+1.25] | TIE |
| Intel | nasa     | -3.48% | -1.33% [-3.00,-0.13] | **WIN(paired)** |
| Intel | monorepo | -3.75% | -0.83% [-3.18,+0.56] | TIE |
| AMD/Zen2 | silesia  | -4.05% | +0.80% [+0.30,+1.02] | **REGRESS(paired)** |
| AMD/Zen2 | nasa     | -3.61% | +1.29% [+0.56,+2.13] | **REGRESS(paired)** |
| AMD/Zen2 | monorepo | -3.69% | +0.42% [-0.21,+1.22] | TIE |

## VERDICT: FALSIFIED as a wall lever (cross-arch)
NRK cuts ~4% instructions (~0.54 instr/B — matches NIGHT32's -0.607, confirming
non-inert) but cyc/B is OPPOSITE-SIGNED across arches: marginally faster on Intel
(nasa only), marginally SLOWER on AMD/Zen2. The per-iteration resumable anchor is
genuinely IPC-SHADOWED (absorbed by spare execution slots); shedding it perturbs the
schedule without moving the wall. This CONFIRMS NIGHT32's cycle-slack result and now
EXTENDS it cross-arch. The ~2 instr/B T1-vs-igzip surplus does NOT convert to cyc/B
via resumable-contract removal — the gap is the intrinsic pure-Rust-vs-ISA-L codegen
floor. Default routing NRK would REGRESS AMD, so it must NOT be default-routed.

## Disposition
NRK kept as env-gated opt-in (GZIPPY_T1_NRK=1), NOT default (would regress AMD).
This is the deepest structural attack on the igzip-T1 floor; its falsification is
the definitive answer that the floor is codegen-intrinsic, not contract-tax.
