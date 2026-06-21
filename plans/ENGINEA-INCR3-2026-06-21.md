# ENGINE-A INCREMENT-3 — literal per-symbol excess: INSTRUMENT-FIRST verdict — 2026-06-21

**Branch:** kernel-converge-A (HEAD `1c24aacf`). **Scope:** macOS aarch64 (Apple
Silicon, quiet). **Build:** `--no-default-features --features gzippy-native`,
path=ParallelSM, -p1. **Instrument:** `scripts/bench/standing/enginea_converge_mac.py`
(Gate-0: byte-exact all arms == `gzip -d` + same /dev/null sink + ParallelSM routing;
Gate-1: best-of-N, instr/B load-immune [spread 0.04–0.7%], cyc/B HYPOTHESIS-tier).
Frontier comparator = libdeflate-gunzip. **Tier:** gated HYPOTHESIS. **NOT-YET-LAW**
(single-arch; AMD + Intel-engine-A-asm-off owed).

## Mandate
Close the +4.21 instr/symbol literal excess (B-1/B-2 top lever) by converging engine
A's literal fastloop (`decode_huffman_fastloop_bounded`, consume_first_decode.rs:1302,
the production aarch64 `-p1` clean-contig path) toward `decompress_template.h`. The
brief was explicit: **INSTRUMENT FIRST** — the 8-deep packed lookahead may be a
deliberate net-positive trade; measure WHICH part of the excess is removable before
blindly porting 8→3 / conditional→unconditional.

## STEP 1 — causal decomposition of the literal excess (4 gated perturbations)

Objdump was not usable: the fastloop inlines fully (release strips to 113 symbols),
and per CLAUDE.md a disassembly read is only HYPOTHESIS-tier anyway. The **verdict is
the causal perturbation** — change one structural element, measure the deterministic
instr/B response on the per-symbol-dominated corpus (decomp_literal) + real anchors.

All numbers: instr/B, /dev/null, -p1, mac aarch64. (Each perturbation moved instr/B
well outside spread, which also CONFIRMS `decode_huffman_fastloop_bounded` is the live
`-p1` decode path.)

| variant (vs decompress_template.h) | literal i/B | silesia i/B | monorepo i/B | verdict |
|---|---|---|---|---|
| **HEAD** — 8-deep packed lookahead, conditional refills | **20.47** | **9.63** | **6.74** | baseline |
| converge #2 DEPTH: cap lookahead 8→3 (libdeflate caps at 3) | 20.81 | 10.09 | 6.89 | **REGRESS** +0.35/+0.46/+0.15 |
| converge #3 REFILL: top-of-loop refill conditional→unconditional | 21.42 | 9.98 | 6.97 | **REGRESS** +0.95/+0.35/+0.23 |
| converge #2 WRITE: depth-8 packed-u64 → individual `*out++` stores | 20.49 | 9.63 | 6.74 | **TIE** (Δ ≤ spread) |

cyc/B (wall proxy, HYPOTHESIS-tier) tracked instr in sign on the two regressions
(cap-at-3 7.27→7.20 noise; uncond-refill 7.27→7.67 worse; individual 7.27→7.21 noise).

### Verdict on the three ranked design divergences
- **#2 lookahead DEPTH (8 vs libdeflate's 3): 8-deep is a NET INSTRUCTION WIN.**
  Capping to libdeflate's 3 REGRESSES instr/B on every corpus (+0.35 literal, +0.46
  silesia, Δ ≫ spread). The deeper batch amortizes the loop/refill/branch overhead
  across more literals. **DO NOT revert — this is exactly the deliberate net-positive
  trade the brief warned about.**
- **#2 WRITE strategy (packed-u64 store vs libdeflate's individual `*out_next++`):
  TIE.** Individual byte stores at depth-8 are byte-identical and instr/cyc-neutral
  (Δ ≤ 0.05% spread) — the compiler lowers both equivalently. Converging it is
  faithful-but-pointless; not landed (large diff × 2 functions + reference model for
  zero benefit + needless byte-exact risk).
- **#3 refill CADENCE (conditional skip-when-sufficient vs unconditional): the
  CONDITIONAL cadence is a WIN.** Forcing the top-of-loop refill unconditional
  REGRESSES +0.95 literal instr/B. (Note: libdeflate is *not* purely unconditional
  either — its offset refill is `if (unlikely(bitsleft < …)) REFILL`; engine A's
  skip-when-sufficient cadence is faithful in spirit and already efficient.)

## STEP 2 — convergence
**NONE LANDED.** The instrument decided: no faithful convergence of the literal path
toward `decompress_template.h` reduces instr/B — the two candidate convergences (#2
depth, #3 refill) REGRESS, and the third (#2 write-strategy) is a zero-benefit TIE.
The structural-convergence budget for engine A's literal path was **already captured**
by increments 1–2 (TABLE_BITS 12→11 / 9→8, banked `ba282489`/`5b04e1c5`).

**The residual +4 instr/symbol literal excess is NOT a removable divergence from the
blueprint** — engine A's literal fastloop is at-or-better than libdeflate's on the
structural axes (depth, write-strategy, refill cadence). The excess is intrinsic to
the Rust port: per-symbol entry-extraction codegen, bounds/`FlatFastloopExit`
resumable-contract bookkeeping, and register-allocation differences vs the C kernel.
Closing it (if it pays at the wall at all — cyc/B gap is ~1.13 on literal, narrower
than instr 1.25) would require **open-territory inner-kernel innovation**, NOT faithful
convergence; and the two naive innovations tried here both regressed.

## Current HEAD gap (best-of-15, /dev/null, gz native vs libdeflate-gunzip, mac aarch64)
| corpus | gz i/B | ld i/B | i-ratio | gz cyc/B | ld cyc/B | c-ratio |
|---|---|---|---|---|---|---|
| decomp_literal | 20.47 | 16.41 | 1.247 | 7.27 | 6.43 | 1.130 |
| silesia | 9.62 | 7.64 | 1.260 | 4.04 | 3.28 | 1.230 |
| monorepo | 6.74 | 5.49 | 1.227 | 2.80 | 2.17 | 1.290 |
| nasa | 3.37 | 3.19 | 1.057 | 1.66 | 1.29 | 1.289 |
| decomp_backref | 1.83 | 1.71 | 1.073 | 0.95 | 0.60 | 1.595 |

## Gates
- Byte-exact: every perturbation's gz sha == `gzip -d` sha on all 5 corpora at -p1
  (the instrument's Gate-0 LOUD-PASSED on every build, incl. the individual-write
  variant — which also proves the edits were on the live path). HEAD unchanged →
  the banked 943-lib-test / differential pass at `1c24aacf` stands.
- Routing: GZIPPY_DEBUG=1 → path=ParallelSM.
- Significance: instr/B spread 0.04–0.7% (load-immune); the two REGRESS deltas
  (+0.35, +0.95 literal) are Δ ≫ spread; the WRITE TIE is Δ ≤ spread on silesia/monorepo.

## What remains
- **No cheap faithful convergence left for engine A's literal path.** Further gain
  needs open-territory kernel innovation (authorized by CLAUDE.md, but the two naive
  attempts regressed — needs a genuinely better scheme, e.g. SIMD literal scatter or
  a codegen-shaped extraction, gated hard against decomp_literal + the resumable tail).
- Cross-ISA LAW: Intel engine-A-asm-off + AMD/Zen2 replication owed (this is a
  single-arch macOS-aarch64 verdict).
