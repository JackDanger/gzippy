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

## COPY-PATH lever — gated FALSIFIED (2026-06-23), confirming the floor exhaustively
perf annotate (AMD Zen2, root bare-metal) located run_contig=80.8% of cycles, with
the backref COPY path ~18% of cyc: the `cmp len,40` dispatch (9% cyc, ~10% of
branch-misses = real mispredict) + the 3× MOVDQU burst (48B written for ~6B mean
match). The recurrence (litlen table load, ~15% cyc) is INTRINSIC (igzip identical;
the serial-Huffman dependency L1_load→consume→index→L1_load).
Two copy convergence attempts, BOTH gated-FALSIFIED:
- cursor #1 (route ≤40 through igzip's cmovg+MOVDQU loop): REVERT (instr UP, monorepo
  cyc +8.2%) — the sub;jle trip-count loop mispredicts worse than the flat burst.
- MINIMAL-WRITE (1 MOVDQU for len≤16 + well-predicted cmp;jbe gates, write 16/32/48
  not always 48): REGRESS (Intel silesia +1.5%, nasa +6.5%, monorepo +4.8% paired;
  byte-exact). The 2 length branches mispredict (varied match lengths) and cost MORE
  than the saved store bandwidth.
=> CONCLUSION: the copy path's cost is BRANCHES, not store bandwidth; the flat
branchless 48B burst is gz's LOCAL OPTIMUM (both length-control variants regress).
Combined with NRK (anchor cycle-slack), dist-rewrite (refuted), literal-core (parity),
EVERY named structural T1 lever is gated-exhausted. The ~9% Intel / ~4-7% AMD
T1-vs-igzip gap is the genuine pure-Rust-vs-ISA-L codegen/schedule FLOOR. The only
untried path is a full-loop igzip-objdump-scheduled MONOLITH (cursor MOVE 2) — and
NRK already showed whole-loop schedule changes regress on AMD (high-risk).

## LITLEN AMORTIZATION — gated INERT (2026-06-23), last optimization lever falsified
Built litlen LUT amortization (memcmp cached lens, skip rebuild on match; byte-safe,
default-on + GZIPPY_LITLEN_AMORT kill-switch + reuse counters). Measured reuse-rate
(GZIPPY_LITLEN_AMORT_STATS, T4): silesia 1/2816, nasa 0/364, monorepo 0/180,
model 0/6472 = ~0% EVERYWHERE. gzip/pigz re-optimize the Huffman tree per block =>
consecutive blocks never share litlen lengths. Amortization only helps repeated-
header streams (BGZF/concatenated/low-entropy), NONE in the benchmark. Default-on it
would only add a per-block memcmp for 0 benefit. REVERTED.

## FINAL EXHAUSTIVE VERDICT — T1-vs-igzip is a confirmed pure-Rust-asm!-vs-hand-asm floor
Every optimization lever gated-falsified or proven intrinsic:
  resumable anchor (NRK): cycle-slack (Intel TIE / AMD regress)
  dist-rewrite: refuted (already optimal)
  copy-shape #1 (igzip loop): regress (trip-count branch)
  copy minimal-write: regress (length branches > store savings)
  table-build: intrinsic (igzip pays same, inlined) — TBUILD_MULT 0.17 cyc/B but shared
  CRC: equal to igzip (4.5% both)
  litlen-amortization: inert (reuse ~0)
  literal core: at igzip parity (37 vs 38 instr/iter, gz leaner discriminator)
  recurrence: intrinsic serial-Huffman litlen-load dependency (~12 cyc/packet)
ROOT CAUSE (objdump-proven): Rust `asm!` lets LLVM pick registers/schedule (NRK
removing 2 ops reshuffled ALL loop registers); ISA-L ships HAND-assembled .s with
hand-tuned register allocation + scheduling. The ~9% Intel / ~4-7% AMD residual on
an instruction-parity loop is the `asm!`-vs-hand-asm codegen delta — NOT optimization-
closable. Corroborates Fulcrum excess recoverable-budget=0.
ONLY remaining path to MEET igzip at T1: a HAND-WRITTEN .s assembly decode kernel
(our own, no C FFI — the branch's "reimplement-isa-l" namesake), giving igzip-grade
register/schedule control that Rust `asm!` structurally cannot. Multi-session build;
NRK evidence flags AMD schedule-regress risk; to MEET (not lose to) ISA-L's hand-asm
likely requires transliterating its decode_huffman_code_block_stateless schedule.
