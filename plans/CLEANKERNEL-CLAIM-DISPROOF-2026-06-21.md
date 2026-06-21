# CLEANKERNEL-CLAIM-DISPROOF — is "the ONE real gz-vs-rg gap is the x86 ISA-L clean INNER-KERNEL" true?

**Date:** 2026-06-21  **Branch:** kernel-converge-A  **gz git (decode path):** dc3158e6
(mac binary) ≡ 9df14077 (Intel binary — verified: `git diff 9df14077..dc3158e6 -- src/`
is EMPTY, byte-identical decode path).
**Boxes:** macOS M1 Pro (quiet, deterministic) + Intel i7-13700T LXC (LOADED, load 5-7).
**Stamp:** NOT-YET-LAW — macOS-aarch64 + Intel-x86 only; AMD/Zen2 owed; the
ISA-L-vs-gz copy-vs-decode-loop split is OWED (rg --verbose folds them).

## CLAIM UNDER TEST (actively attacked, NOT defended)
> "The ONE real multi-thread gz-vs-rapidgzip gap is the x86 ISA-L clean INNER-KERNEL"
> (silesia-T4 +16% on Intel = gz's Huffman clean inner-kernel losing to rg's ISA-L;
> tracks the clean fraction).

Three suspected weaknesses attacked: **W1** "inner-KERNEL" too specific (the support
bucket = WHOLE clean-path Σdecode, which also holds backref COPY, bit reader, CRC);
**W2** "the ONE gap" over-attributes (gz markers +79% heavier abs on silesia);
**W3** under-validated (single-arch, single loaded-LXC).

## Gate-0 (self-validation)
- **gz** path=ParallelSM, build-flavor=parallel-sm+pure (FFI off) at p1/p4 (asserted
  both boxes); byte-EXACT gz==gzip==zcat all arms; /dev/null sink both arms.
- **Removal/inject oracles PROVEN non-inert:** NODECODE replay byte-EXACT (sha==ref) with
  `hits=2807 sil / 7191 squishy, misses=0` (replaces ALL Huffman decode + bit reads +
  LUT builds with a recorded symbol-stream replay whose stores go through the production
  kernel); SLOW_DECODE / SLOW_STORE byte-transparent (sha==ref) and move instructions
  6.5-20.3 G (fire massively). NOCRC byte-exact, removes 0.0-0.5 M instr.
- **Oracle CAVEAT (documented, bounds the read):** NOSTORE is NON-isolating in this
  build (it gates OFF the fast SIMD loop → routes the careful loop → instr goes UP
  2.10→3.80 G at -p1; discarded). NODECODE's replay loop SUBSTITUTES per-op overhead
  for the per-symbol decode loop, so `base−nodecode` isolates the Huffman-decode
  ARITHMETIC (LUT lookup + bit extraction) SPECIFICALLY, not the whole decode call —
  which is exactly the "inner kernel" the claim names. contig_prof is x86-rdtsc-only
  (inert on mac) and classes by iteration-type not decode-vs-copy → not used.

## W1 — gz CLEAN-PATH (-p1) COMPONENT DECOMPOSITION (gz's OWN structure; NO ISA-L)
Deterministic instr (mac, /usr/bin/time -l, N=7, spread ≤1.8%) + wall (Intel best-of-11):

```
                        | mac silesia | mac squishy | Intel silesia
base clean instr/B      |    9.908    |    9.532    |   (wall 843.8 ms)
Huffman-decode+bitread  |    2.55%    |    5.83%    |   (NODECODE Δwall -1.9%)
  (base−nodecode, INSTR)|            decode ARITHMETIC share
CRC (base−nocrc)        |    0.02%    |   -0.01%    |   near-0
copy/store+loop+coord   |   97.45%    |   94.17%    |   ~102%
decode-CYCLE share (mac)|   12.35%    |   27.13%    |   ~0% (Intel wall)
```

**Finding W1-a (CAUSAL removal, byte-exact, non-inert):** removing the ENTIRE Huffman
inner-kernel ARITHMETIC (decode + bit-extraction) changes the clean-path wall by only
**−1.9% (Intel) to +12.4%/+27.1% cyc (mac sil/squishy)** and **2.5-5.8% of instructions**
on both arches. The Huffman inner-kernel is therefore **NOT the dominant clean-path
cost** — the **per-symbol COPY/STORE + loop** carries 73-100% of the clean wall and
94-97% of clean instructions. **CRC is near-free (≤0.02%)** — ruled out as a component.

**Finding W1-b (causal slow-inject, Gate-2, byte-transparent):** the per-event
copy-vs-decode WALL-criticality FLIPS SIGN across arches — mac store/decode ratio
**0.74-0.76** (store/copy site MORE wall-critical) but Intel **0.69** (decode-loop site
MORE wall-critical). The slow-inject is event-count-confounded (decode events ≈1.2-1.3×
store events), so per-event copy and loop are ~comparable, arch-dependent. ⇒ **the
copy-vs-loop split is arch-dependent and NOT cleanly isolable here** — do NOT
over-correct to "it is purely the COPY."

## W2 — marker-portion vs clean-portion contribution to the silesia gz-vs-rg gap
From the marker disproof (MARKER-TAX-DISPROOF-2026-06-21, reconfirmed): the gz/rg gap
**ANTI-correlates** with marker fraction — Intel wall gz/rg@T4 vs marker%: r=−0.999
(nasa 89.8% mkr → 0.960 gz WINS; monorepo 80.9% → 1.002 TIE; silesia 34.5% → 1.161 gz
loses WORST). A linear fit (clean%, gap%) through {nasa,monorepo,silesia} gives slope
≈ +0.36 gap-%/clean-% and a pure-marker intercept of ≈ **−7.7%** (gz would WIN a 100%-
marker stream) vs **+28.6%** at 100% clean. **⇒ the marker portion CONTRIBUTES NEGATIVELY
to the gap (gz is relatively BETTER on markers); silesia's +16% loss is concentrated in
the CLEAN portion.** rg --verbose silesia P4 confirms 65.5% non-marker / 34.5% markers,
copy 0.056 s, apply-window(marker resolve) 0.094 s. **W2 attack FAILS — the marker
portion is not a material contributor to the loss; the clean attribution is correct.**

## W3 — silesia-T4 +16% REPRODUCIBILITY on a LOADED Intel box
Interleaved best-of-15, /dev/null both arms, sha==zcat both arms, **rg-vs-rg A/A
self-test:**
```
gz  best=494.7 ms med=513.4  spr=7.7%
rgA best=431.0 ms med=439.9  spr=5.1%   rgB best=429.8 ms  spr=5.4%
A/A rg(best)/rg(best) = 1.0028  (|AA-1|=0.28%)   A/A med = 1.0056
gz/rg = 1.1479 (best) / 1.1669 (median)  → gz SLOWER 14.8-16.7%
```
**Finding W3:** the +16% silesia-T4 loss **REPRODUCES** (gz/rg 1.148 best / 1.167 med ≈
the STANDING-MATRIX 1.161). The A/A self-test ratio is **1.003** (box load is symmetric/
stable — NOT manufacturing a phantom sign-flip), and the gap **Δ (14.8-16.7%) >> inter-run
spread (5-7.7%)** → significant, NOT a contention artifact. **W3 attack FAILS — the +16%
is real and reproduces** (the harness flags "UNTRUSTED" only because spreads slightly
exceed a 5% tol; the A/A=1.003 + Δ≫spread make the verdict sound). Still single-arch
Intel — **AMD owed for LAW.**

## VERDICT — NEEDS-REFINEMENT (claim's CORE survives; its LABEL is too specific)
- **SURVIVES (W2+W3):** the gap is real, reproduces (+16% silesia-T4), tracks the CLEAN
  fraction (anti-correlates with markers), and is x86-specific (on aarch64 gz WINS
  silesia-T4 — rg has no ISA-L). The "clean-path / x86-ISA-L / the ONE loss" framing holds.
- **DISPROVEN / REFINED (W1):** "**inner-KERNEL**" (the Huffman decode loop) is **too
  specific** — gz's clean-path cost is **NOT** Huffman-decode-arithmetic-bound (that is
  ≤12% of the clean wall, ≤6% of instructions on BOTH arches; CRC ≤0.02%). The clean
  wall is dominated by the **per-symbol COPY/STORE + decode-LOOP throughput**. So the
  gap is the broader **clean per-symbol path**, not the Huffman kernel arithmetic.

### CORRECTED CLAIM
> The ONE real multi-thread gz-vs-rg gap (silesia-T4 +16%, reproduced) is the **x86
> clean-PATH per-symbol throughput** — copy/store + decode-LOOP — where rg uses ISA-L,
> **NOT specifically the Huffman inner-kernel arithmetic** (≤12% of the clean wall).
> WHICH ISA-L technique closes it (wider AVX2 copy vs multi-symbol decode-loop packing)
> is **OWED**: rg --verbose folds decode+backref-copy into `decodeBlock`, and the mac/
> Intel per-event copy-vs-loop criticality flips sign — an ISA-L-vs-gz isolating
> measurement is required before naming the sub-component.

## Owed (NOT-YET-LAW)
- AMD/Zen2 replication of both the +16% gap and the clean-path component ranking.
- A clean ISA-L-vs-gz x86 isolation of copy vs decode-loop (rg verbose folds them; needs
  a per-phase counter inside ISA-L or a gz-side copy-only/decode-loop-only perturbation
  compared to an ISA-L copy/decode micro-measurement).
- A quiet/frozen Intel re-run (this one was on a loaded box; the gap reproduced and the
  A/A self-test passed at 1.003, but a frozen box would tighten the spread).

## Reproduce
```
# W1 mac component decomposition (instr + cyc, N=7)
python3 scripts/bench/standing/cleankernel_decomp_mac.py
# W1 Intel gz clean-path decomposition (wall best-of-N) + rg --verbose
scp scripts/bench/standing/cleankernel_w1_intel.py guest:/dev/shm/ck_w1.py && python3 /dev/shm/ck_w1.py
# W3 Intel silesia-T4 gz/rg reproducibility + rg-vs-rg A/A
scp scripts/bench/standing/_cleankernel_silt4_guest.sh guest:/dev/shm && bash /dev/shm/ck_silt4.sh
```
Artifacts: `artifacts/cleankernel/mac_decomp.txt`.
