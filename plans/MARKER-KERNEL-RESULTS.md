# MARKER-KERNEL codegen lever — RESULTS

**Date:** 2026-06-22. **Branch:** `marker-kernel` (off `kernel-converge-A` @ `671c5752` +
amd-t2t4-locate). **Front:** AMD/Zen2 T>=2 vs rapidgzip, window-absent MARKER decode kernel
(inner Huffman, CLAUDE.md OPEN TERRITORY). **Falsifier:** `plans/MARKER-KERNEL-FALSIFIER.md`
(pre-registered + pushed before any number). Graded against CLAUDE.md gates.

## Boxes / Gate-0 / Gate-4
- **AMD solvency** EPYC 7282 Zen2 `root@10.0.2.240` (PRIMARY — the gap lives here). FROZEN
  gov=performance/boost=0 (NOT no_turbo), bounded auto-restore watchdog; measurement cores
  `8,10,12,14` taskset-pinned away from the roaming `llama-completion` (UNTOUCHED). **RESTORED
  clean at exit: boost=1, gov=ondemand, watchdog killed.**
- **Intel neurotic** i7-13700T LXC `ssh -J 10.0.0.100 root@10.30.0.199` (unfreezable):
  taskset-pinned distinct P-cores + interleaved best-of-15. Not mutated.
- build `--no-default-features --features gzippy-native`, `-C target-cpu=native`,
  flavor=`parallel-sm+pure`, path=ParallelSM. **Baseline** = `671c5752` (`/root/gz-base-bin`
  AMD, `/root/mk-base-bin` Intel); **MINE** = the lever commit. rg comparator = native ELF 0.16.0.
- All arms `sha == zcat` on silesia/monorepo/nasa/squishy at T1 AND T4 (AMD + Intel). In-tree
  multi-oracle differential (flate2 + libdeflate + zlib-ng) PASS incl. the NEW real-silesia
  parallel-SM marker-path test at 4 chunk granularities (256/512/1024/4096 KiB). /dev/null both arms.

---

## LEVER 1 — `#[inline(always)]` on `HuffmanCodingShortBitsCached::decode` → **PARTIAL (KEEP)**

**Change (byte-exact, codegen-only):** force-inline the hot decode wrapper. Eliminates the
per-symbol `call HuffmanCodingShortBitsCached::decode` in the marker careful-tail dist decode
(`marker_inflate.rs:2825`) + the fast-loop kill-switch arm. rg's `Block::read` inlines its decode
(vendor gzip/deflate.hpp:336,1580-1590).

**Gate-0 non-inert (objdump, debuginfo build):** BASELINE had **8** `call …::decode` sites; MINE
has **0** — all 8 remaining calls now target the cold `decode_long` cache-miss path (the hot
peek+LUT+seek body is inlined). Codegen provably changed.

**Gate-2 AMD silesia-T4 (WORK-bound cell, frozen, N=11–13 interleaved, min/best, A/A-gated):**
reproduced across 3 freeze sessions; BASELINE cyc rock-stable 2495.3M every session (anchor).
- cyc: BASE 2495.3M → **MINE 2482.8M = −0.50%** (A/A spread 0.18% ⇒ real, ~3× spread).
- wall: BASE 259.5ms → **MINE 257.0ms = −0.96%** (A/A spread 0.28% ⇒ real this session).
- gz/rg: **wall 1.065 → 1.055**, cyc 1.092 → 1.087. (rg has larger session-to-session wall
  noise; the BASE→MINE delta is the stable quantity.)

**AMD monorepo-T2 (serialization-bound):** cyc −0.5–0.8% (real vs A/A 0.3%); wall flat — expected
(T2 is serialization-bound per the locate, a cyc drop does not move the wall).

**Gate-3 Intel no-regress (neurotic, best-of-15, distinct P-cores):** MINE is **TIE-or-marginally-
faster than BASE everywhere** — silesia T1/T2/T4/T8 and monorepo T2/T4. (MINE even BEATS rg at
silesia-T8 0.965 and monorepo-T4 0.984.) No regression on any cell.

**VERDICT vs falsifier — PARTIAL.** Byte-exact + objdump-confirmed codegen change + a real,
reproducible cyc/B drop (mechanism: per-symbol dist `call` eliminated) + a real ~1% AMD-T4 wall
drop, with NO regression on Intel T1/T2/T4/T8. But it does NOT reach gz/rg <= 1.01 (AMD-T4 stays
~1.055). **KEPT** (byte-exact + real measured win). Remaining bucket: the marker FAST loop (59.8%
of marker decode, already uses the dist table) + the asm clean kernel.

---

## LEVER 2 — careful-tail dist via `DistTable` LUT → **FALSIFIED-per-lever (REVERTED)**

**Change:** route the marker careful-tail distance through the same single-lookup `DistTable` the
fast loop uses (banked P3.1 84.8→61.5 cyc/backref mechanism). Byte-exact (sha==zcat + differential
pass).

**Gate-2 AMD (frozen, N=11):** did NOT shrink the careful tail — it **ADDED** cycles:
- silesia-T4: MINE cyc 2494M back at BASE 2495M (erased Lever 1's 12–16M drop).
- monorepo-T2: cyc gain shrank 5M→2M; slight wall regression.

**Mechanism (HYPOTHESIS):** on this data the careful-tail dist codes are short and already hit
`dist_hc`'s 12-bit cache fast-path; the DistTable path adds a subtable branch + `consume_entry` +
the `dist_valid` gating, and the larger careful-tail body disrupts hot-loop inlining.

**VERDICT — FALSIFIED-per-lever** (no cyc drop beyond noise; a cyc increase that cancels Lever 1;
slight wall regression). **REVERTED** (`git revert`, kept as an honest FALSIFY in history).

---

## LEVER 3 — clean-kernel per-symbol classification branch density → NOT ATTEMPTED (next, HYPOTHESIS-tier)

The locate's annotate (HYPOTHESIS-tier — no per-region perturbation) put ~36% of the clean asm
`run_contig` cycles in per-symbol classification branches (`cmp $0x100`, `cmp $0xf0`, refill mask)
and flagged in the marker fast loop a `test $0x2000000` (LARGE_FLAG, 9.5%) + marker-tag `or $0x38`.
These sit in the asm clean kernel (already efficient at 4.7 cyc/B, near rg) and the SHARED LUT
decode primitive (`lut_huffman::decode_prefilled`) used by both clean and marker paths and by the
separate T1-native front. They are not byte-exact-trivial codegen flips; restructuring them is deep
hand-asm / shared-primitive work with high backfire risk (Lever 2 demonstrated how a plausible
byte-exact lever cancels itself). Left as the next ranked target for a dedicated cycle with its own
pre-registered falsifier + per-region Gate-2 perturbation (not an annotate read).

---

## Net standing
- Marker-kernel cyc-excess closed by Lever 1: AMD-T4 ~0.5% cyc / ~1% wall (gz/rg 1.065→1.055),
  byte-exact, no Intel regression. **Banked.**
- The residual AMD-T4 gz/rg ~1.05 is dominated by the marker FAST loop (the bigger bucket, already
  table-driven) and is largely structural (the u16 marker machinery is faithful rg structure that
  is inherently ~2.5× the clean path; rg pays the same). Two of the three annotate-named sub-targets
  did not pay (Lever 2 falsified) or are deep/shared (Lever 3).

## Rig artifacts
AMD `/root/mk_confirm.sh` (3-arm BASE/MINE/RG interleaved best-of-N, duration+cycles, A/A arms) +
`/root/amd_freeze.sh`. Intel `/root/mk_confirm_intel.sh` (3-arm, taskset, best-of-15). Baselines
`/root/gz-base-bin` (AMD), `/root/mk-base-bin` (Intel); debuginfo `/root/gz-base-dbg`,
`/root/gz-head-dbg/release/gzippy`.
