# MARKER-KERNEL codegen lever — PRE-REGISTERED FALSIFIER

**Date:** 2026-06-22. **Branch:** `marker-kernel` (off `origin/kernel-converge-A` @ `671c5752`
+ amd-t2t4-locate `b2df8e3f`). **Front:** AMD/Zen2 T>=2 vs rapidgzip, window-absent MARKER
decode kernel (inner Huffman, CLAUDE.md OPEN TERRITORY). **Primary box:** solvency AMD EPYC
7282 Zen2 `root@REDACTED_IP`; **secondary:** neurotic Intel `ssh -J REDACTED_IP root@REDACTED_IP`.

## Gated starting point (from project_amd_t2t4_locate_2026_06_22, Gate-2 STRONG)
- AMD-T4 silesia: gz/rg wall 1.079, cyc 1.079 (WORK-bound). Excess is INSIDE gz's window-absent
  decode kernel (61% of cycles); markers/pipeline/CRC ruled out.
- Window-absent MARKER bucket: mfast 59.8% (63.8 cyc/ev) + careful-tail 40.2% (96.4 cyc/ev);
  whole marker loop 11.7 cyc/B = 2.5x the clean kernel's 4.7 cyc/B.
- Annotate (HYPOTHESIS-tier): marker path emits a non-inlined `call HuffmanCodingShortBitsCached::decode`
  (6.2%); rg's Block::read inlines its decode (vendor deflate.hpp:336/1580-1590).

## Levers (each a SEPARATE gated change; AB is the verdict, not the annotate)
- **Lever 1 — INLINE the marker-path per-symbol decode.** `HuffmanCodingShortBitsCached::decode`
  is `#[inline]` (declined by LLVM → standalone symbol + per-symbol call/spill/return). Force
  `#[inline(always)]` on the hot wrapper (cold `decode_long`/`base.decode` stay out-of-line).
  Source of the call: careful-tail dist decode (marker_inflate.rs:2825) + fast-loop kill-switch arm.
  Byte-exact: codegen-only (no logic change). Mirrors the existing `emit_backref_ring`
  `#[inline(always)]` precedent (marker_inflate.rs:4514-4521).
- **Lever 2 (if 1 PARTIAL) — shrink/streamline the careful-tail** (40.2% of marker bucket,
  96.4 cyc/ev): e.g. route its distance through the same `DistTable` LUT the fast loop uses
  (single-lookup vs dist_hc → DISTANCE_EXTRA → refill → DISTANCE_BASE chain).
- **Lever 3 (if still PARTIAL) — clean-kernel per-symbol classification branch density** on Zen2.

## Grading per lever (CLAUDE.md gates)
- **Gate-0:** byte-exact — sha==zcat all arms + flate2 AND libdeflate silesia differential at
  multiple chunk sizes (shipped in the SAME commit); A/A << Δ; /dev/null both arms; objdump proof
  the targeted `call` is GONE (codegen changed → non-inert).
- **Gate-1:** interleaved best-of-N>=7 (AMD), best-of-15 (Intel); report Δ vs inter-run spread;
  Δ < spread ⇒ TIE.
- **Gate-2:** cyc/B freq-pinned = the WORK mechanism (marker-fast-loop / careful cyc/ev toward
  clean kernel); the AB wall = the VERDICT.
- **Gate-3:** AMD primary (T2/T4; T8 if llama-free) + Intel no-regress (T2/T4/T8 AND T1-native).
- **Gate-4:** GZIPPY_DEBUG=1 → path=ParallelSM; feature fingerprint; VERIFY the box binary is
  HEAD+this-commit (grep symbol / confirm sha).

## VERDICTS (pre-registered, per lever)
- **CONFIRMED** iff: byte-exact AND drops the AMD-T4 cyc/B (mechanism: marker-fast-loop / careful
  cyc/B toward the clean kernel's; objdump confirms the codegen change) AND drops the AMD-T4 wall
  gz/rg toward <= 1.01, with NO regression on Intel T>=2, T1-native, or T8.
- **PARTIAL:** a real gated cyc/wall drop not reaching 1.01 → report new ratios + remaining bucket
  + next sub-target; do NOT narrate as closed.
- **FALSIFIED-per-lever:** no cyc-drop beyond noise / no wall move / a regression elsewhere →
  revert that lever, try the next ranked one.

## Box hygiene
AMD: freeze gov=performance + boost=0 (NOT no_turbo), idempotent + bounded auto-restore watchdog,
taskset-pin measurement cores away from the roaming `llama-completion` (cores 8,10,12,14 known-good),
RESTORE at exit (gov=ondemand, boost=1, watchdog killed) + REPORT box-state. Intel LXC unfreezable:
taskset/pin + interleave + best-of-15. NEVER touch bench-lock.sh.
