# WINDOW-ABSENT-CONVERGE — Results

Branch `kernel-converge-A` (subject 8cad4f6b). AMD solvency EPYC 7282 Zen2
`root@REDACTED_IP`, FROZEN gov=performance/boost=0 @2.8GHz, taskset cores 8,10,12,14
away from a roaming `llama-completion` (load 2-3 most of session; periodic spikes).
rg native ELF 0.16.0. All cyc = `perf stat cycles` (boost-off ⇒ fixed-freq ⇒ cyc is
the load-robust verdict metric; wall A/A spread swamped the gz-rg delta all session).

## STARTING-POINT PREMISE — REFUTED (Gate-0/Gate-2, STRONG)
The brief's GATED STARTING POINT ("the marker fast loop runs at 11.7 cyc/B because it
uses a slow NON-INLINED `HuffmanCodingShortBitsCached::decode` interleaved with marker
logic") is FALSE at current HEAD:
- Source: `decode_marker_fast_loop` decodes litlen via `LutLitLenCode` (fast LUT) and
  dist via `DistTable` (fast single-lookup); `dist_hc` (the `HuffmanCodingShortBitsCached`
  class) survives only in the careful-tail dist + a kill-switch arm. Lever 1
  (`#[inline(always)]`, marker-kernel cycle) already landed here.
- perf (silesia-T4, dbg binary, cycles:u): the marker decode is ONE symbol
  `Block::read_internal_compressed = 47.06%` (everything inlined; NO hot sub-call),
  vs the clean asm kernel `run_contig = 30.63%`. There is no hot non-inlined huffman call.
- perf-annotate hot instructions inside `read_internal_compressed`: `test $0x2000000`
  (LARGE_FLAG, the LUT's literal-vs-long classification — SHARED with the clean path,
  not marker-specific), `or $0x38`, and `lea (%rax,%rax,1)` (the *2 u16-index doubling).
  ⇒ the marker overhead vs clean is INHERENT window-absent bookkeeping (u16 stores +
  `distance_marker` tracking + the backward marker scan), which rapidgzip ALSO pays
  (u16 window + `m_distanceToLastMarkerByte` + `resolveBackreference<u16>`), PLUS shared
  LUT classification — NOT a removable slow call. The "inline the slow marker decode"
  lever the brief proposed DOES NOT EXIST at HEAD.

## REPRODUCED GAP (silesia-T4, frozen, N=11, cores 8,10,12,14)
- BASE gz cyc median ~2441M vs rg ~2295-2310M ⇒ **gz/rg cyc ≈ 1.057** (banked 1.05-1.09).
- Wall: gz ~256-259ms, rg ~255-262ms — A/A spread (3-7ms) >= gz-rg delta ⇒ **wall TIE
  under llama load** (cannot get a clean wall verdict this session).
- mfast_prof (silesia-T4): mfast 323.7M cyc / 64.3% / 65.2 cyc/ev; careful 179.7M /
  35.7% / 81.2 cyc/ev; marker-decode total 503.5M cyc (~20% of process cyc).

## LEVER A — backstop-free `decode_prefilled` on the window-absent litlen sites → TIE/PARTIAL, KEPT
Change: the marker fast loop (3 sites) + `lut_litlen_decode` (careful tail + the
`decode_clean_into_contig` Rust-fallback careful loop) now call
`LutLitLenCode::decode_prefilled` instead of `decode`. Every site is IMMEDIATELY
post-`bits.refill()`, so the `available()<32` backstop inside `decode` is provably dead
(lut_huffman.rs:1088-1101) — matching the clean asm path, which already uses
`decode_prefilled`. Faithful convergence toward ONE decode shape.
- Gate-0 byte-exact: sha==zcat silesia/nasa/monorepo/squishy/bignasa @T4 + silesia/nasa
  @T1; in-tree `diff_real_silesia_marker_path_multi_chunk` (flate2/libdeflate oracle,
  multiple chunk sizes) PASSES with the change.
- Gate-0 NON-INERT: `read_internal_compressed` shrank 4846 → 4673 instructions (−173,
  −3.6%) — the backstop branches are genuinely gone (objdump, dbg binaries).
- Gate-2 cyc (mechanism+verdict): silesia-T4 N=11 (load ~2.0): MINE −5.9M (−0.24%),
  Δ > BASE A/A (1.2M) but ~= MINE A/A (3.4M). silesia-T4 N=21 (load ~2.9): MINE +2.4M
  by best-of — OPPOSITE SIGN. silesia-T1: BASE 1213.9M / MINE 1213.1M (TIE). nasa-T4:
  medians ~TIE (MINE marginally faster by min). ⇒ the Δ is BELOW the llama-load noise
  floor in BOTH directions across runs ⇒ **TIE** (not a confirmed win). NO regression
  on any cell (T1, T4, cross-corpus, clean careful tail).
- VERDICT vs pre-registered falsifier: **PARTIAL→TIE.** byte-exact ✓ + non-inert
  codegen reduction ✓, but cyc/wall drop NOT distinguishable from zero under load
  (fails the "cyc drop > spread" arm). KEPT per CLAUDE.md (a byte-exact change may be
  kept on a TIE) as a zero-risk faithful convergence to the clean path's decode shape;
  NOT claimed as a win.

## STANDING CONCLUSION (deterministic)
The residual AMD-T4 gz/rg ~1.05-1.06 cyc gap on the window-absent path is NOT a
removable slow-huffman call (refuted). It is the inherent u16-marker bookkeeping that
both gz and rg pay, plus the fact that gz decodes window-absent chunks with a RUST loop
while it decodes window-PRESENT chunks with a hand-tuned ASM kernel (`run_contig`,
4.7 cyc/B) — whereas rapidgzip uses ONE templated fast loop for both. The faithful-rg
convergence that would actually move this = route the window-absent decode through an
asm kernel (a u16/marker-output variant of `run_contig`), a large multi-session asm
effort (high backfire risk — cf. marker-kernel Lever 2). Prize remains bounded ~3-10%
per the locate (gz's clean-asm over-performance offsets much of the marker delta), and
is currently UN-MEASURABLE at the wall under the box's llama contamination. Small
codegen flips on this path are exhausted (Lever 1 −0.5%, this Lever A TIE).
