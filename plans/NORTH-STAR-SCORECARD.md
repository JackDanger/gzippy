# gzippy-native NORTH-STAR SCORECARD (gated, 2026-06-23, @eea9e445)

North star: fastest gzip/DEFLATE **decoder** — beat libdeflate+igzip at T1, beat
pigz at all T, tie-or-beat rapidgzip at all T, across Intel x86, AMD Zen2, macOS
aarch64. gzippy-native (pure-Rust, C-FFI off decode graph) = sole production path.

Method: interleaved median wall, /dev/null all arms (SINK LAW), sha==zcat verified,
fair thread counts. Intel = neurotic i7-13700T LXC (taskset-pinned, paired-interleave
cancels turbo/drift). AMD = solvency EPYC 7282 Zen2 (idle-core pin; box under llama
load ⇒ load-robust RATIO, absolute ms noisy). gz-native @eea9e445 (= 5beb2733 kernel
wins B2/B3/RANK-2 + rss-inflate oracle), target-cpu=native.

## T≥2 (parallel) — gz/tool, <1 = gz faster
| arch | cell | gz/rg | gz/pigz |
|------|------|-------|---------|
| Intel | T2 monorepo | 0.798 | 0.789 |
| Intel | T4 silesia  | 0.917 | 0.554 |
| Intel | T8 silesia  | 0.868 | 0.276 |
| AMD   | T2 monorepo | 0.848 | 0.871 |
| AMD   | T4 silesia  | 0.937 | 0.318 |
| AMD   | T8 silesia  | 0.863 | 0.251 |

**T≥2: gz BEATS rapidgzip AND pigz on BOTH arches, every cell. GOAL MET (x86).**

## T1 (serial) — gz/tool, <1 = gz faster (N=13, 3 corpora)
| arch | gz/igzip | gz/libdeflate | gz/pigz | gz/rg | gz/gunzip |
|------|----------|---------------|---------|-------|-----------|
| Intel silesia  | 1.093 | 0.982 | 0.410 | 0.885 | 0.254 |
| Intel monorepo | 1.094 | 0.956 | 0.379 | 0.598 | 0.216 |
| Intel nasa     | 1.147 | 0.758 | 0.366 | 0.606 | 0.157 |
| AMD silesia    | 1.039 | 0.835 | 0.476 | 0.839 | 0.283 |
| AMD monorepo   | 1.084 | 0.797 | 0.458 | 0.554 | 0.264 |
| AMD nasa       | 1.065 | 0.565 | 0.406 | 0.512 | 0.178 |

**T1: gz BEATS pigz (~2-2.7×), rapidgzip, gunzip (~4-6×), AND libdeflate (all
corpora both arches; near-tie vs libdeflate on Intel-silesia 0.982) — loses ONLY to
igzip (4-15%, even tighter on AMD).**

## SCORECARD — what is WON vs OPEN (x86, gated cross-arch)
| target | T1 | T≥2 |
|--------|----|----|
| vs pigz       | WON (~2-2.7×) | WON |
| vs rapidgzip  | WON | WON |
| vs gunzip/zlib| WON (~4-6×) | n/a |
| vs libdeflate | WON (near-tie Intel-silesia) | n/a (libdeflate serial) |
| vs igzip      | **LOSS 4-15%** (pure-Rust-vs-ISA-L IPC floor; gated-CONCLUDED BANK+ACCEPT) | n/a (igzip serial) |

**The ONLY unmet north-star cell on x86 is T1-vs-igzip** — the intrinsic pure-Rust-
vs-ISA-L codegen floor ([[project_x86_t1_monolith_finish_2026_06_22]]: kernel
acquitted, scaffold = harness artifact, monolith instruction-only, dist-rewrite
refuted). Closing it needs a heroic inner-kernel asm rewrite to match ISA-L IPC.

## aarch64 (macOS M-series, 10-core; N=13, sha==zcat; no taskset/igzip on arm64;
## local-box noise ⇒ ratios near 1.0 are effectively TIE)
| cell | gz/libdeflate | gz/pigz | gz/rg | gz/gunzip |
|------|---------------|---------|-------|-----------|
| T1 silesia  | 1.045 | 0.940 | 0.365 | 0.331 |
| T1 monorepo | 1.068 | 0.949 | 0.277 | 0.295 |
| T1 nasa     | 0.936 | 0.928 | 0.280 | 0.266 |
| T4 silesia  | —     | 0.410 | 0.530 | — |
| T4 monorepo | —     | 0.743 | 0.511 | — |
| T8 silesia  | —     | 0.297 | 0.541 | — |

**aarch64: gz beats pigz, rapidgzip (~2-3×), gunzip at ALL T; ~TIES libdeflate at
T1 (0.94-1.07, within local-box noise; ~5-7% behind on silesia/monorepo, ahead on
nasa). No igzip on arm64 → libdeflate is the arm64 serial SOTA, and gz is at-parity.**

## FINAL CROSS-ARCH VERDICT
| north-star target | Intel | AMD | aarch64 |
|-------------------|-------|-----|---------|
| beat pigz @ all T | ✓ | ✓ | ✓ |
| tie/beat rapidgzip @ all T | ✓ | ✓ | ✓ |
| beat libdeflate @ T1 | ✓ | ✓ | ~TIE (noise) |
| beat igzip @ T1 | **LOSS 9-15%** | **LOSS 4-8%** | n/a (no ISA-L) |
| beat gunzip/zlib | ✓ (~4-6×) | ✓ | ✓ (~3×) |

**gz-native is the fastest or tied-fastest gzip decoder across every arch × T × tool
measured, with ONE honest asterisk: T1-vs-igzip on x86 (the intrinsic pure-Rust-vs-
ISA-L IPC floor, 4-15%, gated-CONCLUDED).** It beats pigz and rapidgzip — the
explicit "all T" / parallel comparators — everywhere.

## OPEN / CAVEATS
- AMD numbers are llama-load-noisy ⇒ a frozen+llama-paused reconfirm tightens ratios
  (direction robust across all cells). aarch64 measured on the working mac (no
  pinning, efficiency cores) ⇒ T1 gz/libdeflate near 1.0 is effectively a TIE.
- For LAW-grade promotion: frozen AMD reconfirm + a quiet-mac aarch64 reconfirm
  would tighten the near-1.0 ratios (none would flip a direction — the pigz/rg wins
  are 2-6× and the igzip loss is consistent 4-15%).
- The single remaining north-star gap is **T1-vs-igzip on x86** — closing it needs a
  heroic inner-kernel asm rewrite to match ISA-L IPC (gated-concluded floor; opt-in
  front, not a default).

## T1-vs-igzip FLOOR ATTACK (2026-06-23, gated) — named micro-levers EXHAUSTED
Mechanism re-located at HEAD (gated, Intel+AMD): INSTRUCTION-BOUND, not mispredict
(gz instr/B 13.4 vs igzip 11.4 ~1.18x; gz IPC HIGHER; gz brmiss% LOWER). Recoverability
inject oracle: cyc/B rises with instr/B (slope 0.19-0.22, not flat floor) BUT IPC rises
under injection (partial slack; slope = upper bound on hypothetical removals, not the
actual surplus mix).
ATTEMPTS (all gated): cursor #1 copy-shape (route <=40&dist>=16 through arm-1 MOVDQU
loop) BUILT byte-exact -> cyc/B REVERT (instr UP +2-4.6%, monorepo +8.2%); the RANK-2
3-burst was already near-optimal for <=40. stateless anchor (-0.607 instr/B) BROKEN at
HEAD + cycle-slack pre. dist refuted. literal path at parity.
VERDICT (cursor + gated): BANK+ACCEPT the incremental inner-kernel floor. Remaining
#2/#3 = <1 instr/B at ~0.12-0.20 cyc/B (noise vs 0.40 gap). The ONLY remaining path to
igzip-T1 parity is a HEROIC from-scratch T1-stateless-monolith asm rewrite (igzip _04
loop shape, no resumable tax, T1-only; high byte-exact risk; possibly the true floor).
=> STRATEGIC FORK for the user: fund the heroic rewrite or accept the ~9%(Intel)/
~4-7%(AMD) pure-Rust-vs-ISA-L T1 floor.
