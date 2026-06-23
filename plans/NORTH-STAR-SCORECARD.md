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

## OPEN (owed before any "we win" claim)
- **aarch64 (macOS)**: T1 + T≥2 cross-tool matrix — UNMEASURED at HEAD (different
  Rust no-asm kernel; igzip N/A on arm64). cursor's next front (b).
- AMD numbers are llama-load-noisy ⇒ a frozen+llama-paused reconfirm tightens ratios
  (direction is robust). Honest label: "parallel-decode SOTA on x86 (T≥2); serial
  beats all-but-igzip at T1 (pure-Rust floor); aarch64 owed."
