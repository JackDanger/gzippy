# SURPASS STEP B-1 — T1 frontier-gap APPORTIONMENT (both arches, gated) — 2026-06-21

**Subject sha:** `94e79e64` (kernel-converge-A HEAD). The decode binary is
byte-identical to `1e95d911` (the FRONTIER subject sha) — `git diff 1e95d911 94e79e64`
touches ONLY plans/artifacts/scripts, no `src/`. **Build:** gzippy-native
(`--no-default-features --features gzippy-native` → pure-rust-inflate → asm-kernel on
x86; engine A flat clean kernel on aarch64), path=ParallelSM asserted, `-p1`.

**Scope stamp:** macOS aarch64 (Apple Silicon, quiet) + Intel i7-13700T x86_64
(guest 199, **FROZEN** via bench-lock: no_turbo=1, gov=performance, BENCH_LOCK=quiet
runnable_avg=2.00 at acquire; released + RESTORE VERIFIED after). **NOT-YET-LAW** —
single-arch-pair; **AMD/Zen2 owed**. Every number below is the output of a Gate-0
self-validating instrument (byte-exact every arm, /dev/null sink all arms,
path=ParallelSM, oracle non-inert checks) at Gate-1 best-of-N (instr load-immune;
cyc on the FROZEN box only).

---

## INSTRUMENTS (this turn)

- `scripts/bench/standing/clean_core_decomp_mac.py` (existing) — aarch64 core split:
  CRC removal-oracle (`GZIPPY_FOLD_NOCRC`) + per-symbol vs per-copy corpus-contrast.
- `scripts/bench/standing/aarch64_tbuild_scaffold_mac.py` (NEW) — aarch64 conservation
  closer: table-build slope (`GZIPPY_TBUILD_MULT` 2 vs 4) + pipeline-scaffold
  (`GZIPPY_THIN_T1_ORACLE`).
- `scripts/bench/kernel-ab/x86_t1_apportion_guest.py` (NEW) — x86 FROZEN apportionment:
  gz asm `run_contig` vs igzip, CRC/table-build/scaffold/kernel via the same knobs +
  perf cyc/instr.

---

## aarch64 (mac) — gz engine A vs libdeflate (the +25% frontier gap)

`silesia`: gz **9.836 instr/B**, ld **7.605** → **1.293× (excess +2.226 instr/B)**;
cyc 4.114 vs 3.271 = 1.256×. (matches FRONTIER +24.6%.)

| component | result | verdict |
|---|---|---|
| **CRC** | FOLD_NOCRC Δinstr = **+0.017% / +0.012% / −0.062%** (lit/backref/silesia) → INERT | **NON-LEVER — already HW (crc32fast PMULL).** Confirmed at HEAD. |
| **backref-copy** | per-copy extreme (decomp_backref) excess **+0.143 instr/B (1.084×)** | **NON-LEVER — NEON copy exists** (`simd_copy.rs:108`, `copy_match_fast`). Confirmed. |
| **pipeline-scaffold** | full − thin = **0.0016 instr/B (0.02%)** | **NON-LEVER at T1** (marker path never fires at -p1). |
| **table-build** | TBUILD slope = **0.627 instr/B (6.37% of gz)**, non-inert | minor (ld also builds tables → excess is a fraction). |
| **per-symbol Huffman-decode + bitreader** | per-symbol extreme (decomp_literal, ~1 sym/B) excess **+4.26 instr/B (1.260×)**; collapses to +0.143 on the copy extreme (258× fewer symbols) | **THE LEVER** — the excess is per-SYMBOL, in engine A's decode loop + bit reader. ~9.2 instr/B of gz; dominates the silesia +2.226 excess. |

## x86 (Intel, FROZEN) — gz asm `run_contig` vs igzip (the +30% frontier gap)

| corpus | gz cyc/B | ig cyc/B | **cyc gap** | gz i/B | ig i/B | instr gap | CRC% | tbuild% | scaff% | **kernel%** |
|---|---|---|---|---|---|---|---|---|---|---|
| silesia | 5.170 | 4.199 | **+23.1%** | 13.33 | 11.32 | +17.8% | −0.00 | 4.84 | +0.68 | **94.49** |
| monorepo| 3.722 | 2.798 | **+33.0%** | 8.34 | 6.71 | +24.3% | −0.00 | 3.46 | +0.76 | **95.79** |
| nasa | 1.983 | 1.508 | **+31.4%** | 4.38 | 3.64 | +20.2% | −0.00 | 1.80 | +0.71 | **97.48** |

- **CRC: NON-LEVER** (FOLD_NOCRC −0.00% all 3 → x86 crc32fast PCLMUL HW, same as arm).
- **pipeline-scaffold: NON-LEVER** (+0.68–0.76% at T1).
- **table-build: minor** (1.8–4.8% of gz; non-inert slope).
- **kernel (run_contig asm + decode_clean_into_contig wrapper): 94.5–97.5% of instr** —
  the +30% lives in the clean-decode kernel. The **cyc gap (+23–33%) EXCEEDS the instr
  gap (+18–24%)** → part is IPC/codegen (igzip's hand-asm runs hotter), not pure instr
  count. The finer run_contig-vs-`decode_clean_into_contig`-wrapper split is the static
  objdump HYPOTHESIS from `5526585e` (~44% wrapper) — NOT re-derived dynamically this
  turn (no env knob isolates the wrapper from run_contig); flagged for B-2.

---

## VERDICT — the SHARED addressable lever

Both arches: the T1 frontier gap lives in the **per-symbol clean Huffman-decode +
bit reader** (94–97% of x86 instr; the per-symbol-extreme excess on aarch64). **CRC,
backref-copy, and pipeline-scaffold are CONFIRMED NON-LEVERS at HEAD on BOTH arches;
table-build is a minor (1.8–6.4%) component.**

**This REFUTES SURPASS-PLAN Target-4 sub-item #1** ("aarch64 hardware CRC32 … likely
the cleanest single win") — CRC is ALREADY HW on both arches (FOLD_NOCRC inert). The
real target-4 levers are #2 (Huffman multi-symbol/refill) and #3 (bit-reader width).

**The lever = engine A's per-symbol Huffman fastloop + bit reader**
(`consume_first_decode.rs:632` `decode_huffman_libdeflate_style` /
`decode_huffman_fastloop_bounded` :1192; flat `LitLenTable`/`DistTable`
`libdeflate_entry.rs:333/361`). Converge toward `vendor/libdeflate/lib/
decompress_template.h` (refill cadence, preload-ahead-of-dependent-load,
multi-symbol lookahead discipline).

- **Cross-arch reach:** engine A is the aarch64 PRODUCTION clean kernel AND the
  x86-asm-OFF kernel → an engine-A convergence advances aarch64 production directly +
  the x86-asm-off cross-ISA-LAW arm.
- **x86 PRODUCTION (run_contig asm) needs run_contig-specific work** — it is a separate
  artifact (already −1 instr vs igzip `_04` on the emission loop per EMISSION-APPORTION);
  its residual is the diffuse length-tail + IPC + the wrapper (44% objdump HYPOTHESIS).
  An engine-A change does NOT touch x86 production.

---

## B-2 DECISION — STOP after B-1; recommend the engine-A convergence as a follow-on

B-1 names a clear lever, but a faithful **byte-exact** convergence of the per-symbol
Huffman fastloop / bit reader toward `decompress_template.h` is the FULL-DESIGN heroic
build (multi-session) and cannot be completed AND fully gated this turn (sha==zcat
across T1/T4/T8 × silesia+monorepo+nasa × both arches + asm c2/c3 differential +
engine-A reference + prop_structured, plus a frozen-Intel before→after). Per the
governing funnel (don't half-land a heroic kernel rewrite; the apportionment is the
deliverable), **B-1 is delivered; B-2 is recommended for a dedicated follow-on**:

1. Engine A refill-cadence / preload convergence to `decompress_template.h` (advances
   aarch64 prod + x86-asm-off). Gate each sub-component with a removal/slope oracle.
2. SEPARATELY, run_contig-specific x86 work (wrapper/`decode_clean_into_contig` shape
   toward igzip `_04` monolithic; the IPC residual) — pair with the `5526585e` objdump
   split made DYNAMIC (vary T1 chunk size to slope the per-chunk wrapper cost).

**STAMP:** NOT-YET-LAW (Intel-frozen + quiet-mac single-arch-pair). **AMD/Zen2 owed.**
