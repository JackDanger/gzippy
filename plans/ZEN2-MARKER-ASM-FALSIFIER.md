# ZEN2 MARKER-ASM — PRE-REGISTERED FALSIFIER

Branch: `kernel-converge-A` (HEAD bdf85676 at registration).
Front: AMD/Zen2 T>=2-vs-rapidgzip window-absent (marker) decode kernel.
Box: solvency AMD EPYC 7282 Zen2 `root@REDACTED_IP` (PRIMARY), neurotic Intel (no-regress).
Authored: 2026-06-22, BEFORE any STEP-1 measurement.

## Frame
- Goal: AMD/Zen2 gz-native / rapidgzip -> >=0.99 (parity) on the losing cells,
  completing cross-arch rapidgzip parity. Byte-exact + gated measured win is the bar.
- LOCATED CAUSE (prior, Gate-2 STRONG): gz window-absent decode = Rust marker loop
  ~11.7 cyc/B vs clean asm `run_contig` 4.7; rg uses ONE templated fast loop for both.
  Prize BOUNDED ~3-10% (rg also pays inherent u16/marker bookkeeping ~7 cyc/B; gz's
  clean asm over-performs rg's blended, offsetting much of the marker delta).
- PRIOR CAVEAT carried in (do NOT ignore): the AMD-T2 excess was located (Gate-2,
  proportional-to-RSS) as PROCESS-LIFECYCLE TEARDOWN (gz peak RSS 97MB vs rg 59MB),
  NOT decode. So a decode-speed asm is EXPECTED to move T4 (decode-bound) but NOT
  T2 (teardown/RSS-bound). The ceiling oracle is the discriminator.

## CLEAN GATED BASELINE (target cells, project_clean_wall_reconfirm_2026_06_22, AMD clean+llama-paused)
- silesia:  T2 1.048 LOSE / T4 1.041 LOSE / T8 0.982 TIE
- monorepo: T2 1.060 LOSE / T4 0.936 BEATS / T8 0.967 TIE
- squishy:  T4 1.034 LOSE
- nasa: BEATS all T. Underlying real loss ~3-6%.

## STEP 1 — CEILING ORACLE (confirm BEFORE writing asm)
Oracle: `GZIPPY_MARKER_CEILING=1` makes the window-absent (speculative) decode seed a
zeroed 32 KiB window, routing the WHOLE bootstrap chunk through the clean asm
`run_contig` path (~4.7 cyc/B) instead of the 11.7 cyc/B marker loop. Output bytes are
WRONG (backrefs resolve into the zero window) -> sha mismatch is EXPECTED (perturbation,
not a product). The full pipeline runs (block-find, decode, write, drain); CRC fails only
at the very end AFTER all bytes are written, so `perf stat duration_time` captures the
full decode+write wall. This is the ABSOLUTE (over-generous) ceiling: ALL decode at clean
asm speed, marker machinery + marker-resolution pass both removed.

Gate-0 non-inert: `MARKER_CEILING_HITS` counter > 0 (proves the oracle fired on the
window-absent path), reported via GZIPPY_SLOW_HITS=1.

### CEILING VERDICT
- **CONFIRMED (proceed to STEP 2) iff:** the ceiling closes AMD/Zen2 gz/rg to <= ~1.01
  on the decode-bound losing cells (silesia-T4, squishy-T4), llama-paused, interleaved
  best-of-N>=9. (If it over-shoots to gz beating rg, that is consistent with proceed —
  the realistic asm lands between baseline and this generous ceiling.)
- **FALSIFIED (STOP, no asm) iff:** the ceiling does NOT close gz/rg to ~1.01 on the
  decode-bound cells — i.e. even with ALL decode at clean asm speed the wall still
  loses. Then the gap is not decode throughput (serialization / teardown / elsewhere)
  and the marker asm cannot pay. REPORT + STOP.
- T2 cells (silesia-T2, monorepo-T2): EXPECTED not to close (teardown-bound per prior
  Gate-2). If the ceiling DOES close T2, that REVISES the T2 locate — report it.

## STEP 2 — ASM STAGES (only if ceiling CONFIRMED), each gated
- **ASM STAGE CONFIRMED iff:** byte-exact (sha==zcat all arms + flate2/libdeflate/
  current-Rust differential, silesia+monorepo, multiple chunk sizes, SAME commit) AND
  window-absent cyc/B drops toward clean (~4.7-7) AND AMD/Zen2 wall gz/rg drops toward
  parity on the losing cells, with NO regression on Intel-T>=2 / T1 / T8 / clean
  (window-present) path.
- **FALSIFIED-per-stage (revert that stage) iff:** not byte-exact, OR no cyc drop, OR
  no wall move, OR any regression on the no-regress cells.

## Measurement discipline (every run)
Gate-0 (sha==zcat all arms OR documented-wrong-for-oracle + differential same commit +
A/A << Δ + /dev/null both arms + objdump/perf non-inert proof of the asm/oracle),
Gate-1 (INTERLEAVED best-of-N>=9), Gate-2 (cyc/B freq-pinned = mechanism; AB wall =
verdict), Gate-3 (AMD primary + Intel no-regress), Gate-4 (GZIPPY_DEBUG=1 ->
path=ParallelSM; verify box binary built this sha).
AMD: PAUSE llama (SIGSTOP + GUARANTEED-RESUME watchdog + trap; verify RUNNING at exit),
freeze (gov performance + boost off) + restore+verify. Never leave llama stopped / box
frozen.
