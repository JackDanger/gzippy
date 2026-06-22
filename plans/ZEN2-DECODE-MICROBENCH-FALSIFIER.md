# ZEN2 WINDOW-ABSENT DECODE MICROBENCH — PRE-REGISTERED FALSIFIER

Branch `kernel-converge-A` @ d30b2c79. Box solvency AMD EPYC 7282 Zen2
(`root@REDACTED_IP`). Committed BEFORE any number is taken.

## THE QUESTION (definitive confirm/refute of the marker-asm front)

Does gzippy's PRODUCTION window-absent (marker) decode loop
(`read_internal_compressed_specialized::<CONTAINS_MARKERS=true>` →
`decode_marker_fast_loop` + the marker `decode_careful_tail`) run MATERIALLY
more cycles per emitted byte than rapidgzip's window-absent decode
(`Block::readInternalCompressedMultiCached<Window=u16>`) on Zen2, measured on the
SAME corpus, with the SAME metric definition?

- CONFIRM ⇒ the marker-decode speed IS the lever; the asm rewrite is worth building.
- REFUTE ⇒ the marker-decode speed is NOT the gap; the AMD ~3-6% rapidgzip residual
  is elsewhere (resolve / u16 traffic / pipeline / process lifecycle). DO NOT build
  the multi-session marker asm.

## WHY THE PRIOR ORACLES WERE INVALID (starting point)

- The seeded-window STEP-1 ceiling (u8 and u16) routed through a SLOWER second
  engine (2.5-3.0x slower than baseline) — cannot bound a speed-up
  (project_zen2_ceiling_u16_2026_06_22).
- The contested "gz marker loop = 11.7 cyc/B vs rg Block::read = 3.85 cyc/B" 3x gap
  is suspected to be a DENOMINATOR MISMATCH: gz's 11.7 divided cycles by only the
  ~35% MARKER bytes (74.9 MB), while rg's 3.85 divided by ALL output (212 MB). The
  marker loop EMITS 100% of a window-absent chunk's bytes (clean literals + markers),
  so the correct denominator is marker-mode-emitted-bytes, identical for both arms.

## METHOD (cursor-agent design-reviewed — see review notes below)

SYMMETRIC PRODUCTION COUNTERS, identical metric on both binaries, full T4 production
run on the same corpus:

- **gz** (env `GZIPPY_MFAST_PROF=1` + new byte counters, `GZIPPY_VERBOSE=1` to dump):
  cyc/B = `(MFAST_CYC + CAREFUL_CYC) / (MFAST_BYTES + CAREFUL_BYTES)`. rdtsc is taken
  AFTER per-block Huffman table build (`mfast_t0`), so the decode-LOOP cost is
  isolated — symmetric to rg's `readInternalCompressedMultiCached` (tables built in
  `readHeader`, outside this function). Bytes counted at exactly the same sites the
  cycles are counted (consistent both-or-neither on the rare EOB commit! path).
- **rg** (env `RAPIDGZIP_WA_PROF=1`): an rdtsc accumulator wrapping
  `readInternalCompressedMultiCached` gated on `if constexpr (containsMarkerBytes)`
  (u16 window only). cyc/B = `RG_WA_CYC / RG_WA_BYTES` (nBytesRead). Same decode-loop
  scope; table build is in `readHeader`, excluded.

Both are the WHOLE marker decode loop (literals + markers + backrefs), divided by
ALL bytes that loop emits — the denominator the prior measurement got wrong.

### GATE-0 (instrument self-validation, BLOCKING)
- gz: `MFAST_BYTES + CAREFUL_BYTES > 0` and within ~5-15% of the corpus's
  window-absent byte volume; `path=ParallelSM`; sha(out)==zcat. Non-inert: counters
  fire only on the marker path.
- rg: counter `RG_WA_CALLS > 0` and `RG_WA_BYTES > 0` (proves the marker variant is
  the active ISA-L path; if 0 the instrument is inert → FAIL loud). rg output
  sha==zcat.
- A/A self-test: gz-vs-gz cyc/B across interleaved reps; spread reported. Verdict Δ
  must exceed A/A spread.
- OFF == identity: both instruments are byte-transparent and ~zero-cost when the env
  var is unset (one branch).

### GATE-1 (significance): interleaved best-of-N≥7 per cell; report cyc/B + spread.
### GATE-2 (mechanism): cyc/B with the box FROZEN (gov=performance, boost=0) so
  invariant-TSC rdtsc ≈ core cycles; llama SIGSTOP'd for the measurement window
  (removes SMT/cache contention that would inflate in-region rdtsc for both arms).
### GATE-3 (scope): AMD/Zen2 primary (the residual is Zen2-specific per
  project_clean_wall_reconfirm). Corpora: silesia, squishy, monorepo (the LOSING
  cells), T4. Intel replication owed for LAW.
### GATE-4 (production path): gzippy-native build (`--no-default-features --features
  gzippy-native`), `GZIPPY_DEBUG=1 → path=ParallelSM`, sha-verified.

## PRE-REGISTERED DECISION RULE (three-way; cursor-agent)

Let R = (gz marker-decode cyc/B) / (rg marker-decode cyc/B), measured per cell,
both arms decode-loop-only, denominator = marker-mode-emitted-bytes, box frozen +
llama paused, N≥7 interleaved, Δ vs A/A spread.

- **CONFIRM** (build the marker asm) iff R ≥ 1.20 on the majority of the three losing
  cells AND the gap exceeds the A/A spread (i.e. gz materially slower per byte).
- **REFUTE** (do NOT build the asm; residual is elsewhere) iff R ≤ 1.10 across the
  cells (gz ≈ rg per byte).
- **AMBIGUOUS** (1.10 < R < 1.20): do NOT start the multi-session asm from this; report
  as inconclusive and name the next discriminator.

Report BOTH absolute cyc/B numbers (gz, rg) per cell, R, the A/A spread, the
marker-mode byte fraction (marker-mode bytes / total output) for each arm, and the
verdict.

## CONFOUNDS EXPLICITLY ADDRESSED (from cursor-agent review)
1. Denominator: marker-mode-emitted-bytes on BOTH (not total). [adopted]
2. Symmetry: both via in-binary production counters, same metric, same run-style
   (not isolated-rdtsc-vs-perf-annotate). [adopted]
3. perf-annotate inlining fragility: avoided (counters instead). [adopted]
4. Representativeness: full production run over the real corpus (not one looped
   block). [adopted]
5. rdtsc-under-contention: box frozen + llama paused so rdtsc≈cycles and contention
   stalls don't inflate. [adopted]
6. Table-build asymmetry: gz rdtsc starts AFTER table build (mfast_t0); rg builds in
   readHeader; both decode-loop-only. [adopted]

## RE-OPEN / KILL
A REFUTE here closes the marker-asm front at this sha on Zen2. Re-open trigger: a
gated wall measurement showing the marker-decode region on the critical path with a
removal-oracle ceiling > the A/A spread, or an Intel result diverging from this.
