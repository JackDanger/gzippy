# GAP-FINDER IDEAS — fresh-eyes, divergent (2026-06-09)

Role: NOT attack a claim (disproof advisors do that). GENERATE the un-illuminated
gaps by questioning the campaign's FRAMING. The human just caught the
T4-trough/T8-recovery thread-curve shape the whole team missed for weeks. This
file hunts the NEXT such gap.

The recurring blind-spot pattern in this campaign: **a banked conclusion rests on
a region of an axis that was measured noisily, then SUPPRESSED from the scorecard,
or never measured at all.** The thread-curve catch was exactly that. Every gap
below is an instance of the same pattern, ranked by how load-bearing the dropped
region is.

---

## GAP 1 (HIGHEST) — The T16/T32 cells are MEASURED-BUT-SUPPRESSED and CONTRADICTORY; the goal's actual regime is decided on dropped data

**What's wrong with the framing.** The MORNING-BRIEF scorecard is T1/T4/T8 only and
banks "T8 ties (0.990) ⇒ high-T is fine." But the goal text is explicit: *"8 or 16
(or MORE) threads"* + *"small hot-in-cache memory so MANY threads stay in cache."*
T16/T32 IS the target regime, and the campaign DID measure T16 — with wildly
inconsistent results that were then dropped from the headline:
- wall-progress.md:20 (Locked Fulcrum): gzippy 1.202s vs rg 0.478s = **0.40x** at T16.
- orchestrator-status:434: T16 = **0.921x "TIE"**.
- wall-progress:240: T8/T16 "too noisy (spread 24-36%, load crept 0.8→2.1)" ⇒ punted.

A 0.40x↔0.921x spread is not a tie or a loss — it is an UNRESOLVED cell, and it is
the cell the goal actually cares about. T32 appears nowhere.

**(a) Measurement that illuminates it.** Run the EXACT frozen-guest parity spine
(interleaved, sha-verified, N≥13, host-freeze HARD-FAIL) that produced the trusted
T1/T4/T8 cells, but for T2,T3,T4,T5,T6,T8,T12,T16,T24,T32 — including
oversubscription past physical P-cores (this box masks P-cores only; T16+ forces
SMT siblings / E-cores, which is itself an untested axis, GAP 7). Report the cell
the same way the banked cells are reported. Do NOT accept "too noisy" — if T16 is
too noisy to bank, the box/method is not yet adequate for the GOAL'S regime, and
that is the finding.

**(b) Why it might hide a lever / change the verdict.** "T8 ties ⇒ done at high T"
is the campaign's high-T anchor and it is built on the one high-T cell that
happened to land favorably while the noisier, less-favorable T16 cell was dropped.
If T16/T32 is genuinely 0.4–0.9x, the build is NOT at parity in its target regime
and a whole class of high-T levers (below) reopens.

**(c) Goal regime?** This IS the goal regime (8/16/more threads). Maximally
under-covered.

---

## GAP 2 (HIGHEST, mechanism) — The speculative-decode FALLBACK STORM at high T: isal_fallbacks EXCEED isal_chunks at T16

**What's wrong with the framing.** Every low-T attribution in the ledger asserts
`isal_chunks=14/14 fb=0` and reasons from a 0-fallback engine. But the campaign's
own data (orchestrator-status:335): **isal_fallbacks = 1 / 11 / 46-48 / 101-104 at
T2 / T4 / T8 / T16** — at T16 the engine FALLS BACK MORE TIMES THAN THERE ARE
CHUNKS. A fallback is a pure-Rust re-decode, and DIS-14 measured a fallback chunk
at ~7.5x the clean cost. So the entire "engine kernel is byte-identical ISA-L ⇒
engine is not the high-T lever" spine evaporates at T16: the engine is mostly NOT
running ISA-L there — it is re-decoding in pure Rust, repeatedly.

This is the single most important un-illuminated mechanism I found. It is measured,
it is dead-center in the goal regime, and it is INVISIBLE to every instruction-count
/ marker-attribution / de-frag analysis the campaign ran (all done at T4/T8 with
fb≈0). The low-T attribution frame literally cannot see it.

**(a) Measurement.** Plot fallback-RATE (isal_fallbacks / isal_chunks) vs T, on the
frozen box, per corpus. Then decompose: WHY do fallbacks rise with T? Pre-registered
hypotheses to discriminate: (i) more threads → shorter per-worker prefetch horizon →
more chunks start window-ABSENT → speculation guesses wrong → fallback (this couples
to the confirmed-offset-prefetch-gap memory and DIS-6); (ii) finer adaptive chunks
at high T (`adjusted_chunk_size_bytes` shrinks chunks as threads rise) → more
boundary guesses → more misses; (iii) worker contention starves the window-publish
chain so the predecessor window arrives too late to flip-to-clean. The Gantt
(GAP 3) discriminates (i)/(iii); a chunk-size A/B at FIXED T discriminates (ii).

**(b) Lever / verdict.** If the high-T wall is a fallback storm, the lever is NOT
the inner-Huffman asm (VAR_VIII, user-gated, plateau) and NOT de-frag (DIS-21,
refuted) — it is *reducing the fallback rate* (better window propagation / boundary
confirmation before decode), which is a SCHEDULING/prefetch lever the campaign
declared dead at T4 but never re-evaluated at the T where fallbacks actually
dominate. A direction the disproof ledger closed at fb≈0 may be wide open at fb>chunks.

**(c) Goal regime?** Yes — the mechanism only appears at the goal's thread counts.

---

## GAP 3 (HIGH) — The per-chunk Gantt TIMELINE is still unplotted; it is the only thing that separates straggler-tail from uniform-slow

**What's wrong with the framing.** The campaign measured aggregate INSTRUCTIONS
(DIS-17/18/19), aggregate RSS, aggregate MPKI — all integrated-over-the-whole-run
scalars. None of them can see WHEN each chunk starts and finishes. The wall of an
in-order pipeline is set by the critical PATH through the chunk dependency graph,
not by the integral of work. A single straggler (e.g. one fallback chunk at ~7.5x,
GAP 2) at the tail sets the wall while contributing ~nothing to aggregate
instruction share — exactly the kind of phantom CLAUDE.md rule 8 warns about, but in
the time domain.

**(a) Measurement.** Emit per-chunk {decode-start, decode-finish, flip-to-clean,
window-available, fallback?(bool), output-published} timestamps (the trace
infrastructure exists — GZIPPY_LOG_FILE + parallel_sm_log_summary.py). Render a
Gantt, gzippy vs rg, at T4, T8, T16. Overlay the dependency edges (chunk N's
flip-to-clean ← chunk N-1's window). Look for: a fat tail chunk; a diagonal
serial-handoff staircase (dependency-bound, not compute-bound); idle gaps where
workers wait for windows.

**(b) Lever / verdict.** Distinguishes three totally different lever classes the
aggregate cannot: (1) tail-straggler ⇒ split the largest/last chunk or work-steal;
(2) serial-handoff staircase ⇒ the window-publish chain (which wall-progress:179
fingered as "the in-order-consumer coordination chain" but never visualized);
(3) uniform compute ⇒ the engine. The campaign has been arguing (3) from instruction
counts while (1)/(2) are unvisualized. Likely explains the T4-trough directly: at
T4 the dependency staircase has fewer workers to hide it behind; at T8 more workers
overlap the tail.

**(c) Goal regime?** The staircase deepens with chunk count, which grows with T —
so the timeline is most diagnostic at T16/T32.

---

## GAP 4 (HIGH) — Corpus monoculture: the entire verdict rests on ONE 212 MB silesia tarball

**What's wrong with the framing.** Every banked number is silesia-large
(211,968,000 bytes output, ~3.8x ratio, ~14 chunks). The goal explicitly invokes
"small hot-in-cache memory." We have ZERO data on:
- **Compression ratio extremes.** High-ratio input (10x+, e.g. logs/JSON) → each
  compressed chunk explodes to huge output → output-bandwidth-bound, marker buffer
  pressure, RSS blowup. Low-ratio/incompressible (~1x, e.g. pre-compressed media in
  a .gz) → tiny output, decode-bound, per-chunk overhead dominates. The gz-vs-rg
  ratio could FLIP sign across this axis. silesia (3.8x mixed) tells us nothing
  about either end.
- **Small files.** The single-member parallel path is gated at >10 MiB; below that a
  different path runs entirely. Is THAT path at parity? Small + high-T = fewer chunks
  than threads = idle cores (DIS-15 sized T1 pipeline startup at 24% of wall — that
  fixed cost is murderous on small inputs).
- **Many-small-members** (concatenated gz, BGZF), and the squishy corpus (the
  project's canonical source per memory) which the campaign never used.

**(a) Measurement.** Build a corpus matrix: {incompressible, silesia-3.8x,
highly-compressible-10x} × {1 MiB, 30 MiB, 212 MiB, 1 GiB} × {T1,T4,T8,T16}, parity
spine vs rg. One plate; it reframes whether 0.90x is a silesia artifact or universal.

**(b) Lever / verdict.** A sign-flip on any corpus axis is a lever (find what gzippy
is GOOD at and why, then port it to where it's bad) AND a verdict-changer (if gzippy
is at parity on the goal's "small hot" regime and only loses on the big tarball,
the campaign has been optimizing the wrong corpus for weeks).

**(c) Goal regime?** "Small hot-in-cache" is named in the goal and entirely
unmeasured. High under-coverage.

---

## GAP 5 (MEDIUM-HIGH, methodological — feed directly to the leader's T-curve/Amdahl work)

**What's wrong with the framing.** Two banked scaling stories CONTRADICT and neither
was resolved: wall-progress:240 "gap is fairly UNIFORM ~20% across thread counts,
NOT a worsening cliff; the scaling-cliff was a loaded-box noise artifact" vs the
MORNING-BRIEF T4-trough(0.906)/T8-recovery(0.990) shape. They can't both be right.
The "uniform" verdict came from high-T runs the same author called "too noisy."

**Two sharp warnings for the leader's Amdahl S/W fit:**
1. **An Amdahl fit assumes FIXED total work as T varies. gzippy VIOLATES this** —
   the fallback storm (GAP 2) means total work GROWS with T (re-decodes). Fitting a
   fixed-work serial-fraction model to growing-work data will silently absorb the
   fallback re-decode into a FAKE "serial fraction" term and mis-attribute a
   scheduling/prefetch problem to an unfixable Amdahl serial floor. The fit MUST
   carry fallback-rate as a covariate or it will manufacture a phantom serial bound.
2. **Overlay, don't just fit.** On one set of axes vs T, plot: wall-ratio,
   fallback-rate, decode-rate (wall-progress:403 already shows decode rate FALLS
   -24% at T16 = bandwidth/contention), per-thread RSS slope. The Amdahl knee, the
   fallback knee, and the bandwidth knee will or won't coincide — and WHICH knee the
   wall-ratio tracks is the lever identification, not the fitted S/W constant.

**(c) Goal regime?** Resolves whether the trough/cliff is real at the goal's T.

---

## GAP 6 (MEDIUM) — Per-thread working-set residency at T16/T32 — the cache thesis tested only by REMOVAL at T8

**What's wrong with the framing.** The user's north star is "MANY threads stay in
cache." The footprint direction was declared SLACK (wall-progress:179: shrinking
RSS 200 MB recovered ~3% IPC) — but that removal oracle ran at T8, where 8 threads
of per-thread scratch (each: 64Ki-u16 ring = 128 KB + output buffers) may still fit
shared L3. At T16/T32 the per-thread scratch AGGREGATE is 2-4x larger and the
"stays in cache" thesis is precisely where it should break or pay off.
orchestrator-status:2126-2146 explicitly flagged "process RSS is TOO COARSE... per-
thread decode scratch at T16" as the right metric and it was never carried through.

**(a) Measurement.** Per-thread resident scratch (not process RSS) vs T, and L2/L3
occupancy (perf c2c / cache occupancy events) at T16/T32, gzippy vs rg. rg's
footprint is leaner (172 vs 208 MiB at T8); does that gap MULTIPLY per-thread at
T32 and finally bind the wall, or does gzippy's measured-BETTER MPKI keep it slack?

**(b) Lever / verdict.** The footprint "slack" verdict is a T8 statement; if it
breaks at T32 the leaner-structure port reopens specifically for the goal regime.

**(c) Goal regime?** The cache-residency thesis is a T16/T32 phenomenon by
definition; tested only at T8.

---

## GAP 7 (MEDIUM) — Comparator & methodology assumptions never stress-tested

- **Oversubscription / heterogeneity.** All cells mask P-cores only. Real users run
  `-p$(nproc)` onto P+E/SMT. gzippy spawns fixed worker threads; rg may handle the
  asymmetry differently. T16 on an 8P box = forced SMT/E-core — untested, and it's
  the default user invocation.
- **rg chunk-size parity.** Does rapidgzip 0.16.0 use the same chunk granularity as
  gzippy's 4 MiB-compressed default? If rg auto-tunes chunk size to T and gzippy
  doesn't (or shrinks differently via adjusted_chunk_size_bytes), the T16 comparison
  is apples-to-oranges and the gap is a tuning artifact, not an engine/structure
  fact. Confirm both tools' effective chunk count at each T.
- **Wall-ratio as the SOLE metric.** Time-to-first-byte / cold-start matters for
  `gzippy -dc | downstream`; DIS-15's 24% T1 startup tax suggests gzippy is slow to
  first byte. Never compared to rg. Also: is FORCE_PARALLEL_SM (the measurement
  convention for low T) the real production path at T2/T3? DIS-22 just fixed T1; the
  unmeasured curve points may still be measured under a non-production force-flag.
- **pigz / other tools at high T.** Goal says parity with EVERY tool. pigz
  decompression is single-threaded so rg is the only real high-T rival — worth
  STATING (it bounds the comparison) rather than leaving implicit.

---

## RANKED SUMMARY (for the supervisor)
1. **T16/T32 unresolved + suppressed** — goal's actual regime, decided on dropped
   0.40x↔0.921x data. Re-measure with full rigor.
2. **Fallback storm at high T** (isal_fallbacks 101-104 > isal_chunks at T16) — a
   measured, engine-defeating mechanism the entire fb=0 low-T attribution can't see.
   Likely reopens the prefetch/scheduling levers the ledger closed at fb≈0.
3. **Per-chunk Gantt timeline** — only discriminator of straggler-tail vs
   serial-staircase vs uniform; aggregate instruction counts are blind to it; likely
   explains the trough.
4. **Corpus monoculture** — whole verdict is one silesia tarball; ratio/size extremes
   and "small hot-in-cache" (named in the goal) are unmeasured and could flip the sign.
5. **Amdahl-fit trap (methodological, for the leader NOW)** — growing-work fallbacks
   violate Amdahl's fixed-work premise; the fit will manufacture a phantom serial
   floor unless fallback-rate is a covariate. Overlay fallback/decode-rate/residency
   knees against the wall-ratio knee.
6. **Per-thread residency at T16/T32** — cache thesis tested only by removal at T8.
7. **Oversubscription / rg chunk-size parity / TTFB / force-flag** — comparator
   assumptions never stress-tested.

Cross-check vs disproof ledger: none of these re-proposes a dead mechanism AS
STATED. de-frag (DIS-21), placement (open1-disentangle), contention (DIS-17),
offset-prefetch fix (DIS-6), reserve-factor (DIS-14), finer-chunking (TIE-3) were
all refuted AT T4/T8 with fb≈0. GAP 2 argues those verdicts are T4/T8-scoped (the
ledger's own staleness rule + the TIE-5 "don't generalize a T8 tie to other T"
precedent) and must be RE-asked at the T where fallbacks dominate — that is a NEW
mechanism (fallback-rate), not a re-run.
