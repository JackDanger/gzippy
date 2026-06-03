# Structural rewrite to x86 parity — SCOPING (user-chosen 2026-06-01)

> **STATUS: SUPERSEDED (2026-06-03).** Prerequisite measurements moved to Fulcrum memlife/TMA.
> Do not start a blind port without [`sm-parity-gap-matrix.md`](sm-parity-gap-matrix.md) execution order.

---

User chose "attempt the structural rewrite" after the located-lever space was
exhausted (fill lever dead both halves; decode-rate ~15%-bounded; the residual is
the diffuse 1.19/1.45× T8/16 shipped-ISA-L stall gap). Goal per CLAUDE.md: faithful
structural port of rapidgzip's decode pipeline until gzippy's wall matches.

## The paradox the rewrite must resolve (why we can't just start transliterating)
The gap is **1.47× cycles at T8 with EQUAL instructions** ⇒ pure MEMORY STALLS, not
extra work. YET every memory cost we removed was wall-neutral/slack:
- page-fault removal = 0% wall (3 oracles, T4/8/16, advisor-airtight)
- copy elimination (absorb_isal_tail, writev) = wall-neutral
- placement / resolve-ahead = TIE (frontier 94% in-flight, priority -1 shipped)
- segmentation/granule port = REGRESSED

So the stalls are real (cycles say so) but NOT in any region we've perturbed. A
structural rewrite without locating the stall site is a multi-session GUESS at which
structure to port. **Prerequisite: locate the stalls, then port the structure that
owns them.**

## Two scoping measurements BEFORE any port (these aim the rewrite)
### M-A: per-region memory-stall-cycle attribution (S3, Fulcrum TOTAL step 3-4)
Needs the perf feasibility spike: host `sysctl kernel.perf_event_paranoid=1`
(host-global, not namespaced) → guest re-probe `perf stat true`. If it opens,
CYCLE_ACTIVITY.STALLS_MEM_ANY + IPC + branch-MPKI per-TID-bound, time-windowed to
the trace regions. Tells us WHICH region the 1.47×-worth of stall-cycles lives in
(decode? marker-resolve? consumer write? window-map lookup?). Per-TID-bound only
(pooled-by-time is smeared at T>1). This is the missing number — every prior
"memory" lever was attacked by COST-of-region (faults/copies), never by
STALL-CYCLES-of-region. They can diverge: a region with few faults can still stall
on cache-miss loads.

### M-B: engine-vs-data-model discriminating oracle (the global-optimum plan's gate)
Build a known-window parallel-clean decode path: supply each chunk's predecessor
window UP FRONT (no speculation, no u16 markers, no apply_window) and decode every
chunk CLEAN with the fast engine, parallel, in-order consume + writev. This REMOVES
the entire marker/speculation/data-model machinery.
- If known-window-clean MATCHES rapidgzip's wall ⇒ the gap is gzippy's
  marker/speculation/data-model ⇒ rewrite THAT (port MarkerVector + in-place
  applyWindow + toIoVec, retire the 3-buffer u16/u8 model).
- If known-window-clean STILL loses ⇒ the gap is the engine/consumer/window-map
  itself ⇒ rewrite THAT (port the chunk lifecycle + window chaining).
This discriminates the rewrite target with ONE oracle instead of porting the wrong
half for a month. (NOTE: this is a measurement oracle, byte-correctness not
required — it can cheat on windows since we're measuring engine wall, not output.)

## Decision gate
Run M-A (if perf opens) + M-B. They jointly name the structure to port:
- stalls in decode + known-window-clean still loses ⇒ port the ENGINE (inner decode
  + chunk lifecycle), the ~15% decode-rate becomes the down-payment not the ceiling
  because the structural port changes the stall profile, not just instruction count.
- stalls in marker/resolve + known-window-clean matches ⇒ port the DATA MODEL
  (MarkerVector/applyWindow-in-place/writev), retire the u16 triple-pass.
THEN faithful vendor transliteration of the named structure, region by region,
re-measuring the WHOLE on scripts/measure.sh (interleaved, sha-verified, T4/8/16)
after each region — layer-don't-revert.

## Risks / pre-mortem
- perf may stay blocked in the LXC even after host paranoid-lower (seccomp/apparmor
  on the container) ⇒ M-A degrades to the perf-free residual tier; M-B alone still
  discriminates engine-vs-datamodel (it's a wall oracle, no PMU needed).
- known-window-clean oracle could be INVALID if "supply windows up front" changes
  the working set in a way that flatters it (the warm-monolith trap) — positive-
  control it: it must still decode the same bytes, same buffer sizes; only the
  window-source and marker-machinery change.
- a structural port that matches the oracle's wall but the oracle DIDN'T match
  rapidgzip ⇒ we ported the wrong half; that's why M-B gates the direction.
