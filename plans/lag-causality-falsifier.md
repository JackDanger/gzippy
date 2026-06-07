# LAG-CAUSALITY FALSIFIER — PRE-REGISTERED before any run (2026-06-07, placement-rescope leader)

Charter: plans/placement-rescope-diagnosis.md. THE PIVOTAL QUESTION (gates whether
placement is a separate faithful-port lever at all):

> Is the ~318ms consumer-prefetcher lag **STRUCTURAL** (prefetch depth / scheduling /
> in-order serialization rapidgzip avoids) or an **EFFECT of the 2.38× slow engine**
> (the consumer can't keep pace because each chunk decodes slowly)?

- **ENGINE-INDUCED** ⇒ placement partly DISSOLVES into the engine lever (fixing the
  engine speeds the consumer, shrinking the lag) — ~ONE lever, re-rank.
- **STRUCTURAL** ⇒ placement is a genuine separate faithful-port lever (mirror
  rapidgzip's pacing) that pays even before the engine.

## METHOD — causal perturbation with the EXISTING committed slow_knob (no new instrument)

Vary engine (clean-loop) speed via `GZIPPY_SLOW_MODE` ∈ {0, 50, 100} on the LOCKED
guest at T8, kind ∈ {spin, sleep}. The slow_knob is wired into the NATIVE clean decode
arm (read_internal_compressed_specialized::<false> + canonical, commit d0aa1db, proven
∝ clean bytes by GZIPPY_SLOW_HITS). For each (F, kind) measure the LAG RESPONSE.

### The LAG metric — TWO channels, source-defined (NOT guessed)
The literal ~318ms is a bilateral timeline derivation (prefetch-issue-time vs
consumer-reach-time of the SAME partition, project_confirmed_offset_prefetch_gap). We do
NOT have a direct single-counter for that delta. We use TWO available proxies, each of
which the lag CAUSES:

1. **HEAD-OF-LINE STALL COUNT** (`GZIPPY_STALL_RESIDENCY_PROBE`: STALLS_TOTAL − startup,
   = NOT_RESIDENT non-startup count). This is the DISCRETE manifestation: the consumer
   lags → the containing chunk is evicted/never-retained → cold-get stall. A consumer
   that keeps pace has ~0 non-startup cold-gets. **This is the PRIMARY lag proxy** because
   it is COUNT, not time — it does NOT trivially scale with per-decode cost (see confound
   below). Byte-exact, counters-only, OFF==identity.

2. **`consumer.wait.block_fetcher_get` TOTAL ms** (from the Fulcrum T8 trace). The time
   the consumer is BLOCKED on synchronous cold decodes — the direct time-cost of the lag.

### THE CONFOUND (must isolate — this is the crux of the falsifier)
When I slow the engine, EVERY decode slows, including the cold re-decode the consumer
blocks on. So `block_fetcher_get` TIME (channel 2) rises mechanically by ~F even if the
NUMBER of stalls is unchanged — that rise is NOT evidence the lag grew; it is just each
cold decode costing more. Channel 2 alone cannot discriminate. **The discriminating
signal is channel 1 (the COUNT) and the NORMALISED time** (block_fetcher_get / per-chunk
decode-busy, i.e. how many chunk-decode-equivalents of cold blocking — a ratio that
divides out the uniform per-decode slowdown).

## PRE-REGISTERED PREDICTIONS (which way = which verdict) — committed BEFORE running

Baseline (F=0, prior measurement): non-startup cold-get stalls = ~3 (the 3
overshoot-tail stalls + startup), block_fetcher_get ≈ 0.49s at the clean-only point /
≈ 0.49s normal.

### Falsifier table
| observation as F: 0→50→100 (spin) | VERDICT |
|---|---|
| **STALL COUNT rises** (e.g. 3→5→8+) AND normalised block_fetcher_get (÷per-chunk busy) RISES | **ENGINE-INDUCED** — slowing the engine makes the consumer lag MORE, so MORE chunks are evicted/cold before the consumer reaches them. The lag is a downstream effect of decode speed. |
| **STALL COUNT ~FLAT** (3→3→3, within ±1) AND normalised block_fetcher_get ~FLAT (raw time rises ∝F only) | **STRUCTURAL** — the lag (count of cold-gets) is invariant to engine speed; it is set by prefetch depth/scheduling/in-order serialization, not by how fast each chunk decodes. The raw-time rise is just the confound (each cold decode costs more), which the normalised ratio divides out. |
| MIXED (count flat but normalised time rises, or vice versa) | **PARTIAL** — report both; lean on the COUNT (channel 1) as primary per the confound argument; flag for advisor. |

### Frequency-neutral control (MANDATORY, CLAUDE.md PROCESS rule 2)
Re-run the WHOLE sweep with `GZIPPY_SLOW_KIND=sleep` (yields the core, calibrated real
thread::sleep per the prior pre-gate). If the STALL-COUNT trend SURVIVES sleep ⇒ the
count response is a real engine-pace effect, not a busy-spin turbo-depression artifact.
If the count trend appears under spin but VANISHES under sleep ⇒ it was a turbo artifact
and the verdict leans STRUCTURAL (count truly invariant). The count is the robust channel
either way (a turbo depression slows per-decode time, not the NUMBER of evictions, so
spin-vs-sleep should AGREE on the count if the effect is real).

### Cross-check (corroborating, NOT the verdict) — A.2 clean-only operating point
At the clean-only operating point (GZIPPY_SEED_WINDOWS, confirmed boundaries pre-seeded =
faster effective per-chunk, no marker-resolve), is the stall count ALREADY smaller than
the normal-path count? If clean-only already has ~0 non-startup stalls, that corroborates
ENGINE-INDUCED (a faster operating point already shrinks the lag). If clean-only STILL
shows the same ~3 stalls, that corroborates STRUCTURAL (the lag is independent of how the
chunk decodes). NOTE: clean-only pre-seeds boundaries so it ALSO fixes placement — read
this only as a soft corroborator, not the verdict.

## INSTRUMENT VALIDATION (CLAUDE.md PROCESS rule 4) — before trusting any number
- **Positive control for the stall-count counter:** the residency probe's cap=1 positive
  control (STALLS_TOTAL 4→9 as cache shrinks) already validated it tracks residency
  (orchestrator-status:972). Re-confirm it fires (non-zero count) on this build at F=0.
- **slow_knob self-test:** GZIPPY_SLOW_HITS must be non-zero and ∝ clean bytes at F>0
  (already proven, commit d0aa1db: 40-95M hits). Re-confirm F=0 is byte-exact identity
  (sha == ref) every run — a stall-count change with WRONG bytes is void.
- **Conservation:** the probe asserts startup+cached+inflight+absent == total each run.

## DISCIPLINE
- Numbers ONLY from the locked guest harness (run_locked_fulcrum.sh; SLOW_MODE/KIND flow
  through). Guest verified idle before; host freq RESTORE verified after; /dev/shm clean.
- Guest run held by a Bash task that HOLDS the ssh for the run's duration (no
  print-and-exit subagent — SIGHUPs the ssh, orphans the guest run).
- N≥7 interleaved, sha-verified, RUN_TRUSTWORTHY=true required.
- The probe must be threaded through the harness env (it is counters-only, OFF==identity).

## WHAT THIS DOES NOT DECIDE
This decides ONLY whether the lag is engine-induced or structural — i.e. whether
placement is a separate lever. It does NOT re-bound either ceiling (already done: A.2
engine 0.61s, Oracle-P placement 0.56-0.66s) and does NOT authorise any build.
