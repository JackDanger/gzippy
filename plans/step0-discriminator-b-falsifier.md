# STEP-0 DISCRIMINATOR (b) — CONSUMER-BLOCK DECOMPOSE — PRE-REGISTERED FALSIFIER

Status: PRE-REGISTERED BEFORE ANY RUN. Leader (TIER-3 STEP-0), HEAD b8a38e64.
Charter: plans/tier3-step0-authorization.md §THIS-TURN (b) + HARD ESCALATION GATE.
Design ref: tier1-design-v2.md §3 "OWED THIRD MEASUREMENT". This is the
[[project_confirmed_offset_prefetch_gap]] "ROOT CAUSE 2026-06-04" pre-registered next
measurement: "decompose gzippy's consumer per-chunk time into SERIAL-WORK vs DECODE-WAIT."

## THE QUESTION
Split the consumer block (the single in-order consumer thread's per-run wall) into:
  - DECODE-WAIT  = time the consumer spends BLOCKED waiting for a chunk decode to be ready
    (`wait.block_fetcher_get` cold gets + `wait.future_recv` in-flight joins). This SHRINKS
    when the engine speeds up.
  - SERIAL-BOOKKEEPING = the consumer's own in-order CPU work that runs REGARDLESS of decode
    speed: window publish (`consumer.window_publish_marker` + get_last_window_vec),
    `consumer.dispatch_post_process`, `consumer.queue_prefetched_postproc`, the
    speculative-accept/try_take path, CRC/ISIZE finalize, output writes. This DOES NOT shrink
    when the engine speeds up — it is gzippy's true NON-DECODE FLOOR.

The non-decode floor = the SERIAL-BOOKKEEPING term. It is the wall lower bound that survives
BOTH the placement lever AND the engine lever (rule: those levers only attack DECODE-WAIT and
the operating point; serial consumer CPU is untouched).

## HARD ESCALATION GATE (charter, BINDING)
- non-decode SERIAL floor ≤ 0.54s ⇒ continue (tie remains reachable per §3 arithmetic).
- non-decode SERIAL floor > 0.54s ⇒ **STOP and ESCALATE to supervisor** (user-level fork:
  revisit the 1.0× bar / add a 4th off-critical-path consumer lever faithfully mirroring
  rapidgzip / accept FFI). Do NOT start the engine build chasing a tie the consumer forbids.

## INSTRUMENT (byte-exact; reuse the EXISTING trace_v2 span machinery — no new decode path)
The consumer already wraps its phases in trace_v2 SpanGuard spans (chunk_fetcher.rs:1264
consumer.try_take_prefetched, :1367 wait.block_fetcher_get, :1578 consumer.window_publish_
marker, :1631 consumer.dispatch_post_process, :2474 consumer.queue_prefetched_postproc).
The locked-Fulcrum harness (scripts/bench/run_locked_fulcrum.sh + guest_fulcrum_capture.sh)
captures these spans per-thread with self-time (no double-count) — the charter-mandated
"FULLEST Fulcrum test, never a hand-rolled script" (CLAUDE.md PROCESS rule 8).

DECOMPOSE the consumer thread's wall from a captured T8 trace as:
  consumer_wall = Σ(consumer-thread self-time spans)
  DECODE-WAIT   = self-time(wait.block_fetcher_get) + self-time(wait.future_recv)
  SERIAL-BOOKKEEPING = consumer_wall − DECODE-WAIT
                     = Σ(window_publish_marker, get_last_window_vec, dispatch_post_process,
                         queue_prefetched_postproc, try_take_prefetched, finalize, output)
The placement-perfect operating point: measure on the A.2 CLEAN-ONLY oracle
(GZIPPY_SEED_WINDOWS, seed_windows.rs) — every chunk forced clean at a confirmed boundary,
which is EXACTLY the placement-perfect operating point §3 prices the 0.61s consumer block at.
That is the right operating point: it removes the speculation/scan premium so the residual
serial bookkeeping is the placement-PERFECT floor, not today's bottlenecked one. (The A.2
oracle already self-tested: publish chain PRESERVED, byte-exact, orchestrator-status:838.)

## INSTRUMENT VALIDATION (CLAUDE.md rule 4)
- POSITIVE CONTROL: the existing GZIPPY_SLOW knob (d0aa1db, wired into the native clean loop)
  inflates DECODE compute by a known factor. Re-run the decompose with SLOW on: the
  DECODE-WAIT term must RISE by ~the injected factor while SERIAL-BOOKKEEPING stays ~flat. If
  SERIAL-BOOKKEEPING also rises with decode slow-injection, the decompose is mis-attributing
  decode time into the serial bucket — FIX before trusting the floor number.
- SELF/CONSERVATION: busy + idle == span (the Fulcrum self-time invariant, CLAUDE.md rule 8);
  the two buckets must sum to consumer_wall (no double-count, no gap). RUN_TRUSTWORTHY=true
  required.
- The clean-only oracle run must reproduce A.2's 0.61s consumer-block magnitude
  (wait.block_fetcher_get ≈ 0.497s at the clean operating point, step-a2 Target 2) — if the
  consumer block is a wildly different size, the operating point drifted; investigate before
  reading the floor.

## PRE-REGISTERED VERDICT RULE
On the LOCKED GUEST (silesia-large, T8, clean-only oracle, N≥7 interleaved, sha-verified,
RUN_TRUSTWORTHY=true), after the positive control passes:
- Report SERIAL-BOOKKEEPING (the non-decode floor) with its inter-run spread.
- SERIAL floor + spread ≤ 0.54s ⇒ CONTINUE (the engine front is worth gating; report the
  margin to 0.54s).
- SERIAL floor − spread > 0.54s ⇒ ESCALATE (the consumer structurally forbids the tie at the
  current consumer architecture; supervisor fork).
- A value straddling 0.54s within spread is a TIE-at-the-edge finding ⇒ report as marginal and
  flag the off-critical-path consumer lever (vendor runs window-publish/apply-window OFF the
  in-order path, GzipChunkFetcher.hpp:553-575) as the disambiguating next step — do NOT silently
  proceed to a multi-week engine build on a marginal floor.

DISCIPLINE: numbers ONLY from the fullest locked Fulcrum (rule 8), never a hand-rolled script.
A per-span busy attribution is a HYPOTHESIS; the floor claim stands only with conservation +
the SLOW-knob positive control passing.
