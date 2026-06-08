# Disproof brief — fulcrum_total whole-system instrument validity

You are an INDEPENDENT DISPROOF advisor. Your job is to BREAK the claim that
`scripts/fulcrum_total.py` (+ `scripts/bench/fulcrum_total_capture.sh`) is a
trustworthy whole-system instrument that ends the campaign's recurring instrument
failures. Be adversarial. Source-verify against the actual files.

## Context (the failures this instrument must prevent)
The gzippy->rapidgzip parity campaign was repeatedly misled by:
1. slack-masked decodeBlock SUMs read as the wall binder (SUM != wall; a region's
   SUM can be huge AND wall-neutral behind <100% Fill);
2. SEEDED oracles that route to the clean engine and MASK the real window-absent
   marker-bootstrap binder (window_seeded>0 => clean engine, gzip_chunk.rs:790);
3. counter inversions (a misnamed counter read backwards twice);
4. nested-span double-counts (the combine_crc "62ms" phantom = a child's time
   charged to the parent);
5. an instrument that emitted EMPTY output, and a clean-window oracle that
   silently RE-RAN the bootstrap.

## What the instrument claims (verify each against the code)
- C1 NO DOUBLE-COUNT: per-name SELF time subtracts nested children; the SUM column
  is explicitly labeled slack-maskable. Self-test asserts combine_crc SELF=200us
  vs SUM=1000us on a synthetic nested trace.
- C2 busy+idle==span: per-thread breakdown uses LEAF attribution (sweep a boundary
  stack, charge each slice to the deepest open span) so every instant is counted
  exactly once; idle = span - covered. assert_busy_plus_idle_equals_span FAILS the
  analyze() call (raises) if violated.
- C3 WAIT vs COMPUTE vs OUTPUT: a blocking get on a decode future
  (ttp.rx_recv_block, consumer.dispatch_recv, consumer.wait_replaced_markers,
  wait.*, pool.pick.wait) is classified WAIT, never serial compute. UNKNOWN names
  are surfaced, never silently bucketed.
- C4 WINDOW-ABSENT-PRESERVING: reads the GZIPPY_VERBOSE counter sidecar; if
  window_seeded>0 OR isal_oracle_chunks>0 it REFUSES to call the run production
  (prints [REFUSE]); production requires window_seeded=0 AND the window-absent
  bootstrap ran (finished_no_flip/flip_to_clean>0). Inconclusive if no sidecar.
- C5 ORACLE CONTAMINATION CHECK: isal_oracle_fallbacks>0 => flags the ceiling as a
  blend; oracle copy/alloc spans the production path lacks => flags overhead.
- C6 SELF-VALIDATING: --selftest builds synthetic traces and asserts all of the
  above PLUS a positive control (+50% into one stage moves that stage ~1.5x,
  others ~1.0x), a negative control (identical run -> zero delta), and the
  empty-trace failure class RAISES.
- C7 OFF==identity: the analyzer is a read-only post-processor; the capture only
  sets GZIPPY_TIMELINE/GZIPPY_VERBOSE (already-wired trace_v2, proven byte-exact
  locally: trace ON == trace OFF == ref sha).

## Evidence gathered locally (validate or refute)
- --selftest: 21/21 checks PASS.
- Real trace (arm64 pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM, 22MB.gz, T4,
  7637 events): analyze() passed busy+idle==span and no-double-count without
  raising. Consumer wall-critical thread = 98.5% WAIT (ttp.rx_recv_block), 1.0%
  compute -- matching the known project_confirmed_offset_prefetch_gap head-of-line
  stall. SUM-vs-SELF on real data: worker.scan_candidate SUM=226ms SELF=41ms;
  pool.run_task SUM=245ms SELF=100us (proves the slack-mask trap is defused).
- Byte-transparency: trace ON sha == trace OFF sha == gzip ref sha.

## ATTACK THESE (the disproof targets)
1. Does LEAF attribution actually keep busy+idle==span EXACTLY, or only when
   spans nest cleanly? What about overlapping (non-nested) spans on one thread,
   zero-length spans, or B/E mismatch? Could a real trace violate it silently?
2. Is the consumer = "max-span thread" heuristic correct as "the wall"? Could a
   worker thread have a larger span and steal the wall-critical label, inverting
   the WAIT/COMPUTE story?
3. Is the WAIT taxonomy COMPLETE? Name any wired span that is a blocking-on-
   another-thread wait but is currently classified compute/overhead/unknown
   (that would re-introduce the inversion). Conversely, any compute mislabeled
   wait (would hide a real engine cost)?
4. Could the routing guard be FOOLED into certifying a seeded/oracle run as
   production (e.g. a counter line format it doesn't match, so it reads as no
   sidecar => inconclusive but the user proceeds)? Is "window_seeded=1" (benign
   chunk-0 seed) correctly handled vs a fully-seeded oracle run?
5. Does the positive control actually prove the instrument LOCALIZES (moves the
   right stage), or could it pass while smearing? Is the selftest's synthetic
   structure representative enough to catch the real failure classes?
6. The honest gap: this is the PERF-FREE tier (no PMU). Does it OVER-claim? It
   reports descriptive structure + a routing/seeding gate; it does NOT itself run
   a causal perturbation (that is GZIPPY_SLOW_BOOTSTRAP + the locked-guest wall,
   which the owner runs). Is the tool clear that descriptive != causal, or could
   a reader mistake a SELF-time ranking for a binder verdict?

Render a verdict per claim: UPHELD / UPHELD-WITH-CAVEATS / REFUTED, with the
specific code line or reasoning. If REFUTED, say exactly what to fix. Write your
verdict to plans/fulcrum-total-instrument-advisor-verdict.md.
