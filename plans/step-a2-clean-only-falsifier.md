# STEP A.2 — CLEAN-ONLY ENGINE ORACLE — PRE-REGISTERED FALSIFIER (2026-06-07)

Leader (fresh instance). HEAD c829982. Charter: plans/step-a2-clean-only-oracle.md.
Corrected picture: plans/step-a-oracle-advisor-verdict.md + memory
project_pregate_placement_is_dominant_lever — PLACEMENT and ENGINE are CO-PRIMARY.
This oracle bounds the ENGINE (class-C) ceiling cleanly, resolving the grey 0.4-0.7s
that the DEGENERATE Oracle-C (decode≈0 → publish-chain collapse) could not.

## WHAT THIS ORACLE MEASURES (and why it is NOT Oracle-C)
Oracle-C removed the decode COMPUTE (bypass replay, decode≈0). That co-removed the
window-arrival latency → L_resolve collapsed 162µs/chunk → DEGENERATE (publish-chain
artificially free). VOID for bounding the engine.

This oracle does the OPPOSITE: it KEEPS the real Huffman decode running at full cost,
but forces EVERY chunk down the CLEAN (window-present) decode path
(`decode_chunk_with_until_exact`) instead of the markered speculative path
(`speculative_decode_find_boundary`). It does this by providing each worker the CORRECT
predecessor 32KiB window (captured from a prior real decode) when the live WindowMap has
not yet published it — instead of falling to speculation.

Effect:
- Engine COMPUTE is fully preserved (real Huffman decode of every byte, once, via the
  clean path). This measures gzippy's CLEAN per-chunk decode rate at T8 — the exact
  quantity (91 ms/chunk vs rapidgzip 39 ms) the advisor flagged as the cleanest engine
  signal both prior oracles missed.
- The publish-chain is PRESERVED: we do NOT touch the consumer publish/apply_window/
  ordering path. The seeded window is ONLY a clean-path FALLBACK at the worker routing
  decision (chunk_fetcher.rs:2147-2154) when window_map.get() returned None. The
  consumer still publishes real windows, still orders, still writes. There are simply no
  MARKERS to resolve (clean decode emits no u16 markers) — which is the whole point: the
  engine residual is the clean-decode cost with the coordination chain intact.

## MECHANISM (the instrument)
New env, byte-exact (additive only, no decode-behavior change when off):
- `GZIPPY_SEED_WINDOWS_CAPTURE=<file>`: during a normal decode, record every published
  window keyed by encoded_offset_bits (32KiB each) to <file>. Reuses the existing
  WindowMap publish path (consumer.window_publish_clean / publish_end_window_before_post_process).
- `GZIPPY_SEED_WINDOWS=<file>`: load <file> at drive start into a side store. At the
  worker routing decision, when window_map.get(start_bit)==None AND the seed store has
  a window for start_bit, USE the seeded window (clean path, until_exact=stop_hint_is_exact)
  instead of speculation. start_bit==0 unchanged (zero window). On a seed MISS, fall to
  the normal path (speculation) so output stays byte-correct.

Output MUST be byte-identical to a normal decode (sha == ref). The seeded window is the
CORRECT predecessor window, so the clean decode produces identical bytes. This is the
correctness gate — a clean-only oracle that produced wrong bytes is VOID.

## SELF-TEST / POSITIVE CONTROL (CLAUDE.md rule 4 — must pass BEFORE the number counts)
1. **Forced-clean proof (the core self-test).** In the seeded run, the trace
   `worker.decode_mode` mode=window_absent fraction → ~0 and marker-decode events
   (speculative_decode_find_boundary invocations / data_with_markers nonzero chunks) → ~0.
   Concretely: count `worker.decode_mode` instants with mode="window_absent" in the seeded
   T8 trace; assert it is ≪ the unseeded baseline (target: near 0, allowing only the
   irreducible head-of-line chunks that have no captured seed — report the exact count).
   If window_absent fraction is NOT driven to ~0, the oracle did NOT force the clean path
   ⇒ VOID.
2. **Publish-chain-preserved proof (distinguishes from degenerate Oracle-C).** In the
   seeded floor trace, L_resolve / consumer window-publish per-chunk time must remain at
   REAL decode-speed magnitude (NOT collapse to ~162µs as Oracle-C did). Specifically:
   the consumer.window_publish_clean spans + post-process must still occur per chunk and
   the publish chain critpath must be of the same order as a real run, NOT ~0. If the
   publish chain collapsed, the oracle is degenerate ⇒ GREY (same defect as Oracle-C).
3. **Off==identity.** GZIPPY_SEED_WINDOWS unset ⇒ byte-identical AND wall-identical to
   today (the env-gated branch is an inlined early-return). DUAL self-test: seeded sha ==
   unseeded sha == gzip ref.
4. **Hit accounting.** Report seed hits / misses / total chunks. A high miss rate means
   the oracle did not force clean for those chunks (they fell to speculation) ⇒ the wall
   is contaminated by the marker path; <90% clean ⇒ treat as a loose UPPER bound only.

## PREDICTION TO BEAT (pre-registered, before running)
From the advisor's first-hand re-derivation (plans/step-a-oracle-advisor-verdict.md):
gzippy clean = 91 ms/chunk; ~42 chunks; T8 best-case placement floor ≈ 42×91/8 ≈ 0.48s,
×1.36 ramp ≈ 0.65s. So the CLEAN-ONLY T8 wall is predicted ≈ **0.60–0.70s** (clean
engine rate with the publish chain intact, all chunks clean). rapidgzip = 0.524s.

PRE-REGISTERED VERDICT BANDS (T8 clean-only wall, sha-OK, self-tests pass):
- **ENGINE-IS-THE-RESIDUAL (co-primary confirmed):** clean-only wall in [0.58, 0.72]s,
  i.e. > rapidgzip 0.524 by a margin > inter-run spread. ⇒ the engine gap survives even
  with EVERY chunk clean and placement-perfect-ish (windows always available) ⇒ class-C
  is a real co-lever; the 2.3× clean-rate gap is confirmed at the wall, not just per-chunk.
- **ENGINE-NEGLIGIBLE (would REFUTE co-primary):** clean-only wall ≤ 0.55s (TIE-or-better
  with rapidgzip within spread). ⇒ gzippy's clean engine is NOT the residual; the gap is
  ENTIRELY placement/coordination ⇒ revert to placement-sole and DROP the inner-loop ASM
  project. (This would CONTRADICT the advisor's 91-vs-39 per-chunk derivation — reconcile
  before accepting.)
- **AMBIGUOUS:** wall in (0.55, 0.58) grey — lean on the per-chunk clean busy rate
  (isal_stream_inflate / clean decode busy ÷ chunk count) vs rapidgzip's 39 ms/chunk,
  and on the self-test publish-chain magnitude.

## ENGINE CEILING (the deliverable)
Report: (a) the clean-only T8 wall (min/median, N≥9, sha-OK); (b) the per-chunk CLEAN
busy rate (gzippy clean decode busy ÷ chunks) vs rapidgzip 39 ms/chunk, RAMP-CONSISTENT
(apply the same actual/makespan ramp to both, or compare floor-to-floor — NOT floor-vs-
wall, the STEP-A error); (c) the implied engine-bounded T8 wall = clean rate × chunks ÷
8 × ramp. This is the cleanly-bounded ENGINE ceiling.

## DISPROOF ATTEMPTS (actively try to BREAK the conclusion)
- Did pre-seeding make the wall artificially FAST by removing window-arrival latency
  (the dual of Oracle-C)? CHECK: self-test #2 — if the publish chain still costs real
  time per chunk, the latency removal is bounded to the head-of-line stalls only. If the
  wall collapses below the per-chunk clean busy floor, the latency-removal contaminated it
  ⇒ GREY.
- Frequency: run is on the locked guest (no_turbo=1, governor=performance) so turbo
  artifacts are pinned out (same as all locked-harness runs).
- Δ < inter-run spread ⇒ TIE (CLAUDE.md). A clean-only wall within spread of rapidgzip
  REFUTES co-primary; a clean margin CONFIRMS it.

## RUN PROTOCOL
- Build via scripts/cargo-lock.sh (ONE at a time; df -h around). Guest VERIFIED IDLE
  before each run; RESTORE host after (no_turbo=0, thaw, clean /dev/shm).
- Locked guest harness (the same machinery as guest_ceiling.sh); silesia-large 503MB;
  T8; N≥9 interleaved; sha-verified every run (seeded == gzip ref).
- Capture pass (GZIPPY_SEED_WINDOWS_CAPTURE) to /dev/shm (RAM), then interleaved A/B:
  gzip_ref | gzippy_seeded(clean-only) | rapidgzip | gzippy_normal.
- Self-test traces: one seeded T8 trace (assert window_absent→~0, publish-chain intact)
  + the unseeded baseline trace for the window_absent fraction comparison.
