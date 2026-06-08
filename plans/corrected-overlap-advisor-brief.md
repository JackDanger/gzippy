# Disproof-advisor BRIEF — CORRECTED overlap oracle (GZIPPY_PERFECT_OVERLAP)

You are an INDEPENDENT, READ-ONLY disproof advisor. Branch reimplement-isa-l,
HEAD 7aae6c4a + corrected oracle overlay. Source-verify first-hand. Builds/guest
numbers as reported (do NOT re-run; read-only).

Convention: **wall ratio = rg_wall / gzippy_wall.** 1.00 = tie. <1.00 = gzippy
slower. Tie threshold F1 (≤1.05× rg): gzippy_wall ≤ 1.05 × 0.130 = 0.137s.

## What changed vs the PRIOR (advisor-refuted) oracle
The prior oracle ran warm-all-then-drain: it `recv()`'d every chunk to completion
BEFORE the consumer started, serializing decode and drain (an ANTI-overlap; wall
0.225s = 0.58× rg, SLOWER than production 0.177s). You (prior verdict,
plans/perfect-overlap-advisor-verdict.md) correctly refuted it: it could not decide
F1, and its warm-alone lower bound 0.117s was inside the tie zone, so the tie
"should" be reachable via real overlap.

THE CORRECTION (src/decompress/parallel/chunk_fetcher.rs `perfect_overlap_warm` +
perfect_overlap.rs): now dispatch EVERY chunk's decode as an IN-FLIGHT prefetch
up-front via `block_fetcher.submit_prefetch(part_key, rx)` (the vendor
`m_prefetching.emplace` API, BlockFetcher.hpp:558) — NON-BLOCKING, NO recv — then
return IMMEDIATELY. The unchanged in-order `consumer_loop` then runs CONCURRENTLY:
`try_take_prefetched_pumping(&part_key)` takes the head chunk's in-flight receiver
and blocks on it WHILE chunks i+1.. keep decoding on the pool. That is decode↔drain
OVERLAP. Removes ONLY the prefetch-dispatch-DEPTH term (every chunk in flight from
t0; the named project_confirmed_offset_prefetch_gap dispatch-TIMING gap). KEEPS the
real marker engine (NOT window-seeded), the serial resolve chain, drain, write.

## Self-test (Rule 4) — VALIDATED
- OFF==identity AND ON byte-exact: sha 028bd002…cb410f on BOTH arm64-native (local)
  AND x86_64 gzippy-isal native (guest). path=ParallelSM asserted.
- dispatch phase = 0.0006-0.0008s (non-blocking — confirms it returns immediately,
  unlike the prior 0.117s blocking warm).
- warm_hit_frac = 0.882 (15/17; the 2 misses are offset-0 stream-start chunks where
  partition_offset==next_block_offset so the consumer skips the prefetch path — on
  the critical path regardless, same as prior turn).

## THE DECIDER NUMBER (T8, measure.sh interleaved N=11, sha-OK every run, CPUS=0,2,4,6,8,10,12,14, locked guest 10.30.0.199, gov=perf, turbo-on, load 0.5-0.85; 3 runs)
| contender | best wall | vs rg | verdict |
| rg (rapidgzip 0.16.0) | 0.1295-0.1307s | 1.000 | — |
| head (production)     | 0.1763-0.1773s | 0.730 / 0.735 / 0.741 | LOSS/TIE |
| perfovl (CORRECTED)   | 0.1864-0.1891s | 0.686 / 0.695 / 0.693 | LOSS |

**The corrected overlap oracle is sign-stably ~5-7% SLOWER than production and does
NOT reach the tie. It is a TIE-with-production at best, a slight LOSS at worst.**

## WHY (verbose single-run, head vs perfovl)
| metric | head | perfovl |
| decodeBlock SUM | 0.838s | 0.796s |
| std::future::get (in-order consumer WAIT) | 0.1038s | 0.1024s |
| Total Real Decode | 0.140s | 0.1246s |
| Theoretical Optimal (÷T) | 0.1048s | 0.0995s |
| Pool Fill | 74.8% | 79.9% |
| overlapped wall (single-run) | (0.177 interleaved) | 0.1441s single |

The oracle DID improve decode-phase Fill (74.8→79.9%) and shaved Real Decode
(0.140→0.125). But **std::future::get — the in-order consumer head-of-line WAIT —
barely moved (0.1038→0.1024s).** That is the binder, and it does NOT shrink when all
chunks are dispatched at t0, because the head-of-line wait is on each chunk's DECODE
COMPLETING (the slow marker engine), not on the chunk being dispatched.

## MY CLAIMS (disprove them)
- **C1 (DECIDER):** the corrected, validated overlap oracle REFUTES the F1 premise
  "the T8 tie is reachable via decode↔drain overlap / earlier dispatch." Flooding all
  chunks in flight from t0 (the dispatch-depth fix the prior lower-bound implied) does
  NOT collapse the wall — it lands ~equal-to-or-slower-than production. The
  dispatch-TIMING / prefetch-depth term (project_confirmed_offset_prefetch_gap) is NOT
  the T8 binder.
- **C2 (the binder):** the binder is the in-order consumer wait (std::future::get
  ~0.10s, unmoved), which is gated by each chunk's MARKER-engine DECODE TIME — the
  serial resolve+write per-chunk does NOT fully hide under decode (drain ≈ co-equal
  with decode, NOT << decode as the lower bound assumed). The warm-alone lower bound
  0.117s was real but UNACHIEVABLE: it assumed drain hides under decode; it does not.
- **C3 (faithfulness):** the oracle is a faithful overlap (submit_prefetch is the
  vendor in-flight API; same partition keys; same speculative params; marker engine
  kept; byte-exact). Its slight slowness vs production is NOT an artifact of
  over-dispatch contention but a real signal that production's prefetcher already
  saturates the pool, so there is no dispatch-depth headroom to recover.

## DISPROOF ANGLES I WANT YOU TO PRESS
1. Could the oracle's slight SLOWNESS be an over-dispatch artifact (all 17 chunks in
   flight at once vs production's metered pump) rather than a real "no headroom"
   signal? If so, C3 is weakened — but does that change C1 (the oracle still doesn't
   reach the tie either way)?
2. Is std::future::get the right binder, or is it downstream of decode (i.e. it's
   WAITING on decode, so the real binder is the marker engine)? If the binder is the
   engine, that REVERSES the prior several turns (which deprioritized the engine as
   slack-masked). Reconcile with: seedfull (clean engine) ties; here the engine is
   marker (kept) and the wall doesn't move with overlap.
3. The single-run overlapped wall is 0.144s but interleaved best is 0.186s — the
   interleaved/contended number is what counts. Does the 0.144 vs 0.186 gap hide a
   measurement issue, or is it just contention (3 contenders interleaved)?
4. Does this oracle truly test "perfect overlap," or is there a BETTER overlap
   schedule it still fails to test (e.g. resolve-ahead / eager post-process deeper)?
   I.e. is C1 a clean refutation of F1, or only of THIS overlap variant?
5. Faithfulness of submit_prefetch keying: I key by partition_offset_containing_offset
   and skip if already in-flight. The consumer queries the same key. Any way this
   produces a window-seed or clean-flip that production wouldn't (which would make the
   oracle cheat like seedfull)? warm_hit_frac 0.88 + flip_to_clean=12 IDENTICAL in
   head and perfovl verbose argues no.

Give: each claim UPHELD / UPHELD-WITH-CAVEATS / REFUTED, the single most
load-bearing correction, and whether F1 can now be DECIDED (and which way).
Write your verdict to plans/corrected-overlap-advisor-verdict.md.

---

## FOLLOW-UP DATA (post-verdict, addresses advisor's "untested drain-hiding lever")

The advisor named resolve-ahead / off-consumer drain-hiding as the UNTESTED open
lever. First-hand verbose (locked guest, T8) shows it is ALREADY SATURATED:

| metric | head (production) | perfovl (corrected, retention-fixed) |
| Worker resolve-ahead | ok=13 / attempts=13 (100%) | ok=14 / attempts=14 (100%) |
| Eager post-process submitted | 13/17 | 14/17 |
| std::future::get | 0.086s | 0.080-0.100s |
| decodeBlock SUM | 0.827s | 0.827s |
| Theoretical Optimal (÷8) | 0.103s | 0.103s |

ALSO: retention-confound REMOVED (no 4096 cap bump; vendor sizing kept). Re-measured
interleaved 2 runs: perfovl STILL 0.684-0.693× rg (0.187-0.192s) vs head 0.735-0.754×
(0.174-0.177s). The slowness SURVIVED the retention fix — so it was NOT only the
retention blowup; the dispatch-flood itself adds no wall benefit.

NEW READING: drain-hiding (resolve-ahead) is at 100% in BOTH production and the
oracle. The consumer's std::future::get ~0.08-0.10s is a wait on the DECODE future
(marker-engine decode time), NOT on resolve (resolve is already hidden). With BOTH
scheduling levers exhausted (resolve-ahead 100% + dispatch-flood tested), the residual
~0.73× gap is the per-thread MARKER-ENGINE DECODE RATE (decodeBlock 0.827s vs rg
0.50s ≈ 1.6×; Theoretical Optimal 0.103s ÷ already near rg wall). rapidgzip ties
because its u16 marker decode is ~1.6× faster, not because it hides more drain.

QUESTION FOR FOLLOW-UP: does resolve-ahead ok=14/14 REFUTE the "untested drain-hiding
lever" rescue of F1? If drain is already fully hidden, then F1-via-overlap is refuted
and the binder relocates to the marker engine (rg's existence proof then says: match
rg's u16 marker decode rate, not its overlap).
