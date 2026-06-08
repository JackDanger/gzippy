# Disproof-advisor BRIEF — GZIPPY_PERFECT_OVERLAP oracle (the registered decider, FINALLY RUN)

You are an INDEPENDENT, READ-ONLY disproof advisor. Your job is to BREAK the
conclusions below, not ratify them. Cite file:line in this tree
(/home/user/www/gzippy-reimplement-isal, branch reimplement-isa-l, HEAD
7aae6c4a + the oracle overlay this turn). Verdict to
plans/perfect-overlap-advisor-verdict.md.

## Context
The registered decider oracle `GZIPPY_PERFECT_OVERLAP`
(plans/scheduling-ceiling-prereg.md §Oracle) had NEVER been run; the whole
"scheduling/serial is the T8 binder, removing it reaches the tie" direction
rested on it unmeasured (a live Rule-3 violation flagged by the supervisor
coach, plans/SUPERVISOR-FEEDBACK.md BOTTOM LINE #1). This turn it was BUILT,
self-tested (Rule 4), and run on the locked guest.

## The oracle (what was built — verify it is faithful + byte-transparent)
`src/decompress/parallel/perfect_overlap.rs` + wiring in chunk_fetcher.rs
(perfect_overlap_warm, ~line 2120; warm call before the timed consumer_loop,
~line 640; cache cap bump ~line 545; warm-hit/miss self-test counters at the
consumer try_take accept ~line 1325 and the blocking-get None branch ~line 1380).

Mechanism: a fully-parallel WARM phase decodes EVERY chunk on the real
production pool with the REAL window-absent MARKER engine (windows NOT seeded,
so the marker premium is KEPT at its true rate — distinct from seedfull which
flips to the clean engine, gzip_chunk.rs:790 vs :826), parking each finished
chunk in the prefetch cache. THEN the unchanged in-order consumer_loop runs:
every `get` is an instant warm-cache hit (head-of-line decode wait + pool-fill
gap removed); the serial marker-resolution window chain
(resolve_chunk_markers_on_chunk → get_last_window_vec, chunk_fetcher.rs:2553,
chunk i needs chunk i-1's resolved tail = rapidgzip's named serial critical
path GzipChunkFetcher.hpp:559) + drain + write are KEPT. Timed wall = warm +
drain (sequential, NOT overlapped) = a STRICT UPPER BOUND on perfect overlap
(prereg lines 30-34).

## Self-test (Rule 4) — VALIDATED
Output sha 028bd002…cb410f byte-IDENTICAL with/without the oracle on BOTH
arm64-native (local) AND x86_64 gzippy-isal native (guest). warm_hit_frac =
0.88-0.96 (the 2 misses are stream-start chunks at offset 0 with no partition
guess). So the oracle removed the head-of-line wait for ~all chunks and is
byte-transparent.

## NUMBERS (locked guest REDACTED_IP double-ssh, 16c gov=perf, taskset
0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, sha-OK every run, 2 runs)
| cell | wall | vs rg |
| rg (rapidgzip 0.16.0) | 0.131-0.132s | 1.000 |
| head (production) | 0.177s | 0.740-0.745× LOSS |
| **perfovl (PERFECT_OVERLAP oracle)** | **0.225-0.227s** | **0.581-0.583× LOSS** |

Warm/drain split (single un-interleaved guest run, 17 chunks): warm
(full-parallel decode-all, real marker engine, Fill→100%) = **0.117s**; drain
(serial resolve-chain + write) = 0.066s; total 0.183s.

Apples-to-apples --verbose (both T8, taskset-pinned):
- rg Theoretical-Optimal (perfectly-parallel decode floor) = 0.085s; Real
  Decode = 0.104s; decodeBlock SUM = 0.628s.
- gzippy oracle warm phase (= gzippy's perfectly-parallel decode floor) = 0.117s.

## CLAIMS to break (LEAD is the causal oracle number, per PROCESS FIX #2)
- **C1 (the decider result): removing the scheduling/overlap loss does NOT reach
  the tie.** The PERFECT_OVERLAP oracle — which removes the head-of-line decode
  wait + pool-fill gap (warm_hit_frac ~0.9, Fill→100%) — lands at 0.225s =
  0.58× rg, a BIGGER loss than production HEAD (0.74×). The pre-registered F1
  ("scheduling fix reaches the tie", ≤1.05× rg) is FALSIFIED.
- **C2 (the binder is the marker-engine decode floor, NOT scheduling): gzippy's
  perfectly-parallel decode floor (warm = 0.117s) is 1.38× rg's decode floor
  (0.085s) and 0.89× of rg's ENTIRE wall — with ALL scheduling loss removed.**
  So the irreducible term is the window-absent marker-engine COMPUTE rate
  (decodeBlock 0.803 vs rg 0.628 ≈ 1.3× per earlier; the floor gap is 1.38×),
  reaching the wall even under perfect overlap.
- **C3 (honest bound, NOT an over-reach): the oracle is a STRICT UPPER BOUND
  (warm+drain serialized).** Per the advisor's binding correction last turn (an
  upper-bound-loss cannot fire F2), this does NOT prove the perfect-overlap
  FLOOR is a loss. BUT the warm phase alone (0.117s, the genuine LOWER bound on
  the parallel decode floor with the real engine) is ITSELF 1.38× rg's floor —
  a lower bound above the tie — so the engine floor is implicated by a LOWER
  bound, not just the upper-bound sum. Is THAT inference sound? (It is the
  symmetric care the advisor demanded last turn — please check it both ways.)
- **C4 (next binder = the window-absent marker-engine compute rate).** The
  faithful path to the tie is a faster window-absent marker decode (rg ties
  unseeded at the same 34.5% markers because its marker decode is faster). NOT a
  pure-scheduling fix (this oracle refutes that). Caveat: the prior marker
  fast-loop port (04fda86d) was a TIE — so "faster marker engine" is not yet a
  proven lever; it needs its own ceiling oracle.

## Disproof angles requested
1. Is the warm phase a FAITHFUL "perfect overlap" of decode? (It decodes every
   chunk markered at the true rate, no seeding — confirm it does not accidentally
   seed/clean-flip. Check the warm decodes use is_speculative_prefetch=true at
   partition guesses, same as the prefetch path.)
2. Does the warm/drain SERIALIZATION inflate the oracle wall above what true
   perfect overlap would be (the upper-bound caveat)? If so, is C2's LOWER-bound
   argument (warm=0.117s alone) the correct way to rescue a real conclusion?
3. Is 0.117 vs rg's 0.085 a denominator-matched floor-to-floor comparison
   (both perfectly-parallel decode floors)? Any confound (warm includes pool
   spin-up; rg's Theoretical-Optimal is decodeBlock/parallelism)?
4. Does warm_hit_frac 0.88-0.96 (not 1.0) leave enough residual head-of-line
   wait to matter? (The misses are offset-0 stream-start chunks.)
5. Is C1 (F1 falsified) safe given the oracle is an upper bound? (An upper-bound
   LOSS does not by itself fire F2 — but does it safely falsify F1, which is a
   tie claim? A config that is SLOWER than production cannot be the tie-reaching
   config, regardless of bound direction.)
