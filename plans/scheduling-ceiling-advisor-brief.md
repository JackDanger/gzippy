# Disproof-advisor brief — scheduling/serial ceiling (owner turn, HEAD f1aceee1)

You are an INDEPENDENT DISPROOF advisor. Read-only, source-verify, NO build/measure.
Your job is to BREAK the conclusions below, not ratify them. Cite file:line.

## Context
gzippy parallel single-member decode is a faithful port of rapidgzip. The campaign
goal is a T8 whole-system wall TIE with rapidgzip on silesia (68MB, sha
028bd002…cb410f). Prior turn's premise (handed to me by the supervisor): "the
per-thread ENGINE COMPUTE is slack-masked at Fill 87%; the T8 binder is the
SCHEDULING/SERIAL term (pool-fill gap + in-order consumer head-of-line future::get
wait); Total Real Decode 0.137s ≈ rg's whole wall, so closing the overlap reaches
the tie." I was asked to bound that scheduling/serial ceiling with a removal oracle.

## First-hand numbers (locked guest 10.30.0.199, gzippy-isal native, sha-exact every cell, interleaved measure.sh)
T8 verbose (GZIPPY_VERBOSE):
| metric | gzippy HEAD | rapidgzip | ratio |
| decodeBlock SUM | 0.803s | 0.502s | 1.60× |
| Theoretical Optimal (÷T) | 0.100s | 0.068s | 1.47× |
| Total Real Decode | 0.116s | 0.084s | 1.38× |
| std::future::get | 0.089s | 0.064s | 1.39× |
| serial tail (wall−RealDecode) | 0.058s | ~0.043s (0.033 applyWindow + 0.010 checksum) | 1.35× |
| WALL (interleaved best) | 0.174-0.177s | 0.130s | 0.736-0.755× |

T16: HEAD wall 0.162s = 0.885× rg (rg 0.144s — rg gets SLOWER at T16 on 17 chunks);
HEAD future::get 0.089→0.046s (HALVES with cores); Real Decode 0.116→0.104; serial
tail 0.058 (FIXED). seedfull wall 0.128s flat = 1.121× rg WIN.

REMOVAL ORACLE (seedfull = GZIPPY_SEED_WINDOWS, the only faithful overlap-removal
available): all 17 chunks window-seeded → CLEAN engine, 0 spec-failures, Fill 90%.
T8 wall 0.128s = 1.029× rg TIE. sha-exact. BUT seedfull ALSO removes the engine
premium (decode goes clean) — it is an OVER-removal, not a pure scheduling isolation.
CRITICAL: seedfull's future::get = 0.083s ≈ HEAD's 0.089s — the consumer head-of-line
wait is NEARLY ENGINE-INDEPENDENT and persists even in the tying cell.

## Conclusions to BREAK
- **C1 (premise REFUTED):** The prompt's "engine is slack-masked, binder is
  scheduling/serial" is WRONG post-mergefix. The engine premium (Theoretical Optimal
  1.47×) REACHES THE WALL via Real Decode (0.116 vs rg 0.084 = 1.38×). Of the T8
  wall gap (0.174−0.130 = 0.044s), ~0.032s is the DECODE floor (Real Decode delta),
  only ~0.012-0.016s is the serial/scheduling tail. The "Real Decode ≈ rg's whole
  wall" claim used a STALE pre-mergefix 0.137s; the current 0.116s is below rg's WALL
  but ABOVE rg's Real Decode 0.084 — floor-to-floor the engine still binds.
- **C2 (scheduling is NOT the binder):** future::get scales away with cores
  (0.089→0.046 T8→T16) and is nearly engine-independent (0.089 HEAD ≈ 0.083 seedfull),
  i.e. it is mostly OVERLAPPED, not on the critical path. The serial tail (~0.058s) is
  fixed and comparable to rg's (~0.043s); it is a minor term, not the binder.
- **C3 (ceiling):** A PURE scheduling/serial fix (perfect overlap, engine unchanged)
  lands at ~RealDecode+serial ≈ 0.116+0.043 ≈ 0.159s ≈ 1.22× rg = STILL A LOSS (F2
  fired). The tie is NOT reachable by closing the overlap alone. seedfull ties ONLY
  because it ALSO removes the engine premium — and engine-clean is architecturally
  COUPLED to window-overlap (a present window => clean decode; the marker engine runs
  precisely because windows aren't overlapped in time). So the two are inseparable in
  gzippy, and the faithful tie requires the engine to be fast on the WINDOW-ABSENT
  marker path (as rg's is: rg decodeBlock 0.502 vs gzippy 0.803 = 1.6×, both
  window-absent / 34.5% replaced markers).
- **C4 (scoped fix for next loop):** the binder is the WINDOW-ABSENT u16 MARKER ENGINE
  rate (1.6× rg in SUM, reaching the wall via Real Decode 1.38×). NOT the merge (gone),
  NOT the marker fast-loop literal pipelining (landed, TIE — the unchanged O(length)
  backward marker scan in emit_backref_ring::<true> marker_inflate.rs:3006-3027 is the
  prime remaining marker-mode cost per the prior advisor). Next: profile/attack the
  marker-mode back-ref resolve (the backward marker scan) which the literal-only fast
  loop left untouched.

## Falsifier I pre-registered (plans/scheduling-ceiling-prereg.md)
F1 (tie in scheduling) wall ≤1.05× rg; F2 (engine binds) ≥1.15×; F3 partial.
I conclude F2 via the seedfull decomposition + arithmetic (no pure-scheduling oracle
exists because of the window/clean coupling). ATTACK whether the arithmetic
(0.116+0.043≈0.159) is a legitimate ceiling bound or a forbidden extrapolation
through an unlocated knee (Measurement PROCESS Rule 3). Is seedfull a fair removal
oracle for the engine premium? Is the window/clean COUPLING claim correct in source?

Write your verdict (UPHELD / UPHELD-WITH-CAVEATS / REFUTED per claim) to
plans/scheduling-ceiling-advisor-verdict.md.
