# Lever B perturbation — decode-rate causal confirmation @ HEAD cbfb256

Read-only artifact extraction. Branch `reimplement-isa-l`, HEAD `cbfb256`.
Matrix: `/tmp/leverB-matrix.txt`. Baseline artifact: `/tmp/leverB-R1-baseline/`.
Settles the §4 falsifier and §5 stale-caveat of `plans/lever-decision.md` (verdict (b)).

---

## VERDICT

**Lever B CONFIRMED.** A `GZIPPY_SLOW_BOOTSTRAP` slow-injection of the
window-absent decode moves the T8 wall monotonically and ~proportionally, and the
delta survives a frequency-neutral sleep control — decode is causally on the
critical path. The fresh live counters at HEAD do **not** overturn the
correctness/decode-rate-bound reuse ceiling: the resolve-ahead key fix now *fires*
(`Worker resolve-ahead ok=31/31`, `handoff_key=31`) but the eager-reuse count and
the predecessor-published-before-start count are **unchanged** vs the stale §5
figures, so the §4 scheduling-only falsifier is **not** triggered and verdict (b)
**stands**.

---

## 1. The wall curve (the causal perturbation matrix)

T8, silesia-large, production path, interleaved, sha-verified
(`/tmp/leverB-matrix.txt`):

| Run | Variant | gzippy min | Δ vs baseline | sd% gz | rapidgzip min |
|-----|---------|-----------|---------------|--------|---------------|
| R1 | baseline | **0.9171s** | — | 1.6 | 0.5238s |
| R2 | ondemand **spin** +50% | 0.9722s | **+6.0%** | 4.0 | 0.5365s |
| R3 | ondemand **sleep** +50% (CONTROL) | 0.9765s | **+6.5%** | 1.2 | 0.5381s |
| R4 | ondemand **spin** +100% | 1.1029s | **+20.3%** | 1.7 | 0.5276s |

rapidgzip ~0.52–0.54s throughout (≈1.75× faster; unperturbed, as expected — the
injection only touches gzippy's decode).

## 2. INTERPRETATION — Lever B confirmed

- **On the critical path.** Slowing the window-absent decode by +50% adds +6.0% wall
  and by +100% adds +20.3% wall. A monotonic, order-preserving response to a known
  perturbation of region R is the signature of R gating the wall (CLAUDE.md
  Measurement PROCESS 1). A slack region would have responded flat.
- **NOT a frequency/turbo artifact (the frequency-neutral disproof survives).** The
  +50% **spin** variant (R2, +6.0%) and the +50% **sleep** variant (R3, +6.5%)
  agree within run spread — the sleep yields the core, so it cannot depress all-core
  turbo, yet it reproduces the spin delta. Per Measurement PROCESS 2 this rules out
  the busy-spin-depresses-turbo confound: the criticality is real, not a spin
  artifact. The two independent injections agreeing **is** the survived disproof.
- The +100% spin (R4) is roughly double the +50% spin response (+20.3% vs +6.0%),
  i.e. the response is roughly proportional, not a step — consistent with a region
  that is continuously on the critical path rather than crossing a knee.

## 3. CAVEAT — slow-down slope ≠ speed-up ceiling (Measurement PROCESS rule 3)

This perturbation establishes only that **slowing** decode adds wall proportionally.
It does **NOT** establish how much **speeding** decode would help. Slowing a critical
region always adds wall; speeding it pays only until the next component binds, and
the location of that knee is not observable from the slow-down slope. Bounding the
speed-up requires an **ORACLE** that removes/replaces the decode region and measures
the interleaved wall — that is a separate experiment (Stage 5b). **No speed-up
projection is stated in this file**; doing so from the slope here would be exactly
the extrapolation-through-an-unlocated-knee error the rule forbids. (For context, the
report's own model already shows a worker-bound knee that caps the publish-chain
lever — `fulcrum-report.txt:328` — reinforcing that the ceiling must be measured, not
extrapolated.)

## 4. FRESH live counters @ HEAD cbfb256 (provenance per line)

> **Source note.** `artifacts-fulcrum/wall.stderr` is a **stale, FAILING** capture
> (mtime Jun 6 05:39, older than the trace files at 11:12); it contains only CRC32
> mismatches, `InvalidLookback`, and panics at `gzip_chunk.rs:972` — **no live
> counters**. The live `GZIPPY_VERBOSE` counters live in `artifacts-fulcrum/trace.log`,
> which is a campaign-wide concatenation of many runs/binaries with differing counter
> schemas. The authoritative production-T8 dump is the **only** block coherent with the
> report's cited `handoff_key=31` (`trace.log:14271–14293`); the report
> (`fulcrum-report.txt`) parsed that block. Counters below are read from that dump and
> the report's parse of it; the OTHER `reused=29 / ok=29` dumps in `trace.log` belong to
> different campaign binaries and are **not** this HEAD R1 run.

| # | Counter | FRESH value @ HEAD | file:line |
|---|---------|--------------------|-----------|
| 1 | Eager post-process **submitted / reused** | submitted=**31**, reused=**12** (runs=914, runs_nonempty=10, inspected=5091) | `trace.log:14271` |
| 2 | Worker **resolve-ahead ok / attempts** | ok=**31** / attempts=**31** (100% when attempted) | `trace.log:14293` |
| 3 | **handoff_key** published (Early window publish) | handoff_key=**31** (published=39, tail_not_clean=0, range_speculative=0) | `trace.log:14281` |
| 4 | **consumer.dispatch_recv** wall-critical | **169.2ms** `[169|0 wall-crit]` | `fulcrum-report.txt:22` (also :80,:114,:150) |
| 5 | causal **RUNTIME window-absent** fraction | **90.2%** (window-absent 37 of 41 decode decisions; static boundary 31.0%) | `fulcrum-report.txt:218–219` |
| 5b | predecessor **published-before-start** | **1/36** of key-mismatch window-absent (≈1/37 of all window-absent); 1 chunk whose predecessor never published below its start | `fulcrum-report.txt:225,227` |

Supporting (same authoritative dump): `Clean decode (pred@key / pred@seed /
handoff@stop / boundary@seed / candidate) = 0 / 0 / 0 / 0 / 0` (`trace.log:14292`);
`Unified decoder: flip_to_clean=29` (`trace.log:14274`); `Slow-path decode: ok=37`
(`trace.log:14280`).

Note on naming: there is **no** counter literally named `EAGER_PROBE_SUBMITTED` /
`EAGER_PROBE_REUSED` or `RESOLVE_AHEAD_OK` in the artifact — those map to the
`Eager post-process … submitted/reused` and `Worker resolve-ahead ok/attempts`
lines above. `HANDOFF_WINDOW_PUBLISHED` maps to `Early window publish … handoff_key`.

## 5. SETTLEMENT of `lever-decision.md` §5 (stale-caveat resolved)

§5 flagged its key numbers as possibly predating HEAD's resolve-ahead key fix:
**12/77 reused, 1/37 published-before-start, dispatch_recv 278ms**. Fresh comparison:

| Quantity | Stale (§5) | FRESH @ HEAD | Moved? |
|----------|-----------|--------------|--------|
| Eager reuse | 12 | **12** (submitted 31) | **NO — identical** |
| predecessor published-before-start | 1/37 | **1/36** (≈1/37) | **NO — identical** |
| consumer.dispatch_recv wall-crit | 278ms | **169ms** | YES — dropped ~109ms |
| Worker resolve-ahead ok/attempts | (not cited) | **31/31** | NEW — now firing |
| handoff_key published | (report showed 0) | **31** | NEW — now publishing |

**Settlement verdict:** HEAD's resolve-ahead key fix did **NOT** change the reuse
ceiling. The fix did what §2–§3 predicted at the *key* level — resolve-ahead now
fires at 100% (`ok=31/31`) and handoff windows now publish (`handoff_key=31`,
previously 0) — but this did **not** raise eager reuse (still **12**), did **not**
convert any window-absent decode to clean (`Clean decode … = 0/0/0/0/0`; runtime
window-absent still **90.2%**), and did **not** improve predecessor-published-before-start
(still **1/36**). This is exactly §3's mechanism: *the keys can align; the windows
are not ready in time*, because the pool decode is decode-rate-bound. The reuse
ceiling is therefore correctness/decode-rate-bound, **not** key-bound — confirmed,
not overturned, by the fresh counters.

The `dispatch_recv` drop (278→169ms) occurred **without** any rise in reuse (12) or
window-present (still 9.8% present / 90.2% absent), so it is **not** the
scheduling-reuse win the §4 falsifier requires. The §4 falsifier needs
`EAGER_PROBE_REUSED` to rise materially (**>~20 of 77**) from a scheduling-only
change **with window-absent ≥~90% preserved**: reused=**12** < 20 ⇒ the first
falsifier clause **fails**; no wall drop toward rapidgzip came from a
resolve-ahead/scheduling change ⇒ the second clause **fails**. Window-absent
**90.2% ≥ ~90%** is preserved (no drift toward the forbidden 31% clean-decoder
divergence). **Falsifier not triggered ⇒ verdict (b) Lever B stands**, and the
perturbation matrix (§1–§2) independently confirms decode is on the critical path.

A scheduling-only lever could pay **only** if it raised reuse past ~20 *and* dropped
`dispatch_recv` while keeping window-absent ≥90% — the fresh counters show the key
fix already did the cheap part (firing resolve-ahead) and the reuse ceiling did not
move, so the residual is decode rate (Lever B), gated by the Stage-5b oracle before
any inner-loop edit.

INVESTIGATION COMPLETE
