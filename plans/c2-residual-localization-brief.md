# C2 non-engine residual — localization + removal-oracle bound (owner turn 2026-06-08)

## Mandate
Close the 21ms NON-ENGINE residual on the ocl_cf (ISA-L-engine) path. The supervisor
gate said: with the engine REMOVED (ISA-L), WHERE is the 21ms, and is it the named
candidate project_confirmed_offset_prefetch_gap (head-of-line stalls at confirmed
offsets ≠ partition guess, ~40% of T8 wall)? Bound it with a removal oracle.

## Host / build provenance
- Host FROZEN via neurotic /root/clock_freeze.sh (no_turbo=1, gov=performance, cpu pinned 1.4GHz, LXC 111/105 cgroup-frozen). Readback no_turbo=1 governor=performance.
- Build: gzippy-isal feature @ HEAD f98af1f, RUSTFLAGS=-C target-cpu=native, bin_sha=2f8667619679b08c, /root/gzippy-bench (the GUEST_SRC pin).
- Corpus /root/silesia.gz, raw 211968000, raw_sha 028bd002…cb410f (the oracle).
- All runs byte-exact (out_sha == 028bd002… ref) and window-absent-preserving (flip_to_clean=12 finished_no_flip=4 window_seeded=2 on BOTH native and ocl_cf; ocl_cf isal_oracle_chunks=14 isal_oracle_fallbacks=0 — coverage 14/0, symmetry ✓).

## STEP 1 — LOCALIZATION (fulcrum_total trace + consumer_block_decompose.py, consumer tid=1)
Captured TWO production-routed traces (GZIPPY_TIMELINE, no seeding):
- native  (production pure-Rust)
- ocl_cf  (GZIPPY_ISAL_ENGINE_ORACLE=1 — clean tail decoded by REAL ISA-L)

Consumer-thread (tid=1) self-time decompose (conservation gap 0.03%, valid):

| term | native | ocl_cf |
|---|---|---|
| consumer_wall | 0.3899s | 0.3851s |
| DECODE_WAIT (sum of wait.*/rx_recv) | 0.2508s (64.3%) | 0.2463s (64.0%) |
| SERIAL_BOOKKEEPING | 0.1390s (35.7%) | 0.1387s (36.0%) |
| consumer.writev (self) | 0.1206s | 0.1289s |
| wait.block_fetcher_get (self) | 0.2229s | 0.1214s |
| ttp.rx_recv_block (self) | 0.0143s | 0.0980s |
| consumer.window_publish_marker / wait_replaced_markers / get_last_window | ~0.0001s each | ~0.0001s each |

KEY OBSERVATIONS:
1. Swapping the clean-tail engine to ISA-L barely moves consumer_wall (0.3899→0.3851, ~5ms) and does NOT shrink DECODE_WAIT (0.2508→0.2463). The wait just MIGRATES between span types (cold block_fetcher_get → rx_recv pump). The consumer waits ~64% of wall on chunk availability REGARDLESS of engine speed.
2. The serial marker-resolve/publish WORK (window_publish_marker, wait_replaced_markers, get_last_window) is ~0ms on the consumer — apply_window/marker-resolution already runs OFF the consumer critical path (faithful to rapidgzip; queue_prefetched_marker_postprocess on the pool). So the residual is NOT consumer-serial bookkeeping.
3. The dominant SERIAL term is consumer.writev (~0.12-0.13s) — the shared 211 MiB output materialization floor rapidgzip also pays (per the prior output-reconciliation: rg pays ~15.7ms exposure too).

## STEP 2 — REMOVAL-ORACLE BOUND (pre-registered falsifier)
Falsifier (pre-registered): if removing the head-of-line dispatch gap on ocl_cf does
NOT move ocl_cf toward rg (Δ < spread) with warm_hit_frac ~1.0, then
project_confirmed_offset_prefetch_gap is NOT the residual.

GZIPPY_PERFECT_OVERLAP removes term (2) ONLY (the dispatch-TIMING head-of-line gap:
every chunk in flight from t0; KEEPS the marker engine + serial resolve chain + writev).
This is the validated removal oracle for the named candidate.

Interleaved best-of-N=13, same /dev/shm sink, sha-verified (0 divergence):

| contender | min | ratio vs rg |
|---|---|---|
| A ocl_cf | 0.3950s | 0.983× |
| B ocl_cf + PERFECT_OVERLAP | 0.3958s | 0.981× |
| C rapidgzip | 0.3881s | 1.000× |

DELTA A→B (overlap removal) = **-0.0008s (-0.2%) = FLAT** (≪ the 14-18% spread).
self-test warm_hit_frac=0.882 (88% of head-of-line stalls removed — the 2 residual
are the offset-0 startup misses; even removing 88% moved nothing).

Confirmation (clean N=17 ocl_cf vs rg): ocl_cf 0.4015 vs rg 0.3927 = 0.978×, residual ~9ms.

## VERDICT (mine — seeking disproof)
1. **The named candidate (project_confirmed_offset_prefetch_gap / head-of-line dispatch gap) is REFUTED as the 21ms residual.** The PERFECT_OVERLAP removal oracle is FLAT on ocl_cf (Δ -0.2%, warm_hit_frac 0.88). Removing the dispatch-timing gap does not move ocl_cf toward rg.
2. **The residual is NOT consumer-serial bookkeeping** — the resolve/publish spans are ~0ms; apply_window already runs off the consumer crit path (already faithful).
3. **The residual decomposes into the advisor-predicted MIX (clean-rate-ceiling C2 caveat):**
   (a) marker-region pure-Rust BOOTSTRAP compute — the ISA-L engine oracle swaps ONLY the clean tail (finish_decode_chunk), leaving the pre-flip u16 marker decode pure-Rust on all 14 chunks. This is engine-CLASS compute, NOT scheduling. It would close only with a faster marker engine (pure-Rust inner-loop work, the engine fork) or ISA-L on the bootstrap (NOT faithful — bootstrap is window-absent, no ISA-L counterpart).
   (b) the shared consumer.writev / memory-bandwidth output floor (~0.12-0.13s), which rg also pays — gzippy pays marginally more.
4. The residual on THIS frozen host is ~7-12ms (0.978-0.983×), smaller than the banked 21ms (host-load dependent; structure identical: ocl_cf consistently < rg by single-to-low-double-digit ms).

## Implication for the campaign
There is NO faithful scheduling/head-of-line fix that moves the matched wall — the
named lever is dead by removal oracle. The residual is engine-class (marker bootstrap)
+ shared output floor. This means the C2 residual does NOT obviously yield a low-risk
faithful tooth distinct from the engine work; the marker-bootstrap term is the same
inner-loop pure-Rust-vs-asm gap as the engine, just on the u16 marker path. The
"KEEP-GRINDING C2 instead of escalate the engine fork" recommendation rested on C2
being scheduling/marker-region/bootstrap = lower-risk faithful-port territory; the
removal oracle shows the scheduling sub-term is null and the bootstrap sub-term is
engine-class. This is a material update to the fork decision.

## STEP 3 — ADVISOR-OWED MEASUREMENTS (both already-built, ran after disproof pass 1)
The disproof advisor (plans/c2-residual-disproof-verdict.md) flagged two cheap removals
to convert "argued"→"removed". Both ran:

### (3a) warm_miss CAUSE (does the unremoved 12% = the costly overshoot-tails?)
ocl_cf + PERFECT_OVERLAP, verbose: warm_chunks=17 hits=15 **misses=2**, and
**Prefetch guard-rejects = 1**. ⇒ of the 2 warm-misses, exactly 1 is the costly
overshoot-tail discard (matches_encoded_offset==false) and 1 is startup-absent.
The interior-reuse/overshoot mechanism (the memory's CONCLUSIVE root cause) fires on
exactly **ONE** chunk this corpus/config (the old memory's "4 stalls / 3 overshoot"
does not reproduce here). Even with that 1 costly stall present in BOTH overlap arms,
the wall was flat ⇒ interior reuse is a 1-chunk, negligible lever. FACT-A concern RESOLVED.

### (3b) SEEDFULL removal bound (the marker-BOOTSTRAP removal oracle, Rule 3 — not a slope)
ocl_cf vs ocl_cf+GZIPPY_SEED_WINDOWS (seedfull flips every chunk to the clean engine,
REMOVING the pre-flip marker bootstrap). Interleaved, A byte-exact, B masks-binder CEILING.
Two passes (heavy host load this window, spread 85-131%):
- Pass 1 (N=13): A 0.4070 (0.989×) / B 0.4093 (0.983×) / rg 0.4024 → A−B = **−2ms** (flat/neg)
- Pass 2 (N=15): A 0.4192 (0.966×) / B 0.4108 (0.985×) / rg 0.4048 → A−B = **+8ms**
⇒ the marker-BOOTSTRAP term is SMALL and noisy: ≤8ms UPPER BOUND (B is a masks-binder
CEILING, not byte-exact — a faithful removal would help even less). Crucially, even
fully removed, **ocl_cf+seedfull = 0.983-0.985× still does NOT reach 1.0×** — a ~6ms
residual remains BEYOND the bootstrap = the writev/memory-bandwidth output floor (which
the PRIOR output-reconciliation turn established as largely SHARED with rg, ~15.7ms rg
exposure; not re-measured as shared here).

## REVISED VERDICT (post advisor-owed measurements)
- **Scheduling/head-of-line (dispatch DEPTH): DEAD** (PERFECT_OVERLAP flat, validated removal).
- **Interior-reuse/overshoot (dispatch TARGETING): 1 chunk, negligible** (guard-rejects=1; flat in both arms; old memory's 3-for-3 failed patches + near-zero payoff stand).
- **Marker-bootstrap pure-Rust compute: SMALL (≤8ms ceiling, masks-binder upper bound), engine-class, and INSUFFICIENT** — seedfull-removed ocl_cf still 0.983-0.985×, not 1.0×.
- **Dominant residual = the consumer.writev / memory-bandwidth output floor (~0.12-0.13s).** The PRIOR output-reconciliation turn established this floor as largely SHARED with rg (~15.7ms rg exposure); that shared character is INHERITED, not re-measured here. The seedfull plateau (~6ms surviving full bootstrap removal) is fresh removal-based confirmation the remainder is this floor, not by-elimination.
- ocl_cf is at **0.966-0.989× rg (residual ~5-14ms), TIE-by-spread** on this (loaded) host. There is **NO located faithful lever that moves the matched wall toward 1.0×** distinct from the engine — the scheduling teeth are null, the bootstrap tooth is small + engine-class, and the output floor is shared/irreducible.

FORK IMPLICATION: the "KEEP-GRINDING C2 (lower-risk faithful scheduling)" recommendation
is now substantially weakened — C2 has no demonstrated low-risk wall-moving faithful tooth.
The remaining gap to rg is (a) the engine (native→ocl_cf, the inline-asm fork) and (b) a
small shared/irreducible output floor. This does NOT by itself escalate the engine fork
(the residual is small and partly shared, not provably engine-only), but it removes C2 as
the cheaper alternative. NOTE: host was loaded (loadavg ~4, spread 85-131%) — a quiet-box
re-measure would tighten the bootstrap split, but the qualitative verdict (all C2 sub-terms
flat-to-small) is robust across passes.

## Disproof asks
- Is the PERFECT_OVERLAP flat result a valid REMOVAL (Rule 3), or does warm_hit_frac
  0.88 (not 1.0) leave enough residual stalls to mask a real overlap win?
- Is attributing the persistent DECODE_WAIT to the marker bootstrap (vs some other
  worker-side serialization) sound, given I did NOT run a marker-bootstrap removal
  oracle (only the SLOW_MARKER_MODE perturbation knob exists, a slope not a removal)?
- Does the ~5ms consumer_wall move (native→ocl_cf) under-credit the engine because the
  trace is single-shot? (The best-of-N residual is 7-12ms, consistent.)
- Is the conclusion "no faithful low-risk C2 tooth" correct, or is there a faithful
  rapidgzip structural difference in the marker bootstrap or output path I'm missing?
