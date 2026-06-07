# L_resolve investigation — is the ~75× per-link gap a marker-resolve COMPUTE lever or a WAIT artifact?

Read-only. HEAD `cbfb256`, branch `reimplement-isa-l`. Every claim cites a
`file:line` I opened. `cf` = `src/decompress/parallel/chunk_fetcher.rs`,
`sm` = `src/decompress/parallel/segmented_markers.rs`,
`cd` = `src/decompress/parallel/chunk_data.rs`,
`vendor` = `vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`.

---

## VERDICT (one line)

**The premise is wrong. The ~75× "L_resolve" gap (4.89 ms vs 0.065 ms) is a
MEASUREMENT ARTIFACT, not a marker-resolution compute gap.** The fulcrum `model`
defines `L_resolve` as the *inter-publish gap* `t_publish(i)−t_publish(i−1)`
(`plans/parallel-sm-model.md:70`), which telescopes to ≈ `wall / N` by
construction and is dominated by the consumer **blocking** (`wait.future_recv`)
on the pool's not-yet-finished **window-absent DECODE**. The actual
marker→byte replacement is a faithful single-pass LUT port that runs **on the
pool, off the critical path** (measured `apply_window` wall-critical = 0 ms).
There is **no faithful resolve-side change that closes the wall**; the confirmed
lever is decode rate (Lever B, `d_w` 1.77×), already proven by a
`GZIPPY_SLOW_BOOTSTRAP` causal perturbation.

---

## 1. MECHANISM — where the time actually is, with the vendor counterpart

### 1.1 The heavy marker→byte replacement runs ON THE POOL (both tools)

The O(chunk) replacement is `resolve_chunk_markers_on_chunk` (`cf:2355`), whose
core is `chunk.resolve_and_narrow_markers_in_place(predecessor_window)`
(`cf:2367` → `cd:1555` → `sm:461`). Its ONLY callers are:

- `run_post_process_in_place` (`cf:2540`, span `post_process.apply_window`),
- `run_post_process_task` (`cf:2580`, same span),

both dispatched through `submit_post_process_void` / `submit_post_process_task`
(`cf:2090`, `cf:2099`) at `thread_pool.submit(task, /* priority */ -1)`. That is
a byte-for-byte port of vendor's
`submitTaskWithHighPriority([chunkData, window]{ chunkData->applyWindow(...) })`
(`vendor:577-582`). **The resolve LUT pass never executes on the consumer
thread.** Prior read-only artifact at the same HEAD measured gzippy
`apply_window` busy = **1028 ms** but **wall-critical = 0 ms**
(`plans/structural-gap-analysis.md` §2 row 7).

### 1.2 The resolve impl is already faithful-or-better than vendor

`SegmentedU16::resolve_and_narrow_in_place` (`sm:461-497`) is a SINGLE pass over
the marker buffer using a **thread-local-reused 64 KiB u8 LUT**
(`APPLY_WINDOW_LUT`, `sm:467-478`): `base[i] = lut[src[i]]`, fusing resolve
(markers→bytes) and narrow (u16→u8) into one loop. The vendor counterpart is two
conceptual steps — `replaceMarkerBytes` via `std::transform` with
`MapMarkers` (`MarkerReplacement.hpp:49-59`) plus the u8 narrow in
`DecodedData::applyWindow` (`DecodedData.hpp:316-337`). gzippy's fused single
pass is, if anything, *more* efficient. There is **no 75× of compute hiding in
this function** — it is O(elements), branchless, vectorizable, LUT reused across
chunks. (Item (i): one-pass, not two; item (v): no per-call LUT rebuild — the
LUT is thread-local and reused, only the 32 KiB window slice is refreshed,
`sm:476`.)

### 1.3 The cheap serial publish the consumer DOES do is ~free

The consumer's own serial work per link is `get_last_window_vec` (`cf:1564` →
`cd:1185` → `cd:1195` `fill_last_window_into`), an O(W)=32 KiB tail build, plus
the eager scan `queue_prefetched_marker_postprocess` (`cf:2456`). The scan is
genuinely non-blocking: `process_ready_prefetches` is `try_recv` only
(`block_fetcher.rs:996-1026`), `prefetch_cache_contents_sorted` clones+sorts a
≤2·T Arc list (`block_fetcher.rs:583`), and each eligible chunk costs one
`window_map.get` (Arc clone, `window_map.rs:80`) + one 32 KiB
`get_last_window_vec` + one `submit`. The prior artifact measured the consumer's
`consumer.get_last_window` span at **217 µs total for the whole run**
(`structural-gap-analysis.md` §2 row 6). This is the gzippy↔vendor counterpart
of `getLastWindow(*previousWindow)` on the main thread (`vendor:572`) and is
already faithful and cheap.

### 1.4 So what IS the 4.89 ms? The inter-publish gap = consumer block time

The fulcrum `model` measures `L_resolve` as
**`t_publish(i) − t_publish(i−1)`** (`plans/parallel-sm-model.md:70`, verbatim:
"inter-publish gap … overlap/serialism already baked in"). Publishes happen once
per chunk, in order, so this interval is the **per-chunk consumer cycle time**;
summed over N it telescopes to `t_publish(N)−t_publish(0) ≈ wall`. Mean
`L_resolve ≈ wall/N` *by definition*, independent of where time is spent. The
interval is dominated by `wait.future_recv` (`cf:1627`, `cf:3492`) — the
consumer **blocking** on the pool's decode+applyWindow future for chunk i —
because chunk i's work isn't finished ahead of the in-order cursor.

**The 75× comparison is a category error:** vendor's 0.065 ms is
`replaceMarkerBytes` *self-time* (the transform); gzippy's 4.89 ms is a
*publish-cadence interval* that includes the decode/resolve wait. They are not
the same quantity. (`structural-gap-analysis.md` §7 final bullet says exactly
this: the model mean L_resolve is "contaminated by decode-rate stalls.")

### 1.5 Decomposition of the gap (the 75× is not one mechanism)

| component | gzippy | rapidgzip | on consumer critical path? |
|---|---|---|---|
| marker-resolve self-time (`apply_window` busy) | 1028 ms | 12,756 ms | **No** — wall-crit 0 ms (gz) / 2.7 ms (rg) |
| consumer serial publish (`get_last_window`) | 217 µs total | ~equiv | yes, but negligible |
| consumer **blocked** (`dispatch_recv`) | 169 ms (was 278) | ~3 ms | **Yes — this is the gap** |
| window-absent DECODE on crit path (`block_body`) | 507 ms | — | **Yes — root cause** |
| `d_w` window-absent decode latency/chunk | 125.5 ms | 70.95 ms (1.77×) | root cause |

(Counters/figures: `plans/leverB-perturbation.md` §4, `structural-gap-analysis.md`
§1.4/§3.1.) The "gap" is consumer block time, whose root is the 1.77×-slower
**decode** keeping the pool behind the cursor at matched prefetch depth (2·T=16,
`cf:508`).

---

## 2. THE ONE PLACE COMPUTE *COULD* TOUCH THE PATH — and why it doesn't

When resolve-ahead missed a chunk, the consumer takes a synchronous fallback
(`cf:1604-1648`): it `submit_post_process_to_pool` (`cf:1640`) then immediately
`recv_post_process_blocking` (`cf:1647`). Scrutinized: even here the LUT pass
runs on the **pool** (priority −1, `cf:2104`); the consumer thread executes zero
resolve compute. What it pays is the **latency** of a freshly-submitted pool task
with no head-start — a scheduling/WAIT cost, fully inside the inter-publish gap,
not resolve self-time. So the fallback is evidence *for* the wait-artifact
thesis, not against it.

How often does the fallback fire? Fresh HEAD counters
(`plans/leverB-perturbation.md` §4): resolve-ahead `ok=31/31` (fires 100% when
attempted), `handoff_key=31` published, but `EAGER_PROBE_REUSED=12` of ~39
chunks. So the consumer reaches the `cf:1640` fallback for the majority — but the
cost it pays there is **wait for the pool**, and the pool is behind because it is
saturated on slow window-absent decode. This is a resolve-ahead *scheduling*
sub-story, still **downstream of decode rate**, not "the LUT pass is slow."

---

## 3. FAITHFUL MINIMAL CHANGE

**There is no faithful resolve-side change that closes the measured wall**, for
three reasons: (a) resolve self-time is already a faithful-or-better single-pass
LUT port (`sm:461`); (b) it already runs off the consumer thread at vendor's
priority −1 (`cf:2104` ↔ `vendor:579`) — it cannot be moved further off-path;
(c) a saturated pool gated by decode throughput cannot be relieved by faster
resolve.

The faithful lever is **Lever B: speed the WINDOW-ABSENT DECODE inner loop** so
the pool stays ahead of the in-order consumer at matched prefetch depth. This is
`decode_huffman_body_resumable` / the `block_body` bootstrap path, which CLAUDE.md
explicitly authorizes for full inner-Huffman reimplementation. It is a *speed*
lever on an already-faithful structure (the chunk pipeline, prefetcher, window
map, resolve-ahead, and pool-side applyWindow are all already vendor-shaped —
`structural-gap-analysis.md` §2 rows 6/7/8). The vendor existence proof is
`d_w` = 70.95 ms vs gzippy 125.5 ms: same window-absent-everywhere pattern,
1.77× faster per-chunk decode.

A *secondary*, resolve-adjacent (not resolve-compute) option — raise the
resolve-ahead reuse hit-rate so the consumer rarely hits the `cf:1640` fallback —
is bounded by the same pool saturation and per rule 7 is a different mechanism;
it should not be pursued before Lever B, and only after the counters in §2 are
re-measured to show reuse can actually rise without the pool falling behind.

> NOTE: do **not** size Lever B from the slow-down slope. Per CLAUDE.md
> Measurement PROCESS rule 3, the speed-up ceiling must be set by a **removal
> oracle** (replace/remove the window-absent bootstrap decode, measure the
> interleaved wall) — the model shows a worker-bound knee at ~491 ms that caps
> the achievable wall. The existing `drive_clean_window_oracle` (`cf:283`) does
> **not** serve this: it bypasses the speculative scheduler, the bootstrap, AND
> the publish chain (`cf:358-405`), isolating nothing about resolve, and
> CLAUDE.md records a prior version of it silently re-ran the bootstrap. A
> repaired bootstrap-removed oracle is required before any speed-up claim.

---

## 4. PRE-REGISTERED FALSIFIER

The investigation's verdict ("L_resolve is a wait artifact; lever is decode
rate") is **FALSIFIED** (i.e. a resolve-side compute lever genuinely exists) iff,
from a change that touches ONLY marker-resolve compute (the `sm:461` LUT pass or
its dispatch — NO inner decode-loop edit):

- `vs --by-role` `marker-resolve / apply-window` **wall-critical** drops from
  ~300 ms toward rapidgzip's ~3 ms, **AND** the T8 silesia-large wall drops
  materially toward rapidgzip's ~0.52 s; **AND**
- the causal **window-absent fraction stays ~90 %** (`causal` `window_present`,
  `fulcrum-report.txt:218`). If it drifts toward the 31 % static fraction the
  change turned workers into clean-decoders — forbidden divergence, REJECT even
  if the wall drops.

The verdict is **CONFIRMED** (resolve is NOT the lever) iff:

- `apply_window` wall-critical stays ≈ 0 ms and `L_resolve` median does **not**
  move from any resolve-compute/scheduling-only change with resolve-ahead
  verified firing (`RESOLVE_AHEAD_OK > 0`, `handoff_key > 0`, `cf:2509/2525`);
  **AND**
- a `GZIPPY_SLOW_BOOTSTRAP=N` perturbation of `worker.block_body` moves the wall
  ~proportionally and survives a sleep-based frequency-neutral control.

  Thresholds (already observed, `plans/leverB-perturbation.md` §1): +50% spin
  → **+6.0%** wall; +50% sleep (control) → **+6.5%** (within spread ⇒ not a turbo
  artifact); +100% spin → **+20.3%** (≈ proportional). PASS = monotonic +
  control-agreeing. This is met today ⇒ verdict CONFIRMED at HEAD.

Quantitative target if a resolve-side lever were real: `L_resolve` median
4.89 ms → toward 0.065 ms with `dispatch_recv` falling AND window-absent ≥ ~90%
AND silesia sha byte-exact. Reject any candidate that drops the wall by pushing
window-absent toward 31% (clean-decoder divergence) or that fails sha.

---

## 5. SKEPTICAL CAVEAT (the crux: compute vs wait)

**Is the 4.89 ms median mostly WAIT for the predecessor window, which a faster
resolve would NOT fix?** From the code: **yes — it is overwhelmingly wait.**

- The `model`'s `L_resolve` is the inter-publish gap (`parallel-sm-model.md:70`),
  which telescopes to ≈ `wall/N` and is dominated by `wait.future_recv`
  (`cf:1627`, `cf:3492`) — the consumer blocking on the pool's
  decode+applyWindow future, gated upstream by window-absent **decode**
  (`block_body` 507 ms on the critical path, `structural-gap-analysis.md` §1.4).
- The code itself separates the consumer's blocking wait
  (`consumer.wait_replaced_markers`, `cf:1509`) from the publish work span
  (`consumer.window_publish_marker`, `cf:1542`), and the publish-work span is the
  217 µs/run `get_last_window` — *not* 4.89 ms. The 4.89 ms only appears when you
  measure the inter-publish *interval*, i.e. include the wait.

Therefore the L_resolve lever is a **mirage**: a resolve-compute speedup cannot
move a wall whose per-link interval is consumer-block-on-slow-decode. The
campaign would be measuring TIEs again (and did — the FastBootstrap TIE,
`parallel-sm-model.md:38-42`, is the model correctly predicting that cutting
resolve does nothing once the worker-bound term binds).

Disproof attempts I could not fully close (read-only):
1. I could not run a fresh fulcrum capture; the structural claims (resolve on
   pool at priority −1; consumer publish ~217 µs; inter-publish-gap definition)
   are verified directly in source, but the *magnitudes* (1028 ms busy / 0 ms
   wall-crit, d_w 1.77×) are from the cited artifacts at this HEAD, not re-run by
   me.
2. The "rapidgzip" trace is a patched binary emitting gzippy-shaped span names;
   its `apply_window` *busy* figure depends on the patch, but the load-bearing
   *wall-critical* numbers (2.7 ms rg / 0 ms gz) are independent of that
   (`structural-gap-analysis.md` §7).
3. Lever B is "confirmed on the slow-down slope" but its speed-up *ceiling* is
   not yet bounded — that needs the repaired removal oracle (§3 NOTE).

---

## Bottom line

`L_resolve` is **not** a compute lever. The marker→byte resolve is a faithful,
fast, single-pass LUT port (`sm:461`) running off the critical path on the pool
(`cf:2540/2580` ↔ `vendor:579-582`), wall-critical 0 ms. The "75×" is the
inter-publish gap (`parallel-sm-model.md:70`) — a wait artifact telescoping to
≈ wall/N, dominated by the consumer blocking on 1.77×-slower window-absent
DECODE. The faithful minimal change is the authorized inner-Huffman decode-loop
reimplementation (Lever B), with its speed-up ceiling to be bounded by a repaired
bootstrap-removal oracle — **not** any change to `apply_window` / `resolve_*`.
