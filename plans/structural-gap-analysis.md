# Structural-gap analysis — gzippy vs rapidgzip parallel single-member (T8, silesia-large)

Source artifacts: `/tmp/gzippy-locked-fulcrum-20260606-065948/` (fulcrum-report.txt +
trace_{gzippy,rapidgzip}_T8.json). Fulcrum tool source: `/home/user/www/fulcrum/src/`.
Read-only diagnosis — no builds, no edits, no benches. Every claim below is grounded in a
trace field or a `file:line` I actually opened.

---

## 0. TL;DR verdict

1. **The instrument "contradiction" is not real — it is a string-match bug in `model.rs`.**
   gzippy emits `causal.decode_decision` with `mode:"boundary_search"`; `model.rs` only
   recognises the literal `"window_absent"`, so it reads gzippy's window-absent fraction as
   **0 %** when gzippy's own `window_present:false` flag (and the `--verbose` counters) say it
   is **~90 %**. The `causal` view (90.5 %) is the correct number. The `model` view's
   "gzippy f=0 % vs rapidgzip f=97 %" delta is an **artifact of two incompatible measurement
   sources** and must be discarded.

2. **Both tools go window-absent on essentially every chunk** (gzippy ~90 %, rapidgzip ~97 %).
   Window-absent decode is the *shared, faithful* behaviour — it is **NOT the divergence**.

3. **The orchestrator's diagnosis ("key-mismatch is the cause") is WRONG** as a remediation
   direction. The key-mismatch *observation* is real (workers look up the partition seed), but
   it is a **faithful port of rapidgzip** (`decodeBlock` looks up `m_windowMap->get(blockOffset)`
   at the partition/speculative offset too — vendor `GzipChunkFetcher.hpp:712`). The causal
   view's own data kills the remediation: **0 / 37** key-mismatch chunks had their predecessor
   window published *before* the chunk's decode start, so re-keying to the "real" boundary would
   still miss — the window does not exist yet. The proposed fix (clean-decode at seed using a
   predecessor dict) would make gzippy *clean-decode*, which **diverges from** rapidgzip's
   window-absent-everywhere pattern.

4. **The real divergence (on which `vs --by-role` and `model` actually AGREE):** gzippy's
   in-order consumer spends ~300 ms wall-critical *blocked* (`consumer.dispatch_recv` 278 ms +
   `consumer.dispatch_post_process` 21 ms) waiting for marker-resolution / decode results that
   rapidgzip has already finished **off** the in-order path. rapidgzip burns **12,756 ms** of
   `apply_window` busy but only **2.7 ms** of it is wall-critical; gzippy burns **1,028 ms** of
   `apply_window` busy with **~300 ms** of the consumer wall blocked on the chain that feeds it.
   The mechanism is the serial **publish-chain** being gated by window-absent **decode that is
   not finished far enough ahead**, not by *which key* a worker decodes against.

---

## 1. Resolving the contradiction (what each instrument actually counts)

### 1.1 Ground truth from the trace

`trace_gzippy_T8.json` — 42 `causal.decode_decision` instants:

| field | distribution |
|---|---|
| `mode` | **38× `"boundary_search"`**, 4× `"clean"` |
| `window_present` | 38× `false`, 4× `true` |

`trace_rapidgzip_T8.json` — **0** `causal.decode_decision` instants; instead 78 `worker.decode`
spans tagged `mode` (38× `"window_absent"`, 1× `"clean"`) and 39 `causal.window_publish`.

### 1.2 `causal` view (`/home/user/www/fulcrum/src/causal.rs:260-297`)

Counts on the **`window_present` bool**: `Some(false)` → window-absent. → **38/42 = 90.5 %**.
This matches gzippy's reality and is corroborated by the `--verbose` counters in the report:

```
clean-decode paths: pred@seed=0  handoff@stop=0  boundary@seed=0  candidate=0
early window: published=39  handoff_key=0
unified decoder: flip_to_clean=29
```

i.e. **zero** chunks decoded clean via a predecessor window; 29 were decoded with markers and
flipped to clean during resolution. The 90.5 % is real.

### 1.3 `model` view (`/home/user/www/fulcrum/src/model.rs:244-260`)

Counts on the **`mode` string**, hard-coded to `"clean"` / `"window_absent"`:

```rust
match arg_str(&e.args, "mode").as_deref() {
    Some("clean") => n_clean_decisions += 1,
    Some("window_absent") => n_absent_decisions += 1,   // <-- gzippy emits "boundary_search"
    _ => {}                                              //     so all 38 fall through here
}
```

gzippy's 38 window-absent decisions carry `mode:"boundary_search"` (set at
`chunk_fetcher.rs:2156`), which matches **neither** arm, so `n_absent_decisions = 0` →
`window_absent_frac → 0 %` (then printed against the `worker.decode_chunk` span count, hence
"0 of 46"). For rapidgzip, with no `decode_decision` instants at all, `model.rs` falls back to
the `worker.decode` mode-tagged spans (`:229-235`), which the rapidgzip patch *does* tag
`"window_absent"` → 97.4 %.

**So the two tools' fractions are computed from different event sources with an asymmetric
string filter.** They are not comparable, and the gzippy 0 % is simply wrong. Note gzippy
*intentionally* keeps `boundary_search` only on the causal instant and tags the
`worker.decode` span `"window_absent"` (`chunk_fetcher.rs:2146-2152`) precisely so `model.rs`
can read it — but `model.rs`'s fraction path reads the *instant's* mode, not the span's, so the
intent is defeated. **This is an instrument bug in `model.rs` (or, equivalently, gzippy should
emit `mode:"window_absent"` on the instant too).**

### 1.4 Why the two "mechanisms" are actually the same mechanism

Once the bogus fraction is discarded:

- `model`'s binding term for *both* tools is **publish-chain** = `frontier + N·L_resolve`
  (gzippy 856.7 ms ≈ wall 863.5 ms; rapidgzip 446.5 ms ≈ wall 455.7 ms). The lever it names is
  **L_resolve** (20.39 ms vs 10.56 ms per link; median 4.89 ms vs **0.065 ms**).
- `vs --by-role` names **marker-resolve / apply-window** as the top Δwc role: **300.0 ms**
  gzippy wall-critical vs **2.7 ms** rapidgzip.
- `critpath` shows the consumer **62.3 % waiting**, with `blocked-on:worker.block_body`
  507 ms (58.8 %) and `consumer.dispatch_recv` 278 ms (32.2 %).

These are three measurements of **one** thing: the in-order consumer is serialized on a
window/marker-resolution chain whose links stall waiting for window-absent **decode** +
**apply_window** that have not completed ahead of the consumer. rapidgzip keeps that work off
the in-order path; gzippy does not (fully). The `causal` "key-mismatch" framing is the odd one
out, and its remediation points the wrong way (§3).

---

## 2. Two-column window-keying map (rapidgzip ↔ gzippy)

`enc = encodedOffsetInBits` (real, confirmed block boundary). `seed = partition/speculative
offset`. `windowOffset = enc + encodedSizeInBits` (the key a chunk *publishes* its tail under).

| # | Question | rapidgzip (`vendor/.../GzipChunkFetcher.hpp`) | gzippy (`src/decompress/parallel/`) |
|---|---|---|---|
| 1 | **Key a WORKER looks up the predecessor window under, at decode time** | `decodeBlock`: `m_windowMap->get(blockOffset)` where `blockOffset` is the **partition/seed** offset (`getBlock` tries `partitionOffset` first, `:601-602`). **:712** | `run_decode_task`: `window_map.get(params.start_bit)` where `start_bit` = partition **seed** for speculative prefetches. **chunk_fetcher.rs:2141** |
| 2 | **What happens on a worker miss** | `sharedWindow` null → window-absent decode emitting markers (not BGZF). **:712-724** | `window_at_offset = None` → `speculative_decode_find_boundary` (markers). **chunk_fetcher.rs:2206-2213** |
| 3 | **Key the CONSUMER looks up the predecessor window under** | `processNextChunk`: `m_windowMap->get(*nextBlockOffset)` — the **REAL** boundary offset; must exist. **:334-340** | consumer marker branch: `confirmed_predecessor_window(window_map, handoff_bit)` → `window_map.get(handoff_bit)`, `handoff_bit = chunk_consumer_handoff_bit` = `max_acceptable_start_bit`. **chunk_fetcher.rs:2319-2326, :2362-2386** |
| 4 | **Key a chunk PUBLISHES its tail window under** | `queueChunkForPostProcessing`: `windowOffset = enc + encodedSizeInBits`; `m_windowMap->emplace(windowOffset, getLastWindow(prev))`. Subchunk windows at `subchunk.encodedOffset+encodedSize` (`:432, :445`). **:557-574** | `publish_handoff_window`: `window_offset = enc + size`; `window_map.insert_owned_none(window_offset, get_last_window_vec(pred))`. Subchunk windows `populate_subchunk_windows`. **chunk_fetcher.rs:2362-2396, :2688-2694** |
| 5 | **Does the worker attempt a CLEAN decode against a predecessor window?** | **No** for ~97 % of chunks — speculative prefetch at the seed misses → window-absent. Clean only for chunk 0 / cache hits. | **No** for ~90 % — identical. `flip_to_clean=29`; `pred@seed=0`. |
| 6 | **Where is the CHEAP window-chain link computed (serial, in-order)?** | Main/consumer thread: `getLastWindow(*previousWindow)` (32 KiB), emplaced **before** submitting the heavy applyWindow. The comment at **:559-561** marks this the *only* unavoidable serial path. | Consumer thread: `get_last_window_vec(predecessor_window)` (`consumer.get_last_window`, 217 µs total). **chunk_fetcher.rs:1564, :2385** |
| 7 | **Where is the HEAVY full marker resolution (applyWindow) run?** | On the **pool**, `submitTaskWithHighPriority([chunkData,window]{ applyWindow })`. Off the in-order path. **:577-582** | On the **pool**, `submit_post_process_to_pool` → `post_process.apply_window`. `apply_window` busy 1028 ms, **wall-crit 0 ms**. **chunk_fetcher.rs:1640, :2473** |
| 8 | **Resolve-AHEAD (kick off future chunks' applyWindow while waiting)** | `queuePrefetchedChunkPostProcessing`: scan whole prefetch cache; for every chunk whose predecessor window exists, `queueChunkForPostProcessing`. **:520-551** | `queue_prefetched_marker_postprocess` (`consumer.queue_prefetched_postproc`, 680×). **chunk_fetcher.rs:2396-2438** |

**Conclusion of the map:** gzippy has *already* faithfully ported the decisive structure —
rows 6/7/8 (cheap getLastWindow on the consumer; heavy applyWindow on the pool; resolve-ahead).
`apply_window` is genuinely off the consumer's wall (0 ms wall-crit). **The worker key (rows 1-2)
is byte-for-byte the vendor pattern and is correct.** There is no single mis-keyed lookup that
explains the gap.

---

## 3. Where gzippy actually diverges, and the minimal faithful change

### 3.1 The divergence is in *pipelining degree*, not structure

The consumer still blocks ~300 ms (`dispatch_recv` 278 ms) because, when it reaches chunk *N*,
*N*'s `apply_window` is **not yet done** — either it was never resolved-ahead, or its input
decode (`worker.block_body`, the window-absent bootstrap, 507 ms on-critical-path) is not
finished. rapidgzip's equivalent wait is 2.7 ms because resolve-ahead + a faster, deeper-running
pool keep the work ahead of the in-order cursor. Two faithful, vendor-grounded sub-levers:

**Lever A — make resolve-ahead actually find its predecessor window (key alignment on the
resolve-ahead path, NOT the worker path).** Until HEAD this was the *real* live bug:
`chunk_may_resolve_markers_early` and `queue_prefetched_marker_postprocess` looked up the
predecessor window under `chunk.encoded_offset_bits` (the partition **seed**), which — exactly
as the causal view notes — never matches the published key. So resolve-ahead **silently
no-op'd** for range-speculative chunks (`handoff_key=0` in `--verbose`), and every chunk's
applyWindow was deferred until the consumer arrived → the 278 ms `dispatch_recv` wall.
**HEAD commit `85ad00a` is precisely this fix in progress**: it switches both sites to
`chunk_consumer_handoff_bit(chunk) = max_acceptable_start_bit` (vendor counterpart
`GzipChunkFetcher.hpp:544`, `m_windowMap->get(chunkData->encodedOffsetInBits)` where the vendor's
`encodedOffsetInBits` is already the real decode-start). This is the right direction and the
right vendor citation. It needs a Fulcrum re-capture to confirm it fired (see §5).

**Lever B — keep the window-absent decode frontier ahead of the consumer.** `critpath`'s
`blocked-on:worker.block_body` 507 ms and `schedule`'s RATE-dominant verdict say the consumer
ultimately waits on decode. gzippy's `d_w` = 125.5 ms vs rapidgzip 70.95 ms (1.77×); with a
matched prefetch depth (both `2·T = 16`) a 1.77×-slower decode means the pool falls behind the
in-order cursor and the consumer stalls. This is the inner-loop `block_body`/bootstrap path,
which CLAUDE.md explicitly authorises for full reimplementation — but it is a *speed* lever, not
a structural one, and it should be confirmed by perturbation, not assumed.

### 3.2 Minimal faithful change (ranked)

1. **(Likely already landed in `85ad00a`) Resolve-ahead key = consumer handoff bit.**
   Functions `chunk_may_resolve_markers_early`, `queue_prefetched_marker_postprocess`,
   `confirmed_predecessor_window` must key on `max_acceptable_start_bit`, not
   `encoded_offset_bits`. Vendor: `GzipChunkFetcher.hpp:544` + `:520-551`. **Verify it fires**
   (`handoff_key` / `RESOLVE_AHEAD_OK` counters > 0) before claiming anything.

2. **Verify resolve-ahead scan breadth matches `queuePrefetchedChunkPostProcessing`.** Vendor
   scans the **entire** prefetch cache sorted by offset every wait (`:524-529`). Confirm
   gzippy's `queue_prefetched_marker_postprocess` (`:2396-2438`) is not gated to fewer chunks
   than the prefetch cache holds; if it is, that throttles look-ahead.

3. **Do NOT** implement the causal view's PRIMARY remediation (clean-decode at seed via
   `get_predecessor_for_worker`). It diverges from vendor (rows 1-2 are already correct) and
   cannot work (0/37 windows ready in time).

---

## 4. Is the orchestrator's "key-mismatch is the cause" correct?

**Partially correct as an observation, wrong as a remediation, and mis-located.**

- *Correct:* there genuinely is a key mismatch — workers (and, until `85ad00a`, the
  resolve-ahead path) look up the partition seed while windows publish at real boundary keys.
- *Wrong / mis-located on the WORKER path:* the worker-side seed lookup is a **faithful port**
  (`GzipChunkFetcher.hpp:712`); rapidgzip is equally window-absent, so "fix the worker key →
  clean decode" would make gzippy *unlike* rapidgzip and is forbidden by the transliteration
  guardrails. The causal data (0/37 ready-in-time) shows it would not even produce clean decodes.
- *Where the key actually mattered:* the **resolve-ahead** path — and that is what `85ad00a`
  already targets. So the productive reading of "key-mismatch" is "resolve-ahead was looking up
  the wrong key and silently no-op'ing," not "workers should clean-decode at the seed."

---

## 5. Falsifiable predictions for the next Fulcrum capture (pre-register before measuring)

If Lever A (resolve-ahead handoff-key fix, `85ad00a`) is working:

- `--verbose`: `early window … handoff_key` and `RESOLVE_AHEAD ok` move **> 0** (were 0). If they
  stay 0, the fix did **not** fire — reject and investigate before any wall claim.
- `vs --by-role`: `marker-resolve / apply-window` wall-critical drops from **300 ms** toward
  rapidgzip's ~3 ms; `consumer.dispatch_recv` wall-critical drops from **278 ms**.
- `model`: `L_resolve` **median** drops from 4.89 ms toward rapidgzip's ~0.065 ms (the
  *mean*/p95 will move less — it is dominated by decode-rate stalls, Lever B).
- **Window-absent fraction MUST stay ~90 %** (`causal` `window_present`). If a change pushes it
  toward the 31 % static fraction, that is the *wrong* (clean-decode) direction — **reject it**
  even if the wall drops, per the transliteration guardrails.
- **Falsifier:** if `dispatch_recv` / `L_resolve-median` do **not** move after `handoff_key`
  counters confirm resolve-ahead fired, then the consumer stall is decode-rate-bound (Lever B),
  not resolve-placement-bound, and the structural story is exhausted — the residual is inner-loop
  `d_w` (1.77×), to be attacked under the inner-loop reimplementation licence and confirmed by a
  `GZIPPY_SLOW_BOOTSTRAP` perturbation (slow `block_body` by a known factor; wall must respond
  ~proportionally and survive a sleep-based frequency-neutral control).

---

## 6. Assessment of recent commits

- **`85ad00a` (HEAD, "publish-chain overlap and post-process alloc cuts", co-authored Cursor)** —
  **moves TOWARD the faithful pattern and attacks the real lever.** Core change: resolve-ahead
  (`chunk_may_resolve_markers_early`, `queue_prefetched_marker_postprocess`) now keys the
  predecessor lookup on `chunk_consumer_handoff_bit = max_acceptable_start_bit` instead of the
  partition seed `encoded_offset_bits`, citing vendor `GzipChunkFetcher.hpp:544`. This is exactly
  Lever A. Supporting micro-opts (try_recv before the post-process wait spin; defer Arc borrow
  when `eager_completed` already has the key; single rpmalloc prepend for resolved markers;
  skip staging reload memcpy when the anchor is unchanged) are pipelining/alloc overlap, **not**
  structural divergences. No vendor-structure regression spotted.
- **`99ff098`** — always fuse marker resolve+narrow, reuse thread-local LUT. Matches the
  `resolve_and_narrow_markers_in_place` single-pass intent (`chunk_fetcher.rs:2307`); the doc
  there already cites a prior +411 ms wall-critical regression on the two-pass path. Aligned.
- **`a5c2497` / `1dfed80` / `06cf79a` / `d3b692e`** — inner `ResumableInflate2` staging-buffer
  work (Lever B territory: decode speed). Legitimate under the inner-loop licence; not structural.
- **`f3e383e`** — explicitly *restores* vendor `decodeChunkWithRapidgzip` runtime shape (a revert
  toward faithfulness). Good signal.

**No recent commit diverges from vendor structure.** `85ad00a` is an in-progress key fix on the
correct (resolve-ahead) path; the rest are decode-speed and alloc work. The thing missing is a
**Fulcrum re-capture proving `85ad00a` actually fired** (`handoff_key`/`RESOLVE_AHEAD` > 0 and
`dispatch_recv` down) — the current report predates or does not reflect it (`handoff_key=0`).

---

## 7. Skeptical caveats (disproof attempts I could not fully close, read-only)

- I could not run a causal perturbation (read-only), so Levers A/B are **hypotheses** with
  trace evidence, not confirmed levers. Per CLAUDE.md, the verdict is the perturbation; treat §3
  as provisional until §5's falsifiers are run.
- The "rapidgzip" trace is a **patched binary** emitting gzippy-shaped span names
  (`worker.isal_stream_inflate`, `post_process.apply_window`, …). I confirmed it is a *distinct*
  binary (it emits `mode:"window_absent"` where gzippy emits `"boundary_search"`, and 0
  `decode_decision` instants), so it is not gzippy mislabelled. But the apply_window busy figure
  (12,756 ms across 4,602 spans) depends on how the patch instruments rapidgzip's applyWindow;
  the *wall-critical* 2.7 ms (the load-bearing number) is independent of that and trustworthy.
- The `model` mean-`L_resolve` lever and the `vs` wall-critical lever agree in *direction* but
  the mean is contaminated by decode-rate stalls (Lever B), so do not read the full 373 ms
  "L_resolve gap" as recoverable by resolve-placement alone; the model's own knee caveat
  (worker-bound ~491 ms) caps it.
