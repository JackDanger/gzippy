# decode_viz — domain-aware verdict panel for the parallel-SM decode

A **domain-aware** HTML visualization of gzippy's (and rapidgzip's) parallel
single-member DEFLATE decode. NOT a generic flamegraph — it encodes the decode
domain (prefetcher → worker pool → in-order consumer) and is built to help *ask
better questions* without lying.

Design: `plans/wall-progress.md` → "DECODE VISUALIZER design 2026-06-01"
(advisor a8b0c06c). The three blocking corrections are obeyed:

1. **WEIGHT = wall-relevance, not CPU-sum.** The consumer spine (the single
   in-order thread, `consumer.iter` spans) IS the wall and is the saturated top
   row with its absolute µs. Worker decode boxes that finish before their
   consumer turn are **SLACK** → rendered folded/thin-grey (inverted flamegraph
   convention, labeled on-chart). A fat overlapped worker box does NOT read as
   the bottleneck.
2. **Blocking is a labeled HEURISTIC.** These traces carry no causal flow edges
   (`worker.decode_chunk`'s `chunk_id` is the `u64::MAX` sentinel), so the
   rate-vs-placement classification is heuristic: at each frontier
   `wait.future_recv{K}` start, is *any* worker decode still RUNNING (● rate) or
   are all closed (◆ placement)? The model carries `causal:"HEURISTIC"` and the
   HTML labels it. A per-stall glyph row + a rate/placement histogram pick the
   lever.
3. **Cross-tool = 6 canonical phases + coverage.** Native span names differ
   between tools (gzippy ~15, rapidgzip coarser → phantom gap). Both are folded
   to exactly 6 phases (dispatch/decode/resolve/publish/output/wait). Per-tool
   coverage-% is shown; uninstrumented wall is grey-hatched **UNKNOWN**, never
   blank.

Honesty guards (load-bearing UI): wall-reconciliation banner (red if viz wall
differs from a supplied measured wall by >10%), B/E-mismatch counter (a dropped
E makes a fake giant bar — surfaced + affected names listed), decode-mode
(clean / window-absent) as a `~` badge on the **folded worker** track (a
CPU-region fact, kept OFF the wall spine).

## Files

- `reduce.py` — Python reducer. Reuses `scripts/timeline_analyze.py`'s
  `load_events` (handles the trailing-comma / missing-bracket trace breakage).
  Holds all lie-prone classification. Emits a small derived `model.json`.
- `test_reduce.py` — unit tests with synthetic traces where a naive (CPU-sum /
  no-coverage / all-rate) reducer gives a *different, wrong* answer.
- `render.py` — presentational only: embeds `model.json` into a self-contained
  static HTML (inline JS + SVG, DOM-inspectable, no server).
- `headless_check.js` — node DOM-shim that executes the panel JS to assert it
  builds SVG without throwing (CI-able smoke test).
- `decode_verdict_T8.html` + `.png` — the rendered artifact (gz_T8 vs rg_T8).

## Usage

```bash
cd tools/decode_viz
python3 reduce.py gz=/tmp/model_traces/gz_T8.json rg=/tmp/model_traces/rg_T8.json \
    --out model_T8.json \
    --measured-wall-us-gz 720000 --measured-wall-us-rg 460000   # enables red self-check
python3 render.py model_T8.json --out decode_verdict_T8.html
python3 -m unittest test_reduce          # 7 tests
node headless_check.js decode_verdict_T8.html   # smoke test the JS

# raw 40×16 timeline exploration: drag the trace into ui.perfetto.dev (already Chrome-trace format).
```

## What the T8 artifact reveals

- gzippy wall **720.9ms** vs rapidgzip **460.5ms** ≈ **1.57×** (same direction as
  the frozen T8-noSMT 1.47×; this is a different trace run).
- gzippy frontier stalls: **26 RATE-bound (564.4ms)** vs **10 placement (81.9ms)**
  — the consumer waits overwhelmingly on decodes that are *still running*
  (rate), not on ordering. Matches the campaign's "placement causally dead via
  oracle TIE" finding.
- Both tools run the same workload shape (gzippy 38 window-absent / 4 clean vs
  rapidgzip 38 / 1) → the gap is decode **RATE**, not extra machinery.
- The 6-phase bars show gzippy's **decode** phase wider than rapidgzip's,
  consistent with the slower per-chunk decode.
- **Thread-count caveat (T16):** at T16 gzippy flips to **3 rate / 33 placement**
  — the lever differs by T (T4/T8 = decode rate; T16 = consumer ordering). The
  panel makes this visible at a glance.
- **Honesty note:** the TLB/page-walk finding (DTLB store-walk 3.26×, 99.6%
  faults in worker decode) is a **perf-counter** fact NOT present in these
  timeline traces — it is deliberately NOT drawn. The decode-mode counts ARE in
  the trace and badge the folded worker track.
- **Cross-tool caveat surfaced by the panel:** rapidgzip's B/E-mismatch (78 on
  `pool.pick`) means its dispatch-phase numbers are unreliable.

## Scope note (flow-event instrumentation)

The advisor's prereq #2 (add `partition_idx` + Chrome flow events `ph:s`/`ph:f`
to `worker.decode_chunk` so the blocking edge becomes DATA, not inference)
requires a gzippy source change + a regenerated trace on neurotic. Per the
mission this build is **scoped to the heuristic-only viz on the existing
traces** (neurotic is owned by other agents; do not contend). The viz degrades
gracefully: with no flow edges the rate/placement classification is labeled
`HEURISTIC` everywhere. If/when the instrumentation lands and a richer trace is
captured, the same reducer would consume the flow edges to upgrade the label to
`MEASURED` (the join key `id=partition` replaces the "any decode running"
approximation in `classify_stalls`).
