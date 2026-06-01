#!/usr/bin/env python3
"""Decode-visualizer REDUCER for gzippy / rapidgzip parallel-SM traces.

Reads one or two Chrome trace_v2 JSON files and emits a SMALL derived JSON
(the "verdict model") consumed by the static HTML verdict panel.

This file holds the LIE-PRONE classification logic so it can be unit-tested
(see test_reduce.py). The HTML is purely presentational.

What it computes (per the advisor-validated design, plans/wall-progress.md
"DECODE VISUALIZER design 2026-06-01"):

  1. CONSUMER SPINE = wall.  The single in-order consumer thread's `consumer.iter`
     spans (plus the waits/publishes nested in them) tile the wall end-to-end.
     We surface the spine as an ordered list of segments with absolute us.
     WEIGHT = wall-relevance: a worker box that closed before its consumer turn
     is SLACK, not weight.

  2. PER-STALL RATE-vs-PLACEMENT classification (HEURISTIC, no causal edges).
     The gzippy trace has `wait.future_recv{chunk_id=K}`.  At each such wait's
     START we ask: was there a worker decode span still OPEN (running) at that
     instant?  If yes  -> RATE-bound  (consumer waiting on a still-running decode).
     If no   -> PLACEMENT-bound (decode already done; the wait is scheduling/order).
     This is a heuristic because the trace carries no chunk_id on the worker
     decode span (chunk_id is the u64::MAX sentinel) -> NO causal flow edge.
     The model marks `causal="HEURISTIC"` so the HTML labels it, never as fact.

  3. SIX-PHASE CANONICALIZATION + COVERAGE.  Native span names differ between
     tools (gzippy ~15 names, rapidgzip coarser), which manufactures a phantom
     cross-tool gap.  We fold every span into exactly 6 phases
     (dispatch/decode/resolve/publish/output/wait) and report per-tool COVERAGE
     (instrumented wall / total wall).  Uninstrumented wall is UNKNOWN, never 0.

  4. HONESTY GUARDS: B/E-mismatch count; wall reconciliation (viz wall vs an
     externally-supplied measured wall, if any).

Usage:
  python3 reduce.py gz=/tmp/model_traces/gz_T8.json rg=/tmp/model_traces/rg_T8.json \\
      --out model.json [--measured-wall-us-gz 950000 --measured-wall-us-rg 646000]
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# Reuse the battle-tested loader (handles trailing-comma / missing-bracket).
sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__file__), "..", "..", "scripts"))
from timeline_analyze import load_events  # noqa: E402


# ---------------------------------------------------------------------------
# B/E pairing with explicit mismatch accounting (the design's honesty guard).
# ---------------------------------------------------------------------------
def pair_spans_counted(events: List[dict]) -> Tuple[List[dict], dict]:
    """Pair B/E into spans AND count mismatches.

    Returns (spans, mismatch_report) where mismatch_report has:
      unmatched_b: count of B events with no matching E (still on stack at end)
      unmatched_e: count of E events with no open B
      name_mismatch: count of E whose name != the B it popped
      affected_names: Counter of span names touched by any mismatch
    """
    stacks: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    spans: List[dict] = []
    unmatched_e = 0
    name_mismatch = 0
    affected: Counter = Counter()
    for e in events:
        ph = e.get("ph")
        if ph == "B":
            stacks[(e.get("pid", 0), e.get("tid", 0))].append(e)
        elif ph == "E":
            key = (e.get("pid", 0), e.get("tid", 0))
            if not stacks[key]:
                unmatched_e += 1
                affected[e.get("name", "<?>")] += 1
                continue
            b = stacks[key].pop()
            if b["name"] != e.get("name"):
                name_mismatch += 1
                affected[b["name"]] += 1
                affected[e.get("name", "<?>")] += 1
            parent = stacks[key][-1]["name"] if stacks[key] else "<root>"
            spans.append({
                "name": b["name"],
                "parent": parent,
                "pid": b.get("pid", 0),
                "tid": b.get("tid", 0),
                "ts_start": b.get("ts", 0),
                "ts_end": e.get("ts", 0),
                "dur": e.get("ts", 0) - b.get("ts", 0),
                "args": b.get("args", {}),
            })
    unmatched_b = sum(len(v) for v in stacks.values())
    for v in stacks.values():
        for b in v:
            affected[b["name"]] += 1
    return spans, {
        "unmatched_b": unmatched_b,
        "unmatched_e": unmatched_e,
        "name_mismatch": name_mismatch,
        "affected_names": dict(affected),
        "total_mismatch": unmatched_b + unmatched_e + name_mismatch,
    }


# ---------------------------------------------------------------------------
# 6-phase canonicalization.  The ONLY honest cross-tool comparison.
# ---------------------------------------------------------------------------
# dispatch : prefetcher emits/dispatches chunk seeds + pool plumbing
# decode   : worker actually inflating bytes (clean or window-absent bootstrap)
# resolve  : marker resolution / post-process (apply_window, crc on markers)
# publish  : consumer publishes the 32KB tail window for the next chunk
# output   : consumer writes decoded bytes in order + crc combine
# wait     : consumer blocked on the frontier chunk (the wall-relevant stall)
PHASE_RULES = [
    ("wait",     lambda n: n.startswith("wait.")),
    ("dispatch", lambda n: n.startswith("coord.") or n.startswith("pool.")
                 or n == "worker.seed_first"
                 or n.startswith("consumer.process_prefetches")
                 or n.startswith("consumer.try_take_prefetched")
                 or n.startswith("consumer.block_finder_get")
                 or n.startswith("consumer.arc_take_or_clone")
                 or n.startswith("ttp.")),
    ("decode",   lambda n: n in ("worker.decode_chunk", "worker.decode",
                                  "worker.bootstrap", "worker.isal_stream_inflate",
                                  "worker.block_header", "worker.block_body",
                                  "worker.scan_candidate", "worker.scan_run",
                                  "worker.absorb_isal_tail")),
    ("resolve",  lambda n: n.startswith("post_process.")
                 or n == "consumer.wait_replaced_markers"
                 or n == "consumer.dispatch_post_process"),
    ("publish",  lambda n: n.startswith("consumer.window_publish")
                 or n == "consumer.publish_windows"),
    ("output",   lambda n: n in ("consumer.write_data", "consumer.write_narrowed",
                                  "consumer.combine_crc")),
]
PHASES = ["dispatch", "decode", "resolve", "publish", "output", "wait"]


def canon_phase(name: str) -> Optional[str]:
    for phase, pred in PHASE_RULES:
        if pred(name):
            return phase
    return None  # uncategorized -> not counted toward a phase (but counted in coverage gap separately)


def is_consumer_tid(spans: List[dict]) -> int:
    """The consumer is the thread that owns consumer.iter."""
    c = Counter(s["tid"] for s in spans if s["name"] == "consumer.iter")
    if c:
        return c.most_common(1)[0][0]
    # fallback: thread with most consumer.* spans
    c = Counter(s["tid"] for s in spans if s["name"].startswith("consumer."))
    return c.most_common(1)[0][0] if c else 1


def decode_span_names() -> set:
    return {"worker.decode_chunk", "worker.decode", "worker.bootstrap",
            "worker.isal_stream_inflate"}


# ---------------------------------------------------------------------------
# Rate-vs-placement classification of frontier stalls (HEURISTIC).
# ---------------------------------------------------------------------------
def classify_stalls(spans: List[dict], consumer_tid: int) -> dict:
    """For each consumer frontier wait, decide rate vs placement.

    rate-bound (RUNNING glyph): at the wait's start, >=1 worker decode span on
        ANOTHER thread is still open -> the consumer is blocked on a decode that
        is still executing => the decode RATE gates the wall.
    placement-bound (READY glyph): no worker decode span open at wait start ->
        the producing work is already done; the stall is ordering/scheduling.

    Heuristic: we cannot join a wait to ITS specific chunk's decode (no chunk_id
    on the worker decode span), so we approximate "is the producer running?" by
    "is ANY worker decode running?".  This is honest about rate-vs-placement at
    the aggregate, which is what picks the lever; per-stall is indicative.
    """
    dnames = decode_span_names()
    worker_decodes = [s for s in spans
                      if s["name"] in dnames and s["tid"] != consumer_tid]
    # consumer frontier waits = the blocking waits on the consumer thread.
    waits = [s for s in spans
             if s["tid"] == consumer_tid and s["name"].startswith("wait.")]
    waits.sort(key=lambda s: s["ts_start"])

    stalls = []
    n_rate = 0
    n_place = 0
    rate_us = 0.0
    place_us = 0.0
    for w in waits:
        t0 = w["ts_start"]
        # any worker decode open at t0?
        running = [d for d in worker_decodes
                   if d["ts_start"] <= t0 < d["ts_end"]]
        cls = "rate" if running else "placement"
        if cls == "rate":
            n_rate += 1
            rate_us += w["dur"]
        else:
            n_place += 1
            place_us += w["dur"]
        stalls.append({
            "ts_start": t0,
            "dur": w["dur"],
            "name": w["name"],
            "chunk_id": w["args"].get("chunk_id"),
            "cls": cls,
            "running_decodes": len(running),
        })
    return {
        "causal": "HEURISTIC",  # never MEASURED until flow edges exist
        "n_rate": n_rate,
        "n_placement": n_place,
        "rate_us": rate_us,
        "placement_us": place_us,
        "stalls": stalls,
    }


# ---------------------------------------------------------------------------
# Consumer spine.
# ---------------------------------------------------------------------------
def build_spine(spans: List[dict], consumer_tid: int, t_origin: int) -> dict:
    """The consumer.iter spans tile the wall; each iter's nested children are
    classified to a phase so the spine renders as a 6-phase-colored bar."""
    iters = sorted((s for s in spans
                    if s["tid"] == consumer_tid and s["name"] == "consumer.iter"),
                   key=lambda s: s["ts_start"])
    # All consumer-thread spans, for per-iter phase breakdown.
    ctid_spans = [s for s in spans if s["tid"] == consumer_tid]
    segments = []
    spine_total = 0
    for it in iters:
        spine_total += it["dur"]
        # children fully contained in this iter (exclude the iter itself)
        children = [s for s in ctid_spans
                    if s is not it
                    and s["ts_start"] >= it["ts_start"]
                    and s["ts_end"] <= it["ts_end"]
                    and s["dur"] >= 0]
        # phase time = leaf-ish: use the wait spans + top consumer ops.
        # We attribute by the *deepest* named span at each phase using simple
        # max over phase among direct/indirect children durations.
        phase_us = defaultdict(float)
        for c in children:
            ph = canon_phase(c["name"])
            if ph:
                phase_us[ph] += c["dur"]
        wait_us = sum(c["dur"] for c in children if c["name"].startswith("wait."))
        segments.append({
            "ts": it["ts_start"] - t_origin,
            "dur": it["dur"],
            "wait_us": wait_us,
            "phase_us": {p: phase_us.get(p, 0.0) for p in PHASES},
        })
    return {
        "consumer_tid": consumer_tid,
        "n_iter": len(iters),
        "spine_total_us": spine_total,
        "segments": segments,
    }


def build_workers(spans: List[dict], consumer_tid: int, t_origin: int) -> dict:
    """Worker decode boxes, flagged SLACK if they close before the consumer's
    wall position reaches them.  Rendered FOLDED by the HTML; weight != duration.

    Slack heuristic: a decode that E-closes before the LAST consumer wait that
    started after it began = overlapped => slack.  We compute, per decode, the
    overlap with consumer wait windows; non-overlapping decode time is slack.
    """
    dnames = decode_span_names()
    decodes = [s for s in spans if s["name"] in dnames and s["tid"] != consumer_tid]
    waits = [s for s in spans
             if s["tid"] == consumer_tid and s["name"].startswith("wait.")]
    total_decode_us = sum(d["dur"] for d in decodes)
    # decode time that overlaps a consumer wait = "wall-relevant decode"
    overlap_us = 0.0
    for d in decodes:
        for w in waits:
            ov = min(d["ts_end"], w["ts_end"]) - max(d["ts_start"], w["ts_start"])
            if ov > 0:
                overlap_us += ov
    slack_us = max(0.0, total_decode_us - overlap_us)
    by_tid = defaultdict(lambda: {"n": 0, "us": 0.0})
    for d in decodes:
        by_tid[d["tid"]]["n"] += 1
        by_tid[d["tid"]]["us"] += d["dur"]
    return {
        "n_decode": len(decodes),
        "total_decode_us": total_decode_us,
        "wall_relevant_decode_us": overlap_us,
        "slack_decode_us": slack_us,
        "by_tid": {str(k): v for k, v in by_tid.items()},
    }


def phase_breakdown(spans: List[dict], consumer_tid: int) -> dict:
    """6-phase canonical breakdown + COVERAGE.

    Coverage = fraction of the consumer-thread wall that is covered by a
    consumer.iter span.  (The consumer spine IS the wall; uninstrumented wall =
    gaps between iters on the consumer thread.)
    Phase totals are computed over the WHOLE trace (all threads), folded to the
    6 canonical phases; native-name detail kept for drill-down.
    """
    phase_us = defaultdict(float)
    uncategorized_us = 0.0
    native = defaultdict(float)
    for s in spans:
        ph = canon_phase(s["name"])
        if ph:
            phase_us[ph] += s["dur"]
        else:
            uncategorized_us += s["dur"]
        native[s["name"]] += s["dur"]

    # Coverage on the consumer thread.
    citers = sorted((s for s in spans
                     if s["tid"] == consumer_tid and s["name"] == "consumer.iter"),
                    key=lambda s: s["ts_start"])
    drives = [s for s in spans if s["tid"] == consumer_tid and s["name"] == "drive"]
    if drives:
        wall_start = min(s["ts_start"] for s in drives)
        wall_end = max(s["ts_end"] for s in drives)
    elif citers:
        wall_start = citers[0]["ts_start"]
        wall_end = citers[-1]["ts_end"]
    else:
        wall_start = min(s["ts_start"] for s in spans)
        wall_end = max(s["ts_end"] for s in spans)
    consumer_wall = wall_end - wall_start
    # instrumented = union of consumer.iter intervals
    covered = 0.0
    last_end = None
    for it in citers:
        a, b = it["ts_start"], it["ts_end"]
        if last_end is None or a > last_end:
            covered += b - a
            last_end = b
        elif b > last_end:
            covered += b - last_end
            last_end = b
    coverage = covered / consumer_wall if consumer_wall else 0.0
    return {
        "phase_us": {p: phase_us.get(p, 0.0) for p in PHASES},
        "uncategorized_us": uncategorized_us,
        "coverage": coverage,
        "consumer_wall_us": consumer_wall,
        "instrumented_us": covered,
        "native_top": dict(sorted(native.items(), key=lambda x: -x[1])[:25]),
    }


def decode_mode_counts(events: List[dict], spans: List[dict]) -> dict:
    """Clean vs window-absent decode counts — a CORE domain fact (the
    speculative bootstrap is the slow path).  Lives on the FOLDED worker track
    as a badge (a CPU-region fact, not wall-spine weight).

    Source: `worker.decode_mode` instant events (gzippy) carry args.mode; if
    absent, fall back to `worker.decode` span args.mode (rapidgzip)."""
    c = Counter()
    for e in events:
        if e.get("ph") == "i" and e.get("name") == "worker.decode_mode":
            c[e.get("args", {}).get("mode", "?")] += 1
    if not c:
        for s in spans:
            if s["name"] in ("worker.decode", "worker.decode_chunk"):
                m = s["args"].get("mode")
                if m:
                    c[m] += 1
    return dict(c)


def reduce_tool(path: str, tool: str, measured_wall_us: Optional[float]) -> dict:
    events = load_events(path)
    spans, mismatch = pair_spans_counted(events)
    if not spans:
        return {"tool": tool, "path": path, "error": "no spans"}
    consumer_tid = is_consumer_tid(spans)
    t_origin = min(s["ts_start"] for s in spans)
    wall_us = max(s["ts_end"] for s in spans) - t_origin

    spine = build_spine(spans, consumer_tid, t_origin)
    workers = build_workers(spans, consumer_tid, t_origin)
    workers["decode_modes"] = decode_mode_counts(events, spans)
    stalls = classify_stalls(spans, consumer_tid)
    phases = phase_breakdown(spans, consumer_tid)

    # wall reconciliation
    recon = {"viz_wall_us": wall_us, "measured_wall_us": measured_wall_us}
    if measured_wall_us:
        diff = abs(wall_us - measured_wall_us)
        recon["delta_us"] = wall_us - measured_wall_us
        recon["delta_pct"] = 100.0 * diff / measured_wall_us
        recon["ok"] = recon["delta_pct"] <= 10.0  # spread tolerance
    else:
        recon["ok"] = None

    return {
        "tool": tool,
        "path": path,
        "wall_us": wall_us,
        "n_events": len(events),
        "n_spans": len(spans),
        "mismatch": mismatch,
        "spine": spine,
        "workers": workers,
        "stalls": stalls,
        "phases": phases,
        "reconciliation": recon,
    }


def main():
    args = sys.argv[1:]
    pos = {}
    out = "model.json"
    measured = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out":
            out = args[i + 1]; i += 2; continue
        if a == "--measured-wall-us-gz":
            measured["gz"] = float(args[i + 1]); i += 2; continue
        if a == "--measured-wall-us-rg":
            measured["rg"] = float(args[i + 1]); i += 2; continue
        if "=" in a:
            k, v = a.split("=", 1)
            pos[k] = v
        i += 1
    if not pos:
        print(__doc__)
        sys.exit(1)
    tools = []
    for tool, path in pos.items():
        tools.append(reduce_tool(path, tool, measured.get(tool)))
    model = {
        "schema": "decode-viz/1",
        "phases": PHASES,
        "tools": tools,
    }
    with open(out, "w") as f:
        json.dump(model, f, indent=2)
    # short stderr summary
    for t in tools:
        if "error" in t:
            print(f"{t['tool']}: ERROR {t['error']}", file=sys.stderr)
            continue
        s = t["stalls"]
        print(f"{t['tool']}: wall={t['wall_us']/1000:.1f}ms "
              f"spine={t['spine']['spine_total_us']/1000:.1f}ms "
              f"coverage={t['phases']['coverage']*100:.0f}% "
              f"stalls rate/place={s['n_rate']}/{s['n_placement']} "
              f"({s['rate_us']/1000:.1f}/{s['placement_us']/1000:.1f}ms) "
              f"BE-mismatch={t['mismatch']['total_mismatch']}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
