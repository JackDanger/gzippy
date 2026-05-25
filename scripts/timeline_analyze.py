#!/usr/bin/env python3
"""Timeline analyzer for gzippy / rapidgzip cross-tool trace_v2 logs.

Input: one or two Chrome trace JSON files emitted by trace_v2.rs
(or the rapidgzip-side equivalent — same schema, just different `pid`).

Output (single-file mode):
  - Per-thread utilization summary
  - Top span totals (sum of duration)
  - Top wait totals (sum of `wait.*` spans)
  - Critical-path estimate (longest chain through wait_for edges)
  - Lock contention summary (sum of `lock.wait` per `args.lock`)
  - Allocation hotspots (sum of `alloc` event bytes per `args.site`)

Output (diff mode, two files):
  - Side-by-side comparison of each metric above
  - "Where does gzippy spend N% more time?" ranked list

Usage:
  python3 scripts/timeline_analyze.py /tmp/gzippy.json
  python3 scripts/timeline_analyze.py /tmp/gzippy.json /tmp/rapidgzip.json
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_events(path: str) -> List[dict]:
    """Load Chrome-trace JSON. Handles trailing comma + missing close bracket."""
    with open(path) as f:
        s = f.read()
    s = s.strip()
    if s.startswith("[") and not s.endswith("]"):
        s = s.rstrip(",\n") + "\n]"
    elif s.endswith(","):
        s = s.rstrip(",\n") + "\n"
        if not s.endswith("]"):
            s += "]"
    if not s.startswith("["):
        s = "[" + s
    return json.loads(s)


def pair_spans(events: List[dict]) -> List[dict]:
    """Pair B/E events into spans with duration. Returns list of:
    {name, tid, pid, ts_start, ts_end, dur, args}
    """
    stacks: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    spans = []
    for e in events:
        if e.get("ph") == "B":
            stacks[(e.get("pid", 0), e.get("tid", 0))].append(e)
        elif e.get("ph") == "E":
            key = (e.get("pid", 0), e.get("tid", 0))
            if stacks[key]:
                b = stacks[key].pop()
                if b["name"] != e["name"]:
                    # Mismatched — skip (could log warning).
                    pass
                spans.append({
                    "name": b["name"],
                    "pid": b.get("pid", 0),
                    "tid": b.get("tid", 0),
                    "ts_start": b.get("ts", 0),
                    "ts_end": e.get("ts", 0),
                    "dur": e.get("ts", 0) - b.get("ts", 0),
                    "args": b.get("args", {}),
                })
    return spans


def instant_events(events: List[dict]) -> List[dict]:
    return [e for e in events if e.get("ph") == "i"]


def summarize(path: str) -> dict:
    events = load_events(path)
    spans = pair_spans(events)
    instants = instant_events(events)

    if not spans:
        return {"path": path, "error": "no spans paired"}

    wall = max(s["ts_end"] for s in spans) - min(s["ts_start"] for s in spans)

    # Per-name totals.
    by_name = defaultdict(lambda: {"count": 0, "sum_us": 0, "max_us": 0})
    for s in spans:
        n = by_name[s["name"]]
        n["count"] += 1
        n["sum_us"] += s["dur"]
        n["max_us"] = max(n["max_us"], s["dur"])
    by_name_list = sorted(
        ((n, v["sum_us"], v["count"], v["max_us"]) for n, v in by_name.items()),
        key=lambda x: -x[1],
    )

    # Per-thread utilization (rough): per-thread sum of non-wait spans /
    # wall = busy fraction.
    by_thread_busy = defaultdict(int)
    by_thread_wait = defaultdict(int)
    for s in spans:
        key = (s["pid"], s["tid"])
        if s["name"].startswith("wait.") or s["name"].startswith("lock.wait"):
            by_thread_wait[key] += s["dur"]
        elif not s["name"].startswith("lock.held"):
            by_thread_busy[key] += s["dur"]

    # Lock contention.
    lock_wait = defaultdict(lambda: {"count": 0, "sum_us": 0, "max_us": 0})
    for s in spans:
        if s["name"] == "lock.wait":
            lock = s.get("args", {}).get("lock", "<unknown>")
            entry = lock_wait[lock]
            entry["count"] += 1
            entry["sum_us"] += s["dur"]
            entry["max_us"] = max(entry["max_us"], s["dur"])
    lock_list = sorted(
        ((k, v["sum_us"], v["count"], v["max_us"]) for k, v in lock_wait.items()),
        key=lambda x: -x[1],
    )

    # Allocation hotspots.
    alloc_by_site = defaultdict(lambda: {"count": 0, "bytes": 0})
    for e in instants:
        if e.get("name") == "alloc":
            site = e.get("args", {}).get("site", "<unknown>")
            sz = e.get("args", {}).get("bytes", 0)
            alloc_by_site[site]["count"] += 1
            alloc_by_site[site]["bytes"] += sz
    alloc_list = sorted(
        ((s, v["bytes"], v["count"]) for s, v in alloc_by_site.items()),
        key=lambda x: -x[1],
    )

    return {
        "path": path,
        "wall_us": wall,
        "n_events": len(events),
        "n_spans": len(spans),
        "n_instants": len(instants),
        "by_name": by_name_list,
        "by_thread_busy": dict(by_thread_busy),
        "by_thread_wait": dict(by_thread_wait),
        "lock_wait": lock_list,
        "alloc_hotspots": alloc_list,
    }


def critical_path(events: List[dict]) -> List[Tuple[str, int, int, int]]:
    """Naive critical-path estimate: for each thread, find its longest
    chain of work, attribute waits to their producer thread, repeat.

    Returns top contributors with (span_name, tid, dur_us, fraction_of_wall).

    A real critical path algorithm would build the DAG of wait_for edges
    and run longest-path. This is a simpler approximation that ranks
    waits + their causing-thread's work.
    """
    spans = pair_spans(events)
    if not spans:
        return []
    wall = max(s["ts_end"] for s in spans) - min(s["ts_start"] for s in spans)
    # For now, just return the spans that overlap the most with the consumer
    # thread's wait periods.
    consumer_waits = [s for s in spans if s["name"].startswith("wait.") and s["tid"] == 1]
    contributors = []
    for w in consumer_waits:
        # Find spans on other threads that overlap this wait window.
        for s in spans:
            if s["tid"] == w["tid"]:
                continue
            overlap = max(0, min(w["ts_end"], s["ts_end"]) - max(w["ts_start"], s["ts_start"]))
            if overlap > 0:
                contributors.append((s["name"], s["tid"], overlap, overlap / wall if wall else 0))
    # Aggregate
    agg = defaultdict(int)
    for n, t, o, _ in contributors:
        agg[(n, t)] += o
    rank = sorted(agg.items(), key=lambda x: -x[1])
    return [(n, t, dur, dur / wall if wall else 0) for (n, t), dur in rank[:20]]


def fmt_us(us: int) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1000:
        return f"{us / 1000:.1f}ms"
    return f"{us}us"


def print_summary(s: dict):
    print(f"\n=== {s['path']} ===")
    if "error" in s:
        print(f"  ERROR: {s['error']}")
        return
    print(f"wall          : {fmt_us(s['wall_us'])}")
    print(f"events        : {s['n_events']}")
    print(f"spans         : {s['n_spans']}")
    print(f"instants      : {s['n_instants']}")

    print("\nTop spans by sum duration:")
    print(f"  {'name':40s} {'sum':>10s} {'count':>8s} {'max':>10s}")
    for n, sum_us, count, max_us in s["by_name"][:20]:
        print(f"  {n:40s} {fmt_us(sum_us):>10s} {count:>8d} {fmt_us(max_us):>10s}")

    print("\nPer-thread busy / wait:")
    print(f"  {'pid/tid':>10s} {'busy':>10s} {'wait':>10s}")
    keys = set(s["by_thread_busy"].keys()) | set(s["by_thread_wait"].keys())
    for k in sorted(keys):
        b = s["by_thread_busy"].get(k, 0)
        w = s["by_thread_wait"].get(k, 0)
        print(f"  {f'{k[0]}/{k[1]}':>10s} {fmt_us(b):>10s} {fmt_us(w):>10s}")

    if s["lock_wait"]:
        print("\nLock contention (sum wait per lock):")
        print(f"  {'lock':32s} {'sum':>10s} {'count':>8s} {'max':>10s}")
        for n, sum_us, count, max_us in s["lock_wait"][:20]:
            print(f"  {n:32s} {fmt_us(sum_us):>10s} {count:>8d} {fmt_us(max_us):>10s}")

    if s["alloc_hotspots"]:
        print("\nAllocation hotspots (sum bytes per site):")
        print(f"  {'site':40s} {'bytes':>12s} {'count':>8s}")
        for site, byts, count in s["alloc_hotspots"][:20]:
            print(f"  {site:40s} {byts:>12d} {count:>8d}")


def print_diff(left: dict, right: dict):
    print("\n=== DIFF ===")
    print(f"  {'left':>20s}    {'right':>20s}")
    print(f"wall          : {fmt_us(left['wall_us']):>20s}    {fmt_us(right['wall_us']):>20s}")

    # Map name → sum_us per side for comparison.
    L = {n: s for n, s, _, _ in left["by_name"]}
    R = {n: s for n, s, _, _ in right["by_name"]}
    all_names = sorted(set(L.keys()) | set(R.keys()))
    print("\nSpan-name sum duration (left vs right; sorted by diff |left - right| desc):")
    print(f"  {'name':40s} {'left':>10s} {'right':>10s} {'delta':>10s}")
    ranked = sorted(all_names, key=lambda n: -abs(L.get(n, 0) - R.get(n, 0)))
    for n in ranked[:25]:
        l = L.get(n, 0)
        r = R.get(n, 0)
        print(
            f"  {n:40s} {fmt_us(l):>10s} {fmt_us(r):>10s} {fmt_us(l - r):>10s}"
        )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    summaries = [summarize(p) for p in sys.argv[1:]]
    for s in summaries:
        print_summary(s)
    if len(summaries) == 2:
        print_diff(*summaries)
        print("\nCritical-path contributors (waits overlapping consumer thread):")
        events = load_events(sys.argv[1])
        cp = critical_path(events)
        print(f"  {'name':40s} {'tid':>4s} {'dur':>10s} {'pct_wall':>10s}")
        for n, t, d, pct in cp:
            print(f"  {n:40s} {t:>4d} {fmt_us(d):>10s} {100 * pct:>9.1f}%")


if __name__ == "__main__":
    main()
