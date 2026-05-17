#!/usr/bin/env python3
"""Summarize a GZIPPY_LOG_FILE produced by the parallel SM decoder.

Reads a JSON-lines log written by `src/decompress/parallel/trace.rs`
and prints aggregates: event counts, per-event duration percentiles,
phase wall time, and partition-level outcome breakdown.

Usage:
    python3 scripts/parallel_sm_log_summary.py /path/to/log.jsonl
    python3 scripts/parallel_sm_log_summary.py /path/to/log.jsonl --partition 5
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path


def iter_events(path):
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"WARN: bad line: {exc} -> {raw!r}", file=sys.stderr)


def fmt_us(us):
    if us < 1000:
        return f"{us:.0f}us"
    if us < 1_000_000:
        return f"{us / 1000:.1f}ms"
    return f"{us / 1_000_000:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to JSON-lines log file")
    ap.add_argument("--partition", type=int, default=None,
                    help="show timeline for one partition")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"no such log: {log_path}", file=sys.stderr)
        sys.exit(2)

    by_event = collections.Counter()
    durations_by_event = collections.defaultdict(list)
    partition_events = collections.defaultdict(list)
    first_t = None
    last_t = None

    for ev in iter_events(log_path):
        by_event[ev["ev"]] += 1
        if "duration_us" in ev:
            durations_by_event[ev["ev"]].append(ev["duration_us"])
        if "partition_idx" in ev:
            partition_events[ev["partition_idx"]].append(ev)
        t = ev["t_ns"]
        if first_t is None or t < first_t:
            first_t = t
        if last_t is None or t > last_t:
            last_t = t

    total_wall_ns = (last_t or 0) - (first_t or 0)
    print(f"=== summary: {log_path} ===")
    print(f"total events:     {sum(by_event.values()):,}")
    print(f"wall span:        {fmt_us(total_wall_ns / 1000)}")
    print(f"partitions seen:  {len(partition_events)}")
    print()
    print("event counts:")
    for ev, count in by_event.most_common():
        print(f"  {ev:32s} {count:,}")
    print()

    print("event durations (microseconds, summed across all instances):")
    print(f"  {'event':32s} {'count':>8s} {'sum_ms':>10s} {'p50_us':>10s} {'p95_us':>10s} {'max_us':>10s}")
    for ev, durs in sorted(durations_by_event.items(),
                           key=lambda kv: -sum(kv[1])):
        durs_sorted = sorted(durs)
        n = len(durs_sorted)
        p50 = durs_sorted[n // 2]
        p95 = durs_sorted[min(n - 1, int(n * 0.95))]
        print(
            f"  {ev:32s} {n:>8d} {sum(durs) / 1000:>10.1f} "
            f"{p50:>10.0f} {p95:>10.0f} {max(durs):>10.0f}"
        )
    print()

    # Per-partition outcome: speculative_accept / speculative_mismatch /
    # speculative_missing — diagnoses whether speculation pays off.
    accepts = sum(1 for ev in partition_events.values()
                  if any(e["ev"] == "speculative_accept" for e in ev))
    mismatches = sum(1 for ev in partition_events.values()
                     if any(e["ev"] == "speculative_mismatch" for e in ev))
    missing = sum(1 for ev in partition_events.values()
                  if any(e["ev"] == "speculative_missing" for e in ev))
    total = len(partition_events)
    print("speculation outcomes (per partition):")
    print(f"  accepted:        {accepts:>4d}  ({100 * accepts / max(total, 1):.1f}%)")
    print(f"  mismatched:      {mismatches:>4d}  ({100 * mismatches / max(total, 1):.1f}%)")
    print(f"  missing:         {missing:>4d}  ({100 * missing / max(total, 1):.1f}%)")
    print()

    # Per-chunk overshoot when speculative_mismatch fires: how far off
    # was the speculative start? Reveals the BlockFinder phantom rate.
    overshoots = []
    for events in partition_events.values():
        for e in events:
            if e["ev"] == "speculative_mismatch":
                overshoots.append(abs(e["speculative_start"] - e["expected_start"]))
    if overshoots:
        print(f"phantom boundary offset (bits, |speculative - expected|):")
        print(f"  count={len(overshoots)}, mean={statistics.mean(overshoots):.0f}, "
              f"median={statistics.median(overshoots):.0f}, "
              f"max={max(overshoots)}")
        print()

    if args.partition is not None:
        idx = args.partition
        evs = partition_events.get(idx, [])
        if not evs:
            print(f"no events for partition {idx}")
            return
        print(f"=== partition {idx} timeline ===")
        for e in sorted(evs, key=lambda x: x["t_ns"]):
            t_ms = e["t_ns"] / 1_000_000
            extra = {k: v for k, v in e.items()
                     if k not in ("t_ns", "thread", "ev")}
            print(f"  [{t_ms:>9.2f}ms] {e['thread']:>14s}  {e['ev']:<22s}  {extra}")


if __name__ == "__main__":
    main()
