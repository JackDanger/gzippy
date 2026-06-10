#!/usr/bin/env python3
"""Extract post_process.apply_window overlap/gating facts from gzippy trace files.

Usage:
  python3 extract_apply_window.py TIMELINE_JSON SM_LOG_JSONL VERBOSE_TXT

TIMELINE_JSON: GZIPPY_TIMELINE output (Chrome trace JSON B/E events)
SM_LOG_JSONL:  GZIPPY_LOG_FILE output (JSON-lines structured events)
VERBOSE_TXT:   GZIPPY_VERBOSE stderr output
"""

import json
import sys
import re
from collections import defaultdict

def load_timeline(path):
    """Load Chrome-trace JSON. Handles trailing comma + missing close bracket."""
    with open(path) as f:
        s = f.read()
    s = s.strip()
    if s.startswith("[") and not s.endswith("]"):
        s = s.rstrip(",\n") + "\n]"
    elif s.endswith(","):
        s = s.rstrip(",\n")
        if not s.endswith("]"):
            s += "]"
    if not s.startswith("["):
        s = "[" + s
    return json.loads(s)

def load_jsonl(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events

def pair_spans_by_tid(events):
    """Pair B/E events into spans. Returns list of dicts with name,tid,ts_start,ts_end,dur_us,args."""
    stacks = defaultdict(list)  # (pid,tid) -> [(name, ts, args)]
    spans = []
    for e in events:
        ph = e.get("ph")
        pid = e.get("pid", 0)
        tid = e.get("tid", 0)
        name = e.get("name", "")
        ts = e.get("ts", 0.0)
        args = e.get("args", {})
        key = (pid, tid)
        if ph == "B":
            stacks[key].append((name, ts, args))
        elif ph == "E":
            st = stacks[key]
            if st:
                bn, bts, bargs = st.pop()
                if bn == name:
                    spans.append({
                        "name": name,
                        "pid": pid,
                        "tid": tid,
                        "ts_start": bts,
                        "ts_end": ts,
                        "dur_us": ts - bts,
                        "args": bargs,
                    })
    return spans

def load_instant_events(events):
    """Return all instant events."""
    insts = []
    for e in events:
        if e.get("ph") == "i":
            insts.append(e)
    return insts

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_apply_window.py TIMELINE_JSON SM_LOG_JSONL [VERBOSE_TXT]")
        sys.exit(1)

    timeline_path = sys.argv[1]
    smlog_path = sys.argv[2]
    verbose_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"=== Loading timeline: {timeline_path}")
    events = load_timeline(timeline_path)
    print(f"  Total events: {len(events)}")

    spans = pair_spans_by_tid(events)
    print(f"  Paired spans: {len(spans)}")

    instants = load_instant_events(events)

    # Collect post_process.apply_window spans
    aw_spans = [s for s in spans if s["name"] == "post_process.apply_window"]
    print(f"  post_process.apply_window spans: {len(aw_spans)}")

    # Collect post_process.task spans (carry start_bit in args)
    task_spans = [s for s in spans if s["name"] == "post_process.task"]
    print(f"  post_process.task spans: {len(task_spans)}")

    # Collect causal.tax instants (carry start_bit, resolve_us)
    tax_events = [e for e in instants if e.get("name") == "causal.tax"]
    print(f"  causal.tax instants: {len(tax_events)}")

    # Match each apply_window span to its enclosing task span (same tid, task contains apply)
    def find_task_for_apply(aw):
        best = None
        for t in task_spans:
            if t["tid"] != aw["tid"]:
                continue
            if t["ts_start"] <= aw["ts_start"] and t["ts_end"] >= aw["ts_end"]:
                # take the tightest enclosing
                if best is None or (t["ts_end"] - t["ts_start"]) < (best["ts_end"] - best["ts_start"]):
                    best = t
        return best

    # Match causal.tax to apply_window by tid and proximity (tax emitted just after apply ends)
    def find_tax_for_apply(aw):
        best = None
        best_gap = float("inf")
        for tx in tax_events:
            if tx.get("tid") != aw["tid"]:
                continue
            ts = tx.get("ts", 0.0)
            # tax is emitted just after the apply window ends; gap should be small and positive
            gap = ts - aw["ts_end"]
            if 0 <= gap < 1000 and gap < best_gap:  # within 1ms after end
                best = tx
                best_gap = gap
        return best

    print()
    print("=== apply_window spans (sorted by start time) ===")
    aw_sorted = sorted(aw_spans, key=lambda s: s["ts_start"])

    rows = []
    for i, aw in enumerate(aw_sorted):
        task = find_task_for_apply(aw)
        tax = find_tax_for_apply(aw)
        start_bit = None
        marker_bytes = None
        if task:
            start_bit = task["args"].get("start_bit")
            marker_bytes = task["args"].get("marker_bytes")
        if tax:
            targs = tax.get("args", {})
            if start_bit is None:
                start_bit = targs.get("start_bit")
            if marker_bytes is None:
                marker_bytes = targs.get("marker_bytes")
        rows.append({
            "idx": i,
            "tid": aw["tid"],
            "start_us": aw["ts_start"],
            "end_us": aw["ts_end"],
            "dur_us": aw["dur_us"],
            "start_bit": start_bit,
            "marker_bytes": marker_bytes,
        })
        print(f"  [{i:2d}] tid={aw['tid']:3d}  start={aw['ts_start']:>10.1f}us  end={aw['ts_end']:>10.1f}us  dur={aw['dur_us']:>7.1f}us  start_bit={start_bit}  markers={marker_bytes}")

    # Overlap matrix
    print()
    print("=== Overlap analysis ===")
    # How many spans overlap each other?
    overlap_counts = []
    for i, a in enumerate(rows):
        count = 0
        for j, b in enumerate(rows):
            if i == j:
                continue
            # overlap = not (a ends before b starts, or b ends before a starts)
            if not (a["end_us"] <= b["start_us"] or b["end_us"] <= a["start_us"]):
                count += 1
        overlap_counts.append(count)

    max_concurrent = max(overlap_counts) + 1 if overlap_counts else 0
    print(f"  Max concurrent applies (including self): {max_concurrent}")
    print(f"  Overlaps per span: {overlap_counts}")

    # Check whether applies span multiple distinct tids
    tids_seen = set(r["tid"] for r in rows)
    print(f"  Distinct tids that ran apply_window: {sorted(tids_seen)}")

    # Serial chain check: are all applies non-overlapping (serial)?
    sorted_by_start = sorted(rows, key=lambda r: r["start_us"])
    is_serial = True
    for i in range(1, len(sorted_by_start)):
        if sorted_by_start[i]["start_us"] < sorted_by_start[i-1]["end_us"]:
            is_serial = False
            break
    print(f"  All applies strictly serial (no overlaps): {is_serial}")

    # Gating analysis: for each apply, what gated its start?
    # We need decode completion and window publish times.
    # From the sm log: chunk_decode_done events with t_ns and start_bit
    # From timeline: causal.window_publish instants

    print()
    print("=== Loading sm log for decode timing ===")
    smlog_events = load_jsonl(smlog_path)
    print(f"  SM log events: {len(smlog_events)}")

    decode_done = {}  # start_bit -> t_ns (nanoseconds from epoch)
    for ev in smlog_events:
        if ev.get("ev") == "chunk_decode_done":
            sb = ev.get("start_bit")
            if sb is not None:
                decode_done[sb] = ev.get("t_ns", 0)

    # window_publish instants from timeline
    wp_events = [e for e in instants if e.get("name") == "causal.window_publish"]
    print(f"  causal.window_publish instants in timeline: {len(wp_events)}")

    # The timeline anchor and sm log epoch are different (different Instant::now() calls).
    # We can't directly compare t_ns (sm log nanoseconds from sm-log-epoch)
    # with ts (timeline microseconds from timeline-anchor).
    # However, we CAN compare within each system, and we can look at relative timing.
    # The key question for gating: is the apply_window start gap from the end of
    # the task span's materialize phase (i.e., how long after task started did apply start)?
    # Also: is apply_window start close to task start? (task = materialize + apply)
    # The task span covers materialize + apply; the gap = apply.start - task.start = materialize time.

    print()
    print("=== Gating attribution ===")
    print("  (apply start - task start = materialize phase duration)")
    print("  (cannot correlate sm-log t_ns with timeline ts directly — different epochs)")
    print()
    print(f"  {'idx':>4} {'start_bit':>14} {'task_start_us':>14} {'apply_start_us':>14} {'materialize_us':>14} {'dur_us':>10}")
    for i, aw in enumerate(aw_sorted):
        task = find_task_for_apply(aw)
        mat_us = (aw["ts_start"] - task["ts_start"]) if task else float("nan")
        start_bit = rows[i]["start_bit"]
        task_start = task["ts_start"] if task else float("nan")
        print(f"  {i:>4} {str(start_bit):>14} {task_start:>14.1f} {aw['ts_start']:>14.1f} {mat_us:>14.1f} {aw['dur_us']:>10.1f}")

    # Window publish times from timeline — match to apply window by start_bit
    # causal.window_publish has args start_bit (the chunk's start_bit) and end_bit
    wp_by_startbit = {}
    for e in wp_events:
        args = e.get("args", {})
        sb = args.get("start_bit")
        if sb is not None:
            wp_by_startbit[sb] = e.get("ts", 0.0)

    print()
    print("=== Window publish timing (from timeline) ===")
    print(f"  window_publish events by start_bit: {sorted(wp_by_startbit.keys())}")
    print()
    print(f"  {'idx':>4} {'start_bit':>14} {'wp_ts_us':>12} {'apply_start_us':>14} {'gap_wp_to_apply_us':>20}")
    for i, row in enumerate(rows):
        sb = row["start_bit"]
        wp_ts = wp_by_startbit.get(sb, None)
        gap = (row["start_us"] - wp_ts) if wp_ts is not None else float("nan")
        print(f"  {i:>4} {str(sb):>14} {str(wp_ts if wp_ts else 'N/A'):>12} {row['start_us']:>14.1f} {str(f'{gap:.1f}' if wp_ts is not None else 'N/A'):>20}")

    # Task-level gap: time between task end and next task start on same tid
    print()
    print("=== Gap between consecutive tasks on same tid ===")
    tasks_by_tid = defaultdict(list)
    for t in task_spans:
        tasks_by_tid[t["tid"]].append(t)
    for tid in sorted(tasks_by_tid.keys()):
        ts_sorted = sorted(tasks_by_tid[tid], key=lambda x: x["ts_start"])
        for j in range(1, len(ts_sorted)):
            gap = ts_sorted[j]["ts_start"] - ts_sorted[j-1]["ts_end"]
            print(f"  tid={tid} task[{j-1}]->task[{j}]: gap={gap:.1f}us  ({ts_sorted[j-1]['ts_end']:.1f} -> {ts_sorted[j]['ts_start']:.1f})")

    if verbose_path:
        print()
        print("=== Verbose counter lines ===")
        with open(verbose_path) as f:
            for line in f:
                line = line.rstrip()
                # Print lines matching eager/post_process/unified/isal/early/submit
                if any(k in line.lower() for k in [
                    "eager", "post_process", "post-process", "unified", "isal",
                    "early window", "submit_post", "prefetch post", "harvest",
                    "promoted", "already-resolved", "blockfetcher",
                    "window_seeded", "flip_to_clean", "finished_no_flip",
                ]):
                    print(f"  {line}")

    print()
    print("=== Done ===")

if __name__ == "__main__":
    main()
