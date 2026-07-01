#!/usr/bin/env python3
"""Supplement: for each apply_window chunk, find predecessor's window publish time."""
import json, sys
from collections import defaultdict

path = sys.argv[1]  # GZIPPY_TIMELINE JSON

with open(path) as f:
    s = f.read().strip()
if s.startswith("[") and not s.endswith("]"):
    s = s.rstrip(",\n") + "\n]"
events = json.loads(s)

# Collect causal.window_publish instants: start_bit -> ts
wp = {}  # start_bit -> ts_us
for e in events:
    if e.get("ph") == "i" and e.get("name") == "causal.window_publish":
        args = e.get("args", {})
        sb = args.get("start_bit")
        if sb is not None:
            wp[sb] = e.get("ts", 0.0)

print("All causal.window_publish start_bit -> ts_us:")
for sb in sorted(wp.keys()):
    print(f"  start_bit={sb:>14}  ts={wp[sb]:>12.1f}us")

# Pair spans
stacks = defaultdict(list)
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
                spans.append({"name": name, "tid": tid, "ts_start": bts, "ts_end": ts, "args": bargs})

# apply spans + task spans
aw_spans = sorted([s for s in spans if s["name"] == "post_process.apply_window"], key=lambda x: x["ts_start"])
task_spans = [s for s in spans if s["name"] == "post_process.task"]

def find_task(aw):
    for t in task_spans:
        if t["tid"] == aw["tid"] and t["ts_start"] <= aw["ts_start"] and t["ts_end"] >= aw["ts_end"]:
            return t
    return None

# All start_bits in order
all_start_bits = sorted(wp.keys())

print()
print("Chunk predecessor chain and gating analysis:")
print(f"{'idx':>4} {'start_bit':>14} {'pred_start_bit':>14} {'pred_wp_ts':>12} {'apply_start':>12} {'gap_us':>10} {'gap_note'}")
for i, aw in enumerate(aw_spans):
    task = find_task(aw)
    start_bit = task["args"].get("start_bit") if task else None
    if start_bit is None:
        print(f"  {i:>4}  no task found")
        continue
    # predecessor is the largest start_bit in all_start_bits that is < start_bit
    pred_sb = None
    for sb in reversed(all_start_bits):
        if sb < start_bit:
            pred_sb = sb
            break
    pred_wp_ts = wp.get(pred_sb, None) if pred_sb is not None else None
    apply_start = aw["ts_start"]
    gap = (apply_start - pred_wp_ts) if pred_wp_ts is not None else float("nan")
    gap_note = ""
    if pred_wp_ts is not None:
        if gap < 0:
            gap_note = "APPLY BEFORE PRED_WP (early-publish not yet emitted?)"
        elif gap < 500:
            gap_note = "tight (<0.5ms after pred window ready)"
        elif gap < 5000:
            gap_note = "moderate (0.5-5ms)"
        else:
            gap_note = f"LONG ({gap/1000:.0f}ms delay)"
    print(f"  {i:>4} {start_bit:>14} {str(pred_sb):>14} {str(f'{pred_wp_ts:.1f}' if pred_wp_ts else 'N/A'):>12} {apply_start:>12.1f} {str(f'{gap:.0f}' if pred_wp_ts else 'N/A'):>10}  {gap_note}")
