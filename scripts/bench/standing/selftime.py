#!/usr/bin/env python3
# Exclusive (self) time per span name from a partial chrome-trace, per (pid,tid)
# stack. Self-time = a span's duration minus the time covered by its children,
# so self-times are additive (sum == sum of top-level span wall across threads)
# -> conservation-respecting component breakdown.
import json, sys
from collections import defaultdict

path = sys.argv[1]
events = []
for raw in open(path):
    s = raw.strip().rstrip(",")
    if not s or s in "[]" or not s.endswith("}"):
        continue
    try:
        events.append(json.loads(s))
    except Exception:
        pass

# group by (pid,tid), keep order
byk = defaultdict(list)
for e in events:
    if e.get("ph") in ("B", "E"):
        byk[(e.get("pid"), e.get("tid"))].append(e)

self_ms = defaultdict(float)
cnt = defaultdict(int)
total_top = 0.0
for k, evs in byk.items():
    # stack of [name, start_ts, child_time_accum]
    stack = []
    for e in evs:
        ts = float(e.get("ts", 0))
        if e.get("ph") == "B":
            stack.append([e.get("name"), ts, 0.0])
        else:  # E
            # pop matching name
            idx = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == e.get("name"):
                    idx = i
                    break
            if idx is None:
                continue
            # close everything above idx as anomalies (ignore)
            frame = stack[idx]
            dur = ts - frame[1]
            selft = dur - frame[2]
            self_ms[frame[0]] += selft
            cnt[frame[0]] += 1
            stack = stack[:idx]
            if stack:
                stack[-1][2] += dur  # add full child dur to parent's child_time
            else:
                total_top += dur

tot = sum(self_ms.values())
print(f"total self-time across threads: {tot/1000.0:.1f} ms")
print(f"{'span':40s} {'self_ms':>10s} {'share':>7s} {'count':>6s}")
for name in sorted(self_ms, key=lambda x: -self_ms[x]):
    sh = 100 * self_ms[name] / tot if tot else 0
    if self_ms[name] / 1000.0 < 0.05:
        continue
    print(f"{name:40s} {self_ms[name]/1000.0:>10.2f} {sh:>6.1f}% {cnt[name]:>6d}")
