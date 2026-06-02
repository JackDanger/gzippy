#!/usr/bin/env python3
# Aggregate per-span total duration from a trace_v2 Chrome-trace JSON (B/E events).
# Spans nest per-tid; duration = matched E.ts - B.ts. Reports count, sum_ms, p50/p95/max us.
import json, sys, statistics, re
from collections import defaultdict

path = sys.argv[1]
want = set(sys.argv[2:]) if len(sys.argv) > 2 else None

# file is "[\n{..},\n{..},\n" possibly unterminated -> parse line by line.
durs = defaultdict(list)
stacks = defaultdict(list)  # tid -> [(name, ts), ...]
nlines = 0
with open(path) as f:
    for line in f:
        line = line.strip().rstrip(',')
        if not line or line == '[' or line == ']':
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        nlines += 1
        ph = e.get('ph'); tid = e.get('tid'); name = e.get('name'); ts = e.get('ts')
        if ph == 'B':
            stacks[tid].append((name, ts))
        elif ph == 'E':
            st = stacks[tid]
            # pop until matching name (defensive against drops)
            while st:
                bn, bts = st.pop()
                if bn == name:
                    durs[name].append(ts - bts)  # us
                    break

print(f"parsed {nlines} events; {len(durs)} distinct spans")
print(f"{'span':34s} {'count':>8s} {'sum_ms':>10s} {'p50_us':>10s} {'p95_us':>10s} {'max_us':>10s}")
items = sorted(durs.items(), key=lambda kv: -sum(kv[1]))
for name, d in items:
    if want and name not in want:
        continue
    d2 = sorted(d)
    p50 = statistics.median(d2)
    p95 = d2[min(len(d2)-1, int(0.95*len(d2)))]
    print(f"{name:34s} {len(d):>8d} {sum(d)/1000:>10.1f} {p50:>10.1f} {p95:>10.1f} {max(d):>10.1f}")
