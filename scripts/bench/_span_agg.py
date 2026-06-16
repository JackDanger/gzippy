#!/usr/bin/env python3
import json, sys, collections
path = sys.argv[1]
# chrome trace: events with ph B/E (begin/end) per (pid,tid). Pair them.
events = []
with open(path) as f:
    data = f.read().strip()
# file may be a JSON array or one-object-per-line; handle both
try:
    arr = json.loads(data)
    if isinstance(arr, dict) and "traceEvents" in arr:
        arr = arr["traceEvents"]
except Exception:
    arr = []
    for line in data.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in "[]":
            continue
        try:
            arr.append(json.loads(line))
        except Exception:
            pass

dur = collections.defaultdict(float)   # name -> total us
cnt = collections.defaultdict(int)
stacks = collections.defaultdict(list) # (pid,tid) -> stack of (name, ts)
have_dur = False
for e in arr:
    ph = e.get("ph")
    name = e.get("name","")
    if ph == "X" and "dur" in e:
        dur[name] += e["dur"]; cnt[name]+=1; have_dur=True
    elif ph == "B":
        stacks[(e.get("pid"),e.get("tid"))].append((name, e.get("ts",0)))
    elif ph == "E":
        st = stacks[(e.get("pid"),e.get("tid"))]
        if st:
            n, ts0 = st.pop()
            dur[n] += (e.get("ts",0)-ts0); cnt[n]+=1

print(f"{'span':40s} {'total_ms':>12s} {'count':>8s} {'avg_us':>10s}")
for name in sorted(dur, key=lambda k:-dur[k]):
    ms = dur[name]/1000.0
    c = cnt[name]
    print(f"{name:40s} {ms:12.2f} {c:8d} {(dur[name]/c if c else 0):10.1f}")
