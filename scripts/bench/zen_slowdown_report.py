#!/usr/bin/env python3
"""Marker-loop slowdown sweep: best-of-N wall per (cell, kind, mode) + slope.
CSV cols: corp,T,kind(spin|sleep|rg),mode,run,ns.
For each cell: baseline = mode0 (spin==sleep at 0). Report wall vs mode for spin
and sleep, the per-mode delta vs mode0, and rg. VERDICT: monotonic rise that the
sleep (freq-neutral) control reproduces => marker loop on the T4 critical path."""
import sys, csv
from collections import defaultdict

d = defaultdict(list)  # (corp,T,kind,mode) -> [ms]
rg = defaultdict(list)  # (corp,T) -> [ms]
for corp, T, kind, mode, run, ns in csv.reader(open(sys.argv[1])):
    try:
        ms = float(ns) / 1e6
    except ValueError:
        continue
    if kind == 'rg':
        rg[(corp, int(T))].append(ms)
    else:
        d[(corp, int(T), kind, int(mode))].append(ms)

best = lambda xs: min(xs) if xs else float('nan')
cells = sorted({(c, t) for (c, t, _, _) in d})
for (c, t) in cells:
    rgb = best(rg[(c, t)])
    base = best(d[(c, t, 'spin', 0)] + d[(c, t, 'sleep', 0)])
    print(f"\n=== {c} T{t} ===  baseline(mode0)={base:.1f}ms  rg={rgb:.1f}ms  gz/rg={base/rgb:.3f}")
    print(f"  {'mode':>5}{'spin_ms':>9}{'Δspin':>8}{'sleep_ms':>9}{'Δsleep':>8}")
    modes = sorted({m for (cc, tt, k, m) in d if cc == c and tt == t})
    sp0 = best(d[(c, t, 'spin', 0)]); sl0 = best(d[(c, t, 'sleep', 0)])
    for m in modes:
        sp = best(d.get((c, t, 'spin', m), []))
        sl = best(d.get((c, t, 'sleep', m), []))
        print(f"  {m:>5}{sp:>9.1f}{sp-sp0:>+8.1f}{sl:>9.1f}{sl-sl0:>+8.1f}")
    # crude marker wall-share estimate from +100% spin (doubling marker per-event cost
    # adds ~= marker loop's current wall contribution).
    sp100 = best(d.get((c, t, 'spin', 100), []))
    sl100 = best(d.get((c, t, 'sleep', 100), []))
    if sp100 == sp100 and base == base:
        share_spin = sp100 - sp0
        share_sleep = sl100 - sl0 if sl100 == sl100 else float('nan')
        gap_ms = base - rgb
        print(f"  est marker-wall-contribution (+100% delta): spin~{share_spin:.1f}ms "
              f"sleep~{share_sleep:.1f}ms ; current gz-rg gap={gap_ms:+.1f}ms")
        print(f"  NOTE asm captures only the EXCESS (marker 11.7->~7 cyc/B ~= 40% of marker "
              f"time): ceiling wall-save ~= 0.4*marker_contribution")
