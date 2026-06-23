#!/usr/bin/env python3
"""Analyze the T>=2 tri-falsifier CSV (paired-interleaved perturbation sweep).

Verdicts:
  * ORTHOGONALITY (the fixation-exposer): MARKER perturbation should move the T4
    cell wall but NOT the T2 cell; RSS perturbation should move T2 but NOT T4. If
    so, the two mechanisms are orthogonal and a single 'unifying' port cannot
    close both -> the lead's synthesis is FALSIFIED.
  * CEILINGS: MARKER@100 paired Δ% = upper bound on the marker-loop wall prize;
    RSS-inflate slope (Δ% per MiB) inverted at ΔRSS=(gz-rg peak) = teardown prize.
  * sleep-control: the spin Δ must exceed the freq-neutral sleep Δ to be real.

Gate: a level 'MOVES' only if |median paired Δ%| > the BASE inter-rep spread%
(the per-cell noise floor) AND the paired Δ distribution excludes 0.
"""
import sys, csv, statistics as st
from collections import defaultdict

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith('#') or line.startswith('cell,'):
            continue
        p = line.rstrip('\n').split(',')
        if len(p) < 10:
            continue
        rows.append(p)

# index: (cell,pert,level,kind,rep) -> wall ; BASE per (cell,rep)
wall = {}
base = {}            # (cell,rep)->wall
peak = defaultdict(list)  # (cell,who)->[kib]
cells = []
for cell, corpus, thr, pert, level, kind, rep, w, sha, hits in rows:
    if pert == 'PEAKRSS':
        peak[(cell, level)].append(float(w)); continue
    rep = int(rep); w = float(w)
    if pert == 'BASE':
        base[(cell, rep)] = w
        if cell not in cells: cells.append(cell)
    else:
        wall[(cell, pert, level, kind, rep)] = w

def med(xs): return st.median(xs) if xs else float('nan')

# BASE noise floor per cell: inter-rep (max-min)/median %
base_spread = {}
base_med = {}
for cell in cells:
    bs = [base[(cell, r)] for (c, r) in base if c == cell]
    base_med[cell] = med(bs)
    base_spread[cell] = (max(bs) - min(bs)) / med(bs) * 100 if bs else float('nan')

def paired_delta_pct(cell, pert, level, kind):
    """median paired Δ% vs same-rep BASE, plus min/max paired Δ% (CI proxy)."""
    ds = []
    for (c, p, l, k, r), w in wall.items():
        if (c, p, l, k) == (cell, pert, level, kind) and (cell, r) in base:
            b = base[(cell, r)]
            ds.append((w - b) / b * 100)
    if not ds:
        return None
    return med(ds), min(ds), max(ds), len(ds)

print("="*72)
print("BASE wall (median s) and noise floor (inter-rep spread %)")
for cell in cells:
    print(f"  {cell}: base_med={base_med[cell]*1000:.2f}ms  spread={base_spread[cell]:.2f}%  (noise floor)")
print("="*72)

def verdict(cell, dpct):
    fl = base_spread[cell]
    md, lo, hi, n = dpct
    moves = abs(md) > fl and (lo > 0 or hi < 0)
    return "MOVES " if moves else "tie   ", md, lo, hi, n

print("\nPERTURBATION RESPONSES (paired Δ% vs same-rep baseline):")
for cell in cells:
    print(f"\n--- cell {cell} (noise {base_spread[cell]:.2f}%) ---")
    for pert, level, kind in [('MARKER','50','spin'),('MARKER','100','spin'),
                              ('MARKER','100','sleep'),
                              ('RSS','20','mmap'),('RSS','40','mmap'),('RSS','60','mmap')]:
        d = paired_delta_pct(cell, pert, level, kind)
        if not d: continue
        v, md, lo, hi, n = verdict(cell, d)
        print(f"  {pert:6} {level:>3} {kind:5}  {v}  Δ={md:+6.2f}%  [{lo:+.2f},{hi:+.2f}]  n={n}")

# ---- ONE-VS-TWO-LEVERS VERDICT (fixation-exposer, corrected) ----
def get(cell, pert, level, kind):
    d = paired_delta_pct(cell, pert, level, kind)
    return d[0] if d else None
def mv(cell, x): return x is not None and abs(x) > base_spread[cell]

print("\n" + "="*72)
print("ONE-VS-TWO INDEPENDENT LEVERS (fixation-exposer):")
m_t4 = get('T4','MARKER','100','spin'); m_t2 = get('T2','MARKER','100','spin')
r_t2 = get('T2','RSS','40','mmap');     r_t4 = get('T4','RSS','40','mmap')
sl_t4 = get('T4','MARKER','100','sleep'); sl_t2 = get('T2','MARKER','100','sleep')
print(f"  MARKER@100 (DECODE-compute pert): T4 Δ={m_t4}  T2 Δ={m_t2}")
print(f"  RSS@40     (RSS/teardown pert)  : T2 Δ={r_t2}  T4 Δ={r_t4}")
print(f"  sleep-control MARKER@100        : T4 Δ={sl_t4}  T2 Δ={sl_t2}  (spin must exceed sleep)")
marker_real = mv('T4', m_t4) or mv('T2', m_t2)
rss_real    = mv('T2', r_t2) or mv('T4', r_t4)
if marker_real and rss_real:
    print("  => BOTH perturbations move the wall ⇒ (at least) TWO INDEPENDENT levers exist:")
    print("     (1) decode-compute (the marker fast-loop) and (2) RSS/teardown.")
    print("     A single 'unifying decoupled-marker port' cannot be ASSUMED to capture both —")
    print("     RSS/teardown responds to a pure-RSS perturbation that does NOT touch decode.")
    print("     The lead's one-port synthesis is OVER-CLAIMED: each lever needs its own")
    print("     gated removal-oracle prize before any port. (NOT 'orthogonal by cell' —")
    print("     both cells decode markers — but orthogonal by MECHANISM.)")
elif marker_real and not rss_real:
    print("  => Only DECODE moves the wall; RSS/teardown does NOT replicate on this arch.")
    print("     The teardown finding is arch-specific or below this box's noise; focus marker.")
elif rss_real and not marker_real:
    print("  => Only RSS/teardown moves; marker-loop NOT critical here. Focus RSS.")
else:
    print("  => Neither moves beyond noise — re-check N / freeze / non-inert.")

# ---- CEILINGS ----
print("\nCEILINGS / CRITICALITY (NOTE: a slow-down slope is CRITICALITY, not a")
print("speed-up ceiling — Gate-2. To BOUND the marker speed-up, build the byte-exact")
print("removal oracle, do not extrapolate the slow slope.):")
for cell in cells:
    m = get(cell,'MARKER','100','spin'); s = get(cell,'MARKER','100','sleep')
    if m is not None:
        real = m - (s or 0)
        print(f"  marker-loop {cell}: slow@100 spin Δ={m:+.2f}% − sleep {s:+.2f}% "
              f"= ~{real:+.2f}% real-compute on the critical path (criticality, not ceiling)")
# RSS slope
rss_pts = [(int(l), get('T2','RSS',l,'mmap')) for l in ['20','40','60'] if get('T2','RSS',l,'mmap') is not None]
if len(rss_pts) >= 2:
    xs = [p[0] for p in rss_pts]; ys = [p[1] for p in rss_pts]
    # slope %/MiB via least squares
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    slope = (n*sxy - sx*sy)/(n*sxx - sx*sx) if (n*sxx-sx*sx) else float('nan')
    print(f"  Leg-2 teardown (T2): wall slope = {slope:.4f} %/MiB  pts={rss_pts}")
    # peak RSS gap
    gz = med(peak.get(('T2','gz'),[])); rg = med(peak.get(('T2','rg'),[]))
    if gz and rg:
        dmib = (gz-rg)/1024
        print(f"     peak RSS T2: gz={gz/1024:.1f}MiB rg={rg/1024:.1f}MiB ratio={gz/rg:.2f} ΔRSS={dmib:.1f}MiB")
        print(f"     => [CONFLATED first-touch+teardown — run() times the WHOLE process;")
        print(f"         this OVER-states the prize. Use _teardown_split_guest.sh for the")
        print(f"         PURE-teardown slope (~0.084 ms/MiB Intel => ~0.9% of T2 wall).]")
        print(f"         conflated estimate: {slope*dmib:+.2f}% of T2 wall")

print("\nPEAK RSS (median KiB) all cells:")
for (cell, who), v in sorted(peak.items()):
    print(f"  {cell} {who}: {med(v)/1024:.1f} MiB")
