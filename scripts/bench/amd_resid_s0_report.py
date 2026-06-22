#!/usr/bin/env python3
"""S0 report: per (corpus,threads) best-of-N cycles gz vs rg, ratio, A/A self-test,
inter-run spread, GHz-stability gate. Premise CONFIRM if gz/rg >= 1.04 & Δ>spread."""
import sys, csv, statistics
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/amd-resid-s0/raw.csv"
rows = list(csv.DictReader(open(path)))
g = defaultdict(lambda: defaultdict(list))  # (corpus,th) -> tool -> [(cyc,insn,ghz)]
for r in rows:
    try:
        g[(r["corpus"], r["threads"])][r["tool"]].append(
            (int(r["cycles"]), int(r["instructions"]), float(r["ghz"])))
    except (ValueError, KeyError):
        continue

def best(xs):  # min cycles = least contended
    return min(xs, key=lambda t: t[0])
def spreadpct(xs):
    v = [t[0] for t in xs]
    return 100.0*(max(v)-min(v))/min(v) if v and min(v) else 0.0

print(f"{'corpus':<10} {'T':>2} {'gz_cyc(best)':>14} {'rg_cyc(best)':>14} {'gz/rg':>7} "
      f"{'gzspr%':>7} {'rgspr%':>7} {'A/A%':>6} {'gzGHz':>6} {'rgGHz':>6} verdict")
for (corpus, th) in sorted(g):
    d = g[(corpus, th)]
    if "gz" not in d or "rg" not in d:
        continue
    gzb, rgb = best(d["gz"]), best(d["rg"])
    ratio = gzb[0]/rgb[0] if rgb[0] else 0
    gzspr, rgspr = spreadpct(d["gz"]), spreadpct(d["rg"])
    aa = 0.0
    if "gzAA" in d:
        aab = best(d["gzAA"]); aa = 100.0*abs(gzb[0]-aab[0])/min(gzb[0],aab[0])
    gzghz = statistics.median([t[2] for t in d["gz"]])
    rgghz = statistics.median([t[2] for t in d["rg"]])
    # Stability = WITHIN-tool rep spread (cycles is freq-invariant work; gz-vs-rg
    # aggregate cyc/wall differ legitimately by tool core-occupancy, so do NOT gate
    # on gz-vs-rg GHz). Reproducible when max(gzspr,rgspr) small and A/A tiny.
    delta = 100.0*(ratio-1.0)
    spread = max(gzspr, rgspr)
    if spread > 8.0 or aa > 3.0:
        v = f"NOISY(spr{spread:.1f}/AA{aa:.1f})"
    elif ratio >= 1.04 and delta > spread:
        v = "CONFIRM gz>rg"
    elif ratio <= 0.99:
        v = "gz<=rg"
    elif abs(delta) <= spread:
        v = "TIE(<spread)"
    else:
        v = f"weak(+{delta:.1f}%)"
    print(f"{corpus:<10} {th:>2} {gzb[0]:>14,} {rgb[0]:>14,} {ratio:>7.3f} "
          f"{gzspr:>7.1f} {rgspr:>7.1f} {aa:>6.1f} {gzghz:>6.3f} {rgghz:>6.3f} {v}")
