#!/usr/bin/env python3
"""ZEN2-DECODE-MICROBENCH report: median cyc/B per (tool,corpus), A/A spread,
gz/rg ratio R, three-way verdict vs the pre-registered falsifier."""
import sys, csv, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/zen2-mb-out/raw.csv"
rows = list(csv.DictReader(open(path)))
by = defaultdict(list)            # (tool,corpus) -> [cyc_per_byte]
bytes_by = defaultdict(list)
for r in rows:
    try:
        by[(r["tool"], r["corpus"])].append(float(r["cyc_per_byte"]))
        bytes_by[(r["tool"], r["corpus"])].append(int(r["bytes"]))
    except (ValueError, KeyError):
        pass

corpora = sorted({c for (_, c) in by})
def med(xs): return st.median(xs) if xs else float("nan")
def spread(xs):
    return (max(xs) - min(xs)) / med(xs) * 100 if len(xs) > 1 and med(xs) else 0.0

print(f"{'corpus':<10} {'gz cyc/B':>9} {'rg cyc/B':>9} {'R=gz/rg':>8} "
      f"{'A/A%':>6} {'gz_wa_MB':>8} {'rg_wa_MB':>8} verdict")
verdicts = []
for c in corpora:
    gz = by[("gz", c)]; rg = by[("rg", c)]; aa = by[("gzAA", c)]
    gzm, rgm = med(gz), med(rg)
    R = gzm / rgm if rgm else float("nan")
    # A/A spread across gz and gzAA pooled medians
    aa_pct = spread(gz + aa)
    gz_mb = med(bytes_by[("gz", c)]) / 1e6 if bytes_by[("gz", c)] else 0
    rg_mb = med(bytes_by[("rg", c)]) / 1e6 if bytes_by[("rg", c)] else 0
    if R >= 1.20: v = "CONFIRM(gz slower)"
    elif R <= 1.10: v = "REFUTE(parity/gz-faster)"
    else: v = "AMBIGUOUS"
    verdicts.append(R)
    print(f"{c:<10} {gzm:>9.3f} {rgm:>9.3f} {R:>8.3f} {aa_pct:>5.1f}% "
          f"{gz_mb:>8.1f} {rg_mb:>8.1f} {v}")

print("\n--- OVERALL ---")
allR = [r for r in verdicts if r == r]
if allR:
    print(f"median R across corpora = {st.median(allR):.3f}  (R=gz/rg; <1 = gz faster/byte)")
    if all(r <= 1.10 for r in allR):
        print("VERDICT: REFUTE — gz marker decode cyc/B <= rg on all cells; do NOT build the marker asm.")
    elif sum(r >= 1.20 for r in allR) > len(allR) / 2:
        print("VERDICT: CONFIRM — gz materially slower; marker asm worth building.")
    else:
        print("VERDICT: mixed/AMBIGUOUS — see per-cell.")
