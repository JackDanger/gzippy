#!/usr/bin/env python3
"""Analyze ZEN ceiling raw.csv: per-cell best-of-N walls + ratios + A/A spread.
Arms: GZ/GZ2 (baseline, oracle off), GZC (ceiling), RG/RG2.
ratio_base = best(GZ,GZ2)/best(RG,RG2);  ratio_ceil = best(GZC)/best(RG,RG2).
A/A spread (sanity floor) = |best(GZ)-best(GZ2)| and |best(RG)-best(RG2)| / best."""
import sys, csv
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))  # (corp,T) -> arm -> [ms]
with open(sys.argv[1]) as f:
    for corp, T, arm, r, w in csv.reader(f):
        try:
            rows[(corp, int(T))][arm].append(float(w) / 1e6)  # ns -> ms
        except ValueError:
            pass

def best(xs): return min(xs) if xs else float('nan')

print(f"{'cell':<16}{'gz_base':>9}{'gz_ceil':>9}{'rg':>9}{'r_base':>8}{'r_ceil':>8}"
      f"{'AAgz%':>7}{'AArg%':>7}  verdict")
for (corp, T) in sorted(rows):
    a = rows[(corp, T)]
    gz = best(a.get('GZ', []) + a.get('GZ2', []))
    gz1, gz2 = best(a.get('GZ', [])), best(a.get('GZ2', []))
    gzc = best(a.get('GZC', []))
    rg = best(a.get('RG', []) + a.get('RG2', []))
    rg1, rg2 = best(a.get('RG', [])), best(a.get('RG2', []))
    rb = gz / rg if rg else float('nan')
    rc = gzc / rg if rg else float('nan')
    aagz = abs(gz1 - gz2) / min(gz1, gz2) * 100 if a.get('GZ') and a.get('GZ2') else 0
    aarg = abs(rg1 - rg2) / min(rg1, rg2) * 100 if a.get('RG') and a.get('RG2') else 0
    # verdict on ceiling closing to parity
    if rc <= 1.01:
        v = "CEIL->PARITY (decode lever real)"
    elif rc < rb - 0.005:
        v = f"ceil moves toward parity (rb {rb:.3f}->rc {rc:.3f}) but not <=1.01"
    else:
        v = "ceil FLAT (decode NOT the lever here)"
    print(f"{corp+' T'+str(T):<16}{gz:>9.1f}{gzc:>9.1f}{rg:>9.1f}{rb:>8.3f}{rc:>8.3f}"
          f"{aagz:>7.1f}{aarg:>7.1f}  {v}")
