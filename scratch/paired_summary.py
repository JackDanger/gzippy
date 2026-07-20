#!/usr/bin/env python3
"""Per-cell paired summary of the M1 directional log (sf6 vs sf7)."""
import re, statistics, sys

log = open(sys.argv[1]).read()
cells = re.split(r"== (\S+ L\d+ T\d+) ==", log)[1:]
for name, body in zip(cells[0::2], cells[1::2]):
    pairs = re.findall(r"pair \d+ sf6=([\d.]+) sf7=([\d.]+)", body)
    if not pairs:
        continue
    ratios = [float(b) / float(a) for a, b in pairs]
    deltas = [float(b) - float(a) for a, b in pairs]
    med_r = statistics.median(ratios)
    med_d = statistics.median(deltas)
    n = len(ratios)
    lo, hi = min(ratios), max(ratios)
    wins = sum(1 for r in ratios if r < 1.0)
    print(f"{name}: n={n} median_ratio(sf7/sf6)={med_r:.4f} median_delta={med_d:+.3f}s "
          f"range=[{lo:.3f},{hi:.3f}] sf7-faster-pairs={wins}/{n}")
