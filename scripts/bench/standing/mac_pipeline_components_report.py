#!/usr/bin/env python3
"""Reduce mac_pipeline_components.sh samples to the per-component tax breakdown.

Per corpus, best-of-N (min instructions = least-noise warm sample) per arm, with
the within-arm spread (max-min)/min as the A/A trust gate (>0.5% => UNTRUSTED).
Then the causal decomposition:
  tax_total   = base4 - base1               (the T1->T instruction inflation)
  marker_mach = base4 - noprefetch4         (CAUSAL: speculation->marker machinery)
  marker_share= marker_mach / tax_total     (fraction of tax that is the markers)
"""
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
# samples[(corpus,arm)] = list of instr (rep>0 only)
samp = defaultdict(list)
cyc = defaultdict(list)
for r in rows:
    if int(r["rep"]) == 0:
        continue
    if r["instr"] and r["instr"] != "0":
        samp[(r["corpus"], r["arm"])].append(int(r["instr"]))
        cyc[(r["corpus"], r["arm"])].append(int(r["cyc"]))

corpora = []
for r in rows:
    if r["corpus"] not in corpora:
        corpora.append(r["corpus"])
arms = ["base1", "base4", "noprefetch4", "thin1", "nocrc4"]

def best(c, a):
    v = samp.get((c, a), [])
    return min(v) if v else None

def spread(c, a):
    v = samp.get((c, a), [])
    return (max(v) - min(v)) / min(v) if v else None

print("\n=== PIPELINE-TAX COMPONENT DECOMPOSITION (instructions retired, best-of-N) ===")
print("    base1=clean floor  base4=full pipeline  noprefetch4=no-speculation(clean)\n")
hdr = f"{'corpus':9s} {'base1(M)':>9s} {'base4(M)':>9s} {'noPF4(M)':>9s} {'thin1(M)':>9s} | {'tax':>7s} {'mkrMach':>8s} {'mkr/tax':>8s} {'infl':>5s} | spread%"
print(hdr)
print("-" * len(hdr))
for c in corpora:
    b1, b4, npf, th = best(c, "base1"), best(c, "base4"), best(c, "noprefetch4"), best(c, "thin1")
    if None in (b1, b4, npf):
        print(f"{c:9s}  (missing arms)")
        continue
    tax = b4 - b1
    mm = b4 - npf
    mfrac = mm / tax if tax else 0
    infl = b4 / b1
    sp = max(spread(c, a) or 0 for a in ["base1", "base4", "noprefetch4"])
    # Gate-1: the marker-machinery signal must dwarf the A/A instr spread.
    # spread is on base4 (~1e9 instr); signal is the absolute delta mm.
    spread_abs = sp * b4
    sig_over_spread = mm / spread_abs if spread_abs else float("inf")
    flag = "" if sig_over_spread >= 10 else "  <signal NOT >> spread>"
    print(f"{c:9s} {b1/1e6:9.1f} {b4/1e6:9.1f} {npf/1e6:9.1f} {(th/1e6 if th else 0):9.1f} | "
          f"{tax/1e6:7.1f} {mm/1e6:8.1f} {mfrac*100:7.1f}% {infl:5.2f} | {sp*100:.2f}  sig/spread={sig_over_spread:.0f}x{flag}")

print("\n  tax = base4-base1 (T1->T inflation, instr).  mkrMach = base4-noprefetch4 ")
print("  (CAUSAL: removing speculation removes the window-absent u16-marker machinery).")
print("  mkr/tax = share of the tax that IS the marker machinery.  infl = base4/base1.")
# nocrc diagnostic — NON-ISOLATING (FOLD_NOCRC also skips the drain/changes the
# post-path, so its delta is not a clean CRC cost; observed NEGATIVE = nocrc heavier).
# CRC is not a tax component anyway (it runs at T1 too). Reported only for honesty.
print("\n  nocrc4 diagnostic (base4 - nocrc4) — NON-ISOLATING (path side-effect, not clean CRC):")
for c in corpora:
    b4, nc = best(c, "base4"), best(c, "nocrc4")
    if b4 and nc:
        print(f"    {c:9s} base4-nocrc4 = {(b4-nc)/1e6:+.1f}M")
