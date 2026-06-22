#!/usr/bin/env python3
"""Analyze ZEN STEP-1 ceiling raw.csv: per-cell best-of-N walls + ratios + A/A spread.

Arms:
  GZ/GZ2 = baseline (oracle off, production)
  GZC    = u8 ceiling (over-generous: clean decode, NO u16, NO resolve)
  GZCU   = U16 ceiling, consumer-serial resolve (PESSIMISTIC resolve location)
  GZCW   = U16 ceiling, worker-parallel resolve (OPTIMISTIC resolve location)
  RG/RG2 = rapidgzip comparator

Bracket:  baseline > {GZCU, GZCW} > GZC.   ratio = arm_best / best(RG,RG2)  (>1 = gz slower).
The corrected U16-preserving ceiling is {r_u16c (pessimistic), r_u16w (optimistic)};
the real pool-parallel-but-consumer-blocked resolve lies between them.

Pre-registered verdict (per cell):
  CONFIRM (asm worth building) iff BOTH u16 arms <= ~1.01 (decode dominates).
  REFUTE  (stop, no asm)       iff NEITHER u16 arm gets near parity (u16+resolve traffic dominates).
  PARTIAL/AMBIGUOUS            iff the u16 arms straddle ~1.01 (location-sensitive; faithful
                                 pool arm needed) OR u8 closes but u16 does not (asm-decode alone
                                 insufficient; resolve/u16 traffic is the real cost)."""
import sys, csv
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))
with open(sys.argv[1]) as f:
    for corp, T, arm, r, w in csv.reader(f):
        try:
            rows[(corp, int(T))][arm].append(float(w) / 1e6)  # ns -> ms
        except ValueError:
            pass

def best(xs): return min(xs) if xs else float('nan')

hdr = (f"{'cell':<14}{'gz':>8}{'u8':>8}{'u16c':>8}{'u16w':>8}{'rg':>8}"
       f"{'r_base':>8}{'r_u8':>7}{'r16c':>7}{'r16w':>7}{'AAg%':>6}{'AAr%':>6}  verdict")
print(hdr)
print("-" * len(hdr))
for (corp, T) in sorted(rows):
    a = rows[(corp, T)]
    gz  = best(a.get('GZ', []) + a.get('GZ2', []))
    gz1, gz2 = best(a.get('GZ', [])), best(a.get('GZ2', []))
    u8  = best(a.get('GZC', []))
    u16c = best(a.get('GZCU', []))
    u16w = best(a.get('GZCW', []))
    rg  = best(a.get('RG', []) + a.get('RG2', []))
    rg1, rg2 = best(a.get('RG', [])), best(a.get('RG2', []))
    rb  = gz / rg if rg else float('nan')
    r8  = u8 / rg if rg else float('nan')
    r16c = u16c / rg if rg else float('nan')
    r16w = u16w / rg if rg else float('nan')
    aag = abs(gz1 - gz2) / min(gz1, gz2) * 100 if a.get('GZ') and a.get('GZ2') else 0
    aar = abs(rg1 - rg2) / min(rg1, rg2) * 100 if a.get('RG') and a.get('RG2') else 0

    near = lambda x: x <= 1.01
    if rb <= 1.01:
        v = "baseline already TIE/beats (not a losing cell)"
    elif near(r16c) and near(r16w):
        v = "CONFIRM: both u16 arms -> parity (decode dominates)"
    elif near(r16w) and not near(r16c):
        v = "AMBIGUOUS: location-sensitive (u16w closes, u16c not)"
    elif (not near(r16c)) and (not near(r16w)):
        if near(r8):
            v = "REFUTE-ish: u8 closes but u16 does NOT -> resolve/u16 traffic is the cost, not decode"
        else:
            v = "REFUTE: even u8 ceiling does not reach parity -> not decode-bound at all"
    else:
        v = "mixed"
    print(f"{corp+' T'+str(T):<14}{gz:>8.1f}{u8:>8.1f}{u16c:>8.1f}{u16w:>8.1f}{rg:>8.1f}"
          f"{rb:>8.3f}{r8:>7.3f}{r16c:>7.3f}{r16w:>7.3f}{aag:>6.1f}{aar:>6.1f}  {v}")
