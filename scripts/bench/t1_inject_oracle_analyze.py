#!/usr/bin/env python3
"""T1 inject recoverability: regress cyc/B on instr/B across inject levels.
slope d(cyc/B)/d(instr/B) ~ 1/IPC (proportional) => instructions on retiring
critical path => surplus RECOVERABLE. slope ~0 (flat cyc/B) => slack/floor."""
import sys, csv, statistics as st
from collections import defaultdict
d=defaultdict(lambda:defaultdict(lambda:defaultdict(list))); rb={}
lines=[l for l in open(sys.argv[1]) if not l.lstrip().startswith('#')]
for r in csv.DictReader(lines):
    if not r.get('corpus'): continue
    c=r['corpus']; rb[c]=float(r['rawbytes']); inj=int(r['inject'])
    d[c][inj]['cyc'].append(float(r['cyc'])); d[c][inj]['instr'].append(float(r['instr']))
def med(x): return st.median(x)
for c in d:
    print(f"\n=== {c} ===")
    pts=[]
    for inj in sorted(d[c]):
        cyc=med(d[c][inj]['cyc'])/rb[c]; ins=med(d[c][inj]['instr'])/rb[c]
        pts.append((ins,cyc,inj))
        print(f"  inject={inj}: instr/B={ins:.3f}  cyc/B={cyc:.3f}  IPC={ins/cyc:.3f}")
    # linear regression cyc/B vs instr/B
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; n=len(xs)
    sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    slope=(n*sxy-sx*sy)/(n*sxx-sx*sx)
    base_ipc=pts[0][0]/pts[0][1]
    print(f"  -> slope d(cyc/B)/d(instr/B) = {slope:.3f}  (proportional ~1/IPC={1/base_ipc:.3f})")
    verdict = "RECOVERABLE (instr on retiring critical path)" if slope>0.5/base_ipc else ("PARTIAL" if slope>0.2 else "SLACK/FLOOR (cyc flat vs instr)")
    print(f"  -> {verdict}")
    # implied wall recovery for cutting 2 instr/B
    print(f"  -> cutting 2.0 instr/B would save ~{slope*2.0:.3f} cyc/B ({slope*2.0/pts[0][1]*100:.1f}% of baseline cyc/B)")
