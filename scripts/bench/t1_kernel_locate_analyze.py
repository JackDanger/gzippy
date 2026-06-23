#!/usr/bin/env python3
"""T1 kernel-locate: gz vs igzip cyc/B, instr/B, IPC, brmiss/kB per corpus.
Mechanism read: gz IPC < igzip IPC => mispredict/stall-bound (Intel pattern);
gz instr/B >> igzip instr/B with IPC>= => instruction-bound (Zen2 pattern)."""
import sys, csv, statistics as st
from collections import defaultdict
d=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
rb={}
lines=[l for l in open(sys.argv[1]) if not l.lstrip().startswith('#')]
for r in csv.DictReader(lines):
    if not r.get('corpus'): continue
    c,t=r['corpus'],r['tool']; rb[c]=float(r['rawbytes'])
    d[c][t]['cyc'].append(float(r['cyc'])); d[c][t]['instr'].append(float(r['instr']))
    d[c][t]['br'].append(float(r['branches'])); d[c][t]['brm'].append(float(r['brmiss']))
def med(x): return st.median(x)
print(f"{'corpus':9} {'tool':6} {'cyc/B':>7} {'instr/B':>8} {'IPC':>6} {'brmiss/kB':>10} {'brmiss%':>8}")
for c in d:
    for t in ['GZ','IGZIP']:
        if t not in d[c]: continue
        cyc=med(d[c][t]['cyc']); ins=med(d[c][t]['instr']); br=med(d[c][t]['br']); brm=med(d[c][t]['brm'])
        print(f"{c:9} {t:6} {cyc/rb[c]:7.3f} {ins/rb[c]:8.3f} {ins/cyc:6.3f} {brm/rb[c]*1000:10.3f} {brm/br*100:8.3f}")
    g,i=d[c]['GZ'],d[c]['IGZIP']
    gcyc,icyc=med(g['cyc']),med(i['cyc']); gins,iins=med(g['instr']),med(i['instr'])
    gipc,iipc=gins/gcyc, iins/icyc
    print(f"  -> {c}: cyc gap {gcyc/icyc:.3f}x  instr gap {gins/iins:.3f}x  IPC gz {gipc:.3f} vs ig {iipc:.3f} "
          f"({'gz IPC<ig => mispredict/stall-bound' if gipc<iipc else 'gz IPC>=ig => instruction-bound'})")
