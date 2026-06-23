#!/usr/bin/env python3
"""A0 marker-ceiling analyzer: (base - arm)/base = marker speed-up CEILING.
A ceiling arm decodes window-absent chunks via the FAST clean asm path, so it
should be FASTER than baseline; the paired Δ is the UPPER BOUND on what porting
the marker loop could recover. Δ < noise on the U16 arms ⇒ port DEAD.
"""
import sys, csv, statistics as st
from collections import defaultdict
base=defaultdict(dict); arm=defaultdict(lambda: defaultdict(dict))
for r in csv.reader(open(sys.argv[1])):
    if not r or r[0].startswith('#') or r[0]=='cell': continue
    cell,corp,thr,a,rep,w = r[0],r[1],r[2],r[3],int(r[4]),float(r[5])
    if a=='BASE': base[cell][rep]=w
    else: arm[cell][a][rep]=w
def med(x): return st.median(x)
for cell in base:
    bs=list(base[cell].values()); bm=med(bs); noise=(max(bs)-min(bs))/bm*100
    print(f"\n=== {cell}  base_med={bm*1000:.2f}ms  noise={noise:.2f}% ===")
    print(f"  {'arm':6} {'ceiling Δ%':>10}  {'[min,max]':>16}  verdict (speed-up upper bound)")
    for a in ['U16','U16W','U8']:
        if a not in arm[cell]: continue
        ds=[(base[cell][rep]-w)/base[cell][rep]*100 for rep,w in arm[cell][a].items() if rep in base[cell]]
        m,lo,hi=med(ds),min(ds),max(ds)
        moves = abs(m)>noise and (lo>0 or hi<0)
        print(f"  {a:6} {m:+10.2f}  [{lo:+6.2f},{hi:+6.2f}]  {'RECOVERABLE' if moves else 'tie (<noise) -> NO PRIZE'}")
    # verdict
    u16=[ (base[cell][rep]-w)/base[cell][rep]*100 for rep,w in arm[cell].get('U16',{}).items() if rep in base[cell]]
    u16w=[ (base[cell][rep]-w)/base[cell][rep]*100 for rep,w in arm[cell].get('U16W',{}).items() if rep in base[cell]]
    if u16 and u16w:
        a,b=med(u16),med(u16w)
        print(f"  -> realistic marker prize bracketed [{min(a,b):+.2f}%, {max(a,b):+.2f}%] (U16W..U16)")
print("\nNOTE: ceiling arms output WRONG bytes by design (perturbation). Non-inert")
print("proof = HITS/RESOLVE_BYTES in the collector stderr, NOT sha. A0 over-credits")
print("(reroutes whole chunk); if even this brackets < noise, the port is DEAD.")
