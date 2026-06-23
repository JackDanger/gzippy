#!/usr/bin/env python3
"""T1 cross-tool analyzer: gz/<tool> median ratios per corpus + north-star verdict.
gz/<tool> < 1 => gz faster (good). Falsifier: gz/pigz must be < 1 ('beat pigz at all T')."""
import sys, csv, statistics as st
from collections import defaultdict
d=defaultdict(lambda:defaultdict(list))
for r in csv.reader(open(sys.argv[1])):
    if not r or r[0].startswith('#') or r[0]=='corpus': continue
    d[r[0]][r[1]].append(float(r[2+1]))
tools=['IGZIP','LIBDEFLATE','PIGZ','RG','GUNZIP']
def med(x): return st.median(x) if x else float('nan')
print(f"{'corpus':10} {'gz_ms':>8} " + " ".join(f"{'gz/'+t:>12}" for t in tools))
agg=defaultdict(list)
for c in d:
    gz=med(d[c]['GZ'])
    row=f"{c:10} {gz*1000:8.1f} "
    for t in tools:
        tm=med(d[c].get(t,[]))
        r=gz/tm if tm else float('nan')
        agg[t].append(r)
        row+=f"{r:12.3f} "
    print(row)
print("\nNorth-star T1 verdicts (gz/tool < 1.0 = gz faster):")
for t in tools:
    rs=[x for x in agg[t] if x==x]
    if not rs: continue
    mx=max(rs); verdict = "gz BEATS (all corpora)" if mx<0.99 else ("MIXED" if min(rs)<1.0<mx else "gz LOSES (all)")
    flag = "  <-- FALSIFIER" if t=='PIGZ' else ""
    print(f"  gz vs {t:11}: ratios {[f'{x:.3f}' for x in rs]} worst={mx:.3f}  {verdict}{flag}")
