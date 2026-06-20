#!/usr/bin/env python3
import csv, random, statistics as st
from collections import defaultdict
OUT="/dev/shm/n38_slope"
B=int(open(f"{OUT}/sizes.txt").read().split("=")[1])
d=defaultdict(dict)
for r in csv.DictReader(open(f"{OUT}/raw.csv")):
    d[(r["tool"],int(r["mult"]))][int(r["rep"])]=float(r["cycles"])/B
def boot(v,n=20000):
    bs=sorted(sum(random.choice(v) for _ in v)/len(v) for _ in range(n))
    return sum(v)/len(v),bs[int(.025*n)],bs[int(.975*n)]
def slope(tool):
    reps=sorted(set(d[(tool,1)])&set(d[(tool,4)]))
    per=[(d[(tool,4)][r]-d[(tool,1)][r])/3 for r in reps]  # per extra build
    return boot(per)
print(f"=== NIGHT38 TBUILD per-build slope (cyc/B per extra litlen rebuild), silesia bytes={B} ===")
for tool in ("base","conv"):
    m1=st.median(list(d[(tool,1)].values())); m4=st.median(list(d[(tool,4)].values()))
    s,lo,hi=slope(tool)
    print(f"  {tool}: cyc/B m1={m1:.4f} m4={m4:.4f}  per-build slope={s:+.4f} [{lo:+.4f},{hi:+.4f}]")
# did conv's slope drop vs base?
reps=sorted(set(d[("conv",1)])&set(d[("conv",4)])&set(d[("base",1)])&set(d[("base",4)]))
dd=[((d[("conv",4)][r]-d[("conv",1)][r])-(d[("base",4)][r]-d[("base",1)][r]))/3 for r in reps]
m,lo,hi=boot(dd)
sig="conv slope LOWER (build work dropped)" if hi<0 else ("conv slope HIGHER" if lo>0 else "slopes TIE (no measurable build-work change)")
print(f"  Δslope(conv-base) per-build: {m:+.4f} [{lo:+.4f},{hi:+.4f}] -> {sig}")
