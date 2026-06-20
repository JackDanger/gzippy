#!/usr/bin/env python3
import csv, sys, random, statistics as st
from collections import defaultdict
OUT=sys.argv[1] if len(sys.argv)>1 else "/dev/shm/n38_ab"
B={}
for line in open(f"{OUT}/sizes.txt"):
    if line.startswith("SIL_BYTES"):
        for kv in line.split():
            k,v=kv.split("=")
            if k=="SIL_BYTES":B["silesia"]=int(v)
            if k=="MONO_BYTES":B["monorepo"]=int(v)
data=defaultdict(dict)
for r in csv.DictReader(open(f"{OUT}/raw.csv")):
    try: ins=float(r["instructions"]);cyc=float(r["cycles"]);tc=float(r["task_clock_ms"]);llc=float(r["llc_misses"] or 0)
    except: continue
    data[(r["corpus"],r["tool"])][int(r["rep"])]=(ins,cyc,tc,llc)
def boot(d,n=20000):
    bs=[]
    for _ in range(n):
        s=[random.choice(d) for _ in d];bs.append(sum(s)/len(s))
    bs.sort();return sum(d)/len(d),bs[int(.025*n)],bs[int(.975*n)]
def paired(c,a,b):
    reps=sorted(set(data[(c,a)])&set(data[(c,b)]))
    da=[data[(c,a)][r][1]/B[c] for r in reps];db=[data[(c,b)][r][1]/B[c] for r in reps]
    diffs=[x-y for x,y in zip(da,db)];m,lo,hi=boot(diffs)
    return len(reps),st.median(da),st.median(db),m,lo,hi
print("=== NIGHT38 production A/B (cyc/B) — redundant litlen-clear convergence ===")
for c in ("silesia","monorepo"):
    if (c,"conv") not in data: continue
    print(f"\n--- {c} (bytes={B[c]}) ---")
    for tool in ("conv","base","conv2","igzip"):
        if (c,tool) not in data: continue
        vals=list(data[(c,tool)].values())
        cb=st.median([cyc/B[c] for (_,cyc,_,_) in vals])
        ib=st.median([ins/B[c] for (ins,_,_,_) in vals])
        ipc=st.median([ins/cyc for (ins,cyc,_,_) in vals])
        ghz=[cyc/(tc*1e6) for (_,cyc,tc,_) in vals]
        ghzspread=(max(ghz)-min(ghz))/st.median(ghz)*100
        print(f"  {tool:5s}: cyc/B={cb:.4f} instr/B={ib:.4f} IPC={ipc:.3f} GHz={st.median(ghz):.3f} (spread {ghzspread:.2f}%)")
    n,ma,mb,m,lo,hi=paired(c,"conv2","conv")
    sig="OK(brackets 0)" if (lo<0<hi) else "!! SELF-TEST FAIL (box noisy)"
    print(f"  [Gate-0 self-test] conv2-conv cyc/B Δ: {m:+.4f} [{lo:+.4f},{hi:+.4f}] N={n} -> {sig}")
    n,ma,mb,m,lo,hi=paired(c,"conv","base")
    sig="CI-DISJOINT(win)" if hi<0 else ("CI-DISJOINT(regress)" if lo>0 else "TIE")
    print(f"  [VERDICT] conv-base cyc/B Δ: {m:+.4f} [{lo:+.4f},{hi:+.4f}] N={n} -> {sig}")
    n,ma,mb,m,lo,hi=paired(c,"conv","igzip")
    print(f"  conv-igzip remaining gap: {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
    n,ma,mb,m,lo,hi=paired(c,"base","igzip")
    print(f"  base-igzip gap (ref):     {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
