#!/usr/bin/env python3
import sys, glob, statistics, random, subprocess
OUTDIR=sys.argv[1]; CORPORA=sys.argv[2:] or ["silesia","monorepo"]
def zb(c): return int(subprocess.check_output(f"zcat /root/{c}.gz|wc -c",shell=True))
def parse(p):
    ins=cyc=tc=llc=None
    for line in open(p):
        a=line.split(",")
        if len(a)<3: continue
        try: v=float(a[0])
        except: continue
        e=a[2]
        if "instructions" in e: ins=v
        elif "cycles" in e: cyc=v
        elif "task-clock" in e: tc=v
        elif "LLC-load-misses" in e: llc=v
    return ins,cyc,tc,llc
def ci(v,n=10000):
    if not v: return (float('nan'),)*3
    random.seed(7); L=len(v); m=[]
    for _ in range(n): m.append(sum(v[random.randrange(L)] for _ in range(L))/L)
    m.sort(); return statistics.median(v),m[int(.025*n)],m[int(.975*n)]
ARMS=["IG","IG2","T1","T1b","T2","T4"]
for corp in CORPORA:
    by=zb(corp); data={a:{} for a in ARMS}; reps=set()
    for a in ARMS:
        for f in glob.glob(f"{OUTDIR}/{corp}.{a}.*.csv"):
            r=int(f.rsplit(".",2)[1]); ins,cyc,tc,llc=parse(f)
            if ins is None or cyc is None: continue
            data[a][r]=(cyc/by,ins/by,(cyc/(tc*1e6)) if tc else float('nan')); reps.add(r)
    reps=sorted(reps)
    if not reps: print(f"{corp}: NO DATA"); continue
    print(f"\n### {corp} bytes={by} reps={len(reps)}")
    med={}
    print(f"{'arm':5} {'cyc/B':>9} {'instr/B':>9} {'GHz':>6}")
    for a in ARMS:
        c=[data[a][r][0] for r in reps if r in data[a]]; i=[data[a][r][1] for r in reps if r in data[a]]; g=[data[a][r][2] for r in reps if r in data[a]]
        if not c: continue
        med[a]=(statistics.median(c),statistics.median(i)); print(f"{a:5} {med[a][0]:9.4f} {med[a][1]:9.4f} {statistics.median(g):6.3f}")
    def paired(a,b):return [data[a][r][0]-data[b][r][0] for r in reps if r in data[a] and r in data[b]]
    m,lo,hi=ci(paired("IG2","IG")); print(f"  SELF-TEST IG2-IG = {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'PASS' if lo<0<hi else 'FAIL'}")
    m,lo,hi=ci(paired("T1b","T1")); print(f"  SELF-TEST T1b-T1 = {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'PASS' if lo<0<hi else 'FAIL'}")
    print(f"  GAP T1-IG cyc/B = {med['T1'][0]-med['IG'][0]:+.4f}")
    # per-build slope: T2-T1 (1 extra build), (T4-T1)/3
    s2=[data["T2"][r][0]-data["T1"][r][0] for r in reps if r in data["T2"] and r in data["T1"]]
    s4=[(data["T4"][r][0]-data["T1"][r][0])/3 for r in reps if r in data["T4"] and r in data["T1"]]
    m,lo,hi=ci(s2); print(f"  PER-BUILD slope T2-T1   = {m:+.4f} [{lo:+.4f},{hi:+.4f}] cyc/B {'ON-WALL(CI>0)' if lo>0 else 'flat'}")
    m,lo,hi=ci(s4); print(f"  PER-BUILD slope (T4-T1)/3= {m:+.4f} [{lo:+.4f},{hi:+.4f}] cyc/B {'ON-WALL(CI>0)' if lo>0 else 'flat'}")
