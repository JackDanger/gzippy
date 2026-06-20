#!/usr/bin/env python3
# Analyze NIGHT35 injector sweep. Paired-by-rep cyc/B, instr/B, IPC, GHz, LLC.
# Slope = Δcyc/B per Δinjected-instr/B with bootstrap CI (must exclude 0).
import sys, glob, os, statistics, random

OUTDIR = sys.argv[1]
CORPORA = sys.argv[2:] if len(sys.argv) > 2 else ["silesia", "monorepo"]
BYTES = {}

def zbytes(corp):
    if corp not in BYTES:
        import subprocess
        BYTES[corp] = int(subprocess.check_output(f"zcat /root/{corp}.gz|wc -c", shell=True))
    return BYTES[corp]

def parse(csvpath):
    ins = cyc = tc = llc = None
    for line in open(csvpath):
        p = line.split(",")
        if len(p) < 3: continue
        val, unit, ev = p[0], p[1], p[2]
        try: v = float(val)
        except: continue
        if "instructions" in ev: ins = v
        elif "cycles" in ev: cyc = v
        elif "task-clock" in ev: tc = v   # ms
        elif "LLC-load-misses" in ev: llc = v
    return ins, cyc, tc, llc

def boot_ci(vals, n=10000):
    if not vals: return (float('nan'), float('nan'), float('nan'))
    random.seed(12345)
    means = []
    L = len(vals)
    for _ in range(n):
        s = [vals[random.randrange(L)] for _ in range(L)]
        means.append(sum(s)/L)
    means.sort()
    return (statistics.median(vals), means[int(0.025*n)], means[int(0.975*n)])

ARMS = ["IG","IG2","M0","M0b","M1","M2","M4"]
INJMULT = {"IG":None,"IG2":None,"M0":0,"M0b":0,"M1":1,"M2":2,"M4":4}

for corp in CORPORA:
    by = zbytes(corp)
    # collect per-rep per-arm
    data = {a: {} for a in ARMS}  # arm -> rep -> (cycPB, insPB, ghz, llcPKB)
    reps = set()
    for a in ARMS:
        for f in glob.glob(f"{OUTDIR}/{corp}.{a}.*.csv"):
            r = int(f.rsplit(".",2)[1])
            ins,cyc,tc,llc = parse(f)
            if ins is None or cyc is None: continue
            ghz = (cyc/(tc*1e6)) if tc else float('nan')
            data[a][r] = (cyc/by, ins/by, ghz, (llc/by*1024) if llc else float('nan'))
            reps.add(r)
    if not reps:
        print(f"\n### {corp}: NO DATA"); continue
    reps = sorted(reps)
    print(f"\n### {corp}  bytes={by}  reps={len(reps)}")
    print(f"{'arm':5} {'mult':>4} {'cyc/B':>9} {'instr/B':>9} {'IPC':>6} {'GHz':>6} {'LLCmiss/KB':>10}")
    med = {}
    for a in ARMS:
        cyc = [data[a][r][0] for r in reps if r in data[a]]
        ins = [data[a][r][1] for r in reps if r in data[a]]
        ghz = [data[a][r][2] for r in reps if r in data[a]]
        llc = [data[a][r][3] for r in reps if r in data[a]]
        if not cyc: continue
        mc, mi = statistics.median(cyc), statistics.median(ins)
        med[a] = (mc, mi)
        ipc = mi/mc if mc else float('nan')
        print(f"{a:5} {str(INJMULT[a]):>4} {mc:9.4f} {mi:9.4f} {ipc:6.3f} {statistics.median(ghz):6.3f} {statistics.median([x for x in llc if x==x] or [float('nan')]):10.3f}")
    # Gate-0 self-tests (paired)
    def paired(a,b,idx):
        return [data[a][r][idx]-data[b][r][idx] for r in reps if r in data[a] and r in data[b]]
    if "IG" in data and "IG2" in data:
        m,lo,hi = boot_ci(paired("IG2","IG",0)); print(f"  SELF-TEST IG2-IG cyc/B = {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'PASS(spans0)' if lo<0<hi else 'FAIL'}")
    if "M0" in data and "M0b" in data:
        m,lo,hi = boot_ci(paired("M0b","M0",0)); print(f"  SELF-TEST M0b-M0 cyc/B = {m:+.4f} [{lo:+.4f},{hi:+.4f}] {'PASS(spans0)' if lo<0<hi else 'FAIL'}")
    # gap vs igzip
    if "IG" in med and "M0" in med:
        print(f"  GAP M0-IG cyc/B = {med['M0'][0]-med['IG'][0]:+.4f}  instr/B = {med['M0'][1]-med['IG'][1]:+.4f}")
    # SLOPE: paired per-rep Δcyc/B vs Δinstr/B relative to M0, across M1/M2/M4.
    # slope_point(arm) = (cyc[arm]-cyc[M0]) / (instr[arm]-instr[M0]) per rep
    print("  --- INJECTOR SLOPE (Δcyc/B per Δinjected-instr/B vs M0) ---")
    for a in ["M1","M2","M4"]:
        if a not in data: continue
        slopes=[]
        for r in reps:
            if r in data[a] and r in data["M0"]:
                dins = data[a][r][1]-data["M0"][r][1]
                dcyc = data[a][r][0]-data["M0"][r][0]
                if dins>0.5: slopes.append(dcyc/dins)
        if slopes:
            m,lo,hi = boot_ci(slopes)
            inj = med[a][1]-med["M0"][1]
            print(f"    {a}: +{inj:.2f} instr/B  slope={m:+.4f} [{lo:+.4f},{hi:+.4f}] cyc/instr  {'PROPORTIONAL(CI>0)' if lo>0 else ('FLAT' if hi<0.02 else 'ambig')}")
