#!/usr/bin/env python3
"""NIGHT22 4-arm paired perf analyzer.
Reads perf-stat -x, CSVs from <outdir>/<corp>.<arm>.<rep>.csv (arms IG IG2 A OLD N19),
computes per-arm cyc/B, IPC, instr/B, GHz, and PAIRED diffs (X - IG) with bootstrap
95% CI + the IG2-IG self-test (must include 0). cyc/B is frequency-invariant.
"""
import sys, os, glob, random, statistics as st

def parse_csv(path):
    cyc=ins=tc=None
    try:
        with open(path) as f:
            for line in f:
                p=line.strip().split(',')
                if len(p)<3: continue
                v,ev=p[0],p[2]
                try: x=float(v)
                except ValueError: continue
                if 'instructions' in ev: ins=x
                elif 'cycles' in ev: cyc=x
                elif ev=='task-clock': tc=x  # milliseconds
    except FileNotFoundError:
        return None
    if cyc is None or ins is None: return None
    return cyc,ins,tc

def boot_ci(diffs, n=5000):
    if not diffs: return (float('nan'),)*3
    m=st.median(diffs)
    meds=[]
    L=len(diffs)
    for _ in range(n):
        s=[diffs[random.randrange(L)] for _ in range(L)]
        meds.append(st.median(s))
    meds.sort()
    lo=meds[int(0.025*n)]; hi=meds[int(0.975*n)]
    return m,lo,hi

def main():
    outdir=sys.argv[1]; corpora=sys.argv[2:]
    bytes_map={}
    with open(os.path.join(outdir,'bytes.txt')) as f:
        for line in f:
            c,b=line.split(); bytes_map[c]=int(b)
    arms=['IG','IG2','A','OLD','N19']
    random.seed(1234)
    print("\n================ NIGHT22 DECISIVE PERF (cyc/B = cycles/decompressed-byte; lower=faster) ================")
    for corp in corpora:
        B=bytes_map[corp]
        # gather per-rep per-arm metrics
        reps=sorted({int(os.path.basename(p).split('.')[-2])
                     for p in glob.glob(os.path.join(outdir,f"{corp}.*.*.csv"))})
        per={a:{'cycb':[],'ipc':[],'insb':[],'ghz':[]} for a in arms}
        paired={'A':[], 'OLD':[], 'N19':[], 'IG2':[]}  # X - IG cyc/B, per rep
        for r in reps:
            vals={}
            ok=True
            for a in arms:
                d=parse_csv(os.path.join(outdir,f"{corp}.{a}.{r}.csv"))
                if d is None: ok=False; break
                cyc,ins,tc=d
                vals[a]=(cyc,ins,tc)
                per[a]['cycb'].append(cyc/B)
                per[a]['insb'].append(ins/B)
                per[a]['ipc'].append(ins/cyc if cyc else float('nan'))
                if tc: per[a]['ghz'].append(cyc/(tc*1e6))
            if not ok: continue
            ig_cycb=vals['IG'][0]/B
            paired['A'].append(vals['A'][0]/B - ig_cycb)
            paired['OLD'].append(vals['OLD'][0]/B - ig_cycb)
            paired['N19'].append(vals['N19'][0]/B - ig_cycb)
            paired['IG2'].append(vals['IG2'][0]/B - ig_cycb)
        print(f"\n--- {corp}  (decomp bytes={B}, N={len(reps)}) ---")
        print(f"{'arm':<5}{'cyc/B':>9}{'IPC':>8}{'instr/B':>10}{'GHz':>7}")
        for a in arms:
            def mn(k):
                v=per[a][k]; return st.mean(v) if v else float('nan')
            print(f"{a:<5}{mn('cycb'):>9.4f}{mn('ipc'):>8.4f}{mn('insb'):>10.4f}{mn('ghz'):>7.3f}")
        print("  PAIRED Δcyc/B (X - igzip), median[95%CI]:  NEG ⇒ faster than igzip")
        for k in ['A','OLD','N19','IG2']:
            m,lo,hi=boot_ci(paired[k])
            tag=''
            if k=='IG2':
                tag='  <- SELF-TEST (must include 0)'+('  OK' if lo<=0<=hi else '  !!FAIL!!')
            print(f"    {k:<4} vs IG: {m:+.4f} [{lo:+.4f},{hi:+.4f}]{tag}")
        # A vs OLD and A vs N19 (paired same-rep)
        for other in ['OLD','N19']:
            dd=[]
            for r in reps:
                da=parse_csv(os.path.join(outdir,f"{corp}.A.{r}.csv"))
                db=parse_csv(os.path.join(outdir,f"{corp}.{other}.{r}.csv"))
                if da and db: dd.append(da[0]/B - db[0]/B)
            m,lo,hi=boot_ci(dd)
            verdict = 'A FASTER' if hi<0 else ('A SLOWER' if lo>0 else 'TIE')
            print(f"    A vs {other}: Δcyc/B {m:+.4f} [{lo:+.4f},{hi:+.4f}]  => {verdict}")
    print("\n================ END NIGHT22 PERF ================")

if __name__=='__main__':
    main()
