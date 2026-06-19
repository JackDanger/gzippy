#!/usr/bin/env python3
"""TASK2 oracle attribution analyzer. Arms IG IG2 A ORC.
cyc/B, instr/B, IPC per arm + paired deltas (X-IG, ORC-A) with bootstrap 95% CI.
Attribution: fraction of A's +instr/B gap to igzip closed by removing the glue (A-ORC)."""
import sys, os, glob, random, statistics as st

def parse(path):
    cyc=ins=tc=None
    try:
        for line in open(path):
            p=line.strip().split(',')
            if len(p)<3: continue
            v,ev=p[0],p[2]
            try: x=float(v)
            except ValueError: continue
            if 'instructions' in ev: ins=x
            elif 'cycles' in ev: cyc=x
            elif ev=='task-clock': tc=x
    except FileNotFoundError: return None
    if cyc is None or ins is None: return None
    return cyc,ins,tc

def ci(d,n=5000):
    if not d: return (float('nan'),)*3
    m=st.median(d); L=len(d); meds=[]
    for _ in range(n):
        s=[d[random.randrange(L)] for _ in range(L)]; meds.append(st.median(s))
    meds.sort(); return m,meds[int(.025*n)],meds[int(.975*n)]

def main():
    outdir=sys.argv[1]; corpora=sys.argv[2:]; arms=['IG','IG2','A','ORC']
    bytes_map={}
    for line in open(os.path.join(outdir,'bytes.txt')):
        c,b=line.split(); bytes_map[c]=int(b)
    random.seed(1234)
    print("\n========= TASK2 STATELESS-REMOVAL-ORACLE ATTRIBUTION (cyc/B, instr/B; lower=faster) =========")
    for corp in corpora:
        B=bytes_map[corp]
        reps=sorted({int(os.path.basename(p).split('.')[-2]) for p in glob.glob(os.path.join(outdir,f"{corp}.*.*.csv"))})
        per={a:{'cycb':[],'insb':[],'ipc':[]} for a in arms}
        pIG={a:{'cycb':[],'insb':[]} for a in ['IG2','A','ORC']}  # X-IG
        pORCA={'cycb':[],'insb':[]}  # ORC-A
        for r in reps:
            v={}; ok=True
            for a in arms:
                d=parse(os.path.join(outdir,f"{corp}.{a}.{r}.csv"))
                if d is None: ok=False; break
                v[a]=d
            if not ok: continue
            for a in arms:
                cyc,ins,tc=v[a]; per[a]['cycb'].append(cyc/B); per[a]['insb'].append(ins/B)
                per[a]['ipc'].append(ins/cyc if cyc else float('nan'))
            igc=v['IG'][0]/B; igi=v['IG'][1]/B
            for a in ['IG2','A','ORC']:
                pIG[a]['cycb'].append(v[a][0]/B-igc); pIG[a]['insb'].append(v[a][1]/B-igi)
            pORCA['cycb'].append(v['ORC'][0]/B-v['A'][0]/B); pORCA['insb'].append(v['ORC'][1]/B-v['A'][1]/B)
        print(f"\n--- {corp} (bytes={B}, N={len(reps)}) ---")
        print(f"{'arm':<5}{'cyc/B':>9}{'instr/B':>10}{'IPC':>8}")
        for a in arms:
            mn=lambda k: st.mean(per[a][k]) if per[a][k] else float('nan')
            print(f"{a:<5}{mn('cycb'):>9.4f}{mn('insb'):>10.4f}{mn('ipc'):>8.4f}")
        print("  PAIRED median[95%CI]:")
        m,lo,hi=ci(pIG['IG2']['insb']); print(f"    instr/B IG2-IG: {m:+.4f}[{lo:+.4f},{hi:+.4f}]  SELF-TEST {'OK' if lo<=0<=hi else '!!FAIL!!'}")
        aI=ci(pIG['A']['insb']);  print(f"    instr/B A  -IG: {aI[0]:+.4f}[{aI[1]:+.4f},{aI[2]:+.4f}]  (the gap)")
        oI=ci(pIG['ORC']['insb']);print(f"    instr/B ORC-IG: {oI[0]:+.4f}[{oI[1]:+.4f},{oI[2]:+.4f}]  (residual after glue removed)")
        gI=ci(pORCA['insb']);     print(f"    instr/B ORC-A : {gI[0]:+.4f}[{gI[1]:+.4f},{gI[2]:+.4f}]  (the glue removed; NEG=fewer)")
        if aI[0]!=0:
            frac=(-gI[0])/aI[0]*100
            print(f"    ==> ATTRIBUTION: glue closes {frac:.1f}% of A's +{aI[0]:.4f} instr/B gap to igzip")
        aC=ci(pIG['A']['cycb']);  print(f"    cyc/B   A  -IG: {aC[0]:+.4f}[{aC[1]:+.4f},{aC[2]:+.4f}]")
        oC=ci(pIG['ORC']['cycb']);print(f"    cyc/B   ORC-IG: {oC[0]:+.4f}[{oC[1]:+.4f},{oC[2]:+.4f}]")
        gC=ci(pORCA['cycb']);     print(f"    cyc/B   ORC-A : {gC[0]:+.4f}[{gC[1]:+.4f},{gC[2]:+.4f}]  (glue cyc/B; NEG=oracle faster)")
    print("\n========= END TASK2 ATTRIBUTION =========")

if __name__=='__main__': main()
