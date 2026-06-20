#!/usr/bin/env python3
# NIGHT35 isolated kernel_ab A/B reproduction. Reports per-arm median cyc/B,
# run-to-run SPREAD (stdev + min..max + IQR), paired a-b CI. Answers: is the
# NIGHT31 +0.89 real or swamped by ±0.4 box noise?
import sys, csv, statistics, random
path = sys.argv[1] if len(sys.argv)>1 else "/dev/shm/kab35.csv"
rows = list(csv.DictReader(open(path)))
arms = {}
for r in rows:
    arms.setdefault(r["arm"], []).append((int(r["run"]), float(r["cyc_per_byte"]), float(r["instr_per_byte"]), float(r["ipc"])))
def ci(v,n=10000):
    if not v: return (float('nan'),)*3
    random.seed(3); L=len(v); m=[sum(v[random.randrange(L)] for _ in range(L))/L for _ in range(n)]; m.sort()
    return statistics.median(v), m[int(.025*n)], m[int(.975*n)]
print(f"file={path}")
for a in sorted(arms):
    cyc=[x[1] for x in arms[a]]; ins=[x[2] for x in arms[a]]; ipc=[x[3] for x in arms[a]]
    sd=statistics.pstdev(cyc) if len(cyc)>1 else 0
    cyc_s=sorted(cyc); iqr=cyc_s[int(.75*len(cyc_s))]-cyc_s[int(.25*len(cyc_s))]
    print(f"arm={a} N={len(cyc)} cyc/B med={statistics.median(cyc):.4f} mean={statistics.mean(cyc):.4f} "
          f"SD={sd:.4f} min={min(cyc):.4f} max={max(cyc):.4f} range={max(cyc)-min(cyc):.4f} IQR={iqr:.4f} "
          f"instr/B={statistics.median(ins):.4f} IPC={statistics.median(ipc):.3f}")
# paired a-b by run
da={r:c for r,c,_,_ in arms.get("a",[])}; db={r:c for r,c,_,_ in arms.get("b",[])}
pair=[da[r]-db[r] for r in da if r in db]
if pair:
    m,lo,hi=ci(pair)
    print(f"\nPAIRED a-b (gz run_contig - igzip _04) cyc/B = {m:+.4f} [{lo:+.4f},{hi:+.4f}]  N={len(pair)}")
    print(f"  run-to-run SPREAD of the a-b delta: SD={statistics.pstdev(pair):.4f} min={min(pair):+.4f} max={max(pair):+.4f} range={max(pair)-min(pair):.4f}")
# instr/B paired
ia={r:i for r,_,i,_ in arms.get("a",[])}; ib={r:i for r,_,i,_ in arms.get("b",[])}
pi=[ia[r]-ib[r] for r in ia if r in ib]
if pi:
    m,lo,hi=ci(pi); print(f"PAIRED a-b instr/B = {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
