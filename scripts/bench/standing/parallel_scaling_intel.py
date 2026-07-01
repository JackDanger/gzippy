#!/usr/bin/env python3
# Intel quiet-window scaling curve: does T2<T1 reproduce uncontended?
import subprocess, time, statistics, os, hashlib, sys
BIN = "/root/gzippy/target/release/gzippy"
DN = open(os.devnull, "wb")
CORP = {"nasa": "/root/nasa.gz", "monorepo": "/root/monorepo.gz", "silesia": "/root/silesia.gz"}
N = 11

def nbytes(p):
    return len(subprocess.run(["gzip","-dc",p],stdout=subprocess.PIPE).stdout)
def refsha(p):
    return hashlib.sha256(subprocess.run(["gzip","-dc",p],stdout=subprocess.PIPE).stdout).hexdigest()
def gzsha(p,t):
    return hashlib.sha256(subprocess.run([BIN,"-dc","-p%d"%t,p],stdout=subprocess.PIPE,stderr=DN).stdout).hexdigest()
def wall(t,p):
    a=[BIN,"-dc","-p%d"%t,p]; t0=time.perf_counter()
    r=subprocess.run(a,stdout=DN,stderr=DN)
    if r.returncode: raise RuntimeError("rc")
    return time.perf_counter()-t0

# gate-0
dbg=subprocess.run([BIN,"-dc","-p1",CORP["nasa"]],stdout=DN,stderr=subprocess.PIPE,env={**os.environ,"GZIPPY_DEBUG":"1"}).stderr.decode()
assert "path=ParallelSM" in dbg, dbg
print("PASS path=ParallelSM; flavor:", [l for l in dbg.splitlines() if "flavor" in l])
print("load:", os.getloadavg())
for c,p in CORP.items():
    assert refsha(p)==gzsha(p,4), "sha mismatch "+c
    print("PASS sha %s"%c)
print()
hdr="%-10s %3s %9s %8s %6s %9s %8s"%("corpus","T","wall_ms","med_ms","spr%","MB/s","speedup")
for c,p in CORP.items():
    nb=nbytes(p); base=None
    print("### %s (%d MB) ###"%(c,nb//(1<<20))); print(hdr)
    for t in (1,2,4,8):
        for _ in range(2): wall(t,p)  # warmup
        s=sorted(wall(t,p) for _ in range(N))
        mn=s[0]; med=statistics.median(s)
        p10=s[max(0,int(0.1*(len(s)-1)))]; p90=s[int(0.9*(len(s)-1))]
        spr=(p90-p10)/med*100
        if t==1: base=mn
        print("%-10s %3d %9.1f %8.1f %5.1f %9.1f %7.2fx"%(c,t,mn*1e3,med*1e3,spr,nb/mn/1e6,base/mn))
    print()
print("binary:", subprocess.run(["sha256sum",BIN],stdout=subprocess.PIPE).stdout.decode().split()[0])
print("load:", os.getloadavg())
