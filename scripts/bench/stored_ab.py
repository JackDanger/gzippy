#!/usr/bin/env python3
"""Gated interleaved A/B for the StoredParallel copy path.

Arms: BASE (baseline gz), FIX (fused copy+crc gz), RG (rapidgzip native).
Self-test arms BASE2/FIX2 (binary-vs-itself ~1.0) license trusting ratios.
/dev/null sink all arms; even-P-core pin; sha-verified == zcat (Gate-0).
Reports min(floor)+median wall + inter-run spread; Delta<spread => TIE (Gate-1).
"""
import subprocess, sys, time, statistics, os, hashlib

BASE = "/dev/shm/sp-target/release/gzippy"
FIX  = "/dev/shm/sp-fixed/release/gzippy"
RG   = "/root/oracle_c/rapidgzip-native"
CORP = "/root"
N = int(os.environ.get("N", "15"))

def pin(t):
    return ",".join(str(2*i) for i in range(t))

def cmd(arm, corp, t):
    f = f"{CORP}/{corp}.gz"
    m = pin(t)
    if arm in ("BASE","BASE2"):
        return ["taskset","-c",m,"env","GZIPPY_FORCE_PARALLEL_SM=1",BASE,"-d","-c",f"-p{t}",f]
    if arm in ("FIX","FIX2"):
        return ["taskset","-c",m,"env","GZIPPY_FORCE_PARALLEL_SM=1",FIX,"-d","-c",f"-p{t}",f]
    if arm == "RG":
        return ["taskset","-c",m,RG,"-d","-c",f"-P{t}",f]
    raise ValueError(arm)

def sha_of(arm, corp, t):
    f = f"{CORP}/{corp}.gz"
    out = subprocess.run(cmd(arm,corp,t), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    return hashlib.sha256(out).hexdigest()[:16], len(out)

def zcat_sha(corp):
    f = f"{CORP}/{corp}.gz"
    p = subprocess.run(["zcat",f], stdout=subprocess.PIPE)
    return hashlib.sha256(p.stdout).hexdigest()[:16], len(p.stdout)

def run_wall(arm, corp, t):
    devnull = open("/dev/null","wb")
    t0 = time.perf_counter()
    subprocess.run(cmd(arm,corp,t), stdout=devnull, stderr=subprocess.DEVNULL)
    dt = time.perf_counter()-t0
    devnull.close()
    return dt*1000.0

def spread(xs):
    xs=sorted(xs); med=statistics.median(xs)
    return (max(xs)-min(xs))/med*100.0 if med else 0.0

def main():
    corpora = sys.argv[1:] or ["pure_stored","storedheavy","storedmix"]
    threads = [1,2,4,8]
    arms = ["BASE","FIX","RG","BASE2","FIX2"]
    print(f"# StoredParallel A/B  N={N}  arms={arms}")
    # Gate-0 correctness
    print("== GATE-0 sha == zcat ==")
    for corp in corpora:
        ref,nbytes = zcat_sha(corp)
        line=[f"{corp} ref={ref} bytes={nbytes}"]
        ok=True
        for arm in ("BASE","FIX","RG"):
            s,n = sha_of(arm,corp,4)
            good = (s==ref)
            ok = ok and good
            line.append(f"{arm}={'OK' if good else 'BAD('+s+')'}")
        print("  "+" ".join(line))
        if not ok:
            print("GATE0_FAIL"); sys.exit(2)
    print("GATE0 PASS")
    print("== MEASURE (interleaved) ==")
    data = {(a,c,t):[] for a in arms for c in corpora for t in threads}
    for r in range(N):
        for corp in corpora:
            for t in threads:
                for arm in arms:
                    data[(arm,corp,t)].append(run_wall(arm,corp,t))
    print(f"{'corpus':12} {'T':>2} | {'BASE ms':>8} {'FIX ms':>8} {'RG ms':>8} | "
          f"{'FIX/BASE':>8} {'FIX/RG':>7} {'BASE/RG':>7} | {'AAb%':>5} {'AAf%':>5} {'spr%':>5} verdict")
    for corp in corpora:
        for t in threads:
            b=min(data[("BASE",corp,t)]); f=min(data[("FIX",corp,t)]); g=min(data[("RG",corp,t)])
            bmed=statistics.median(data[("BASE",corp,t)])
            fmed=statistics.median(data[("FIX",corp,t)])
            gmed=statistics.median(data[("RG",corp,t)])
            aab=abs(min(data[("BASE2",corp,t)])-b)/b*100.0
            aaf=abs(min(data[("FIX2",corp,t)])-f)/f*100.0
            spr=max(spread(data[("BASE",corp,t)]),spread(data[("FIX",corp,t)]),spread(data[("RG",corp,t)]))
            fb=f/b; fg=f/g; bg=b/g
            # verdict on FIX vs BASE using median ratio vs spread
            dfb=abs(fmed-bmed)/bmed*100.0
            if dfb<=spr: v="TIE(fix~base)"
            elif fmed<bmed: v=f"FIX FASTER {(1-fmed/bmed)*100:.1f}%"
            else: v=f"FIX SLOWER {(fmed/bmed-1)*100:.1f}%"
            # vs rg
            dfg=abs(fmed-gmed)/gmed*100.0
            vrg = "TIE-rg" if dfg<=spr else ("fix<rg" if fmed<gmed else "fix>rg")
            unt = " UNTRUSTED" if (aab>5 or aaf>5) else ""
            print(f"{corp:12} {t:>2} | {b:8.1f} {f:8.1f} {g:8.1f} | "
                  f"{fb:8.3f} {fg:7.3f} {bg:7.3f} | {aab:5.1f} {aaf:5.1f} {spr:5.1f} {v} | {vrg}{unt}")

if __name__=="__main__":
    main()
