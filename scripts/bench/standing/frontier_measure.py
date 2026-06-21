#!/usr/bin/env python3
# frontier_measure.py — FRONTIER-CONFIRM rig.
#
# "What is the per-arch DECODE frontier (fastest single-stream T1 competitor) gzippy
# must beat, and does the field tier libdeflate≈igzip > zlib-ng > pigz/zlib HOLD on
# OUR boxes/corpora?"  -> one self-validating measurement.
#
# Runs ON a box (Intel guest under freeze, or quiet macOS). Measures gz-native vs ALL
# present decode comparators {rapidgzip, igzip, libdeflate, zlib-ng minigzip, pigz} on
# the corpus spread, interleaved best-of-N, /dev/null sink for EVERY arm, wall via
# time.perf_counter. Identifies the FRONTIER per (corpus,T) and gzippy's gap to it.
#
# Gate-0 (BLOCKING; a cell that fails emits no number):
#   (a) every arm sha16(stdout) == zcat ref (byte-exact, non-inert);
#   (b) same /dev/null sink for ALL arms (DEVNULL on every timed run);
#   (c) A/A: each arm's own run-to-run spread (CV) is the binary-vs-itself self-test —
#       if the fastest arm's CV > NOISE_CV the cell is flagged NOISY (ratios untrusted);
#   (d) gz routed via GZIPPY_FORCE_PARALLEL_SM=1 (production parallel engine).
# Gate-1: interleaved best-of-N>=13; Δ vs spread -> TIE if |a-b| < spread(a)+spread(b).
#
# Single-stream-only tools (igzip / libdeflate / zlib-ng) are measured at T1 only and
# pinned to ONE core; rapidgzip/pigz/gz are measured at every requested T.
import argparse, subprocess, time, os, sys, statistics, hashlib, json, shutil

NOISE_CV = 0.08  # fastest-arm coefficient-of-variation gate

def sha16(b): return hashlib.sha256(b).hexdigest()[:16]

def which(p): return p if (os.path.isabs(p) and os.access(p, os.X_OK)) else shutil.which(p)

def run_wall(argv, env=None, stdin_path=None):
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    try:
        t0 = time.perf_counter()
        r = subprocess.run(argv, stdin=fin, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, env=env)
        t1 = time.perf_counter()
    finally:
        if stdin_path: fin.close()
    return (t1 - t0), r.returncode

def run_sha(argv, env=None, stdin_path=None):
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    try:
        r = subprocess.run(argv, stdin=fin, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, env=env)
    finally:
        if stdin_path: fin.close()
    return sha16(r.stdout), r.returncode, len(r.stdout)

def pin_mask(pinbase, T, step=2):
    return ",".join(str(pinbase + i*step) for i in range(T))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["intel", "mac"])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--threads", default="1 2 4")
    ap.add_argument("-N", type=int, default=13)
    ap.add_argument("--gz", required=True)
    ap.add_argument("--rg", default="")
    ap.add_argument("--igzip", default="")
    ap.add_argument("--libdeflate", default="")
    ap.add_argument("--pigz", default="")
    ap.add_argument("--zng", default="")  # zlib-ng minigzip
    ap.add_argument("--pin", action="store_true")  # taskset (intel)
    ap.add_argument("--pinbase", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpora = args.corpora.split()
    threads = [int(t) for t in args.threads.split()]
    pinbase = args.pinbase

    # ---- resolve arms present ----
    gz = which(args.gz)
    if not gz: print("FRONTIER_FAIL gz binary missing", flush=True); sys.exit(2)
    arms = []  # (label, parallel, builder)  builder(path,T,mask)-> (argv, env, stdin_path)
    def taskset(mask, argv):
        return (["taskset", "-c", mask] + argv) if args.pin else argv

    gzenv = dict(os.environ); gzenv["GZIPPY_FORCE_PARALLEL_SM"] = "1"
    arms.append(("gz", True, lambda p,T,m: (taskset(m, [gz,"-d","-c",f"-p{T}",p]), gzenv, None)))
    arms.append(("gz2", True, lambda p,T,m: (taskset(m, [gz,"-d","-c",f"-p{T}",p]), gzenv, None)))  # A/A
    rg = which(args.rg) if args.rg else None
    if rg: arms.append(("rapidgzip", True, lambda p,T,m: (taskset(m, [rg,"-d","-c",f"-P{T}",p]), None, None)))
    ig = which(args.igzip) if args.igzip else None
    if ig: arms.append(("igzip", False, lambda p,T,m: (taskset(str(pinbase), [ig,"-d","-c",p]), None, None)))
    ld = which(args.libdeflate) if args.libdeflate else None
    if ld: arms.append(("libdeflate", False, lambda p,T,m: (taskset(str(pinbase), [ld,"-d","-c",p]), None, None)))
    pg = which(args.pigz) if args.pigz else None
    if pg: arms.append(("pigz", True, lambda p,T,m: (taskset(m, [pg,"-d","-c",f"-p{T}",p]), None, None)))
    zng = which(args.zng) if args.zng else None
    if zng: arms.append(("zlib-ng", False, lambda p,T,m: (taskset(str(pinbase), [zng,"-d"]), None, p)))

    present = [a[0] for a in arms]
    print(f"== FRONTIER rig  arch={args.arch}  arms={present}  N={args.N}  threads={threads} ==", flush=True)
    print(f"   loadavg_start={open('/proc/loadavg').read().split()[0] if os.path.exists('/proc/loadavg') else '?'}", flush=True)

    # ---- Gate-0: byte-exact sha per arm vs zcat ref + bytes ----
    print("--- GATE-0 byte-exact (sha16 == zcat) ---", flush=True)
    corp_path, corp_bytes = {}, {}
    for c in corpora:
        f = os.path.join(args.corpus_dir, f"{c}.gz")
        if not os.path.isfile(f): print(f"  {c}: MISSING {f} (skip)", flush=True); continue
        ref = subprocess.run(["zcat", f], stdout=subprocess.PIPE).stdout if shutil.which("zcat") \
              else subprocess.run(["gzip","-dc", f], stdout=subprocess.PIPE).stdout
        refh, nb = sha16(ref), len(ref); del ref
        ok = True
        for label, par, build in arms:
            if label == "gz2": continue
            argv, env, sp = build(f, 1, str(pinbase))
            h, rc, _ = run_sha(argv, env, sp)
            good = (h == refh)
            print(f"  {c:10s} {label:11s} {'OK' if good else 'BAD '+h}", flush=True)
            if not good: ok = False
        if not ok: print(f"FRONTIER_FAIL {c} a comparator is not byte-exact", flush=True); sys.exit(2)
        corp_path[c] = f; corp_bytes[c] = nb
    if not corp_path: print("FRONTIER_FAIL no corpora present", flush=True); sys.exit(2)
    print(f"   ref bytes: " + ", ".join(f"{c}={corp_bytes[c]}" for c in corp_path), flush=True)

    # ---- Gate-1: interleaved best-of-N walls ----
    samples = {}  # (corp,T,label) -> [walls]
    for c in corp_path:
        f = corp_path[c]
        for T in threads:
            mask = pin_mask(pinbase, T)
            cell_arms = [a for a in arms if a[1] or T == 1]  # single-stream only at T1
            print(f"--- MEASURE {c} T{T} mask={mask if args.pin else '(no-pin)'} arms={[a[0] for a in cell_arms]} ---", flush=True)
            for label,_,_ in cell_arms: samples[(c,T,label)] = []
            for rep in range(args.N):
                for label, par, build in cell_arms:
                    argv, env, sp = build(f, T, mask)
                    w, rc = run_wall(argv, env, sp)
                    samples[(c,T,label)].append(w)

    loadend = open('/proc/loadavg').read().split()[0] if os.path.exists('/proc/loadavg') else '?'
    out = {"arch": args.arch, "N": args.N, "threads": threads, "corpora": list(corp_path),
           "bytes": corp_bytes, "present": present, "loadend": loadend, "samples": {}}
    for k, v in samples.items():
        out["samples"]["|".join(map(str,k))] = v
    with open(args.out, "w") as fh: json.dump(out, fh)
    print(f"== wrote {args.out} ==", flush=True)
    report(out)

def stat(walls):
    mn = min(walls); md = statistics.median(walls)
    cv = (statistics.pstdev(walls)/statistics.mean(walls)) if len(walls) > 1 else 0.0
    spread = (max(walls)-mn)  # absolute spread
    return mn, md, cv, spread

def report(out):
    print("\n############ FRONTIER REPORT (arch=%s, N=%d) ############" % (out["arch"], out["N"]), flush=True)
    S = out["samples"]
    def get(c,T,l):
        k = "|".join([c,str(T),l]); return S.get(k)
    comparators_order = ["rapidgzip","igzip","libdeflate","zlib-ng","pigz"]
    for c in out["corpora"]:
        nb = out["bytes"][c]; mb = nb/1e6
        for T in out["threads"]:
            gz = get(c,T,"gz")
            if not gz: continue
            gz2 = get(c,T,"gz2")
            gmn,gmd,gcv,gsp = stat(gz)
            aa = (min(gz2)/gmn) if gz2 else float('nan')
            print(f"\n== {c}  T{T}  ({mb:.1f} MB)   gz min={gmn*1000:.1f}ms cv={gcv*100:.1f}%  A/A(gz2/gz)={aa:.3f} ==", flush=True)
            print(f"   {'arm':12s} {'min_ms':>9s} {'MB/s':>8s} {'cv%':>6s} {'gz/arm':>8s} {'verdict':>10s}", flush=True)
            gz_mbps = mb/gmn
            print(f"   {'gzippy':12s} {gmn*1000:9.1f} {gz_mbps:8.1f} {gcv*100:6.1f} {'1.000':>8s} {'(subject)':>10s}", flush=True)
            cell = []
            for label in comparators_order:
                v = get(c,T,label)
                if not v: continue
                mn,md,cv,sp = stat(v)
                ratio = gmn/mn  # gz/arm on min wall; >1 => gz SLOWER (loses)
                mbps = mb/mn
                # TIE if |gz-arm| < gz_spread + arm_spread
                tie = abs(gmn-mn) < (gsp + sp)
                verdict = "TIE" if tie else ("gz WINS" if ratio < 1 else "gz LOSES")
                noisy = " NOISY" if cv > NOISE_CV else ""
                print(f"   {label:12s} {mn*1000:9.1f} {mbps:8.1f} {cv*100:6.1f} {ratio:8.3f} {verdict:>10s}{noisy}", flush=True)
                cell.append((label, mn, cv))
            # frontier = fastest comparator (min wall) in this cell
            if cell:
                fl, fmn, fcv = min(cell, key=lambda x: x[1])
                fr = gmn/fmn
                print(f"   --> FRONTIER={fl} ({fmn*1000:.1f}ms)  gz gap to frontier: gz/frontier={fr:.3f} "
                      f"({'gz beats frontier' if fr<1 else f'gz is {(fr-1)*100:.1f}% slower'})", flush=True)

if __name__ == "__main__":
    main()
