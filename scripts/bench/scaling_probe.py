#!/usr/bin/env python3
"""Self-speedup + native-vs-rg scaling probe.

Pure-wall measurement: decode to /dev/null, N interleaved rounds, one sha-verify
per config (deterministic decode => per-config verify is sound; sha of a 200MB
output would otherwise dominate a sub-100ms decode wall).

Usage:
  scaling_probe.py --gz BIN --rg BIN --file F.gz --sha SHA \
      --tlist 1,4,8,16 --rounds 9 [--gz-extra ARG ...] [--chunk-kib N]
"""
import argparse, hashlib, subprocess, sys, time, os

def mask(t):
    return "0" if t <= 1 else f"0-{t-1}"

def run_capture_sha(cmd, env):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    return hashlib.sha256(p.stdout).hexdigest(), p.returncode

def run_timed(cmd, env):
    devnull = open(os.devnull, "wb")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=devnull, stderr=subprocess.DEVNULL, env=env)
    dt = time.perf_counter() - t0
    devnull.close()
    return dt, p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", required=True)
    ap.add_argument("--rg", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--tlist", default="1,4,8,16")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--chunk-kib", default=None)
    ap.add_argument("--tools", default="native,rg")
    ap.add_argument("--no-pin", action="store_true", help="do not taskset-pin (for messy topology e.g. Intel LXC)")
    args = ap.parse_args()

    tlist = [int(x) for x in args.tlist.split(",")]
    tools = args.tools.split(",")

    base_env = dict(os.environ)
    gz_env = dict(base_env)
    if args.chunk_kib:
        gz_env["GZIPPY_CHUNK_KIB"] = str(args.chunk_kib)

    configs = []  # (label, cmd, env)
    for t in tlist:
        ms = [] if args.no_pin else ["taskset", "-c", mask(t)]
        if "native" in tools:
            configs.append((f"native_T{t}",
                ms + [args.gz, "-d", "-c", "-p", str(t), args.file], gz_env))
        if "rg" in tools:
            configs.append((f"rg_T{t}",
                ms + [args.rg, "-d", "-c", "-P", str(t), args.file], base_env))

    # sha-verify each config once
    print("== sha verify (per config) ==")
    for label, cmd, env in configs:
        sh, rc = run_capture_sha(cmd, env)
        ok = "OK" if (sh == args.sha and rc == 0) else f"MISMATCH rc={rc} sha={sh[:12]}"
        print(f"  {label:14s} {ok}")
        if sh != args.sha or rc != 0:
            print(f"!! {label} FAILED verification; aborting", file=sys.stderr)
            sys.exit(1)

    # interleaved timed rounds
    samples = {label: [] for label, _, _ in configs}
    for r in range(args.rounds):
        for label, cmd, env in configs:
            dt, rc = run_timed(cmd, env)
            if rc != 0:
                print(f"!! {label} rc={rc} round {r}", file=sys.stderr)
                sys.exit(1)
            samples[label].append(dt)

    # report
    print(f"\n== results (chunk_kib={args.chunk_kib}, rounds={args.rounds}) ==")
    print(f"{'config':14s} {'min_ms':>9s} {'med_ms':>9s} {'spread%':>8s}")
    res = {}
    for label, _, _ in configs:
        s = sorted(samples[label])
        mn = s[0] * 1000
        med = s[len(s)//2] * 1000
        spread = (s[-1] - s[0]) / s[0] * 100
        res[label] = mn
        print(f"{label:14s} {mn:9.1f} {med:9.1f} {spread:8.1f}")

    # self-speedup S(tN)=wall(t1)/wall(tN) per tool
    print("\n== self-speedup S(tN) = wall_min(T1)/wall_min(tN) ==")
    for tool in tools:
        if f"{tool}_T1" not in res:
            continue
        base = res[f"{tool}_T1"]
        row = [f"{tool}:"]
        for t in tlist:
            k = f"{tool}_T{t}"
            if k in res:
                row.append(f"T{t}={base/res[k]:.2f}")
        print("  " + "  ".join(row))

    # native/rg wall ratio per T
    if "native" in tools and "rg" in tools:
        print("\n== native/rg wall ratio (>1 = native slower) ==")
        for t in tlist:
            nk, rk = f"native_T{t}", f"rg_T{t}"
            if nk in res and rk in res:
                print(f"  T{t}: {res[nk]/res[rk]:.3f}")

if __name__ == "__main__":
    main()
