#!/usr/bin/env python3
"""Load-immune interleaved paired-ratio wall measurement (AMD/Zen2, llama-load).

Each rep runs every arm BACK-TO-BACK for a given (corpus, T) so llama drift hits
all arms equally; the verdict-bearing quantity is the median over reps of the
per-rep PAIRED ratio (gz_on / comparator). Absolutes are HYPOTHESIS-grade (VOID
under load); only the ratios are reported as gated. /dev/null sink both arms,
sha-verified each gz arm vs `gzip -dc`. A/A arm (gz_on twice) gives the noise floor.

Arms:
  gz_on  = selector default (arch-dispatched constants)            gzippy -d -c -p T
  gz_off = selector disabled (parallel-always = old regressing)    MARGIN=0
  rg     = rapidgzip -P T
  igzip  = igzip -d -c  (single-thread, T-invariant)
"""
import argparse
import os
import statistics
import subprocess
import time
import hashlib
import sys


def run_timed(cmd, env=None, capture=False):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    t0 = time.perf_counter()
    if capture:
        p = subprocess.run(cmd, env=full_env, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t0
        return dt, p.stdout
    else:
        with open(os.devnull, "wb") as dn:
            subprocess.run(cmd, env=full_env, stdout=dn, stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t0
        return dt, None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--rg", required=True)
    ap.add_argument("--igzip", default="igzip")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tlist", required=True)
    ap.add_argument("--cores", default="0-31")
    ap.add_argument("--n", type=int, default=11)
    args = ap.parse_args()

    tlist = [int(x) for x in args.tlist.split()]
    taskset = ["taskset", "-c", args.cores]

    # Gate-0: sha-verify gz_on vs gzip -dc once.
    _, gz_out = run_timed(taskset + [args.bin, "-d", "-c", "-p", "8", args.corpus],
                          capture=True)
    ref = subprocess.run(["gzip", "-dc", args.corpus], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout
    if sha(gz_out) != sha(ref):
        print(f"SHA MISMATCH on {args.corpus} — ABORT", file=sys.stderr)
        sys.exit(1)
    print(f"# {os.path.basename(args.corpus)} sha OK (gz==gzip)")

    print(f"# corpus={os.path.basename(args.corpus)} n={args.n} cores={args.cores}")
    print(f"{'T':>3} {'gz_on':>8} {'gz_off':>8} {'rg':>8} {'igzip':>8} "
          f"{'on/off':>7} {'on/rg':>7} {'on/igz':>7} {'aa':>6}")

    for T in tlist:
        on, off, rg, ig, aa = [], [], [], [], []
        for _ in range(args.n):
            # interleaved: every arm once, back to back, this rep
            d_on, _ = run_timed(taskset + [args.bin, "-d", "-c", "-p", str(T), args.corpus])
            d_aa, _ = run_timed(taskset + [args.bin, "-d", "-c", "-p", str(T), args.corpus])
            d_off, _ = run_timed(taskset + [args.bin, "-d", "-c", "-p", str(T), args.corpus],
                                 env={"GZIPPY_PARALLEL_CROSSOVER_MARGIN": "0"})
            d_rg, _ = run_timed(taskset + [args.rg, "-d", "-c", "-P", str(T), args.corpus])
            d_ig, _ = run_timed(taskset + [args.igzip, "-d", "-c", args.corpus])
            on.append(d_on); aa.append(d_aa); off.append(d_off); rg.append(d_rg); ig.append(d_ig)

        # per-rep paired ratios, then median (load-immune)
        r_off = statistics.median([on[i] / off[i] for i in range(args.n)])
        r_rg = statistics.median([on[i] / rg[i] for i in range(args.n)])
        r_ig = statistics.median([on[i] / ig[i] for i in range(args.n)])
        r_aa = statistics.median([on[i] / aa[i] for i in range(args.n)])
        m = lambda x: statistics.median(x) * 1000.0
        print(f"{T:>3} {m(on):>8.1f} {m(off):>8.1f} {m(rg):>8.1f} {m(ig):>8.1f} "
              f"{r_off:>7.3f} {r_rg:>7.3f} {r_ig:>7.3f} {r_aa:>6.3f}")


if __name__ == "__main__":
    main()
