#!/usr/bin/env python3
"""Generic best-of-N timer with sha-verify. Each label is a shell command.
Reads label=cmd pairs from argv after --file/--sha/--rounds. Interleaves rounds.

Usage: timeit.py --sha SHA --rounds N LABEL1 "CMD1" LABEL2 "CMD2" ...
CMD is run via /bin/bash -c; stdout hashed for verify, /dev/null for timing.
"""
import argparse, hashlib, subprocess, sys, time, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("pairs", nargs="+")
    args = ap.parse_args()
    assert len(args.pairs) % 2 == 0, "need label cmd pairs"
    configs = [(args.pairs[i], args.pairs[i+1]) for i in range(0, len(args.pairs), 2)]

    print("== sha verify ==")
    for label, cmd in configs:
        p = subprocess.run(["/bin/bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sh = hashlib.sha256(p.stdout).hexdigest()
        ok = "OK" if (sh == args.sha and p.returncode == 0) else f"MISMATCH rc={p.returncode} {sh[:12]}"
        print(f"  {label:22s} {ok}")
        if sh != args.sha or p.returncode != 0:
            sys.exit(1)

    samples = {l: [] for l, _ in configs}
    dn = open(os.devnull, "wb")
    for r in range(args.rounds):
        for label, cmd in configs:
            t0 = time.perf_counter()
            subprocess.run(["/bin/bash", "-c", cmd], stdout=dn, stderr=subprocess.DEVNULL)
            samples[label].append(time.perf_counter() - t0)
    print(f"\n== results (rounds={args.rounds}) ==")
    print(f"{'label':22s} {'min_ms':>9s} {'med_ms':>9s} {'spread%':>8s}")
    for label, _ in configs:
        s = sorted(samples[label])
        print(f"{label:22s} {s[0]*1000:9.1f} {s[len(s)//2]*1000:9.1f} {(s[-1]-s[0])/s[0]*100:8.1f}")

if __name__ == "__main__":
    main()
