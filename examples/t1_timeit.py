#!/usr/bin/env python3
# THROWAWAY T1 timing harness. Interleaved best-of-N, stdout -> /dev/null.
# Usage: t1_timeit.py <reps> <corpus.gz> <tool_label:argv...> [<tool>...]
#   tool spec: label==space-joined-argv, with %F replaced by the corpus path.
# Env passthrough: a tool argv may be prefixed with "ENV=VAR=val,VAR2=val2|"
import subprocess, sys, time, statistics, os

reps = int(sys.argv[1])
corpus = sys.argv[2]
specs = sys.argv[3:]

tools = []
for s in specs:
    label, rest = s.split("==", 1)
    env = dict(os.environ)
    if rest.startswith("ENV="):
        envpart, rest = rest[4:].split("|", 1)
        for kv in envpart.split(","):
            k, v = kv.split("=", 1)
            env[k] = v
    argv = [a.replace("%F", corpus) for a in rest.split(" ")]
    tools.append((label, argv, env))

results = {label: [] for label, _, _ in tools}

# warm-up (one untimed run each)
for label, argv, env in tools:
    with open("/dev/null", "wb") as dn:
        subprocess.run(argv, stdout=dn, stderr=subprocess.DEVNULL, env=env)

for r in range(reps):
    for label, argv, env in tools:
        with open("/dev/null", "wb") as dn:
            t0 = time.perf_counter()
            p = subprocess.run(argv, stdout=dn, stderr=subprocess.DEVNULL, env=env)
            dt = (time.perf_counter() - t0) * 1000.0
        if p.returncode != 0:
            print(f"  !! {label} rc={p.returncode}")
        results[label].append(dt)

print(f"\n== corpus={os.path.basename(corpus)} reps={reps} (process wall ms, /dev/null) ==")
print(f"{'tool':<28} {'min':>9} {'median':>9} {'max':>9}")
rows = []
for label, _, _ in tools:
    ts = results[label]
    rows.append((label, min(ts), statistics.median(ts), max(ts)))
for label, mn, md, mx in rows:
    print(f"{label:<28} {mn:>9.2f} {md:>9.2f} {mx:>9.2f}")
