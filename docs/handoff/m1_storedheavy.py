#!/usr/bin/env python3
# Clean interleaved gz-vs-rg measurement on M1 (ns timer, best-of-N, paired).
import subprocess, time, sys, statistics

GZ = "/Users/jackdanger/www/gzippy-reimplement-isal/scratchpad/wt-m1score/target/release/gzippy"
RG = "/opt/homebrew/bin/rapidgzip"
CORPUS = "/tmp/storedheavy.gz"
N = 15

def run_ms(cmd, env=None):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, r.returncode

import os
base_env = dict(os.environ); base_env["GZIPPY_FORCE_PARALLEL_SM"] = "1"

for T in (1, 4, 8):
    gz_cmd = [GZ, "-d", "-c", "-p", str(T), CORPUS]
    rg_cmd = [RG, "-d", "-c", "-f", "-P", str(T), CORPUS]
    gz_walls, rg_walls = [], []
    # interleaved, warmup dropped
    for i in range(N + 1):
        g, grc = run_ms(gz_cmd, base_env)
        r, rrc = run_ms(rg_cmd)
        if grc != 0 or rrc != 0:
            print(f"T{T}: NONZERO EXIT gz={grc} rg={rrc}"); break
        if i == 0:
            continue
        gz_walls.append(g); rg_walls.append(r)
    if not gz_walls:
        continue
    gz_best, rg_best = min(gz_walls), min(rg_walls)
    gz_med, rg_med = statistics.median(gz_walls), statistics.median(rg_walls)
    ratio = rg_best / gz_best
    verdict = "PASS(gz>=rg)" if ratio >= 0.99 else "FAIL(gz slower)"
    print(f"T{T}: gz best={gz_best:.1f}ms med={gz_med:.1f}  rg best={rg_best:.1f}ms med={rg_med:.1f}  ratio(rg/gz)={ratio:.3f}  {verdict}")
