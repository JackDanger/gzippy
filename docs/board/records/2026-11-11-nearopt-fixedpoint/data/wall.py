#!/usr/bin/env python3
# wall.py — interleaved paired wall medians BASE vs SKIP (both arms to /dev/null),
# alternating arm order per rep, medians + raw samples printed. Follows ab.sh
# phase-5's shape. Args configure cells; defaults below.
import json, os, subprocess, sys, time

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-fixpoint"
FIX = "/Users/jackdanger/www/gzippy/.review-tmp/fixpt"
A = f"{WT}/.rel-base2/release/gzippy"   # skip OFF (final source)
B = f"{WT}/.rel-skip2/release/gzippy"   # skip ON (final source)

# (input, level, threads, reps)
CELLS = [
    (f"{FIX}/in16m.bin", 9, 4, 9),
    (f"{FIX}/in16m.bin", 10, 4, 9),
    (f"{FIX}/in16m.bin", 11, 4, 9),
    (f"{FIX}/in16m.bin", 12, 4, 9),
    (f"{FIX}/in16m.bin", 10, 2, 7),
    (f"{FIX}/in16m.bin", 11, 2, 7),
    (f"{FIX}/in16m.bin", 12, 2, 7),
    (f"{FIX}/in16m.bin", 9, 2, 7),
    (f"{FIX}/silesia-agg.bin", 9, 4, 9),
    (f"{FIX}/silesia-agg.bin", 11, 4, 9),
    (f"{FIX}/silesia-agg.bin", 12, 4, 9),
    (f"{FIX}/logs16m.bin", 12, 4, 9),
    (f"{FIX}/logs16m.bin", 11, 4, 7),
    (f"{FIX}/logs16m.bin", 12, 2, 7),
    (f"{FIX}/logs64m.bin", 12, 4, 7),
    (os.path.expanduser("~/www/gzippy-bench/corpus/weights.safetensors"), 11, 4, 7),
    (os.path.expanduser("~/www/gzippy-bench/corpus/weights.safetensors"), 12, 4, 7),
    (os.path.expanduser("~/www/gzippy-bench/corpus/weights.safetensors"), 12, 2, 7),
    (os.path.expanduser("~/www/gzippy-bench/corpus/monorepo.tar"), 12, 4, 7),
]

def run(binpath, lvl, th, inp):
    t0 = time.perf_counter()
    subprocess.run([binpath, "-n", f"-{lvl}", f"-p{th}", "-c", inp],
                   stdout=subprocess.DEVNULL, check=True)
    return time.perf_counter() - t0

receipt_path = f"{FIX}/wall.log"
rp = open(receipt_path, "a")

def log(s):
    rp.write(s + "\n"); rp.flush()

sha = lambda p: subprocess.run(["shasum", "-a", "256", p], capture_output=True, text=True).stdout.split()[0]
stamp = (f"--- fixpt wall: A={sha(A)} B={sha(B)} levels={ {c[1] for c in CELLS} } "
         f"threads={ {c[2] for c in CELLS} } cpu={os.cpu_count()} load={os.getloadavg()[0]:.1f} ---")
print(stamp, flush=True); log(stamp)

rows = []
for inp, lvl, th, reps in CELLS:
    name = os.path.basename(inp)
    run(A, lvl, th, inp); run(B, lvl, th, inp)  # warm both
    ta, tb = [], []
    for i in range(reps):
        if i % 2 == 0:
            ta.append(run(A, lvl, th, inp)); tb.append(run(B, lvl, th, inp))
        else:
            tb.append(run(B, lvl, th, inp)); ta.append(run(A, lvl, th, inp))
    a = sorted(ta)[len(ta) // 2]
    b = sorted(tb)[len(tb) // 2]
    raw = f"  [raw] {name} L{lvl} p{th} A={[round(x,4) for x in ta]} B={[round(x,4) for x in tb]}"
    row = (f"  {name} L{lvl} -p{th}: base {a*1000:9.1f} ms | skip {b*1000:9.1f} ms | "
           f"B/A = {b/a:.4f}")
    print(raw, flush=True); print(row, flush=True)
    log(raw); log(row)
    rows.append((name, lvl, th, a*1000, b*1000, b/a, os.getloadavg()[0]))
print("--- summary ---", flush=True); log("--- summary ---")
for name, lvl, th, a, b, r, ld in rows:
    line = f"  {name} L{lvl} p{th} base={a:.1f}ms skip={b:.1f}ms B/A={r:.4f} load={ld:.1f}"
    print(line); log(line)
