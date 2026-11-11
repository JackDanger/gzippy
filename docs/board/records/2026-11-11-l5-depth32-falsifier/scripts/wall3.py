#!/usr/bin/env python3
"""Wall block 3: l16 vs d32g8 (same tune binary, env-only) + d32 vs d32g8."""
import os, statistics, subprocess, time

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-l5"
TUNE = f"{WT}/target-tune/release/gzippy"
PLAYERS = {
    "l16":   (TUNE, {"GZIPPY_LDX_CFG": "5:16:30"}),
    "d32":   (TUNE, {"GZIPPY_LDX_CFG": "5:32:30"}),
    "d32g8": (TUNE, {"GZIPPY_LDX_CFG": "5:32:30:8"}),
}
def run(who, path):
    binary, env = PLAYERS[who]
    e = {**os.environ, **env}
    t0 = time.perf_counter()
    with open(path, "rb") as fin:
        subprocess.run([binary, "-n", "-5", "-p1", "-c"], stdin=fin, stdout=subprocess.DEVNULL, check=True, env=e)
    return time.perf_counter() - t0
PAIRS = [("logs.txt-card26m", "l16", "d32g8", 9),
         ("webster", "l16", "d32g8", 9),
         ("logs.txt-card26m", "d32", "d32g8", 9),
         ("webster", "d32", "d32g8", 9)]
with open(f"{WT}/scratch/wall.log", "a") as log:
    log.write("--- wall block 3: good8 arm same-binary pairs ---\n")
    for member, a, b, reps in PAIRS:
        path = os.path.join(WT, "scratch", "members", member)
        run(a, path); run(b, path)
        ta, tb = [], []
        for i in range(reps):
            if i % 2 == 0:
                ta.append(run(a, path)); tb.append(run(b, path))
            else:
                tb.append(run(b, path)); ta.append(run(a, path))
        ma = statistics.median(ta) * 1000
        mb = statistics.median(tb) * 1000
        raw = f"[raw3] {member} A={a}={[round(x, 4) for x in ta]} B={b}={[round(x, 4) for x in tb]}"
        log.write(raw + "\n"); print(raw, flush=True)
        line = (f"{member:18s} {a} median={ma:8.1f} ms | {b} median={mb:8.1f} ms | B/A={mb/ma:.4f} ({(mb/ma-1)*100:+.2f}%)"
                " [same-binary env-only pair; HYPOTHESIS-TIER]")
        log.write(line + "\n"); print(line, flush=True)
