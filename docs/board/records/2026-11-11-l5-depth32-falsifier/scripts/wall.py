#!/usr/bin/env python3
"""Interleaved wall A/B for the L5 depth-ladder falsifier (charter: interleave,
medians, alternate arm order per rep; laptop wall = HYPOTHESIS-TIER).

Pairs:
  calib  = ship-vanilla vs tune@(16,30)   — same bytes-by-proof, bounds the
            instrumented-binary wall premium that could pollute cross-binary
            pairings (charter: a tuned build 1.17x slower; NEVER quote one
            against a rival — here both arms are OUR builds and the delta is
            itself measured).
  depth  = ship-vanilla vs tune@GZIPPY_LDX_CFG=5:32:30
  good8  = ship-vanilla vs tune@GZIPPY_LDX_CFG=5:32:30:8
  sameb  = tune@(16,30) vs tune@5:32:30 (single-binary, env-only arm)
Raw reps retained; 9 reps interleaved, alternating order.
"""
import os, statistics, subprocess, sys, time

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-l5"
SHIP = f"{WT}/target-ship/release/gzippy"
TUNE = f"{WT}/target-tune/release/gzippy"
PLAYERS = {
    "ship":  (SHIP, {}),
    "l16":   (TUNE, {"GZIPPY_LDX_CFG": "5:16:30"}),
    "d32":   (TUNE, {"GZIPPY_LDX_CFG": "5:32:30"}),
    "d32g8": (TUNE, {"GZIPPY_LDX_CFG": "5:32:30:8"}),
}
MEMBERS = f"{WT}/scratch/members"
PAIRS = [
    ("logs.txt-card26m", [("ship", "l16")], 9),
    ("logs.txt-card26m", [("ship", "d32")], 9),
    ("logs.txt-card26m", [("ship", "d32g8")], 9),
    ("logs.txt",         [("ship", "d32")], 9),
    ("software.archive", [("ship", "d32")], 9),
    ("webster",          [("ship", "d32")], 9),
    ("webster",          [("ship", "d32g8")], 7),
    ("mozilla",          [("ship", "d32")], 7),
    ("dickens",          [("ship", "d32")], 7),
    ("xml",              [("ship", "d32")], 7),
]
LOG = f"{WT}/scratch/wall.log"

def run(who, path):
    binary, env = PLAYERS[who]
    e = {**os.environ, **env}
    t0 = time.perf_counter()
    with open(path, "rb") as fin:
        subprocess.run([binary, "-n", "-5", "-p1", "-c"], stdin=fin, stdout=subprocess.DEVNULL, check=True, env=e)
    return time.perf_counter() - t0

rev = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(LOG, "a") as log:
    log.write(f"--- wall block rev={rev} ---\n")
    print(f"--- interleaved wall (medians, alternating) rev={rev[:12]} ---", flush=True)
    for member, pair, reps in PAIRS:
        a, b = pair[0]
        path = os.path.join(MEMBERS, member)
        run(a, path); run(b, path)  # warm both
        ta, tb = [], []
        for i in range(reps):
            if i % 2 == 0:
                ta.append(run(a, path)); tb.append(run(b, path))
            else:
                tb.append(run(b, path)); ta.append(run(a, path))
        ma = statistics.median(ta) * 1000
        mb = statistics.median(tb) * 1000
        raw = f"[raw] {member} A={a}={[round(x, 4) for x in ta]} B={b}={[round(x, 4) for x in tb]}"
        log.write(raw + "\n")
        print(raw, flush=True)
        line = (f"{member:18s} {a} median={ma:8.1f} ms | {b} median={mb:8.1f} ms "
                f"| B/A={mb/ma:.4f} ({(mb/ma-1)*100:+.2f}%)"
                " [HYPOTHESIS-TIER laptop; sub-5% = local tie, frozen box adjudicates]")
        log.write(line + "\n")
        print(line, flush=True)
