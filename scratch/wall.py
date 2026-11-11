#!/usr/bin/env python3
"""L4 deflate_medium pricing sweep — wall phase (HYPOTHESIS-TIER).

Interleaved paired A/B at `-4 -p1`, alternating arm order per rep, medians,
raw per-rep samples logged. Arms:
  A = shipped greedy(16,30)  : vanilla binary, no env
  B = medium rung            : ladder-tune binary + GZIPPY_LDX_MEDIUM=1

Charter note (CLAUDE.md, TIE BAND box): on an unfrozen laptop a same-binary
interleaved median can swing 3-24% under co-runner contention, so a sub-5%
wall delta is a LOCAL TIE and the frozen box adjudicates; a tie reads NO-SHIP.
Wall rows are receipts for the mechanism's wall cost class, never a verdict.
"""
import os
import subprocess
import sys
import time

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-medium"
VANILLA = os.path.join(WT, "target-vanilla/release/gzippy")
TUNE = os.path.join(WT, "target-tune/release/gzippy")
SCRATCH = os.path.join(WT, "scratch")
L5_SET = [os.path.join(SCRATCH, "members", n) for n in [
    "dickens", "logs.txt", "logs.txt-card26m", "mozilla", "mr", "nci", "ooffice",
    "osdb", "reymont", "samba", "sao", "software.archive", "webster", "x-ray",
    "xml",
]]
CAMPAIGN = os.environ.get("CAMPAIGN_CORPUS_ROOT",
                          os.path.expanduser("~/www/gzippy-bench/corpus"))
EXTRA = ["data.sqlite", "aozora.txt", "data.json", "tool.bin", "minjs.min.js", "weights.safetensors"]
EXTRA_PATHS = [(n, os.path.join(CAMPAIGN, n)) for n in EXTRA
               if os.path.isfile(os.path.join(CAMPAIGN, n))]


def run(bin, path, env=None, threads=("-p1",)):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.perf_counter()
    subprocess.run([bin, "-n", "-4", *threads, "-c", path],
                   stdout=subprocess.DEVNULL, check=True, env=e)
    return time.perf_counter() - t0


def load1():
    return os.getloadavg()[0]  # 1-minute load, portable (macOS/Linux)


def paired(members, reps):
    logf = open(os.path.join(SCRATCH, "wall.log"), "a")
    logf.write(f"--- wall sweep {time.strftime('%Y-%m-%dT%H:%M:%S')} load1={load1():.1f} "
               f"vanilla={VANILLA} tune={TUNE} ---\n")
    for name, path in members:
        a, b = [], []
        run(VANILLA, path)
        run(TUNE, path, {"GZIPPY_LDX_MEDIUM": "1"})
        loads = []
        for i in range(reps):
            if i % 2 == 0:
                a.append(run(VANILLA, path))
                b.append(run(TUNE, path, {"GZIPPY_LDX_MEDIUM": "1"}))
            else:
                b.append(run(TUNE, path, {"GZIPPY_LDX_MEDIUM": "1"}))
                a.append(run(VANILLA, path))
            loads.append(load1())
        a_s = sorted(a)
        b_s = sorted(b)
        ma = a_s[len(a_s) // 2] * 1000
        mb = b_s[len(b_s) // 2] * 1000
        line = (f"L4 -p1 {name:22s} ship {ma:8.1f} ms | medium {mb:8.1f} ms | "
                f"B/A = {mb / ma:.4f} | load1 med {sorted(loads)[len(loads) // 2]:.1f}")
        logf.write(f"[raw] {name} loads={[round(x, 1) for x in loads]} A={[round(x, 4) for x in a]} "
                   f"B={[round(x, 4) for x in b]}\n")
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)
    logf.close()


if __name__ == "__main__":
    reps_default = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    members = [("l5/" + os.path.basename(p), p) for p in L5_SET] + EXTRA_PATHS
    # Big members get fewer reps to keep the battery bounded; swap point singles
    # out the two >150 MB ones.
    big = {p for _, p in members if os.path.getsize(p) > 100_000_000}
    ok = 0
    for name, path in members:
        reps = 5 if path in big else reps_default
        paired([(name, path)], reps)
        ok += 1
    print(f"wall battery: {ok} members done", flush=True)
