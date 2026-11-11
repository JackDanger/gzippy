#!/usr/bin/env python3
"""L4 deflate_medium pricing sweep — wall phase v3 (HYPOTHESIS-TIER).

THREE interleaved arms at `-4 -p1`, alternating arm order per rep, medians,
raw per-rep samples + load stamps logged:

  ship   : vanilla binary, no env          — shipped bytes, shipped codegen
  tune   : ladder-tune binary, no env      — the SAME shipped bytes (A/A), the
                                             tune build's own codegen (the
                                             cfg'd software-pipelined chain walk)
                                           — this pair isolates the instrument
                                             premium from the mechanism
  medium : ladder-tune binary + GZIPPY_LDX_MEDIUM=1
                                           — the mechanism (fizzle + probe) on
                                             top of the tune codegen

The MECHANISM'S wall cost is the SAME-BINARY pair `medium / tune` (the
instrument premium cancels). `medium / ship` is what a real arming would
inherit ONLY if the lever also switched the hc walk to the pipelined variant —
it would not — so it is the pessimistic upper bound, reported for context.
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

ARMS = [
    ("ship", VANILLA, None),
    ("tune", TUNE, None),
    ("medium", TUNE, {"GZIPPY_LDX_MEDIUM": "1"}),
]


def run(arm, path):
    name, bin, env_override = arm
    e = dict(os.environ)
    if env_override:
        e.update(env_override)
    t0 = time.perf_counter()
    subprocess.run([bin, "-n", "-4", "-p1", "-c", path],
                   stdout=subprocess.DEVNULL, check=True, env=e)
    return time.perf_counter() - t0


def load1():
    return os.getloadavg()[0]


def paired(members, reps):
    """Rotating arm order: every rep runs the three arms in a rotated order
    (rep i rotation i%3), so each arm sits first about a third of the time."""
    logf = open(os.path.join(SCRATCH, "wall.log"), "a")
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    logf.write(f"--- wall v3 {stamp} ship={VANILLA} tune={TUNE} "
               f"[same-binary medium/tune isolates the mechanism] ---\n")
    for name, path in members:
        samples = {name_: [] for name_, _, _ in ARMS}
        loads = []
        for arm in ARMS:  # warm all three
            run(arm, path)
        for i in range(reps):
            order = ARMS[i % 3:] + ARMS[:i % 3]
            for arm in order:
                samples[arm[0]].append(run(arm, path))
            loads.append(load1())
        line = [f"L4 -p1 {name:22s}"]
        for name_, _, _ in ARMS:
            s = sorted(samples[name_])
            line.append(f"{name_} {s[len(s) // 2] * 1000:8.1f} ms")
        def med_of(k):
            v = sorted(samples[k])
            return v[len(v) // 2]
        mt = med_of("medium")
        tn = med_of("tune")
        vs = med_of("ship")
        line.append(f"| mechanism B/A (medium/tune, same binary) = {mt / tn:.4f}"
                    f" | full B/A (medium/ship) = {mt / vs:.4f}"
                    f" | tune/ship (instrument) = {tn / vs:.4f}"
                    f" | load1 med {sorted(loads)[len(loads) // 2]:.1f}")
        logf.write(f"[raw] {name} loads={[round(x, 1) for x in loads]}\n")
        for name_, _, _ in ARMS:
            logf.write(f"[raw] {name} {name_}={[round(x, 4) for x in samples[name_]]}\n")
        logf.write(" / ".join(line) + "\n")
        logf.flush()
        print(" / ".join(line), flush=True)
    logf.close()


if __name__ == "__main__":
    reps_default = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    members = [("l5/" + os.path.basename(p), p) for p in L5_SET] + EXTRA_PATHS
    for name, path in members:
        reps = 5 if os.path.getsize(path) > 100_000_000 else reps_default
        paired([(name, path)], reps)
    print("wall battery v3 done", flush=True)
