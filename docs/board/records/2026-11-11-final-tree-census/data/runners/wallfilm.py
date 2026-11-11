#!/usr/bin/env python3
# WALL FILM — final-tree census section 3 (HYPOTHESIS-TIER: unfrozen laptop).
# Cells: {in16m, logs-card26m, binary26m, text26m} x L1..L12 x {-p1,-p2,-p4}.
# Per cell: one warm run per arm (uncounted), then 8 interleaved reps in the
# order A,B,B,A (16 timed samples per arm); median wall per arm; delta =
# A_med/B_med - 1. Any cell outside the +/-5% local-tie band is rerun at 14
# reps (28/arm) and the rerun replaces the first pass (charter falsifier
# discipline, wave-B wall-census method). Per sample: load1/5/15 logged.
# Byte stamp: once per cell both arms' streams are sha256'd; a DIFF is a
# census byte-failure (the trio is default-off; expect identity) and the
# streams are roundtripped through python3 gzip before being reported.
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = "/Users/jackdanger/www/gzippy"
RUN = f"{ROOT}/.review-tmp/wC-final/run"
A_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-a/release/gzippy"
B_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-b/release/gzippy"
RAWDATA = f"{ROOT}/.review-tmp/wC-walls/corpus"
IN16M = f"{ROOT}/.review-tmp/in16m.bin"
LOG = f"{RUN}/wall-raw.log.txt"

INPUTS = {
    "in16m": IN16M,
    "logs-card26m": f"{RAWDATA}/logs-card26m.bin",
    "binary26m": f"{RAWDATA}/binary26m.bin",
    "text26m": f"{RAWDATA}/text26m.bin",
}


def loadavg():
    r = subprocess.run(["sysctl", "-n", "vm.loadavg"], stdout=subprocess.PIPE)
    choose = [p.strip(".") for p in
              r.stdout.decode().replace("{", "").replace("}", "").split()][:3]
    return "/".join(choose)


AA = False  # set True by `main aa`: same binary for both arms (A/A calibration)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one_run(binpath, name, level, p):
    t0 = time.perf_counter()
    with open(INPUTS[name], "rb") as fin, open(os.devnull, "wb") as devnull:
        r = subprocess.run([binpath, "--level", str(level), "-p", str(p), "-c"],
                           stdin=fin, stdout=devnull, stderr=subprocess.PIPE)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError(f"{binpath} failed: {r.stderr.decode()[:200]}")
    return dt


def logline(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")


def stamp_bytes(name, level, p):
    shas = {}
    sizes = {}
    tmp = f"{RUN}/stamp.tmp.gz"
    for arm, binpath in (("A", A_BIN), ("B", B_BIN if not AA else A_BIN)):
        with open(INPUTS[name], "rb") as fin, open(tmp, "wb") as fout:
            r = subprocess.run([binpath, "--level", str(level), "-p", str(p),
                                "-c"], stdin=fin, stdout=fout,
                               stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(f"stamp encode failed: {r.stderr.decode()[:150]}")
        shas[arm] = sha_file(tmp)
        sizes[arm] = os.path.getsize(tmp)
    os.remove(tmp)
    return shas, sizes


def film_cell(name, level, p, reps, tag):
    one_run(A_BIN, name, level, p)   # warmup, uncounted
    one_run(B_BIN if not AA else A_BIN, name, level, p)
    sample_a, sample_b = [], []
    order = ["A", "B", "B", "A"] * reps
    for arm in order:
        dt = one_run(A_BIN if arm == "A" else B_BIN, name, level, p)
        (sample_a if arm == "A" else sample_b).append(dt)
        logline(f"[{tag}] [raw] {arm} {dt:.1f} load={loadavg()}")
    med_a = statistics.median(sample_a)
    med_b = statistics.median(sample_b)
    delta = med_a / med_b - 1.0
    size_a, size_b = len(sample_a), len(sample_b)
    logline(f"[{tag}] med A={med_a:.1f} B={med_b:.1f} delta={delta:+.2%} "
            f"(n={size_a}/{size_b}) load={loadavg()}")
    shas, bsz = stamp_bytes(name, level, p)
    bytes_equal = shas["A"] == shas["B"]
    logline(f"[{tag}] [sha] A={shas['A'][:16]} B={shas['B'][:16]} "
            f"szA={bsz['A']} szB={bsz['B']} equal={bytes_equal}")
    return {
        "cell": tag, "input": name, "level": level, "p": p,
        "reps": reps, "a_samples": size_a, "b_samples": size_b,
        "raw_a": sample_a, "raw_b": sample_b,
        "med_a_ms": round(med_a, 1), "med_b_ms": round(med_b, 1),
        "delta_pct": round(delta * 100, 2),
        "load": loadavg(),
        "sha_a": shas["A"], "sha_b": shas["B"],
        "size_a": bsz["A"], "size_b": bsz["B"],
        "bytes_equal": bytes_equal,
    }


def main():
    global AA
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cells = []
    for name in INPUTS:
        for level in range(1, 13):
            for p in (1, 2, 4):
                cells.append((name, level, p))
    if which == "aa":
        AA = True
        results = []
        # A/A calibration: same binary both arms quantifies today's noise
        # floor (wave-B wall-census method; cells pick two fast + two nearopt;
        # "aa8" adds the one A/B cell that stayed outside the band at rerun —
        # it is the falsifier separating box noise from a real tree delta)
        if len(sys.argv) > 2:
            name, level, p = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
            aa_cells = [(name, level, p)]
        else:
            # base calibration + the one A/B cell that stayed outside the
            # band at rerun (the falsifier separating box noise from a real
            # tree delta — same binary both arms, same protocol)
            aa_cells = [("in16m", 9, 2), ("in16m", 11, 4),
                        ("text26m", 4, 2), ("text26m", 9, 4),
                        ("in16m", 8, 2)]
        for name, level, p in aa_cells:
            tag = f"AA_{name}_L{level}_p{p}"
            res = film_cell(name, level, p, 6, tag)
            results.append(res)
            print(f"[aa] {tag} A={res['med_a_ms']} B={res['med_b_ms']} "
                  f"delta={res['delta_pct']:+.2f}%", flush=True)
        with open(f"{RUN}/wall_aa.json", "w") as f:
            json.dump(results, f, indent=1)
        print("AA-DONE", flush=True)
        return
    results = []
    rerun_needed = []
    if which in ("all", "primary"):
        for name, level, p in cells:
            tag = f"{name}_L{level}_p{p}"
            res = film_cell(name, level, p, 8, tag)
            results.append(res)
            print(f"[cell] {tag} A={res['med_a_ms']} B={res['med_b_ms']} "
                  f"delta={res['delta_pct']:+.2f}% bytes_eq={res['bytes_equal']}",
                  flush=True)
            if abs(res["delta_pct"]) > 5.0:
                rerun_needed.append((name, level, p))
        with open(f"{RUN}/wall_primary.json", "w") as f:
            json.dump(results, f, indent=1)
    if which in ("all", "rerun"):
        rerun_results = []
        for name, level, p in rerun_needed:
            tag = f"{name}_L{level}_p{p}"
            res = film_cell(name, level, p, 14, tag + "[rerun]")
            rerun_results.append(res)
            print(f"[rerun] {tag} A={res['med_a_ms']} B={res['med_b_ms']} "
                  f"delta={res['delta_pct']:+.2f}%", flush=True)
        with open(f"{RUN}/wall_rerun.json", "w") as f:
            json.dump(rerun_results, f, indent=1)
    print("WALL-DONE", flush=True)


if __name__ == "__main__":
    main()
