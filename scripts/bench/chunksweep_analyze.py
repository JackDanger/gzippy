#!/usr/bin/env python3
"""Analyze chunk-granularity sweep samples (wall + task-clock per arm).

Per arm file `samples_<arm>.txt`, each line: "<wall_s> <taskclock_ms> <sha>".
Reports per-arm: N, median wall (ms), wall spread% (max-min)/min, median P
(avg busy CPUs = taskclock_ms / wall_ms), and ratios vs the default (k0) and
vs rapidgzip (rg). Significance: a wall delta counts only if |Δ| > the larger
of the two arms' spreads (Gate 1). The A/A arm ("aa") must ratio ~1.0 vs k0.
"""
import sys
import os
import glob
import statistics


def load(path):
    walls, tcs = [], []
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            w = float(parts[0]) * 1000.0  # ms
            tc = float(parts[1])          # ms
        except ValueError:
            continue
        walls.append(w)
        tcs.append(tc)
    return walls, tcs


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def spread_pct(xs):
    if not xs:
        return float("nan")
    lo, hi = min(xs), max(xs)
    return (hi - lo) / lo * 100.0 if lo > 0 else float("nan")


def main(artdir):
    arms = {}
    for f in sorted(glob.glob(os.path.join(artdir, "samples_*.txt"))):
        name = os.path.basename(f)[len("samples_"):-len(".txt")]
        w, tc = load(f)
        if w:
            arms[name] = (w, tc)
    if not arms:
        print("no samples found in", artdir)
        return 1

    rows = {}
    for name, (w, tc) in arms.items():
        mw = med(w)
        # per-run P then median (robust)
        ps = [tc[i] / w[i] for i in range(len(w)) if w[i] > 0]
        rows[name] = {
            "n": len(w),
            "wall_ms": mw,
            "spread": spread_pct(w),
            "P": med(ps),
        }

    base = rows.get("k0")
    rg = rows.get("rg")
    print(f"artifact: {artdir}")
    print(f"{'arm':>6} {'N':>3} {'wall_ms':>9} {'spr%':>5} {'P':>5} "
          f"{'vs_k0':>7} {'sig':>4} {'gz/rg':>7}")
    # order: k0, k8192, k4096, k2048, k1024, k512, aa, rg
    order = ["k0", "k8192", "k4096", "k2048", "k1024", "k512", "aa", "rg"]
    names = [n for n in order if n in rows] + [n for n in rows if n not in order]
    for name in names:
        r = rows[name]
        vs_k0 = sig = gzrg = ""
        if base and name != "k0":
            ratio = r["wall_ms"] / base["wall_ms"]
            vs_k0 = f"{ratio:.4f}"
            delta = abs(r["wall_ms"] - base["wall_ms"])
            tol = max(r["spread"], base["spread"]) / 100.0 * base["wall_ms"]
            sig = "Y" if delta > tol else "tie"
        if rg and name not in ("rg",):
            gzrg = f"{r['wall_ms'] / rg['wall_ms']:.4f}"
        print(f"{name:>6} {r['n']:>3} {r['wall_ms']:>9.2f} {r['spread']:>5.1f} "
              f"{r['P']:>5.2f} {vs_k0:>7} {sig:>4} {gzrg:>7}")

    # rig self-test
    if base and "aa" in rows:
        aa_ratio = rows["aa"]["wall_ms"] / base["wall_ms"]
        ok = abs(aa_ratio - 1.0) <= max(rows["aa"]["spread"], base["spread"]) / 100.0
        print(f"\nA/A self-test: aa/k0={aa_ratio:.4f}  -> "
              f"{'PASS (rig sound)' if ok else 'FAIL (rig noisy/biased)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
