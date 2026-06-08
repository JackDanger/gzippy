#!/usr/bin/env python3
"""Per-T RSS summary + linear regression for rss_vs_t.sh.

Reads a whitespace data file with columns:
    T  peak_bytes  plateau_bytes  sha_ok  path_ok
(one row per trial). Prints a per-T table (best/min, median, spread%) for both
peak and plateau RSS, then a linear regression of MEDIAN plateau vs T
(slope bytes/thread, R^2) and the same for peak. A 2-point slope can't separate
constant overhead from per-thread growth, so we regress over the full grid and
report linearity (R^2).

Any row with sha_ok!=1 or path_ok<1 is VOID (dropped) and counted.
"""
import sys
from collections import defaultdict


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return float(xs[n // 2]) if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def linregress(pts):
    """Ordinary least squares. pts = [(x,y),...]. Returns (slope, intercept, r2)."""
    n = len(pts)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for _, y in pts)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def fmt_mib(b):
    return f"{b / (1024 * 1024):.2f} MiB"


def main() -> int:
    path = sys.argv[1]
    peak = defaultdict(list)
    plateau = defaultdict(list)
    void = 0
    total = 0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            total += 1
            T = int(parts[0])
            pk = int(parts[1])
            pl = int(parts[2])
            sha_ok = parts[3] == "1"
            path_ok = int(parts[4]) >= 1
            if not (sha_ok and path_ok):
                void += 1
                continue
            if pk > 0:
                peak[T].append(pk)
            if pl > 0:
                plateau[T].append(pl)

    Ts = sorted(peak.keys())
    print(f"\n# trials={total}  void(sha/path){void}\n")
    print(
        f"{'T':>3} | {'peak_best':>14} {'peak_med':>14} {'pk_spread%':>10}"
        f" | {'plat_best':>14} {'plat_med':>14} {'pl_spread%':>10}"
    )
    print("-" * 92)

    peak_med_pts = []
    plat_med_pts = []
    for T in Ts:
        pk = peak[T]
        pl = plateau.get(T, [])
        pk_best = min(pk) if pk else 0
        pk_med = median(pk)
        pk_spread = 100.0 * (max(pk) - min(pk)) / min(pk) if pk and min(pk) else 0.0
        pl_best = min(pl) if pl else 0
        pl_med = median(pl)
        pl_spread = 100.0 * (max(pl) - min(pl)) / min(pl) if pl and min(pl) else 0.0
        peak_med_pts.append((T, pk_med))
        if pl_med > 0:
            plat_med_pts.append((T, pl_med))
        print(
            f"{T:>3} | {pk_best:>14} {int(pk_med):>14} {pk_spread:>10.1f}"
            f" | {pl_best:>14} {int(pl_med):>14} {pl_spread:>10.1f}"
        )

    print("\n## Linear regression vs T (median across trials)")
    ps, pi, pr2 = linregress(peak_med_pts)
    print(
        f"PEAK    slope = {ps:>14.0f} bytes/thread ({fmt_mib(ps)}/thread)  "
        f"intercept = {fmt_mib(pi)}  R^2 = {pr2:.4f}"
    )
    if plat_med_pts:
        ls, li, lr2 = linregress(plat_med_pts)
        print(
            f"PLATEAU slope = {ls:>14.0f} bytes/thread ({fmt_mib(ls)}/thread)  "
            f"intercept = {fmt_mib(li)}  R^2 = {lr2:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
