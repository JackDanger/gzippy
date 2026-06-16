#!/usr/bin/env python3
"""Analyze settlement artifacts: per-cell, per-sink gz/rg ratio + 95% CI.

Ratio convention: rg_time / gz_time  (matches the campaign's 0.836/0.923 numbers;
<1 means gz is SLOWER than rg, i.e. a loss; >=0.99 = parity).

CI: nonparametric bootstrap (10k resamples) of the ratio of medians, paired by
interleave index where both arms have a sample. Reports median ratio + 95% CI
and the per-arm median/IQR. VERDICT per the decorrelated rule:
  gz slower than rg with NON-OVERLAPPING -> loss; ratio CI straddling/above 0.99
  with overlap -> parity.
"""
import os
import sys
import glob
import random
import statistics as st

random.seed(1234)


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(line))
            except ValueError:
                pass
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def iqr(xs):
    if len(xs) < 2:
        return 0.0
    s = sorted(xs)
    n = len(s)
    q1 = s[int(0.25 * (n - 1))]
    q3 = s[int(0.75 * (n - 1))]
    return q3 - q1


def boot_ratio_ci(rg, gz, B=10000):
    """Bootstrap CI for median(rg)/median(gz). Resample each arm independently."""
    if not rg or not gz:
        return (float("nan"), float("nan"))
    ratios = []
    nr, ng = len(rg), len(gz)
    for _ in range(B):
        rs = med([rg[random.randrange(nr)] for _ in range(nr)])
        gs = med([gz[random.randrange(ng)] for _ in range(ng)])
        if gs > 0:
            ratios.append(rs / gs)
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios))]
    return (lo, hi)


def analyze_cell(cdir):
    name = os.path.basename(cdir).replace("cell_", "")
    rows = []
    for sink in ("devnull", "regfile"):
        gz = load(os.path.join(cdir, f"{sink}_gz.txt"))
        rg = load(os.path.join(cdir, f"{sink}_rg.txt"))
        if not gz or not rg:
            continue
        ratio = med(rg) / med(gz) if med(gz) > 0 else float("nan")
        lo, hi = boot_ratio_ci(rg, gz)
        verdict = "PARITY" if hi >= 0.99 else (
            "LOSS" if hi < 0.99 else "?")
        # stricter: loss only if upper CI < 0.99 (non-overlapping with parity)
        rows.append({
            "cell": name, "sink": sink,
            "gz_med": med(gz), "gz_iqr": iqr(gz),
            "rg_med": med(rg), "rg_iqr": iqr(rg),
            "ratio": ratio, "lo": lo, "hi": hi,
            "n_gz": len(gz), "n_rg": len(rg), "verdict": verdict,
        })
    # champions
    champs = {}
    for cf in glob.glob(os.path.join(cdir, "regfile_*.txt")):
        tool = os.path.basename(cf)[len("regfile_"):-len(".txt")]
        if tool in ("gz", "rg"):
            continue
        champs[tool] = med(load(cf))
    return rows, champs


def main():
    art = sys.argv[1]
    cells = sorted(glob.glob(os.path.join(art, "cell_*")))
    print(f"# settlement analysis: {art}\n")
    hdr = (f"{'cell':<18}{'sink':<10}{'gz_med':>9}{'rg_med':>9}"
           f"{'rg/gz':>9}{'95%CI':>18}{'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))
    all_champs = {}
    for cdir in cells:
        rows, champs = analyze_cell(cdir)
        for r in rows:
            ci = f"[{r['lo']:.3f},{r['hi']:.3f}]"
            print(f"{r['cell']:<18}{r['sink']:<10}{r['gz_med']:>9.4f}"
                  f"{r['rg_med']:>9.4f}{r['ratio']:>9.3f}{ci:>18}"
                  f"{r['verdict']:>9}")
        if champs:
            all_champs[os.path.basename(cdir).replace('cell_', '')] = champs
    if all_champs:
        print("\n# champions (regfile median secs; lower=faster):")
        for cell, ch in all_champs.items():
            parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(ch.items()))
            print(f"  {cell}: {parts}")


if __name__ == "__main__":
    main()
