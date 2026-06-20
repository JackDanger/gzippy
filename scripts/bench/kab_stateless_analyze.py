#!/usr/bin/env python3
"""NIGHT32 paired-CI analyzer for the isolated stateless-kernel A/B.

Args: <measured.csv> [<selftest.csv>]
CSV cols: arm,run,cycles,instructions,bytes,cyc_per_byte,instr_per_byte,ipc
Arms: a (resumable), as (stateless), b (igzip _04). selftest uses a,a2.

Reports per-arm median+IQR and PAIRED (by run index) deltas with a bootstrap
95% CI: as-a (does shedding help), a-b, as-b. Paired pairing decorrelates box
drift across rounds. A-vs-A self-test ratio must be ~1.0 to trust any delta.
"""
import csv
import sys
import statistics as st
import random


def load(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[int(r["run"])] = r
    return by


def med(x):
    return st.median(x)


def q(x, p):
    x = sorted(x)
    i = max(0, min(len(x) - 1, int(round(p * (len(x) - 1)))))
    return x[i]


def paired(by, a, b, key):
    runs = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    return [(float(by[a][r][key]) - float(by[b][r][key])) for r in runs], runs


def boot_ci(diffs, iters=20000):
    if not diffs:
        return (0.0, 0.0)
    n = len(diffs)
    meds = []
    for _ in range(iters):
        s = [diffs[random.randrange(n)] for _ in range(n)]
        meds.append(st.median(s))
    meds.sort()
    return (meds[int(0.025 * iters)], meds[int(0.975 * iters)])


def arm_summary(by, arm):
    if arm not in by:
        return None
    rows = by[arm].values()
    cpb = [float(r["cyc_per_byte"]) for r in rows]
    ipb = [float(r["instr_per_byte"]) for r in rows]
    ipc = [float(r["ipc"]) for r in rows]
    return dict(n=len(cpb), cpb=cpb, ipb=ipb, ipc=ipc)


def main():
    random.seed(1234)
    by = load(sys.argv[1])
    labels = {"a": "gz run_contig (resumable)",
              "as": "gz run_contig_stateless (D-1 shed)",
              "b": "igzip _04"}
    print("PER-ARM (median [IQR]):")
    for arm in ("a", "as", "b"):
        s = arm_summary(by, arm)
        if not s:
            print(f"  {arm}: MISSING")
            continue
        print(f"  {arm:2s} {labels[arm]:38s} n={s['n']:2d}  "
              f"cyc/B={med(s['cpb']):.4f} [{q(s['cpb'],.25):.4f},{q(s['cpb'],.75):.4f}]  "
              f"instr/B={med(s['ipb']):.4f}  IPC={med(s['ipc']):.4f}")

    def report_pair(a, b, key, unit):
        d, runs = paired(by, a, b, key)
        if not d:
            print(f"  {a}-{b} {key}: no paired runs")
            return None
        lo, hi = boot_ci(d)
        spread = (max(d) - min(d))
        disjoint = (lo > 0) or (hi < 0)
        verdict = "CI-DISJOINT from 0" if disjoint else "TIE (CI includes 0)"
        print(f"  {a}-{b} {key:13s} median={med(d):+.4f} {unit}  "
              f"95%CI=[{lo:+.4f},{hi:+.4f}]  paired-spread={spread:.4f}  -> {verdict}")
        return med(d)

    print("\nPAIRED DELTAS (by run index; negative => first arm faster/fewer):")
    das_a = report_pair("as", "a", "cyc_per_byte", "cyc/B")
    report_pair("as", "a", "instr_per_byte", "instr/B")
    da_b = report_pair("a", "b", "cyc_per_byte", "cyc/B")
    report_pair("a", "b", "instr_per_byte", "instr/B")
    das_b = report_pair("as", "b", "cyc_per_byte", "cyc/B")
    report_pair("as", "b", "instr_per_byte", "instr/B")

    # How much of the A residual the stateless variant closes.
    if das_a is not None and da_b is not None and abs(da_b) > 1e-9:
        closed = -das_a / da_b * 100.0  # as-a is negative when it helps
        print(f"\nFRACTION OF THE (A - igzip) cyc/B RESIDUAL CLOSED BY SHEDDING D-1: {closed:+.1f}%")
        if das_b is not None:
            print(f"  remaining (as - igzip) cyc/B = {das_b:+.4f}")

    if len(sys.argv) > 2:
        sby = load(sys.argv[2])
        d, runs = paired(sby, "a", "a2", "cyc_per_byte")
        if d:
            ratios = []
            for r in runs:
                ra = float(sby["a"][r]["cyc_per_byte"])
                rb = float(sby["a2"][r]["cyc_per_byte"])
                ratios.append(ra / rb)
            lo, hi = boot_ci(d)
            ok = (lo < 0 < hi)
            print(f"\nA-vs-A SELF-TEST (resumable both): n={len(d)}  "
                  f"median ratio={med(ratios):.4f}  paired cyc/B delta median={med(d):+.4f} "
                  f"95%CI=[{lo:+.4f},{hi:+.4f}]  -> {'PASS (~1.0, CI spans 0)' if ok else 'FAIL (box unstable)'}")


if __name__ == "__main__":
    main()
