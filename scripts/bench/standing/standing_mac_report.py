#!/usr/bin/env python3
"""standing_mac_report.py — stats + gated table for the MacOS-local standing rig.

Reads a CSV of raw per-run samples produced by standing_mac.sh and prints ONE
gated table: per (corpus, arm) instr/B + cyc/B (BEST-OF-N) with the floor
agreement, the gz/libd ratios, and per-gate PASS/FAIL.

ESTIMATOR = best-of-N (MINIMUM). Both signals carry only ADDITIVE noise: an
interrupt / context switch / parallel-coordination spin adds instructions and
cycles but never removes them, so the minimum is the cleanest estimate of the
true per-run work. (Confirmed empirically: rep outliers are always ABOVE the
floor — e.g. a context-switched run shows +4% instr AND +27% cyc together.)

A/A DETERMINISM SELF-TEST = the relative range of the FLOOR_K lowest instr
samples (the two/three cleanest runs of an arm vs each other = a true gz-vs-gz /
libd-vs-libd A/A). On the M1 this is ~0.04% on heavy corpora (silesia) and
<=0.1% even on fast corpora where -p1's 2-thread handoff adds a little spin.
cyc floor range above CYC_TRUST_PCT marks cyc UNTRUSTED for that arm (instr
stays valid — instr is the deterministic primitive).

CSV columns (header required):
    corpus,bytes,arm,threads,rep,instr,cyc,real

Usage:  standing_mac_report.py <samples.csv>
"""
import csv
import math
import statistics
import sys
from collections import defaultdict

# Gate thresholds.
#   instr floor gate 0.25%: instructions-retired on the M1 is deterministic to
#   ~0.01-0.06% on heavy corpora (silesia gzippy 0.010%); on a FAST corpus
#   (big, ~0.07 s) gzippy's -p1 2-thread chunk handoff adds a little timing-
#   dependent SPIN whose instructions are counted, raising the floor to ~0.2%.
#   That is a real, documented property of the parallel path, still an order
#   finer than the cyc floor (0.4-7%), so instr remains the deterministic
#   primitive. A genuinely non-deterministic instrument would be multi-%.
INSTR_DETERMINISM_PCT = 0.25   # <= this: instr floor PASS (deterministic primitive proven)
INSTR_SUSPECT_PCT = 1.00       # > this: instr floor FAIL (instrument suspect; a broken
                               #   instrument is multi-%). 0.25-1.0% = WARN (elevated -p1
                               #   coordination jitter on a fast corpus; instr still >> cyc).
CYC_TRUST_PCT = 3.0            # floor (lowest-K) cyc range above this -> cyc UNTRUSTED
ECORE_OUTLIER_FACTOR = 1.40    # cyc > factor * median -> E-core / throttle outlier (info only)


def floor_k(n):
    """How many of the lowest samples define the reproducible floor."""
    return max(2, min(3, math.ceil(n / 3)))


def floor_range_pct(vals):
    """Relative range of the lowest-K samples = the A/A floor-agreement."""
    if len(vals) < 2:
        return float("nan")
    lo = sorted(vals)[: floor_k(len(vals))]
    if lo[0] == 0:
        return float("nan")
    return (lo[-1] - lo[0]) / lo[0] * 100.0


def main():
    if len(sys.argv) != 2:
        print("usage: standing_mac_report.py <samples.csv>", file=sys.stderr)
        sys.exit(2)

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for r in csv.DictReader(fh):
            r["bytes"] = int(r["bytes"])
            r["rep"] = int(r["rep"])
            r["instr"] = int(r["instr"])
            r["cyc"] = int(r["cyc"])
            r["real"] = float(r["real"])
            rows.append(r)

    if not rows:
        print("standing_mac_report: NO SAMPLES — FAIL", file=sys.stderr)
        sys.exit(1)

    # key = (corpus, arm, threads)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["corpus"], r["arm"], r["threads"])].append(r)

    # Per-group reduced stats (best-of-N = minimum).
    stats = {}
    dropped_total = 0
    for key, rs in grouped.items():
        bytes_ = rs[0]["bytes"]
        instr = [x["instr"] for x in rs]
        cyc_all = [x["cyc"] for x in rs]
        n_all = len(cyc_all)
        # E-core / throttle outlier count (info only — best-of-N already ignores them).
        cyc_med0 = statistics.median(cyc_all)
        dropped = sum(1 for c in cyc_all if c > ECORE_OUTLIER_FACTOR * cyc_med0)
        dropped_total += dropped

        instr_min = min(instr)
        cyc_min = min(cyc_all)
        stats[key] = {
            "bytes": bytes_,
            "n": n_all,
            "dropped": dropped,
            "instr_min": instr_min,
            "cyc_min": cyc_min,
            "instr_pb": instr_min / bytes_,
            "cyc_pb": cyc_min / bytes_,
            "instr_floor": floor_range_pct(instr),
            "cyc_floor": floor_range_pct(cyc_all),
        }

    # ---- GATE-0: A/A determinism (instr floor) + cyc trust (cyc floor) ----
    instr_gate_ok = True       # FAIL only if an arm is instrument-suspect (> SUSPECT)
    instr_elevated = []        # WARN: 0.25-1.0% (parallel-coordination jitter on fast corpus)
    cyc_untrusted = []
    for key, s in stats.items():
        if s["instr_floor"] > INSTR_SUSPECT_PCT:
            instr_gate_ok = False
        elif s["instr_floor"] > INSTR_DETERMINISM_PCT:
            instr_elevated.append(key)
        if s["cyc_floor"] > CYC_TRUST_PCT:
            cyc_untrusted.append(key)

    corpora = sorted({k[0] for k in stats})
    threads = sorted({k[2] for k in stats}, key=lambda t: int(t))

    print()
    print("=" * 100)
    print("  MACOS-LOCAL STANDING RIG  (Apple M1 Pro, aarch64) — gzippy-native vs libdeflate")
    print("  primitive: /usr/bin/time -l -> instructions retired + cycles elapsed; estimator: best-of-N")
    print("=" * 100)
    hdr = (f"{'corpus':<10} {'arm':<14} {'T':>2} {'N':>3} {'out':>4} "
           f"{'instr/B':>9} {'i-flr%':>7} {'cyc/B':>8} {'c-flr%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for corpus in corpora:
        for key in sorted([k for k in stats if k[0] == corpus], key=lambda k: (k[1], int(k[2]))):
            s = stats[key]
            cyc_tag = "" if key not in cyc_untrusted else " *UNTRUSTED"
            print(f"{key[0]:<10} {key[1]:<14} {key[2]:>2} {s['n']:>3} {s['dropped']:>4} "
                  f"{s['instr_pb']:>9.3f} {s['instr_floor']:>7.3f} "
                  f"{s['cyc_pb']:>8.3f} {s['cyc_floor']:>7.3f}{cyc_tag}")

    # ---- Ratios gzippy(-pN) / libdeflate ----
    print()
    print("  RATIOS  gzippy / libdeflate   (instr is the gated primitive; cyc trust per *UNTRUSTED above)")
    print(f"  {'corpus':<10} {'T':>2} {'instr-ratio':>11} {'cyc-ratio':>10}")
    for corpus in corpora:
        ld = stats.get((corpus, "libdeflate", "1"))
        if not ld:
            continue
        for t in threads:
            gz = stats.get((corpus, "gzippy", t))
            if not gz:
                continue
            ir = gz["instr_pb"] / ld["instr_pb"]
            cr = gz["cyc_pb"] / ld["cyc_pb"]
            print(f"  {corpus:<10} {t:>2} {ir:>10.3f}x {cr:>9.3f}x")

    # ---- GATE summary ----
    print()
    print("  GATE-0 SELF-VALIDATION")
    print(f"    [{'PASS' if instr_gate_ok else 'FAIL'}] instr A/A determinism  (lowest-K instr floor <= {INSTR_DETERMINISM_PCT}% PASS / <= {INSTR_SUSPECT_PCT}% WARN / else instrument-suspect FAIL)")
    if instr_elevated:
        names = ", ".join(f"{k[0]}/{k[1]}({stats[k]['instr_floor']:.3f}%)" for k in sorted(instr_elevated))
        print(f"    [WARN] instr floor elevated (>{INSTR_DETERMINISM_PCT}%, <={INSTR_SUSPECT_PCT}%): {names} — -p1 coordination jitter on a fast corpus; instr still >> cyc precision")
    if cyc_untrusted:
        print(f"    [WARN] cyc trust: {len(cyc_untrusted)} arm(s) exceeded {CYC_TRUST_PCT}% cyc floor range -> cyc UNTRUSTED for those (instr still valid)")
    else:
        print(f"    [PASS] cyc trust: all arms within {CYC_TRUST_PCT}% cyc floor range")
    print(f"    [INFO] above-floor outliers seen (cyc > {ECORE_OUTLIER_FACTOR}x median; ignored by best-of-N): {dropped_total}")
    print("    (byte-exact sha gate + path=ParallelSM + /dev/null-sink + -p1 are enforced by standing_mac.sh BEFORE this report)")
    print("=" * 100)

    sys.exit(0 if instr_gate_ok else 1)


if __name__ == "__main__":
    main()
