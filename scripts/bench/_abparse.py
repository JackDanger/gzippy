#!/usr/bin/env python3
"""Parse a fulcrum-abmeasure JSON: per-arm mean cyc/B, instr/B, IPC + paired ratios."""
import json
import sys
import statistics


def arm_stats(arm):
    cyc_b = [s["cycles"] / s["bytes"] for s in arm["samples"]]
    ins_b = [s["instructions"] / s["bytes"] for s in arm["samples"]]
    ipc = [s["instructions"] / s["cycles"] for s in arm["samples"]]
    return (statistics.mean(cyc_b), statistics.mean(ins_b), statistics.mean(ipc),
            cyc_b, ins_b)


def main():
    d = json.load(open(sys.argv[1]))
    label = sys.argv[2] if len(sys.argv) > 2 else "run"
    arms = {k: d[k] for k in ("base", "after", "rg") if k in d}
    print(f"=== {label} ===")
    stats = {}
    for name, arm in arms.items():
        cb, ib, ipc, cbl, ibl = arm_stats(arm)
        stats[name] = (cb, ib, ipc, cbl, ibl)
        sh = (arm.get('sha') or 'none')[:8]
        print(f"  {arm['label']:>8}: cyc/B {cb:7.3f}  instr/B {ib:7.3f}  IPC {ipc:5.3f}  "
              f"sha {sh}")
    # paired ratios base vs rg and base vs after (A/A)
    if "base" in stats and "rg" in stats:
        b_cb, b_ib = stats["base"][3], stats["base"][4]
        r_cb, r_ib = stats["rg"][3], stats["rg"][4]
        n = min(len(b_cb), len(r_cb))
        cyc_ratio = statistics.median([b_cb[i] / r_cb[i] for i in range(n)])
        ins_ratio = statistics.median([b_ib[i] / r_ib[i] for i in range(n)])
        print(f"  gz/rg  cyc/B {cyc_ratio:.4f}   instr/B {ins_ratio:.4f}")
    if "base" in stats and "after" in stats:
        b_cb = stats["base"][3]
        a_cb = stats["after"][3]
        n = min(len(b_cb), len(a_cb))
        aa = statistics.median([b_cb[i] / a_cb[i] for i in range(n)])
        print(f"  A/A    cyc/B {aa:.4f}")


if __name__ == "__main__":
    main()
