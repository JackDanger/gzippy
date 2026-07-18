#!/usr/bin/env python3
"""Render a gz-vs-competitor ratio ledger from benchmark_decompression.py JSONs.

ratio = gzippy_median / competitor_median  (>1.0 => gz SLOWER => a loss cell).
Reports each cell's wall ratio, gz CV, competitor CV, and MB/s.
"""
import json
import sys
from pathlib import Path


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        atype = data.get("archive_type", jf.stem)
        threads = data.get("threads")
        by_tool = {r["tool"]: r for r in data.get("results", []) if "median" in r}
        gz = by_tool.get("gzippy")
        if not gz:
            print(f"!! {jf.name}: no gzippy result")
            continue
        for tool, r in by_tool.items():
            if tool == "gzippy":
                continue
            ratio = gz["median"] / r["median"] if r["median"] else float("nan")
            rows.append({
                "corpus": atype, "T": threads, "vs": tool,
                "ratio": ratio,
                "gz_cv": gz.get("cv", 0), "cmp_cv": r.get("cv", 0),
                "gz_mbps": gz.get("speed_mbps", 0), "cmp_mbps": r.get("speed_mbps", 0),
                "trials": gz.get("trials", 0),
            })

    print("\n==================== aarch64 gz-vs-competitor WALL ratio ledger ====================")
    print(f"{'corpus':<14}{'T':<5}{'vs':<12}{'gz/cmp':<9}{'verdict':<8}{'gzCV':<8}{'cmpCV':<8}{'gzMB/s':<9}{'cmpMB/s':<9}{'N':<4}")
    for r in sorted(rows, key=lambda x: (x["corpus"], str(x["T"]), x["vs"])):
        v = "LOSS" if r["ratio"] > 1.03 else ("WIN" if r["ratio"] < 0.97 else "TIE")
        print(f"{r['corpus']:<14}{str(r['T']):<5}{r['vs']:<12}"
              f"{r['ratio']:<9.3f}{v:<8}{r['gz_cv']*100:<7.1f}%{r['cmp_cv']*100:<7.1f}%"
              f"{r['gz_mbps']:<9.0f}{r['cmp_mbps']:<9.0f}{r['trials']:<4}")
    print("====================================================================================\n")
    print("HEADMAP_LEDGER_JSON=" + json.dumps(rows))


if __name__ == "__main__":
    main()
