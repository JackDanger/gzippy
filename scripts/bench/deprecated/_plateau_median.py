#!/usr/bin/env python3
"""Median of the RSS plateau from a `ps -o rss=` sample file.

Input: one integer per line, RSS in KiB (macOS `ps` units).
Drops the first/last 20% (ramp/teardown), prints the median in BYTES.
"""
import sys


def main() -> int:
    path = sys.argv[1]
    with open(path) as f:
        vals = [int(x) for x in f.read().split() if x.strip().isdigit()]
    if not vals:
        print(0)
        return 0
    vals.sort()
    n = len(vals)
    lo = n // 5
    hi = n - lo
    core = vals[lo:hi] if hi > lo else vals
    core.sort()
    m = len(core)
    median_kib = core[m // 2] if m % 2 else (core[m // 2 - 1] + core[m // 2]) / 2
    print(int(median_kib * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
