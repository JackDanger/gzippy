#!/usr/bin/env python3
"""
Decompression routing check.

Generates 1MB and 10MB test files, compresses them with gzippy at T1 and T4,
then decompresses with GZIPPY_DEBUG=1 to show which path is taken. Also times
gzippy vs pigz to catch regressions.

Usage:
    python3 scripts/route_check.py <gzippy_bin> <pigz_bin>

Run before ANY decompression code change to verify routing is correct.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time

RUNS = 10


def bench(cmd, runs=RUNS):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def get_route(gzippy, path):
    env = dict(os.environ, GZIPPY_DEBUG="1")
    result = subprocess.run(
        [gzippy, "-d", "-c", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    lines = result.stderr.decode(errors="replace").strip().splitlines()
    # Return the second [gzippy] line — the first is the top-level routing
    # decision (bgzf/multi/procs), the second identifies the algorithm path
    gzippy_lines = [
        l for l in lines if "[gzippy]" in l or "[parallel_sm]" in l
    ]
    target = gzippy_lines[1] if len(gzippy_lines) > 1 else (gzippy_lines[0] if gzippy_lines else "")
    return target.replace("[gzippy] ", "").replace("[parallel_sm] ", "").strip() or "(no debug output)"


def main():
    if len(sys.argv) < 3:
        print("Usage: route_check.py <gzippy_bin> <pigz_bin>")
        sys.exit(1)

    gzippy = sys.argv[1]
    pigz = sys.argv[2]

    print("=" * 60)
    print("  Decompression Routing Check")
    print("  Run before ANY decompression code change.")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        for size_mb in [1, 10]:
            # Generate test file (base64 random — matches perf.py)
            src = os.path.join(tmpdir, f"{size_mb}mb.txt")
            subprocess.run(
                f"head -c {size_mb * 1024 * 1024} /dev/urandom | base64 > {src}",
                shell=True,
                check=True,
            )

            for threads in [1, 4]:
                gz = os.path.join(tmpdir, f"{size_mb}mb.t{threads}.gz")
                subprocess.run(
                    [gzippy, "-6", f"-p{threads}", "-c", src],
                    stdout=open(gz, "wb"),
                    stderr=subprocess.DEVNULL,
                )
                gz_size = os.path.getsize(gz) / 1024 / 1024

                route = get_route(gzippy, gz)
                gzippy_ms = bench([gzippy, "-d", "-c", gz])
                pigz_ms = bench([pigz, "-d", "-c", gz])
                diff_pct = (gzippy_ms / pigz_ms - 1) * 100
                status = "✓" if diff_pct <= 5 else "✗"

                print(f"\n  {size_mb}MB T{threads}  ({gz_size:.1f}MB compressed)")
                print(f"    route:  {route}")
                print(
                    f"    timing: gzippy={gzippy_ms:.1f}ms  pigz={pigz_ms:.1f}ms  "
                    f"{diff_pct:+.1f}% {status}"
                )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
