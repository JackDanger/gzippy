#!/usr/bin/env python3
"""Measure the numbers that appear in README.md.

Runs compression and decompression of the Silesia corpus (and the logs
fixture) against gzippy, pigz, and Apple's /usr/bin/gzip, reporting median
throughput over N runs. Output goes to /dev/null so disk I/O does not
dominate timing.

Usage:
    cargo build --release
    (cd vendor/pigz && make)   # one-time: build the bundled pigz
    python3 scripts/readme_benchmark.py

Requirements:
    target/release/gzippy               — cargo build --release
    vendor/pigz/pigz                    — (cd vendor/pigz && make)
    benchmark_data/silesia.tar          — 202 MB (in-repo fixture)
    benchmark_data/logs.txt             — 211 MB (in-repo fixture)

The README quotes the Silesia row; logs is here for anyone exploring.
"""

import statistics
import subprocess
import sys
import time
from pathlib import Path

def find_repo_root() -> Path:
    """Walk up from CWD looking for Cargo.toml. Lets the script run from any
    worktree (the main one holds target/release/gzippy and vendor/pigz/pigz)."""
    for candidate in [Path.cwd(), *Path.cwd().resolve().parents, Path(__file__).resolve().parent.parent]:
        if (candidate / "Cargo.toml").exists() and (candidate / "target" / "release" / "gzippy").exists():
            return candidate
    raise SystemExit("can't find a gzippy repo with target/release/gzippy — run `cargo build --release` first")


ROOT = find_repo_root()
GZIPPY = ROOT / "target" / "release" / "gzippy"
PIGZ = ROOT / "vendor" / "pigz" / "pigz"
APPLE_GZIP = Path("/usr/bin/gzip")

INPUTS = [
    ("silesia", ROOT / "benchmark_data" / "silesia.tar"),
    ("logs", ROOT / "benchmark_data" / "logs.txt"),
]

RUNS = 15
LEVEL = 6


def run_once(cmd: list[str], stdin_path: Path) -> float:
    with open(stdin_path, "rb") as fin, open("/dev/null", "wb") as fout:
        t0 = time.perf_counter()
        rc = subprocess.run(
            cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL
        ).returncode
        t1 = time.perf_counter()
    if rc != 0:
        raise RuntimeError(f"failed: {' '.join(str(c) for c in cmd)} -> rc={rc}")
    return t1 - t0


def bench(cmd: list[str], stdin_path: Path, runs: int = RUNS) -> tuple[float, float]:
    times = [run_once(cmd, stdin_path) for _ in range(runs)]
    return statistics.median(times), statistics.stdev(times)


def mbs(mb: float, seconds: float) -> float:
    return mb / seconds


def bench_input(label: str, path: Path) -> None:
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"\n{'=' * 72}")
    print(f"INPUT: {label}  ({path.name}, {size_mb:.1f} MB)")
    print(f"{'=' * 72}")

    gz_path = path.with_suffix(".bench.gz")
    with open(gz_path, "wb") as gz:
        subprocess.run(
            [str(GZIPPY), f"-{LEVEL}", "-c", str(path)], stdout=gz, check=True
        )
    gz_mb = gz_path.stat().st_size / 1024 / 1024
    print(f"compressed at -{LEVEL}: {gz_mb:.1f} MB ({100 * gz_mb / size_mb:.1f}% of raw)\n")

    print("-- COMPRESSION (raw -> /dev/null) --")
    for name, cmd in [
        ("gzippy 14T", [str(GZIPPY), f"-{LEVEL}", "-p14", "-c"]),
        ("gzippy 1T",  [str(GZIPPY), f"-{LEVEL}", "-p1",  "-c"]),
        ("pigz 14T",   [str(PIGZ),   f"-{LEVEL}", "-p14", "-c"]),
        ("pigz 1T",    [str(PIGZ),   f"-{LEVEL}", "-p1",  "-c"]),
        ("Apple gzip", [str(APPLE_GZIP), f"-{LEVEL}", "-c"]),
    ]:
        med, sd = bench(cmd, path)
        print(f"  {name:12s}  {mbs(size_mb, med):7.0f} MB/s   "
              f"{med * 1000:7.1f} ms  (sd {sd * 1000:5.1f} ms)")

    print("\n-- DECOMPRESSION (.gz -> /dev/null) --")
    for name, cmd in [
        ("gzippy 14T", [str(GZIPPY), "-d", "-p14", "-c"]),
        ("gzippy 1T",  [str(GZIPPY), "-d", "-p1",  "-c"]),
        ("pigz",       [str(PIGZ),   "-d",         "-c"]),
        ("Apple gzip", [str(APPLE_GZIP), "-d",     "-c"]),
    ]:
        med, sd = bench(cmd, gz_path)
        print(f"  {name:12s}  {mbs(size_mb, med):7.0f} MB/s   "
              f"{med * 1000:7.1f} ms  (sd {sd * 1000:5.1f} ms)")

    gz_path.unlink()


def main() -> int:
    for p in [GZIPPY, PIGZ, APPLE_GZIP, *[p for _, p in INPUTS]]:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1
    print(f"runs:   {RUNS} per tool, median reported")
    print(f"level:  -{LEVEL}")
    subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"])
    for label, path in INPUTS:
        bench_input(label, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
