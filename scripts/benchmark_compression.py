#!/usr/bin/env python3
"""
Compression benchmark runner.

Benchmarks compression performance for gzippy vs pigz, gzip, igzip, zopfli.

Usage:
    python3 scripts/benchmark_compression.py \
        --binaries ./bin \
        --data-file ./data/test-data.bin \
        --level 6 \
        --threads 4 \
        --content-type silesia \
        --output results/compression.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Benchmark configuration
MIN_TRIALS = 10
MAX_TRIALS = 40
TARGET_CV = 0.03  # 3% coefficient of variation


def trimmed_stats(times):
    """Compute stats after dropping the single fastest and slowest trials."""
    if len(times) <= 4:
        t = sorted(times)
    else:
        t = sorted(times)[1:-1]
    mean = statistics.mean(t)
    stdev = statistics.stdev(t) if len(t) > 1 else 0
    cv = stdev / mean if mean > 0 else 1.0
    return t, mean, stdev, cv


def find_binary(binaries_dir: Path, name: str) -> str | None:
    """Find a binary in the binaries directory."""
    candidates = [
        binaries_dir / name,
        binaries_dir / f"{name}-cli",
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def build_compress_cmd(tool: str, bin_path: str, input_file: str,
                       level: int, threads: int) -> list | None:
    """Stdin→stdout compress command for a tool (writes gzip to stdout)."""
    if tool == "gzippy":
        if level >= 10:
            return [bin_path, "--level", str(level), f"-p{threads}", "-c", input_file]
        return [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "pigz":
        return [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "gzip":
        # Cap at L9; at L>=10 gzip runs at its max level as a size baseline
        return [bin_path, f"-{min(9, level)}", "-c", input_file]
    elif tool == "igzip":
        # igzip only supports levels 0-3
        return [bin_path, f"-{min(3, level)}", "-c", input_file]
    elif tool == "zopfli":
        # zopfli: only run once (very slow)
        return [bin_path, "--i5", "-c", input_file]
    return None


def prepare_tool(cmd: list, output_file: str) -> tuple:
    """Warmup compress to a REAL file (SINK LAW, Gate-0d): this is the ONE
    write that touches disk, and output_size/ratio are captured FROM IT — the
    timed trials never write a file, so disk-writeback can't stall the measured
    wall (which on a shared runner used to land in the faster tool's window).
    Returns (output_size, error_str|None)."""
    with open(output_file, 'wb') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        return None, "compression failed on warmup"
    return os.path.getsize(output_file), None


def run_timed(cmd: list) -> tuple:
    """One timed compress to /dev/null (SINK LAW). Returns (seconds, ok)."""
    start = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start, result.returncode == 0


def finalize_stats(tool: str, times: list, converged: bool,
                   output_size: int, input_size: int) -> dict:
    """Build the per-tool result dict (schema unchanged) from its trial times.
    output_size/ratio come from the warmup write, never a timed trial."""
    trimmed, t_mean, t_stdev, t_cv = trimmed_stats(times)
    median = statistics.median(trimmed)
    sorted_trimmed = sorted(trimmed)
    p10 = sorted_trimmed[max(0, len(sorted_trimmed) // 10)]
    p90 = sorted_trimmed[min(len(sorted_trimmed) - 1, len(sorted_trimmed) * 9 // 10)]
    return {
        "tool": tool,
        "operation": "compress",
        "times": times,
        "median": median,
        "mean": t_mean,
        "stdev": t_stdev,
        "cv": t_cv,
        "trials": len(times),
        "trimmed_trials": len(trimmed),
        "converged": converged,
        "output_size": output_size,
        "input_size": input_size,
        "ratio": output_size / input_size,
        "speed_mbps": input_size / median / 1_000_000,
        "p10_speed_mbps": input_size / p90 / 1_000_000,
        "p90_speed_mbps": input_size / p10 / 1_000_000,
        "timed_sink": "devnull",
    }


def main():
    parser = argparse.ArgumentParser(description="Compression benchmark runner")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to uncompressed test data")
    parser.add_argument("--level", type=int, required=True,
                       help="Compression level to test")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--content-type", type=str, required=True,
                       help="Type of content being compressed")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")

    args = parser.parse_args()

    binaries_dir = Path(args.binaries)
    data_file = args.data_file
    input_size = os.path.getsize(data_file)

    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "zopfli": find_binary(binaries_dir, "zopfli"),
        "gzip": "/usr/bin/gzip",
    }

    print(f"=== Compression Benchmark ===")
    print(f"Content: {args.content_type}, Level: {args.level}, Threads: {args.threads}")
    print(f"Input size: {input_size / 1_000_000:.1f} MB")
    print(f"Available tools: {[k for k, v in tools.items() if v]}")
    print()

    results = {
        "level": args.level,
        "threads": args.threads,
        "content_type": args.content_type,
        "input_size": input_size,
        "results": [],
    }

    # Tools to benchmark
    # pigz only supports L1-L9; exclude it at L10+ so it doesn't error out
    comp_tools = ["gzippy"]
    if args.level < 10:
        comp_tools.append("pigz")
    comp_tools.append("gzip")  # runs at L9 when level>=10 (see benchmark_compression())
    if args.level <= 3 and tools["igzip"]:
        comp_tools.append("igzip")
    if args.level >= 9 and tools["zopfli"]:
        comp_tools.append("zopfli")

    # L>=10 compression is too slow for statistical convergence; 1 trial is
    # enough since the ratio (output size) is deterministic and that's all we
    # guard on at those levels.
    effective_min = 1 if args.level >= 10 else MIN_TRIALS
    effective_max = 3 if args.level >= 10 else MAX_TRIALS

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Phase 1 — prepare every multi-trial tool: build cmd + ONE warmup
        # write to a real file (SINK LAW, Gate-0d), capturing output_size/ratio
        # FROM that warmup. zopfli is handled separately below (ratio-only,
        # single deterministic run, OUTSIDE the interleave). Tools that fail
        # the warmup are recorded as errors and excluded from timing. The valid
        # set is then timed INTERLEAVED (round-robin per trial, Gate-1) to
        # /dev/null — the old harness ran all N trials of one tool then all N of
        # the next while writing the full output to disk every trial, so a
        # shared-runner disk-writeback stall landed in the faster tool's window
        # and manufactured a phantom sign-flip.
        valid = []  # [(tool, cmd, output_size)]
        times_by_tool = {}
        for tool in comp_tools:
            if tool == "zopfli":
                continue
            bin_path = tools.get(tool)
            if not bin_path or not os.path.exists(bin_path):
                continue
            cmd = build_compress_cmd(tool, bin_path, data_file, args.level, args.threads)
            if cmd is None:
                continue
            out_file = str(tmpdir / f"{tool}.gz")
            output_size, err = prepare_tool(cmd, out_file)
            if err is not None:
                print(f"  {tool}: {err}")
                results["results"].append({
                    "tool": tool, "operation": "compress",
                    "error": f"{tool} {err}",
                    "level": args.level, "threads": args.threads,
                    "content_type": args.content_type,
                })
                continue
            valid.append((tool, cmd, output_size))
            times_by_tool[tool] = []

        # Phase 2 — interleaved timed trials to /dev/null.
        converged = False
        for trial in range(effective_max):
            for tool, cmd, _ in valid:
                elapsed, ok = run_timed(cmd)
                if not ok:
                    # Should not happen after a passing warmup; record and drop.
                    times_by_tool[tool].append(float("inf"))
                else:
                    times_by_tool[tool].append(elapsed)
            if trial + 1 >= effective_min:
                if all(trimmed_stats(times_by_tool[t])[3] < TARGET_CV for t, _, _ in valid):
                    converged = True
                    break

        # Phase 3 — finalize per-tool stats (schema unchanged); ratio from warmup.
        for tool, _, output_size in valid:
            result = finalize_stats(
                tool, times_by_tool[tool], converged, output_size, input_size,
            )
            print(f"  {tool}: {result['speed_mbps']:.1f} MB/s, "
                  f"ratio {result['ratio']:.3f}, "
                  f"{result['trials']} trials")
            result["level"] = args.level
            result["threads"] = args.threads
            result["content_type"] = args.content_type
            results["results"].append(result)

        # zopfli — ratio-only single deterministic measurement, OUTSIDE the
        # interleave (it is far too slow to converge and is guarded only on
        # output size). It needs a real output file to read the size, so it
        # keeps its single timed write-to-file.
        if "zopfli" in comp_tools:
            bin_path = tools.get("zopfli")
            if bin_path and os.path.exists(bin_path):
                cmd = build_compress_cmd("zopfli", bin_path, data_file, args.level, args.threads)
                out_file = str(tmpdir / "zopfli.gz")
                start = time.perf_counter()
                with open(out_file, 'wb') as f:
                    zr = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
                elapsed = time.perf_counter() - start
                if zr.returncode != 0:
                    print("  zopfli: zopfli failed")
                    results["results"].append({
                        "tool": "zopfli", "operation": "compress",
                        "error": "zopfli failed",
                        "level": args.level, "threads": args.threads,
                        "content_type": args.content_type,
                    })
                else:
                    output_size = os.path.getsize(out_file)
                    result = {
                        "tool": "zopfli",
                        "operation": "compress",
                        "times": [elapsed],
                        "median": elapsed,
                        "mean": elapsed,
                        "stdev": 0,
                        "cv": 0,
                        "trials": 1,
                        "converged": True,
                        "output_size": output_size,
                        "input_size": input_size,
                        "ratio": output_size / input_size,
                        "speed_mbps": input_size / elapsed / 1_000_000,
                    }
                    print(f"  zopfli: {result['speed_mbps']:.1f} MB/s, "
                          f"ratio {result['ratio']:.3f}, "
                          f"{result['trials']} trials")
                    result["level"] = args.level
                    result["threads"] = args.threads
                    result["content_type"] = args.content_type
                    results["results"].append(result)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
