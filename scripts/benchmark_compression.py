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


def benchmark_compression(
    tool: str,
    bin_path: str,
    input_file: str,
    output_file: str,
    level: int,
    threads: int,
) -> dict:
    """Benchmark compression for a single tool."""

    # Build command based on tool
    if tool == "gzippy":
        if level >= 10:
            cmd = [bin_path, "--level", str(level), f"-p{threads}", "-c", input_file]
        else:
            cmd = [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "pigz":
        cmd = [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "gzip":
        # Cap at L9; at L>=10 gzip runs at its max level as a size baseline
        cmd = [bin_path, f"-{min(9, level)}", "-c", input_file]
    elif tool == "igzip":
        # igzip only supports levels 0-3
        igzip_level = min(3, level)
        cmd = [bin_path, f"-{igzip_level}", "-c", input_file]
    elif tool == "zopfli":
        # zopfli: only run once (very slow)
        cmd = [bin_path, "--i5", "-c", input_file]
    else:
        return {"error": f"unknown tool: {tool}"}

    # For zopfli, just run once
    if tool == "zopfli":
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            return {"error": "zopfli failed"}

        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)

        return {
            "tool": tool,
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

    # Normal adaptive benchmark
    def run_compress():
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    # L>=10 (zopfli) is too slow for statistical convergence; 1 trial is enough
    # since the ratio (output size) is deterministic and that's all we guard on.
    effective_min = 1 if level >= 10 else MIN_TRIALS
    effective_max = 3 if level >= 10 else MAX_TRIALS

    times = []
    converged = False

    # Warmup
    run_compress()

    for trial in range(effective_max):
        start = time.perf_counter()
        if not run_compress():
            return {"error": f"{tool} compression failed"}
        times.append(time.perf_counter() - start)

        if len(times) >= effective_min:
            _, _, _, cv = trimmed_stats(times)
            if cv < TARGET_CV:
                converged = True
                break

    output_size = os.path.getsize(output_file)
    input_size = os.path.getsize(input_file)
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for tool in comp_tools:
            bin_path = tools.get(tool)
            if not bin_path or not os.path.exists(bin_path):
                continue

            out_file = str(tmpdir / f"{tool}.gz")
            result = benchmark_compression(
                tool, bin_path, data_file, out_file,
                args.level, args.threads
            )

            if "error" not in result:
                print(f"  {tool}: {result['speed_mbps']:.1f} MB/s, "
                      f"ratio {result['ratio']:.3f}, "
                      f"{result['trials']} trials")
            else:
                print(f"  {tool}: {result['error']}")

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
