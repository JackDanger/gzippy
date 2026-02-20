#!/usr/bin/env python3
"""
Decompression benchmark runner.

Benchmarks decompression performance for gzippy vs pigz, gzip, igzip, rapidgzip.

Usage:
    python3 scripts/benchmark_decompression.py \
        --binaries ./bin \
        --compressed-file ./archive/compressed.gz \
        --original-file ./archive/original.bin \
        --threads 4 \
        --archive-type silesia-dynamic \
        --output results/decompression.json
"""

import argparse
import filecmp
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


def benchmark_decompression(
    tool: str,
    bin_path: str,
    compressed_file: str,
    output_file: str,
    original_file: str,
    threads: int,
) -> dict:
    """Benchmark decompression for a single tool."""

    original_size = os.path.getsize(original_file)
    compressed_size = os.path.getsize(compressed_file)

    # Build command based on tool
    if tool == "gzippy":
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "pigz":
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "unpigz":
        cmd = [bin_path, f"-p{threads}"]
    elif tool == "gzip":
        cmd = [bin_path, "-d"]
    elif tool == "igzip":
        cmd = [bin_path, "-d"]
    elif tool == "rapidgzip":
        cmd = [bin_path, "-d", "-P", str(threads)]
    else:
        return {"error": f"unknown tool: {tool}"}

    def run_decompress():
        with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
            result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    # Warmup
    if not run_decompress():
        return {"error": f"{tool} decompression failed on warmup"}

    # Verify correctness
    if not filecmp.cmp(original_file, output_file, shallow=False):
        return {"error": f"{tool} decompression produced incorrect output", "status": "fail"}

    # Adaptive benchmark with trimmed statistics
    times = []
    converged = False

    for trial in range(MAX_TRIALS):
        start = time.perf_counter()
        if not run_decompress():
            return {"error": f"{tool} decompression failed"}
        times.append(time.perf_counter() - start)

        if len(times) >= MIN_TRIALS:
            _, _, _, cv = trimmed_stats(times)
            if cv < TARGET_CV:
                converged = True
                break

    trimmed, t_mean, t_stdev, t_cv = trimmed_stats(times)
    median = statistics.median(trimmed)
    sorted_trimmed = sorted(trimmed)
    p10 = sorted_trimmed[max(0, len(sorted_trimmed) // 10)]
    p90 = sorted_trimmed[min(len(sorted_trimmed) - 1, len(sorted_trimmed) * 9 // 10)]

    return {
        "tool": tool,
        "operation": "decompress",
        "times": times,
        "median": median,
        "mean": t_mean,
        "stdev": t_stdev,
        "cv": t_cv,
        "trials": len(times),
        "trimmed_trials": len(trimmed),
        "converged": converged,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "speed_mbps": original_size / median / 1_000_000,
        "p10_speed_mbps": original_size / p90 / 1_000_000,
        "p90_speed_mbps": original_size / p10 / 1_000_000,
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description="Decompression benchmark runner")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--compressed-file", type=str, required=True,
                       help="Path to compressed file")
    parser.add_argument("--original-file", type=str, required=True,
                       help="Path to original uncompressed file")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--archive-type", type=str, required=True,
                       help="Type of archive being decompressed")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")

    args = parser.parse_args()

    binaries_dir = Path(args.binaries)
    original_size = os.path.getsize(args.original_file)
    compressed_size = os.path.getsize(args.compressed_file)

    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "unpigz": find_binary(binaries_dir, "unpigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "rapidgzip": find_binary(binaries_dir, "rapidgzip"),
        "gzip": "/usr/bin/gzip",
    }

    print(f"=== Decompression Benchmark ===")
    print(f"Archive type: {args.archive_type}")
    print(f"Original size: {original_size / 1_000_000:.1f} MB")
    print(f"Compressed size: {compressed_size / 1_000_000:.1f} MB")
    print(f"Threads: {args.threads}")
    print(f"Available tools: {[k for k, v in tools.items() if v]}")
    print()

    results = {
        "archive_type": args.archive_type,
        "threads": args.threads,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "results": [],
    }

    # Decompression tools to benchmark
    decomp_tools = []
    if tools["gzippy"]:
        decomp_tools.append(("gzippy", tools["gzippy"]))
    if tools["unpigz"]:
        decomp_tools.append(("unpigz", tools["unpigz"]))
    elif tools["pigz"]:
        decomp_tools.append(("pigz", tools["pigz"]))
    if tools["igzip"]:
        decomp_tools.append(("igzip", tools["igzip"]))
    if tools["rapidgzip"]:
        decomp_tools.append(("rapidgzip", tools["rapidgzip"]))
    decomp_tools.append(("gzip", tools["gzip"]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for tool_name, bin_path in decomp_tools:
            # Skip multi-threaded benchmark for single-threaded tools
            if tool_name in ("gzip", "igzip") and args.threads > 1:
                continue

            out_file = str(tmpdir / f"out-{tool_name}.bin")
            result = benchmark_decompression(
                tool_name, bin_path, args.compressed_file, out_file,
                args.original_file, args.threads
            )

            if "error" not in result:
                ci = ""
                if "p10_speed_mbps" in result:
                    ci = f" [{result['p10_speed_mbps']:.0f}-{result['p90_speed_mbps']:.0f}]"
                print(f"  {tool_name}: {result['speed_mbps']:.1f} MB/s{ci}, "
                      f"{result['trials']} trials, CV={result.get('cv', 0):.1%}")
            else:
                print(f"  {tool_name}: {result.get('error', 'failed')}")

            result["archive_type"] = args.archive_type
            result["threads"] = args.threads
            results["results"].append(result)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
