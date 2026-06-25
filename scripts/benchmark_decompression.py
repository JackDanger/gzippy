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


# Benchmark configuration (overridable via CLI args)
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


def build_decompress_cmd(tool: str, bin_path: str, threads: int) -> list | None:
    """Stdin→stdout decompress command for a tool (no file args)."""
    if tool in ("gzippy", "pigz"):
        return [bin_path, "-d", f"-p{threads}"]
    elif tool == "unpigz":
        return [bin_path, f"-p{threads}"]
    elif tool in ("gzip", "igzip"):
        return [bin_path, "-d"]
    elif tool == "rapidgzip":
        return [bin_path, "-d", "-P", str(threads)]
    return None


def prepare_tool(cmd: list, compressed_file: str, output_file: str, original_file: str) -> str | None:
    """Warmup decode to a real file + correctness check. Returns an error
    string, or None on success. The timed trials write to /dev/null (SINK
    LAW); this is the ONLY decode that touches the disk, so the 221 MB write
    cannot inject disk-writeback stalls into the measured wall."""
    with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
        result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        return "decompression failed on warmup"
    if not filecmp.cmp(original_file, output_file, shallow=False):
        return "decompression produced incorrect output"
    return None


def run_timed(cmd: list, compressed_file: str) -> tuple:
    """One timed decode to /dev/null (SINK LAW). Returns (seconds, ok)."""
    with open(compressed_file, 'rb') as fin:
        start = time.perf_counter()
        result = subprocess.run(cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elapsed = time.perf_counter() - start
    return elapsed, result.returncode == 0


def finalize_stats(tool: str, times: list, converged: bool,
                   original_size: int, compressed_size: int) -> dict:
    """Build the per-tool result dict (schema unchanged) from its trial times."""
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
        "timed_sink": "devnull",
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
    parser.add_argument("--min-trials", type=int, default=None,
                       help="Override minimum trials (default: 10)")
    parser.add_argument("--max-trials", type=int, default=None,
                       help="Override maximum trials (default: 40)")
    parser.add_argument("--target-cv", type=float, default=None,
                       help="Override target CV (default: 0.03)")

    args = parser.parse_args()

    global MIN_TRIALS, MAX_TRIALS, TARGET_CV
    if args.min_trials is not None:
        MIN_TRIALS = args.min_trials
    if args.max_trials is not None:
        MAX_TRIALS = args.max_trials
    if args.target_cv is not None:
        TARGET_CV = args.target_cv

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

        # Phase 1 — prepare every tool: warmup decode to a real file + correctness
        # check. Tools that fail here are recorded as errors and excluded from the
        # timed loop. The valid set is then timed INTERLEAVED (round-robin per
        # trial) to /dev/null — the project's Gate-0d SINK LAW + Gate-1
        # interleaving. The old harness ran all N trials of one tool then all N of
        # the next, writing 221 MB to disk every trial; on a shared runner the
        # disk-writeback stalls landed in the FASTER tool's window (gzippy),
        # making its distribution bimodal and manufacturing a phantom sign-flip.
        valid = []  # [(tool_name, cmd)]
        times_by_tool = {}
        for tool_name, bin_path in decomp_tools:
            # gzip can't parallelise; skip it in multi-threaded runs (pigz covers that comparison).
            # igzip is also single-threaded but runs in every job so guards can assert
            # that gzippy at T>1 beats igzip at T1.
            if tool_name == "gzip" and args.threads > 1:
                continue
            cmd = build_decompress_cmd(tool_name, bin_path, args.threads)
            if cmd is None:
                continue
            out_file = str(tmpdir / f"out-{tool_name}.bin")
            err = prepare_tool(cmd, args.compressed_file, out_file, args.original_file)
            if err is not None:
                print(f"  {tool_name}: {err}")
                results["results"].append({
                    "tool": tool_name, "operation": "decompress",
                    "error": f"{tool_name} {err}",
                    "status": "fail" if "incorrect" in err else "error",
                    "archive_type": args.archive_type, "threads": args.threads,
                })
                continue
            valid.append((tool_name, cmd))
            times_by_tool[tool_name] = []

        # Phase 2 — interleaved timed trials to /dev/null.
        converged = False
        for trial in range(MAX_TRIALS):
            for tool_name, cmd in valid:
                elapsed, ok = run_timed(cmd, args.compressed_file)
                if not ok:
                    # Should not happen after a passing warmup; record and drop.
                    times_by_tool[tool_name].append(float("inf"))
                else:
                    times_by_tool[tool_name].append(elapsed)
            if trial + 1 >= MIN_TRIALS:
                if all(trimmed_stats(times_by_tool[t])[3] < TARGET_CV for t, _ in valid):
                    converged = True
                    break

        # Phase 3 — finalize per-tool stats (schema unchanged).
        for tool_name, _ in valid:
            result = finalize_stats(
                tool_name, times_by_tool[tool_name], converged,
                original_size, compressed_size,
            )
            ci = f" [{result['p10_speed_mbps']:.0f}-{result['p90_speed_mbps']:.0f}]"
            print(f"  {tool_name}: {result['speed_mbps']:.1f} MB/s{ci}, "
                  f"{result['trials']} trials, CV={result.get('cv', 0):.1%}")
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
