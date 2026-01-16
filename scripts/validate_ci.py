#!/usr/bin/env python3
"""
CI-friendly validation script for rigz.

Tests that rigz produces gzip-compatible output by running a cross-tool
decompression matrix. Outputs JSON results and exits with non-zero code
on any failure.

Runs multiple trials per test for statistical significance.

Usage:
    python3 scripts/validate_ci.py
    python3 scripts/validate_ci.py --level 6 --threads 4
    python3 scripts/validate_ci.py --output results.json
    python3 scripts/validate_ci.py --trials 7  # More trials for CI
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Default number of trials for statistical significance
DEFAULT_TRIALS = 5


def find_tool(name: str) -> str:
    """Find a tool binary."""
    paths = {
        "gzip": ["./gzip/gzip", shutil.which("gzip") or "gzip"],
        "pigz": ["./pigz/pigz"],
        "rigz": ["./target/release/rigz"],
        "unrigz": ["./target/release/unrigz"],
    }
    for path in paths.get(name, []):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"Could not find {name}")


def compress_once(tool: str, level: int, threads: int, input_file: str, output_file: str) -> tuple:
    """Compress a file once. Returns (success, elapsed_time)."""
    bin_path = find_tool(tool)
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def compress(tool: str, level: int, threads: int, input_file: str, output_file: str, trials: int = DEFAULT_TRIALS) -> dict:
    """Compress a file multiple times. Returns stats dict."""
    times = []
    success = False
    
    for _ in range(trials):
        success, elapsed = compress_once(tool, level, threads, input_file, output_file)
        if not success:
            return {"success": False, "median": 0, "min": 0, "max": 0, "stdev": 0, "times": []}
        times.append(elapsed)
    
    return {
        "success": True,
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) >= 2 else 0,
        "times": times,
    }


def decompress_once(tool: str, input_file: str, output_file: str) -> tuple:
    """Decompress a file once. Returns (success, elapsed_time)."""
    bin_path = find_tool(tool)
    cmd = [bin_path, "-d", "-c", input_file]
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def decompress(tool: str, input_file: str, output_file: str, trials: int = DEFAULT_TRIALS) -> dict:
    """Decompress a file multiple times. Returns stats dict."""
    times = []
    success = False
    
    for _ in range(trials):
        success, elapsed = decompress_once(tool, input_file, output_file)
        if not success:
            return {"success": False, "median": 0, "min": 0, "max": 0, "stdev": 0, "times": []}
        times.append(elapsed)
    
    return {
        "success": True,
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) >= 2 else 0,
        "times": times,
    }


def files_identical(file1: str, file2: str) -> bool:
    """Check if two files are byte-identical."""
    result = subprocess.run(["diff", "-q", file1, file2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def create_test_data(output_path: str, size_mb: int = 10) -> bool:
    """Create test data file."""
    size_bytes = size_mb * 1024 * 1024
    cmd = f"head -c {size_bytes} /dev/urandom | base64 > {output_path}"
    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def format_time(seconds: float) -> str:
    """Format time with appropriate precision."""
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.0f}s"


def format_stats(stats: dict) -> str:
    """Format timing statistics for display."""
    if not stats.get("success"):
        return "FAILED"
    median = stats["median"]
    stdev = stats.get("stdev", 0)
    min_t = stats.get("min", median)
    max_t = stats.get("max", median)
    if stdev > 0:
        return f"med={format_time(median)} (±{format_time(stdev)}, {format_time(min_t)}-{format_time(max_t)})"
    return format_time(median)


def main():
    parser = argparse.ArgumentParser(description="CI validation for rigz")
    parser.add_argument("--level", type=int, default=None,
                       help="Specific compression level (default: test 1, 6, 9)")
    parser.add_argument("--threads", type=int, default=None,
                       help="Specific thread count (default: test 1, max_cores)")
    parser.add_argument("--size", type=int, default=10,
                       help="Test file size in MB (default: 10)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                       help=f"Number of trials per test (default: {DEFAULT_TRIALS})")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Determine thread counts: 1 and max available cores
    max_threads = os.cpu_count() or 4
    levels = [args.level] if args.level else [1, 6, 9]
    thread_counts = [args.threads] if args.threads else [1, max_threads]
    tools = ["gzip", "pigz", "rigz"]
    trials = args.trials
    
    results = {
        "config": {
            "levels": levels,
            "threads": thread_counts,
            "size_mb": args.size,
            "trials": trials,
        },
        "tests": [],
        "compression_stats": [],
        "passed": 0,
        "failed": 0,
        "errors": [],
    }
    
    print(f"=== Cross-Tool Validation Matrix ===")
    print(f"Levels: {levels}, Threads: {thread_counts}, Size: {args.size}MB")
    print(f"Trials: {trials} per test for statistical significance")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.bin"
        
        print(f"Creating {args.size}MB test file...")
        if not create_test_data(str(test_file), args.size):
            print("ERROR: Failed to create test data")
            results["errors"].append("Failed to create test data")
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            return 1
        
        for level in levels:
            for threads in thread_counts:
                print(f"\n--- Level {level}, {threads} thread(s) ---")
                
                # Compress with each tool
                compressed = {}
                for tool in tools:
                    out = tmpdir / f"test.{tool}.l{level}.t{threads}.gz"
                    stats = compress(tool, level, threads, str(test_file), str(out), trials)
                    
                    if stats["success"]:
                        compressed[tool] = out
                        size = os.path.getsize(out)
                        stats_str = format_stats(stats)
                        print(f"  {tool:5} compressed: {size:,} bytes  {stats_str}")
                        
                        # Record stats
                        results["compression_stats"].append({
                            "tool": tool,
                            "level": level,
                            "threads": threads,
                            "size_bytes": size,
                            "median_time": stats["median"],
                            "min_time": stats["min"],
                            "max_time": stats["max"],
                            "stdev": stats["stdev"],
                            "all_times": stats["times"],
                        })
                    else:
                        error = f"{tool} compression failed (L{level} T{threads})"
                        print(f"  {tool:5} compressed: FAILED")
                        results["errors"].append(error)
                        results["failed"] += 1
                        results["tests"].append({
                            "compress_tool": tool,
                            "level": level,
                            "threads": threads,
                            "passed": False,
                            "error": "compression failed",
                        })
                
                # Cross-tool decompression matrix
                print()
                for comp_tool, comp_file in compressed.items():
                    for decomp_tool in tools:
                        out = tmpdir / f"test.{comp_tool}.{decomp_tool}.bin"
                        test_id = f"{comp_tool}→{decomp_tool} (L{level} T{threads})"
                        
                        test_result = {
                            "compress_tool": comp_tool,
                            "decompress_tool": decomp_tool,
                            "level": level,
                            "threads": threads,
                        }
                        
                        stats = decompress(decomp_tool, str(comp_file), str(out), trials)
                        
                        if not stats["success"]:
                            print(f"  ❌ {comp_tool:5} → {decomp_tool:5}: decompression failed")
                            test_result["passed"] = False
                            test_result["error"] = "decompression failed"
                            results["errors"].append(f"{test_id}: decompression failed")
                            results["failed"] += 1
                        elif not files_identical(str(test_file), str(out)):
                            print(f"  ❌ {comp_tool:5} → {decomp_tool:5}: output mismatch")
                            test_result["passed"] = False
                            test_result["error"] = "output mismatch"
                            results["errors"].append(f"{test_id}: output mismatch")
                            results["failed"] += 1
                        else:
                            stats_str = format_stats(stats)
                            print(f"  ✅ {comp_tool:5} → {decomp_tool:5}: OK  {stats_str}")
                            test_result["passed"] = True
                            test_result["median_time"] = stats["median"]
                            test_result["min_time"] = stats["min"]
                            test_result["max_time"] = stats["max"]
                            results["passed"] += 1
                        
                        results["tests"].append(test_result)
                        
                        if out.exists():
                            out.unlink()
        
        # Test unrigz symlink
        print("\n--- Testing unrigz symlink ---")
        try:
            unrigz_path = find_tool("unrigz")
            rigz_compressed = tmpdir / "unrigz_test.gz"
            unrigz_out = tmpdir / "unrigz_test.bin"
            
            comp_stats = compress("rigz", 6, max_threads, str(test_file), str(rigz_compressed), trials=1)
            if comp_stats["success"]:
                times = []
                for _ in range(trials):
                    start = time.perf_counter()
                    cmd = [unrigz_path, "-c", str(rigz_compressed)]
                    with open(unrigz_out, "wb") as f:
                        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                
                median_time = statistics.median(times)
                if result.returncode == 0 and files_identical(str(test_file), str(unrigz_out)):
                    print(f"  ✅ unrigz: OK  med={format_time(median_time)}")
                    results["passed"] += 1
                    results["tests"].append({
                        "tool": "unrigz",
                        "passed": True,
                        "median_time": median_time,
                    })
                else:
                    print("  ❌ unrigz: FAILED")
                    results["failed"] += 1
                    results["errors"].append("unrigz decompression failed or mismatch")
                    results["tests"].append({"tool": "unrigz", "passed": False})
            else:
                raise RuntimeError("rigz compression failed for unrigz test")
        except Exception as e:
            print(f"  ❌ unrigz: ERROR - {e}")
            results["failed"] += 1
            results["errors"].append(f"unrigz test error: {e}")
            results["tests"].append({"tool": "unrigz", "passed": False, "error": str(e)})
    
    # Summary
    total = results["passed"] + results["failed"]
    print()
    print("=" * 60)
    print(f"  Results: {results['passed']}/{total} passed, {results['failed']} failed")
    print(f"  ({trials} trials per test)")
    print("=" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
    
    if results["failed"] > 0:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  • {error}")
        return 1
    
    print("\n✅ All validation tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
