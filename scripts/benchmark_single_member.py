#!/usr/bin/env python3
"""
Benchmark single-member gzip decompression.

This specifically tests the 4-phase hyper-parallel pipeline:
  Phase 1: Window Boot (sequential first chunk for 32KB window)
  Phase 2: Speculative Parallel Decode (markers for unresolved back-refs)
  Phase 3: Window Propagation + SIMD Marker Replacement
  Phase 4: Write Output

Single-member files are created by standard gzip (not pigz/gzippy which create
multi-member files). They're the hardest case for parallel decompression because
there's no natural parallelization boundary.

Usage:
    python3 scripts/benchmark_single_member.py \
        --binaries ./bin \
        --compressed-file /tmp/giant.tar.gz \
        --original-file /tmp/giant.tar \
        --threads 8 \
        --output results/single-member.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


# Benchmark configuration
MIN_TRIALS = 3       # Single-member is slow, fewer trials
MAX_TRIALS = 10
TARGET_CV = 0.05


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


def benchmark_decompress(
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
    
    # Routing-trace audit (gzippy only). Without this, a silent fallback to
    # sequential libdeflate looks identical to "marker pipeline ran but was
    # slow" — both produce correct output at ~libdeflate throughput.
    # Capture stderr from gzippy with GZIPPY_DEBUG=1 and parse for the
    # marker-pipeline log line. We do this once before the timed runs so
    # it doesn't affect the perf numbers.
    routing_trace = None
    if tool == "gzippy":
        env = dict(os.environ)
        env["GZIPPY_DEBUG"] = "1"
        with open(compressed_file, 'rb') as fin:
            trace_result = subprocess.run(
                cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
            )
        stderr_text = trace_result.stderr.decode(errors="replace")
        ran_marker = "[parallel_sm:v0.6]" in stderr_text and "total=" in stderr_text
        # New: typed-routing message added in PR #90 for the BTYPE=01-heavy
        # case (see premortem D1+D5). Older log strings retained for
        # backward compatibility with older gzippy binaries on the path.
        fell_back = (
            "parallel single-member fell back" in stderr_text
            or "parallel single-member failed" in stderr_text
        )
        routing_trace = {
            "ran_marker_pipeline": ran_marker,
            "fell_back_to_sequential": fell_back,
            # Capture a larger window of stderr so the per-phase
            # timing line ("[parallel_sm:v0.6] search=Xms decode=Yms
            # retry=Zms resolve=Wms total=Tms ...") survives — that
            # line is what tells us WHY the bench is slow.
            "stderr_head": stderr_text[:8000],
        }

    # rapidgzip verbose stats: chunk count, pool efficiency, decode times.
    # Run with --verbose once (not during timed runs) to capture internal
    # BlockFetcher statistics that explain the parallelism structure.
    rapidgzip_verbose = None
    if tool == "rapidgzip":
        verbose_cmd = [bin_path, "-d", "-P", str(threads), "--verbose"]
        with open(compressed_file, 'rb') as fin:
            vr = subprocess.run(
                verbose_cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        rapidgzip_verbose = vr.stderr.decode(errors="replace")

    def run_decompress(capture_stderr: bool = False):
        env = None
        if capture_stderr and tool == "gzippy":
            env = dict(os.environ)
            env["GZIPPY_DEBUG"] = "1"
        with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
            result = subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
                env=env,
            )
        stderr_text = ""
        if capture_stderr and result.stderr is not None:
            stderr_text = result.stderr.decode(errors="replace")
        return result.returncode == 0, stderr_text

    # Warmup
    print(f"  {tool}: warming up...", end="", flush=True)
    ok, warmup_stderr = run_decompress(capture_stderr=True)
    if not ok:
        print(" FAILED")
        if tool == "gzippy" and routing_trace is not None:
            head = routing_trace.get("stderr_head", "").strip()
            if head:
                print("      [preflight GZIPPY_DEBUG]")
                print("        " + head[:12000].replace("\n", "\n        "))
        if warmup_stderr.strip():
            print("      [warmup stderr]")
            print("        " + warmup_stderr[:12000].strip().replace("\n", "\n        "))
        return {"error": f"{tool} decompression failed on warmup", "tool": tool}

    # Verify correctness
    import filecmp
    if not filecmp.cmp(original_file, output_file, shallow=False):
        print(" INCORRECT OUTPUT")
        return {"error": f"{tool} decompression produced incorrect output", "tool": tool, "status": "fail"}

    # Surface the routing decision now that the warmup proved correctness.
    # When marker pipeline fell back, the throughput number reflects
    # libdeflate, not gzippy's parallel path — call that out explicitly so
    # the guard report doesn't read like "marker pipeline is slow."
    if tool == "gzippy" and threads > 1 and routing_trace is not None:
        if routing_trace["fell_back_to_sequential"] or not routing_trace["ran_marker_pipeline"]:
            print(" [SILENT FALLBACK: marker pipeline did not run end-to-end]", end="")
        # Always print the GZIPPY_DEBUG stderr after the bench line on
        # parallel runs — so a perf shortfall surfaces per-phase timing
        # (search/decode/retry/resolve/total) without needing to
        # download the JSON artifact.
        head = routing_trace.get("stderr_head", "").strip()
        if head:
            print(f"\n      [GZIPPY_DEBUG]\n        " + head.replace("\n", "\n        "))

    # Print rapidgzip's internal statistics (chunk count, pool efficiency).
    if tool == "rapidgzip" and rapidgzip_verbose:
        key_lines = [
            line.strip() for line in rapidgzip_verbose.splitlines()
            if any(k in line for k in [
                "Parallelization", "Total Fetched", "Prefetched", "On-demand",
                "decodeBlock", "futureWait", "Total Real Decode", "Theoretical Optimal",
                "Pool Efficiency", "Spent", "Decompressed",
            ])
        ]
        if key_lines:
            print(f"\n      [rapidgzip --verbose]\n        " + "\n        ".join(key_lines))
    
    # Benchmark
    times = []
    converged = False

    for trial in range(MAX_TRIALS):
        start = time.perf_counter()
        ok, trial_stderr = run_decompress(capture_stderr=(tool == "gzippy"))
        if not ok:
            if trial_stderr.strip():
                print("    [trial stderr]")
                print("      " + trial_stderr[:12000].strip().replace("\n", "\n      "))
            return {"error": f"{tool} decompression failed on trial {trial}", "tool": tool}
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if len(times) >= MIN_TRIALS:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            if cv < TARGET_CV:
                converged = True
                break

    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    cv = stdev / mean if mean > 0 else 0
    speed = original_size / median / 1_000_000

    # Per-trial raw times so variance spikes are visible.
    raw_s = "  ".join(f"{t:.3f}s" for t in times)
    times_sorted = sorted(times)
    p10 = times_sorted[len(times_sorted) // 10] if len(times_sorted) > 1 else times_sorted[0]
    p90 = times_sorted[int(len(times_sorted) * 0.9)] if len(times_sorted) > 1 else times_sorted[-1]
    fastest = original_size / min(times) / 1_000_000
    slowest = original_size / max(times) / 1_000_000

    print(f" {speed:.1f} MB/s ({len(times)} trials, CV={cv:.2%})")
    print(f"    raw: {raw_s}")
    print(f"    p10={fastest:.0f} MB/s  p50={speed:.0f} MB/s  p90={slowest:.0f} MB/s  "
          f"(min={min(times):.3f}s  max={max(times):.3f}s)")

    # For gzippy: capture one timed trial with GZIPPY_DEBUG to get per-chunk
    # breakdown without distorting the benchmark times above.
    per_chunk_debug = None
    if tool == "gzippy":
        env = dict(os.environ)
        env["GZIPPY_DEBUG"] = "1"
        with open(compressed_file, 'rb') as fin:
            dbg = subprocess.run(
                cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
            )
        per_chunk_debug = dbg.stderr.decode(errors="replace")
        # Print per-chunk breakdown lines (not the full warmup trace again).
        chunk_lines = [
            l
            for l in per_chunk_debug.splitlines()
            if "chunk " in l
            or "imbalance" in l
            or "per-chunk" in l
            or "[parallel_sm:v0.6] counters" in l
            or "[parallel_sm:v0.6] search=" in l
            or "[parallel_sm:v0.6] partition_outcomes" in l
            or "[parallel_sm:v0.6] swallowed" in l
        ]
        if chunk_lines:
            print("    per-chunk phase1b breakdown:")
            for line in chunk_lines:
                print(f"      {line.strip()}")
    
    return {
        "tool": tool,
        "operation": "decompress",
        "threads": threads,
        "times": times,
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "cv": cv,
        "trials": len(times),
        "converged": converged,
        "speed_mbps": speed,
        "fastest_mbps": fastest,
        "slowest_mbps": slowest,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": compressed_size / original_size,
        "status": "pass",
        "routing_trace": routing_trace,
        "per_chunk_debug": per_chunk_debug,
        "rapidgzip_verbose": rapidgzip_verbose,
    }


def main():
    parser = argparse.ArgumentParser(description="Single-member decompression benchmark")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--compressed-file", type=str, required=True,
                       help="Path to single-member gzip file")
    parser.add_argument("--original-file", type=str, required=True,
                       help="Path to original uncompressed file")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    binaries_dir = Path(args.binaries)
    
    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "unpigz": find_binary(binaries_dir, "unpigz"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "rapidgzip": find_binary(binaries_dir, "rapidgzip"),
        "gzip": "/usr/bin/gzip",
    }
    
    original_size = os.path.getsize(args.original_file)
    compressed_size = os.path.getsize(args.compressed_file)
    
    print("=" * 70)
    print("SINGLE-MEMBER DECOMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Original:   {original_size / 1_000_000:.1f} MB")
    print(f"Compressed: {compressed_size / 1_000_000:.1f} MB ({compressed_size * 100 / original_size:.1f}%)")
    print(f"Threads:    {args.threads}")
    print()
    print("This tests the 4-phase hyper-parallel pipeline:")
    print("  Phase 1: Window Boot (sequential)")
    print("  Phase 2: Speculative Parallel Decode")
    print("  Phase 3: Window Propagation + SIMD Marker Replacement")
    print("  Phase 4: Write Output")
    print()
    print("Tools to test:", [k for k, v in tools.items() if v])
    print("-" * 70)
    
    results = {
        "benchmark": "single-member",
        "threads": args.threads,
        "original_size_mb": original_size / 1_000_000,
        "compressed_size_mb": compressed_size / 1_000_000,
        "results": [],
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name
    
    try:
        # Benchmark each tool
        # Order matters: gzippy first, then competitors
        tool_order = ["gzippy", "rapidgzip", "unpigz", "igzip", "gzip"]
        
        for tool_name in tool_order:
            bin_path = tools.get(tool_name)
            if not bin_path:
                continue
            
            # Skip multi-threaded for single-threaded tools
            if tool_name in ("gzip", "igzip") and args.threads > 1:
                continue
            
            # Use unpigz instead of pigz for decompression
            if tool_name == "pigz" and tools.get("unpigz"):
                continue
            
            result = benchmark_decompress(
                tool_name, bin_path,
                args.compressed_file, output_file,
                args.original_file, args.threads
            )
            results["results"].append(result)
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Tool':<12} {'Speed (MB/s)':<15} {'Trials':<8} {'Status'}")
    print("-" * 70)
    
    gzippy_speed = None
    for r in results["results"]:
        if "error" in r:
            print(f"{r['tool']:<12} {'FAILED':<15} {'-':<8} ❌")
        else:
            status = "✅" if r["status"] == "pass" else "❌"
            print(f"{r['tool']:<12} {r['speed_mbps']:<15.1f} {r['trials']:<8} {status}")
            if r["tool"] == "gzippy":
                gzippy_speed = r["speed_mbps"]
    
    # Comparisons
    if gzippy_speed:
        print()
        print("GZIPPY vs COMPETITORS:")
        for r in results["results"]:
            if r["tool"] == "gzippy" or "error" in r:
                continue
            ratio = gzippy_speed / r["speed_mbps"]
            icon = "🏆" if ratio >= 1.0 else "📉"
            print(f"  {icon} vs {r['tool']}: {ratio:.2f}x")
    
    # Pass/fail determination
    passed = True
    reasons = []
    
    gzippy = next((r for r in results["results"] if r["tool"] == "gzippy" and "error" not in r), None)
    rapidgzip = next((r for r in results["results"] if r["tool"] == "rapidgzip" and "error" not in r), None)
    unpigz = next((r for r in results["results"] if r["tool"] == "unpigz" and "error" not in r), None)
    
    # Distinguish "marker pipeline ran but was slow" from "marker pipeline
    # never ran (silent fallback to libdeflate)". Both produce the same
    # throughput number; only the routing trace tells us which.
    if gzippy and args.threads > 1:
        trace = gzippy.get("routing_trace") or {}
        if trace.get("fell_back_to_sequential") or not trace.get("ran_marker_pipeline"):
            passed = False
            reasons.append(
                "marker pipeline did not run end-to-end on this fixture "
                "(silent fallback to sequential libdeflate). Throughput "
                "numbers reflect libdeflate, not the parallel path. "
                "See routing_trace.stderr_head in the JSON output."
            )

    # Universal goals — no hardware-class split. The bootstrap→ISA-L
    # handoff in `decode_chunk_with_handoff` matches rapidgzip's
    # per-chunk design (`vendor/rapidgzip/.../GzipChunk.hpp:413-657`):
    # the marker decoder bootstraps ≤32 KB per chunk, then ISA-L
    # handles the bulk at full single-thread ISA-L speed. There's no
    # structural per-thread gap to compensate for. If a runner can't
    # hit these ratios, the implementation has a regression, not the
    # threshold a hardware excuse.
    rapidgzip_threshold = 0.99
    unpigz_threshold = 1.0
    if gzippy and rapidgzip:
        ratio = gzippy["speed_mbps"] / rapidgzip["speed_mbps"]
        if ratio < rapidgzip_threshold:
            passed = False
            reasons.append(
                f"gzippy {ratio:.2f}x rapidgzip — below universal goal "
                f"of ≥{rapidgzip_threshold:.2f}"
            )

    if gzippy and unpigz:
        ratio = gzippy["speed_mbps"] / unpigz["speed_mbps"]
        if ratio < unpigz_threshold:
            passed = False
            reasons.append(
                f"gzippy {ratio:.2f}x unpigz — below universal goal "
                f"of ≥{unpigz_threshold:.2f}"
            )
    
    results["passed"] = passed
    results["reasons"] = reasons
    
    print()
    print("=" * 70)
    print(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    print("=" * 70)
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to {args.output}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
