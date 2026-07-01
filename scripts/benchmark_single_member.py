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


def build_decompress_cmd(tool: str, bin_path: str, threads: int) -> list | None:
    """Stdin→stdout decompress command for a tool (no file args)."""
    if tool == "gzippy":
        return [bin_path, "-d", f"-p{threads}"]
    elif tool == "pigz":
        return [bin_path, "-d", f"-p{threads}"]
    elif tool == "unpigz":
        return [bin_path, f"-p{threads}"]
    elif tool == "gzip":
        return [bin_path, "-d"]
    elif tool == "igzip":
        return [bin_path, "-d"]
    elif tool == "rapidgzip":
        return [bin_path, "-d", "-P", str(threads)]
    return None


def capture_routing_trace(cmd: list, compressed_file: str) -> dict:
    """gzippy routing audit (GZIPPY_DEBUG=1, untimed, to /dev/null). Run ONCE
    per tool BEFORE the timed interleave so it never perturbs the perf numbers.
    Without it, a silent fallback to sequential libdeflate looks identical to
    "marker pipeline ran but was slow" — both produce correct output at
    ~libdeflate throughput."""
    env = dict(os.environ)
    env["GZIPPY_DEBUG"] = "1"
    with open(compressed_file, 'rb') as fin:
        trace_result = subprocess.run(
            cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
        )
    stderr_text = trace_result.stderr.decode(errors="replace")
    ran_marker = "[parallel_sm:v0.6]" in stderr_text and "total=" in stderr_text
    # Typed-routing message added in PR #90 for the BTYPE=01-heavy case (see
    # premortem D1+D5). Older log strings retained for backward compatibility
    # with older gzippy binaries on the path.
    fell_back = (
        "parallel single-member fell back" in stderr_text
        or "parallel single-member failed" in stderr_text
    )
    return {
        "ran_marker_pipeline": ran_marker,
        "fell_back_to_sequential": fell_back,
        # Capture a larger window of stderr so the per-phase timing line
        # ("[parallel_sm:v0.6] search=Xms decode=Yms retry=Zms resolve=Wms
        # total=Tms ...") survives — that line is what tells us WHY the bench
        # is slow.
        "stderr_head": stderr_text[:8000],
    }


def capture_rapidgzip_verbose(bin_path: str, threads: int, compressed_file: str) -> str:
    """rapidgzip --verbose stats (chunk count, pool efficiency, decode times),
    captured ONCE per tool (untimed, to /dev/null) BEFORE the timed interleave."""
    verbose_cmd = [bin_path, "-d", "-P", str(threads), "--verbose"]
    with open(compressed_file, 'rb') as fin:
        vr = subprocess.run(
            verbose_cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    return vr.stderr.decode(errors="replace")


def prepare_tool(cmd: list, compressed_file: str, output_file: str,
                 original_file: str) -> str | None:
    """Warmup decode to a REAL file + filecmp correctness check (SINK LAW,
    Gate-0d): this is the ONLY decode that touches the disk. silesia-large is
    ~500 MB, so writing the output every timed trial dominated noise and landed
    disk-writeback stalls in the faster tool's window; the timed trials below
    write to /dev/null instead. Returns an error string, or None on success."""
    with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
        result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return "decompression failed on warmup"
    if not filecmp.cmp(original_file, output_file, shallow=False):
        return "decompression produced incorrect output"
    return None


def run_timed(cmd: list, compressed_file: str) -> tuple:
    """One timed decode to /dev/null (SINK LAW, Gate-0d). Returns (seconds, ok)."""
    with open(compressed_file, 'rb') as fin:
        start = time.perf_counter()
        result = subprocess.run(
            cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        elapsed = time.perf_counter() - start
    return elapsed, result.returncode == 0


def capture_per_chunk_debug(cmd: list, compressed_file: str) -> str:
    """gzippy per-chunk GZIPPY_DEBUG breakdown, captured ONCE (untimed, to
    /dev/null) AFTER timing so it doesn't distort the benchmark times."""
    env = dict(os.environ)
    env["GZIPPY_DEBUG"] = "1"
    with open(compressed_file, 'rb') as fin:
        dbg = subprocess.run(
            cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
        )
    return dbg.stderr.decode(errors="replace")


def finalize_stats(tool: str, times: list, converged: bool,
                   original_size: int, compressed_size: int, threads: int,
                   routing_trace, per_chunk_debug, rapidgzip_verbose) -> dict:
    """Build the per-tool result dict (schema unchanged) from its trial times."""
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    cv = stdev / mean if mean > 0 else 0
    speed = original_size / median / 1_000_000
    fastest = original_size / min(times) / 1_000_000
    slowest = original_size / max(times) / 1_000_000
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
        "timed_sink": "devnull",
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
    
    # Order matters: gzippy first, then competitors
    tool_order = ["gzippy", "rapidgzip", "unpigz", "igzip", "gzip"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Phase 1 — prepare each tool: untimed diagnostics (gzippy routing
        # trace / rapidgzip --verbose) captured ONCE outside the timed loop,
        # then a warmup decode to a real file + filecmp correctness. The ~500 MB
        # output is written ONCE here (SINK LAW, Gate-0d); failed/incorrect tools
        # are recorded as errors and excluded from timing. The valid set is then
        # timed INTERLEAVED (round-robin per trial, Gate-1) to /dev/null — the
        # old harness ran every trial of one tool then the next while writing
        # the full output to disk each trial, so disk-writeback stalls landed in
        # the faster tool's window and skewed the ratio.
        valid = []  # [(tool_name, cmd)]
        times_by_tool = {}
        diag_by_tool = {}  # tool_name -> {routing_trace, rapidgzip_verbose}
        for tool_name in tool_order:
            bin_path = tools.get(tool_name)
            if not bin_path:
                continue
            # Skip multi-threaded for single-threaded tools.
            if tool_name in ("gzip", "igzip") and args.threads > 1:
                continue
            # Use unpigz instead of pigz for decompression.
            if tool_name == "pigz" and tools.get("unpigz"):
                continue
            cmd = build_decompress_cmd(tool_name, bin_path, args.threads)
            if cmd is None:
                continue

            # Untimed per-tool diagnostics (captured ONCE, before timing).
            routing_trace = (
                capture_routing_trace(cmd, args.compressed_file)
                if tool_name == "gzippy" else None
            )
            rapidgzip_verbose = (
                capture_rapidgzip_verbose(bin_path, args.threads, args.compressed_file)
                if tool_name == "rapidgzip" else None
            )

            out_file = str(tmpdir / f"out-{tool_name}.bin")
            print(f"  {tool_name}: warming up...", end="", flush=True)
            err = prepare_tool(cmd, args.compressed_file, out_file, args.original_file)
            if err is not None:
                if "incorrect" in err:
                    print(" INCORRECT OUTPUT")
                    results["results"].append({
                        "error": f"{tool_name} {err}", "tool": tool_name,
                        "status": "fail",
                    })
                else:
                    print(" FAILED")
                    if tool_name == "gzippy" and routing_trace is not None:
                        head = routing_trace.get("stderr_head", "").strip()
                        if head:
                            print("      [preflight GZIPPY_DEBUG]")
                            print("        " + head[:12000].replace("\n", "\n        "))
                    results["results"].append({
                        "error": f"{tool_name} {err}", "tool": tool_name,
                    })
                continue
            print()  # newline after a successful warmup

            # Surface the routing decision now that warmup proved correctness.
            # When the marker pipeline fell back, the throughput reflects
            # libdeflate, not gzippy's parallel path — call it out explicitly.
            if tool_name == "gzippy" and args.threads > 1 and routing_trace is not None:
                if routing_trace["fell_back_to_sequential"] or not routing_trace["ran_marker_pipeline"]:
                    print("    [SILENT FALLBACK: marker pipeline did not run end-to-end]")
                head = routing_trace.get("stderr_head", "").strip()
                if head:
                    print(f"      [GZIPPY_DEBUG]\n        " + head.replace("\n", "\n        "))

            # Print rapidgzip's internal statistics (chunk count, pool efficiency).
            if tool_name == "rapidgzip" and rapidgzip_verbose:
                key_lines = [
                    line.strip() for line in rapidgzip_verbose.splitlines()
                    if any(k in line for k in [
                        "Parallelization", "Total Fetched", "Prefetched", "On-demand",
                        "decodeBlock", "futureWait", "Total Real Decode", "Theoretical Optimal",
                        "Pool Efficiency", "Spent", "Decompressed",
                    ])
                ]
                if key_lines:
                    print(f"      [rapidgzip --verbose]\n        " + "\n        ".join(key_lines))

            valid.append((tool_name, cmd))
            times_by_tool[tool_name] = []
            diag_by_tool[tool_name] = {
                "routing_trace": routing_trace,
                "rapidgzip_verbose": rapidgzip_verbose,
            }

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
                def _cv(ts):
                    m = statistics.mean(ts)
                    s = statistics.stdev(ts) if len(ts) > 1 else 0
                    return (s / m) if m > 0 else 1.0
                if all(_cv(times_by_tool[t]) < TARGET_CV for t, _ in valid):
                    converged = True
                    break

        # Phase 3 — finalize per-tool stats (schema unchanged) + untimed
        # per-chunk gzippy breakdown (captured AFTER timing).
        for tool_name, cmd in valid:
            times = times_by_tool[tool_name]
            diag = diag_by_tool[tool_name]
            per_chunk_debug = None
            if tool_name == "gzippy":
                per_chunk_debug = capture_per_chunk_debug(cmd, args.compressed_file)

            result = finalize_stats(
                tool_name, times, converged, original_size, compressed_size,
                args.threads, diag["routing_trace"], per_chunk_debug,
                diag["rapidgzip_verbose"],
            )

            raw_s = "  ".join(f"{t:.3f}s" for t in times)
            print(f"  {tool_name}: {result['speed_mbps']:.1f} MB/s "
                  f"({result['trials']} trials, CV={result['cv']:.2%})")
            print(f"    raw: {raw_s}")
            print(f"    p10={result['fastest_mbps']:.0f} MB/s  "
                  f"p50={result['speed_mbps']:.0f} MB/s  "
                  f"p90={result['slowest_mbps']:.0f} MB/s  "
                  f"(min={min(times):.3f}s  max={max(times):.3f}s)")

            if per_chunk_debug:
                chunk_lines = [
                    l for l in per_chunk_debug.splitlines()
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

            results["results"].append(result)
    
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
