#!/usr/bin/env python3
"""
Statistically-aware decompression bench.

Replaces single-trial bench-sm fragility with confidence-interval reasoning.

Runs N=20 trials per tool in INTERLEAVED random order (so thermal drift /
system load can't bias one tool over another), then bootstraps a 95% CI on
the gzippy/rapidgzip throughput ratio. Output answers "are we at parity
with vendor" with statistical rigor, not "did we get lucky this trial?"

Usage:
    bench_stats.py --compressed FILE --original FILE \\
        --gzippy PATH --rapidgzip PATH --unpigz PATH \\
        --trials 20 --threads 16 \\
        --output target/tooling/bench_stats.json

Output:
- JSON with per-tool times, median, p10/p90, IQR, bootstrap CI.
- Markdown to stdout (or --md output path).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------- Bench primitives ----------

def run_once(cmd: list[str], stdin_path: str, stdout_path: str) -> tuple[float, bool]:
    """Run one decompression trial. Returns (wall_seconds, ok)."""
    t0 = time.monotonic()
    with open(stdin_path, "rb") as fin, open(stdout_path, "wb") as fout:
        r = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
    elapsed = time.monotonic() - t0
    return elapsed, r.returncode == 0


def verify_byte_equivalent(produced: str, expected: str) -> bool:
    """Cheap byte-equality check via file size + first-mismatch detection."""
    if os.path.getsize(produced) != os.path.getsize(expected):
        return False
    chunk = 1 << 20
    with open(produced, "rb") as p, open(expected, "rb") as e:
        while True:
            a = p.read(chunk)
            b = e.read(chunk)
            if a != b:
                return False
            if not a:
                return True


def tool_cmd(tool: str, bin_path: str, threads: int) -> list[str]:
    """Argv for `decompress to stdout` per tool."""
    if tool == "gzippy":
        return [bin_path, "-d", "--processes", str(threads)]
    if tool == "rapidgzip":
        return [bin_path, "-d", "-P", str(threads)]
    if tool == "unpigz":
        # pigz -d -c is the canonical "decompress to stdout" command.
        return [bin_path, "-d", "-c"]
    raise ValueError(f"unknown tool: {tool}")


# ---------- Statistics ----------

def bootstrap_ratio_ci(
    a_samples: list[float],
    b_samples: list[float],
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Bootstrap a (alpha/2, 1-alpha/2) CI for median(b) / median(a).

    Use case: a = rapidgzip times, b = gzippy times. Ratio < 1 means gzippy
    is faster (lower time). For throughput ratio, invert.
    """
    rng = random.Random(0x9e3779b97f4a7c15)
    ratios: list[float] = []
    for _ in range(n_bootstrap):
        a_r = [a_samples[rng.randrange(len(a_samples))] for _ in a_samples]
        b_r = [b_samples[rng.randrange(len(b_samples))] for _ in b_samples]
        ma = statistics.median(a_r)
        mb = statistics.median(b_r)
        if mb > 0:
            ratios.append(ma / mb)
    ratios.sort()
    lo = ratios[int(len(ratios) * (alpha / 2))]
    hi = ratios[int(len(ratios) * (1 - alpha / 2))]
    return statistics.median(ratios), lo, hi


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "min": s[0],
        "max": s[-1],
        "median": s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2,
        "p10": s[int(n * 0.10)],
        "p90": s[int(n * 0.90)] if n > 1 else s[0],
        "mean": statistics.mean(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
        "iqr": s[int(n * 0.75)] - s[int(n * 0.25)] if n > 3 else 0.0,
        "cv": (statistics.stdev(s) / statistics.mean(s)) if n > 1 and statistics.mean(s) > 0 else 0.0,
    }


# ---------- Driver ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compressed", required=True, help="Path to compressed input")
    p.add_argument("--original", required=True, help="Path to uncompressed reference for byte-equality")
    p.add_argument("--gzippy", help="Path to gzippy binary")
    p.add_argument("--rapidgzip", help="Path to rapidgzip binary")
    p.add_argument("--unpigz", help="Path to pigz binary (alias used for unpigz)")
    p.add_argument("--trials", type=int, default=20, help="Trials per tool (default 20)")
    p.add_argument("--threads", type=int, default=16, help="--processes/-P for parallel tools")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per tool (default 1)")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--md", help="Optional markdown output path (stdout if omitted)")
    args = p.parse_args()

    tools = {}
    for name in ("gzippy", "rapidgzip", "unpigz"):
        bp = getattr(args, name)
        if bp and os.access(bp, os.X_OK):
            tools[name] = bp
    if not tools:
        print("ERROR: at least one --gzippy / --rapidgzip / --unpigz required", file=sys.stderr)
        return 2

    print(f"Bench: {args.compressed}", file=sys.stderr)
    print(f"  tools: {list(tools)}", file=sys.stderr)
    print(f"  trials: {args.trials}  threads: {args.threads}  warmup: {args.warmup}", file=sys.stderr)

    # Warmup: discarded.
    with tempfile.TemporaryDirectory() as tmp:
        for tool, bp in tools.items():
            for _ in range(args.warmup):
                out = os.path.join(tmp, f"{tool}.warmup.out")
                run_once(tool_cmd(tool, bp, args.threads), args.compressed, out)

        # Interleaved trial schedule: each tool gets `trials` runs, shuffled order.
        schedule: list[str] = []
        for tool in tools:
            schedule.extend([tool] * args.trials)
        random.Random(0xabad1dea).shuffle(schedule)

        results: dict[str, list[float]] = {tool: [] for tool in tools}
        correctness: dict[str, list[bool]] = {tool: [] for tool in tools}

        for i, tool in enumerate(schedule, 1):
            bp = tools[tool]
            out_path = os.path.join(tmp, f"{tool}.trial{i}.out")
            elapsed, ok = run_once(tool_cmd(tool, bp, args.threads), args.compressed, out_path)
            if ok:
                ok = verify_byte_equivalent(out_path, args.original)
            results[tool].append(elapsed)
            correctness[tool].append(ok)
            os.remove(out_path)
            print(f"  [{i:3d}/{len(schedule)}] {tool:12s} {elapsed*1000:7.1f}ms  {'ok' if ok else 'INCORRECT'}", file=sys.stderr)

    original_size = os.path.getsize(args.original)

    # Per-tool throughput samples (MB/s) — invert time, scale by uncompressed size.
    throughputs: dict[str, list[float]] = {
        tool: [original_size / t / 1e6 for t in times] for tool, times in results.items()
    }

    summary: dict[str, dict] = {}
    for tool, mbps in throughputs.items():
        ok_count = sum(correctness[tool])
        summary[tool] = {
            "time_seconds": summarize(results[tool]),
            "throughput_mbps": summarize(mbps),
            "correctness": {
                "trials": len(correctness[tool]),
                "ok": ok_count,
                "incorrect": len(correctness[tool]) - ok_count,
            },
        }

    # Bootstrap pairwise ratio CIs vs rapidgzip (the reference).
    ratios: dict[str, dict] = {}
    if "rapidgzip" in results:
        ref_times = results["rapidgzip"]
        for tool, times in results.items():
            if tool == "rapidgzip":
                continue
            # Throughput ratio = rapidgzip_time / tool_time
            median_ratio, lo, hi = bootstrap_ratio_ci(ref_times, times)
            ratios[tool] = {
                "ratio_to_rapidgzip_throughput": {
                    "median": median_ratio,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                },
                "interpretation": _interpret(median_ratio, lo, hi),
            }

    report = {
        "fixture": args.compressed,
        "original_size_bytes": original_size,
        "threads": args.threads,
        "trials_per_tool": args.trials,
        "summary": summary,
        "ratios": ratios,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    md = render_markdown(report)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    else:
        print(md)

    return 0


def _interpret(median: float, lo: float, hi: float) -> str:
    """Plain-English read on a 95% CI ratio (gzippy/rapidgzip throughput)."""
    if lo >= 0.99:
        return "AT PARITY OR BETTER (CI lower bound ≥ 0.99×)"
    if hi < 0.99:
        return f"BELOW PARITY (CI upper bound {hi:.2f}× < 0.99×)"
    return f"INDETERMINATE (CI spans 0.99×: {lo:.2f}–{hi:.2f}×, median {median:.2f}×)"


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Bench (statistically-aware)")
    lines.append("")
    lines.append(f"- Fixture: `{report['fixture']}`")
    lines.append(f"- Uncompressed: {report['original_size_bytes']/1e6:.1f} MB")
    lines.append(f"- Threads: {report['threads']}, trials per tool: {report['trials_per_tool']}")
    lines.append("")
    lines.append("## Per-tool throughput (MB/s)")
    lines.append("")
    lines.append("| Tool | median | p10 | p90 | mean | CV | trials ok |")
    lines.append("|---|---|---|---|---|---|---|")
    for tool, s in report["summary"].items():
        tp = s["throughput_mbps"]
        c = s["correctness"]
        lines.append(
            f"| {tool} | {tp['median']:.1f} | {tp['p10']:.1f} | {tp['p90']:.1f} | {tp['mean']:.1f} | {tp['cv']:.2%} | {c['ok']}/{c['trials']} |"
        )
    if report["ratios"]:
        lines.append("")
        lines.append("## Throughput ratio vs rapidgzip (95% bootstrap CI)")
        lines.append("")
        lines.append("| Tool | median ratio | 95% CI lo | 95% CI hi | verdict |")
        lines.append("|---|---|---|---|---|")
        for tool, r in report["ratios"].items():
            rr = r["ratio_to_rapidgzip_throughput"]
            lines.append(
                f"| {tool} | {rr['median']:.3f}× | {rr['ci95_lo']:.3f}× | {rr['ci95_hi']:.3f}× | {r['interpretation']} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
