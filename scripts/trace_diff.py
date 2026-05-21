#!/usr/bin/env python3
"""
Align gzippy's GZIPPY_LOG_FILE trace with rapidgzip --verbose output and
emit a per-phase timing delta.

gzippy emits JSON-lines events with `t_ns`, `ev`, `partition_idx`, etc.
rapidgzip --verbose prints free-form text with summary stats. We parse
each, normalize, and emit a side-by-side table.

Usage:
    trace_diff.py --gzippy-log /tmp/gzippy.trace \\
                  --rapidgzip-stderr /tmp/rapidgzip.stderr \\
                  --output target/tooling/trace_diff.json \\
                  --md docs/trace_diff.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------- gzippy log parsing ----------

def parse_gzippy(path: Path) -> dict:
    """Parse JSON-lines GZIPPY_LOG_FILE trace into a structured digest."""
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Aggregate per-chunk per-worker.
    chunk_decode_times: dict[int, list[float]] = defaultdict(list)
    chunk_apply_window_times: dict[int, list[float]] = defaultdict(list)
    consumer_phase_times: dict[str, list[float]] = defaultdict(list)
    decode_errors = 0
    spec_hits = 0
    spec_prefetches = 0

    for ev in events:
        kind = ev.get("ev", "")
        if kind in ("chunk_decode_done", "decode_ok"):
            dur = ev.get("duration_us")
            idx = ev.get("partition_idx", ev.get("chunk_idx", -1))
            if dur is not None:
                chunk_decode_times[idx].append(dur / 1000.0)  # ms
        elif kind == "apply_window_done":
            dur = ev.get("duration_us")
            idx = ev.get("partition_idx", -1)
            if dur is not None:
                chunk_apply_window_times[idx].append(dur / 1000.0)
        elif kind in ("decode_err", "iterate_err", "bootstrap_failed"):
            decode_errors += 1
        elif kind == "speculative_hit":
            spec_hits += 1
        elif kind == "speculative_prefetch":
            spec_prefetches += 1
        elif ev.get("thread", "").startswith("consumer"):
            consumer_phase_times[kind].append(ev.get("duration_us", 0) / 1000.0)

    n_chunks = len(chunk_decode_times)
    decode_total_ms = sum(sum(v) for v in chunk_decode_times.values())
    apply_window_total_ms = sum(sum(v) for v in chunk_apply_window_times.values())

    # Worker concurrency proxy: total decode CPU-ms / number of unique
    # chunks (≈ avg per-chunk parallel decode time).
    avg_per_chunk_decode_ms = decode_total_ms / max(n_chunks, 1)

    # Find the wall time: max t_ns - min t_ns.
    t_min = min((ev.get("t_ns", 0) for ev in events), default=0)
    t_max = max((ev.get("t_ns", 0) for ev in events), default=0)
    wall_ns = t_max - t_min

    return {
        "n_events": len(events),
        "n_chunks": n_chunks,
        "wall_ms": wall_ns / 1e6,
        "decode_total_cpu_ms": decode_total_ms,
        "apply_window_total_cpu_ms": apply_window_total_ms,
        "avg_per_chunk_decode_ms": avg_per_chunk_decode_ms,
        "decode_errors": decode_errors,
        "speculative_hits": spec_hits,
        "speculative_prefetches": spec_prefetches,
        "speculative_hit_rate": spec_hits / max(spec_prefetches, 1),
        "cpu_to_wall_ratio": decode_total_ms / max(wall_ns / 1e6, 1),
        "consumer_phase_ms_totals": {k: sum(v) for k, v in consumer_phase_times.items()},
    }


# ---------- rapidgzip --verbose parsing ----------

RAPIDGZIP_FIELDS = [
    # (display name, regex, group, type)
    ("total_real_decode_duration_s",
     r"Total Real Decode Duration\s*:\s*([\d.]+)\s*s", 1, float),
    ("theoretical_optimal_duration_s",
     r"Theoretical Optimal Duration\s*:\s*([\d.]+)\s*s", 1, float),
    ("pool_efficiency_pct",
     r"Pool Efficiency \(Fill Factor\)\s*:\s*([\d.]+)\s*%", 1, float),
    ("parallelization",
     r"Parallelization\s+:\s*(\d+)", 1, int),
    ("total_fetched",
     r"Total Fetched\s*:\s*(\d+)", 1, int),
    ("prefetched",
     r"Prefetched\s*:\s*(\d+)", 1, int),
    ("fetched_on_demand",
     r"Fetched On-demand\s*:\s*(\d+)", 1, int),
    ("decode_block_total_s",
     r"decodeBlock\s*:\s*([\d.]+)\s*s", 1, float),
    ("write_to_output_s",
     r"Spent\s+([\d.]+)\s*s writing to output", 1, float),
    ("decompressed_mbps",
     r"Decompressed in total \d+ B in [\d.]+\s*s\s*->\s*([\d.]+)\s*MB/s", 1, float),
]


def parse_rapidgzip(path: Path) -> dict:
    text = open(path).read()
    out: dict = {}
    for name, pattern, group, conv in RAPIDGZIP_FIELDS:
        m = re.search(pattern, text)
        if m:
            try:
                out[name] = conv(m.group(group))
            except (ValueError, IndexError):
                pass
    return out


# ---------- Diff ----------

def diff(gzippy: dict, rapidgzip: dict) -> dict:
    g_wall = gzippy["wall_ms"]
    r_wall = rapidgzip.get("total_real_decode_duration_s", 0) * 1000

    g_cpu = gzippy["decode_total_cpu_ms"]
    r_cpu = rapidgzip.get("decode_block_total_s", 0) * 1000

    pool_eff_gzippy = gzippy["cpu_to_wall_ratio"]
    pool_eff_rapidgzip = rapidgzip.get("pool_efficiency_pct", 0) / 100 * rapidgzip.get("parallelization", 1)

    return {
        "wall_ms": {
            "gzippy": g_wall,
            "rapidgzip": r_wall,
            "ratio_g_over_r": g_wall / r_wall if r_wall else None,
        },
        "decode_cpu_ms": {
            "gzippy": g_cpu,
            "rapidgzip": r_cpu,
            "ratio_g_over_r": g_cpu / r_cpu if r_cpu else None,
        },
        "effective_parallelism_x": {
            "gzippy": pool_eff_gzippy,
            "rapidgzip": pool_eff_rapidgzip,
        },
        "speculation": {
            "gzippy_hit_rate": gzippy["speculative_hit_rate"],
            "rapidgzip_prefetch_consumed_pct":
                ((rapidgzip.get("prefetched", 0) - rapidgzip.get("fetched_on_demand", 0))
                 / max(rapidgzip.get("prefetched", 1), 1)) if rapidgzip.get("prefetched") else None,
        },
        "decode_errors": {
            "gzippy": gzippy["decode_errors"],
            "rapidgzip": 0,
        },
        "verdict": _verdict(g_wall, r_wall, g_cpu, r_cpu, pool_eff_gzippy, pool_eff_rapidgzip),
    }


def _verdict(g_wall, r_wall, g_cpu, r_cpu, g_par, r_par) -> str:
    if not r_wall:
        return "no rapidgzip data"
    ratio_wall = g_wall / r_wall
    if ratio_wall <= 1.01:
        return f"AT PARITY: gzippy wall {g_wall:.0f}ms ≈ rapidgzip {r_wall:.0f}ms"
    parts = []
    parts.append(f"gzippy wall {g_wall:.0f}ms vs rapidgzip {r_wall:.0f}ms ({ratio_wall:.2f}× slower)")
    if r_cpu and g_cpu / r_cpu > 1.2:
        parts.append(f"DECODE OVERHEAD: gzippy CPU {g_cpu:.0f}ms vs rapidgzip {r_cpu:.0f}ms")
    if r_par > 1.2 * g_par:
        parts.append(f"PARALLELISM GAP: gzippy effective {g_par:.1f}× vs rapidgzip {r_par:.1f}×")
    return " — ".join(parts)


# ---------- Render ----------

def render_md(gzippy: dict, rapidgzip: dict, d: dict) -> str:
    lines: list[str] = []
    lines.append("# gzippy ↔ rapidgzip trace diff")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{d['verdict']}**")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | gzippy | rapidgzip | gzippy/rapidgzip |")
    lines.append("|---|---|---|---|")
    w = d["wall_ms"]
    lines.append(
        f"| Wall time (ms) | {w['gzippy']:.0f} | {w['rapidgzip']:.0f} "
        f"| {w['ratio_g_over_r']:.2f}× |" if w["ratio_g_over_r"] else "| Wall time | — | — | — |"
    )
    cpu = d["decode_cpu_ms"]
    lines.append(
        f"| Decode CPU summed (ms) | {cpu['gzippy']:.0f} | {cpu['rapidgzip']:.0f} "
        f"| {cpu['ratio_g_over_r']:.2f}× |" if cpu["ratio_g_over_r"] else "| Decode CPU | — | — | — |"
    )
    par = d["effective_parallelism_x"]
    lines.append(
        f"| Effective parallelism (×) | {par['gzippy']:.2f} | {par['rapidgzip']:.2f} | — |"
    )
    lines.append("")
    lines.append("## gzippy phase breakdown (CPU-ms)")
    lines.append("")
    lines.append("| event | total ms |")
    lines.append("|---|---|")
    for k, v in sorted(gzippy.get("consumer_phase_ms_totals", {}).items(), key=lambda kv: -kv[1]):
        if v > 1:
            lines.append(f"| {k} | {v:.1f} |")
    lines.append("")
    lines.append("## Speculation")
    lines.append("")
    sp = d["speculation"]
    lines.append(f"- gzippy speculation hit rate: **{sp['gzippy_hit_rate']:.1%}**")
    if sp["rapidgzip_prefetch_consumed_pct"] is not None:
        lines.append(f"- rapidgzip prefetch consumption: **{sp['rapidgzip_prefetch_consumed_pct']:.1%}**")
    lines.append("")
    lines.append(f"- gzippy decode errors: {d['decode_errors']['gzippy']}")
    lines.append("")
    lines.append("## Raw")
    lines.append("")
    lines.append("```")
    lines.append("gzippy: " + json.dumps(gzippy, indent=2))
    lines.append("rapidgzip: " + json.dumps(rapidgzip, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gzippy-log", required=True, help="GZIPPY_LOG_FILE path")
    p.add_argument("--rapidgzip-stderr", required=True, help="rapidgzip --verbose stderr capture")
    p.add_argument("--output", required=True, help="JSON output")
    p.add_argument("--md", help="Markdown output (stdout if omitted)")
    args = p.parse_args()

    g = parse_gzippy(Path(args.gzippy_log))
    r = parse_rapidgzip(Path(args.rapidgzip_stderr))
    d = diff(g, r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"gzippy": g, "rapidgzip": r, "diff": d}, f, indent=2)

    md = render_md(g, r, d)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md)
        print(f"Markdown written to {args.md}", file=sys.stderr)
    else:
        print(md)
    print(f"Verdict: {d['verdict']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
