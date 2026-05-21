#!/usr/bin/env python3
"""
Deterministic per-thread timeline from gzippy GZIPPY_LOG_FILE trace.

Reads the JSON-lines event log produced by `src/decompress/parallel/trace.rs`
and emits:

- ASCII Gantt chart showing each thread's state over wall time.
- Critical-path analysis: which chunk's completion determined the wall.
- Per-thread summaries: worker decode/idle time, consumer apply_window/wait
  time, utilization percentages.
- Buffer-pool stats: cache-hits vs cache-misses on the lock-free recycle.

Usage:
    timeline.py --log /tmp/gzippy.trace.jsonl \\
                [--width 100]          # characters wide for ASCII chart
                [--output timeline.md] # markdown summary; stdout if omitted

Why this exists: perf record samples at 999 Hz and misses sub-ms events
in a parallel pipeline. This consumes the deterministic per-event trace
to give ground-truth on where the wall time went per thread.
See docs/PARALLEL_PROFILING_PLAN.md §1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    t_ns: int
    thread: str
    ev: str
    body: dict


def parse_trace(path: Path) -> list[Event]:
    events: list[Event] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = j.pop("t_ns", 0)
            thread = j.pop("thread", "?")
            ev = j.pop("ev", "?")
            events.append(Event(t_ns=t, thread=thread, ev=ev, body=j))
    return events


@dataclass
class ThreadSpan:
    """Closed interval [start_us, end_us] in a thread's wall time."""
    start_us: int
    end_us: int
    label: str          # ascii char to render
    info: str           # human-readable note


@dataclass
class ThreadTrack:
    name: str
    spans: list[ThreadSpan] = field(default_factory=list)


def build_tracks(events: list[Event]) -> tuple[dict[str, ThreadTrack], int, int]:
    """
    Group events by thread, infer span boundaries.

    For each worker thread:
      span_decode: chunk_decode_done event → end = ev.t_ns, duration_us=ev.duration_us
      span_idle: worker_pull_done's pull_us > 1ms → idle bracket
      span_worker_exit: worker_idle event terminates the track.
    For the consumer:
      span_wait: consumer_wait_done.wait_us
      span_apply_window: apply_window_done.duration_us
      span_write: chunk_consume_done.hand_off_us (approx)
    """
    if not events:
        return {}, 0, 0
    t_min = min(e.t_ns for e in events)
    t_max = max(e.t_ns for e in events)

    tracks: dict[str, ThreadTrack] = defaultdict(lambda: ThreadTrack(name=""))

    for ev in events:
        thread = ev.thread
        if thread not in tracks:
            tracks[thread] = ThreadTrack(name=thread)
        tr = tracks[thread]
        t_us = (ev.t_ns - t_min) // 1000

        if ev.ev == "chunk_decode_done":
            dur_us = ev.body.get("duration_us", 0)
            tr.spans.append(ThreadSpan(
                start_us=t_us - dur_us,
                end_us=t_us,
                label="=",
                info=f"decode chunk {ev.body.get('partition_idx')} ({dur_us}us, path={ev.body.get('path')})",
            ))
        elif ev.ev == "decode_err":
            dur_us = ev.body.get("duration_us", 0)
            tr.spans.append(ThreadSpan(
                start_us=t_us - dur_us,
                end_us=t_us,
                label="X",
                info=f"decode_err chunk {ev.body.get('partition_idx')} cand={ev.body.get('cand_idx')}",
            ))
        elif ev.ev == "worker_pull_done":
            pull_us = ev.body.get("pull_us", 0)
            if pull_us > 1000:
                tr.spans.append(ThreadSpan(
                    start_us=t_us - pull_us,
                    end_us=t_us,
                    label="_",
                    info=f"idle pull chunk {ev.body.get('partition_idx')} ({pull_us}us)",
                ))
        elif ev.ev == "worker_idle":
            tr.spans.append(ThreadSpan(
                start_us=t_us,
                end_us=t_us,
                label="z",
                info=f"worker exit (processed {ev.body.get('chunks_processed')} chunks)",
            ))
        elif ev.ev == "consumer_wait_done":
            wait_us = ev.body.get("wait_us", 0)
            if wait_us > 100:
                tr.spans.append(ThreadSpan(
                    start_us=t_us - wait_us,
                    end_us=t_us,
                    label="W",
                    info=f"wait chunk {ev.body.get('partition_idx')} ({wait_us}us)",
                ))
        elif ev.ev == "apply_window_done":
            dur_us = ev.body.get("duration_us", 0)
            tr.spans.append(ThreadSpan(
                start_us=t_us - dur_us,
                end_us=t_us,
                label="A",
                info=f"apply_window chunk {ev.body.get('partition_idx')} ({dur_us}us)",
            ))
        elif ev.ev == "chunk_consume_done":
            dur_us = ev.body.get("hand_off_us") or ev.body.get("narrow_and_write_us", 0)
            if dur_us > 0:
                tr.spans.append(ThreadSpan(
                    start_us=t_us - dur_us,
                    end_us=t_us,
                    label="w",
                    info=f"write chunk {ev.body.get('partition_idx')} ({dur_us}us)",
                ))

    return tracks, t_min, t_max


def render_ascii(tracks: dict[str, ThreadTrack], t_min: int, t_max: int, width: int) -> str:
    wall_us = (t_max - t_min) // 1000
    if wall_us <= 0:
        return "(empty trace)"
    us_per_col = max(1, wall_us // width)
    lines: list[str] = []
    lines.append(f"Wall time: {wall_us:>6} us ({wall_us/1000:.1f} ms), {us_per_col} us/col, {width} cols")
    lines.append("─" * (16 + width + 1))

    def name_key(n: str) -> tuple:
        # worker-0, worker-1, ..., worker-N, then consumer last.
        if n.startswith("worker-"):
            try:
                return (0, int(n.split("-", 1)[1]))
            except ValueError:
                return (0, 99)
        if n == "consumer":
            return (1, 0)
        return (2, 0)

    for name in sorted(tracks, key=name_key):
        tr = tracks[name]
        row = [" "] * width
        for span in tr.spans:
            start_col = max(0, span.start_us // us_per_col)
            end_col = max(start_col, min(width - 1, span.end_us // us_per_col))
            for c in range(start_col, end_col + 1):
                # Highest-precedence char wins so multi-event chars don't get masked.
                cur = row[c]
                if cur == " " or cur == "_":
                    row[c] = span.label
        lines.append(f"{name:<14}  |{''.join(row)}|")
    return "\n".join(lines)


def compute_summary(events: list[Event]) -> dict:
    """Aggregate per-event-kind totals, including continuous-coverage
    phase events (Gap 8) and bootstrap-vs-ISA-L split (Gap 1)."""
    decode_us_per_worker: dict[str, int] = defaultdict(int)
    decode_err_count: dict[str, int] = defaultdict(int)
    worker_lifetimes: dict[str, int] = {}
    chunks_per_worker: dict[str, int] = {}
    buf_pool_pop_hits = 0
    buf_pool_pop_misses = 0
    buf_pool_pushes = 0
    # Gap 8: phase events per thread, summed.
    phase_us_per_thread_phase: dict[tuple[str, str], int] = defaultdict(int)
    # Gap 1: bootstrap vs ISA-L per chunk.
    bootstrap_us_total = 0
    isal_us_total = 0
    isal_bytes_total = 0
    chunks_with_bootstrap = 0
    chunks_bootstrap_only = 0
    # Gap 3: per-chunk resource use.
    minflt_per_worker: dict[str, int] = defaultdict(int)
    majflt_per_worker: dict[str, int] = defaultdict(int)
    vcsw_per_worker: dict[str, int] = defaultdict(int)
    ivcsw_per_worker: dict[str, int] = defaultdict(int)

    for ev in events:
        if ev.ev == "chunk_decode_done":
            decode_us_per_worker[ev.thread] += ev.body.get("duration_us", 0)
        elif ev.ev == "decode_err":
            decode_err_count[ev.thread] += 1
        elif ev.ev == "worker_idle":
            worker_lifetimes[ev.thread] = ev.body.get("worker_lifetime_us", 0)
            chunks_per_worker[ev.thread] = ev.body.get("chunks_processed", 0)
        elif ev.ev == "buffer_pool_pop":
            if ev.body.get("was_empty"):
                buf_pool_pop_misses += 1
            else:
                buf_pool_pop_hits += 1
        elif ev.ev == "buffer_pool_push":
            buf_pool_pushes += 1
        elif ev.ev == "phase":
            phase = ev.body.get("phase", "?")
            phase_us_per_thread_phase[(ev.thread, phase)] += ev.body.get("duration_us", 0)
        elif ev.ev == "chunk_decode_split":
            bootstrap_us_total += ev.body.get("bootstrap_us", 0)
            isal_us_total += ev.body.get("isal_us", 0)
            isal_bytes_total += ev.body.get("isal_bytes", 0)
            chunks_with_bootstrap += 1
            if ev.body.get("isal_us", 0) == 0:
                chunks_bootstrap_only += 1
        elif ev.ev == "chunk_resource_done":
            minflt_per_worker[ev.thread] += ev.body.get("minflt", 0)
            majflt_per_worker[ev.thread] += ev.body.get("majflt", 0)
            vcsw_per_worker[ev.thread] += ev.body.get("vcsw", 0)
            ivcsw_per_worker[ev.thread] += ev.body.get("ivcsw", 0)

    return {
        "decode_us_per_worker": dict(decode_us_per_worker),
        "decode_err_count": dict(decode_err_count),
        "worker_lifetimes_us": worker_lifetimes,
        "chunks_per_worker": chunks_per_worker,
        "buffer_pool_pop_hits": buf_pool_pop_hits,
        "buffer_pool_pop_misses": buf_pool_pop_misses,
        "buffer_pool_pushes": buf_pool_pushes,
        "phase_us_per_thread_phase": {f"{t}:{p}": v
                                      for (t, p), v in phase_us_per_thread_phase.items()},
        "bootstrap_us_total": bootstrap_us_total,
        "isal_us_total": isal_us_total,
        "isal_bytes_total": isal_bytes_total,
        "chunks_with_bootstrap": chunks_with_bootstrap,
        "chunks_bootstrap_only": chunks_bootstrap_only,
        "minflt_per_worker": dict(minflt_per_worker),
        "majflt_per_worker": dict(majflt_per_worker),
        "vcsw_per_worker": dict(vcsw_per_worker),
        "ivcsw_per_worker": dict(ivcsw_per_worker),
    }


def compute_coverage(events: list[Event]) -> dict:
    """Gap 8 — per-thread coverage = sum(phase.duration_us) / thread_lifetime.

    Thread lifetime is from the earliest STARTING-edge of any event
    (= event.t_ns minus its own duration_us if it has one) to the
    latest t_ns. A `phase` event with duration_us=N completes at
    timestamp T and started at T-N — so the thread's effective start
    is min(T - duration_us, other_event.t_ns). This avoids the
    double-count where the first phase's duration overshoots the
    naive (max-min t_ns) window.

    Goal: coverage ≥ 99% per thread.

    Threads with fewer than 2 phase events (e.g. the logical "decode"
    sink that only carries chunk_decode_split emits, not phases) are
    excluded — they're not real OS threads, just labels for grouping
    related events.
    """
    thread_min_start: dict[str, int] = {}
    thread_last_t: dict[str, int] = {}
    thread_phase_us: dict[str, int] = defaultdict(int)
    thread_phase_count: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.ev == "phase":
            dur_ns = ev.body.get("duration_us", 0) * 1000
            start_t = ev.t_ns - dur_ns
            cur = thread_min_start.get(ev.thread, start_t)
            thread_min_start[ev.thread] = min(cur, start_t)
            thread_phase_us[ev.thread] += ev.body.get("duration_us", 0)
            thread_phase_count[ev.thread] += 1
        else:
            cur = thread_min_start.get(ev.thread, ev.t_ns)
            thread_min_start[ev.thread] = min(cur, ev.t_ns)
        thread_last_t[ev.thread] = max(thread_last_t.get(ev.thread, 0), ev.t_ns)
    out: dict[str, dict] = {}
    for t in thread_min_start:
        wall_us = (thread_last_t[t] - thread_min_start[t]) // 1000
        covered_us = thread_phase_us[t]
        coverage = (covered_us / wall_us) if wall_us > 0 else 1.0
        # Exclude logical (non-thread) sinks that carry no phases.
        is_logical = thread_phase_count[t] == 0
        out[t] = {
            "wall_us": wall_us,
            "covered_us": covered_us,
            "coverage_pct": coverage * 100,
            "is_logical": is_logical,
            "ok": is_logical or coverage >= 0.99 or wall_us < 1000,
        }
    return out


def render_summary(summary: dict, wall_us: int, coverage: dict) -> str:
    lines: list[str] = []
    lines.append(f"\n## Summary (wall {wall_us/1000:.1f} ms)\n")
    decode_total = sum(summary["decode_us_per_worker"].values())
    n_workers = len(summary["decode_us_per_worker"]) or 1
    lines.append(f"- Workers: {n_workers}")
    lines.append(f"- Worker decode (CPU summed):  {decode_total/1000:>7.1f} ms")
    lifetime_total = sum(summary["worker_lifetimes_us"].values())
    if lifetime_total > 0 and decode_total > 0:
        lines.append(f"- Worker lifetime (CPU summed): {lifetime_total/1000:>7.1f} ms")
        if wall_us > 0:
            lines.append(
                f"- Effective parallelism: **{decode_total / wall_us:.2f}×** "
                f"(perfect = {n_workers}×)"
            )
    lines.append("")
    lines.append("### Phase breakdown (continuous-coverage, sum per thread)")
    lines.append("")
    phase_per_thread: dict[str, dict[str, int]] = defaultdict(dict)
    for k, v in summary["phase_us_per_thread_phase"].items():
        t, p = k.split(":", 1)
        phase_per_thread[t][p] = v
    for thread in sorted(phase_per_thread):
        lines.append(f"- **{thread}**:")
        for phase, us in sorted(phase_per_thread[thread].items(), key=lambda kv: -kv[1]):
            lines.append(f"    - {phase}: {us/1000:>7.1f} ms")
    lines.append("")
    lines.append("### Bootstrap vs ISA-L split (Gap 1)")
    lines.append("")
    boot_us = summary["bootstrap_us_total"]
    isal_us = summary["isal_us_total"]
    n_split = summary["chunks_with_bootstrap"]
    n_boot_only = summary["chunks_bootstrap_only"]
    if n_split > 0:
        lines.append(
            f"- Chunks with bootstrap (Rust marker decoder): **{n_split}**"
            f" (of which {n_boot_only} bootstrap-only, no ISA-L handoff)"
        )
        lines.append(f"- Bootstrap CPU summed: {boot_us/1000:>7.1f} ms ({boot_us/n_split/1000:.1f} ms avg/chunk)")
        if isal_us > 0:
            lines.append(f"- ISA-L CPU summed:    {isal_us/1000:>7.1f} ms ({isal_us/n_split/1000:.1f} ms avg/chunk)")
            lines.append(f"- Bootstrap share of split-chunk decode: **{boot_us/(boot_us+isal_us):.1%}**")
        bytes_out = summary["isal_bytes_total"]
        if isal_us > 0 and bytes_out > 0:
            mbps = bytes_out / isal_us
            lines.append(f"- ISA-L throughput (summed): {mbps:.1f} MB/s")
    else:
        lines.append("- No chunks took the bootstrap path (chunk 0 fast-path only).")
    lines.append("")
    lines.append("### Buffer pool")
    hits = summary["buffer_pool_pop_hits"]
    misses = summary["buffer_pool_pop_misses"]
    pushes = summary["buffer_pool_pushes"]
    total = hits + misses
    hit_rate = (hits / total) if total else 0
    lines.append(f"- Pop hits: {hits}, pop misses (had to alloc fresh): {misses}, hit rate: **{hit_rate:.1%}**")
    lines.append(f"- Pushes (recycle returns): {pushes}")
    lines.append("")
    err_total = sum(summary["decode_err_count"].values())
    if err_total:
        lines.append(f"- Decode errors (workers retrying candidates): {err_total}")
    lines.append("")
    lines.append("### Per-thread resource use (Gap 3, Linux RUSAGE_THREAD)")
    lines.append("")
    minflt_total = sum(summary["minflt_per_worker"].values())
    vcsw_total = sum(summary["vcsw_per_worker"].values())
    ivcsw_total = sum(summary["ivcsw_per_worker"].values())
    if minflt_total > 0:
        lines.append(f"- Minor page faults (page commits) summed: **{minflt_total}**"
                     f" ({minflt_total * 4 / 1024:.1f} MiB committed)")
        lines.append(f"- Voluntary context switches: {vcsw_total}")
        lines.append(f"- Involuntary context switches: {ivcsw_total}")
        for thread in sorted(summary["minflt_per_worker"]):
            mf = summary["minflt_per_worker"][thread]
            v = summary["vcsw_per_worker"][thread]
            iv = summary["ivcsw_per_worker"][thread]
            lines.append(f"    - {thread}: minflt={mf}, vcsw={v}, ivcsw={iv}")
    else:
        lines.append("- (no per-chunk resource events — not on Linux or feature disabled)")
    lines.append("")
    lines.append("### Wall-time coverage (Gap 8 — should be ≥99% per thread)")
    lines.append("")
    lines.append("| thread | wall (ms) | covered (ms) | coverage |")
    lines.append("|---|---|---|---|")
    all_ok = True
    for thread in sorted(coverage):
        c = coverage[thread]
        if c["is_logical"]:
            flag = "(logical sink — no phase events expected)"
            lines.append(f"| {thread} | — | — | {flag} |")
            continue
        flag = "✓" if c["ok"] else "**INCOMPLETE**"
        lines.append(f"| {thread} | {c['wall_us']/1000:.1f} | {c['covered_us']/1000:.1f} | {c['coverage_pct']:.1f}% {flag} |")
        if not c["ok"]:
            all_ok = False
    lines.append("")
    if all_ok:
        lines.append("**✓ Coverage check PASSED** — trace accounts for ≥99% of every thread's wall.")
    else:
        lines.append("**✗ Coverage check FAILED** — some threads have unattributed wall time. "
                     "Conclusions about bottlenecks may be wrong by up to the unattributed remainder.")
    return "\n".join(lines)


def critical_path_chunk(events: list[Event]) -> int | None:
    """The chunk whose consumer pool_push lap sits latest in wall time
    is the one that determined the wall — last chunk to fully consume.

    Uses `phase` events with phase="pool_push" (= consumer chunk-done).
    Falls back to chunk_decode_done if pool_push events aren't present
    (older trace formats).
    """
    latest_t = -1
    latest_idx = None
    for ev in events:
        if ev.ev == "phase" and ev.body.get("phase") == "pool_push":
            if ev.t_ns > latest_t:
                latest_t = ev.t_ns
                latest_idx = ev.body.get("partition_idx")
    if latest_idx is None:
        for ev in events:
            if ev.ev == "chunk_decode_done":
                if ev.t_ns > latest_t:
                    latest_t = ev.t_ns
                    latest_idx = ev.body.get("partition_idx")
    return latest_idx


def render_critical(events: list[Event]) -> str:
    chunk_idx = critical_path_chunk(events)
    if chunk_idx is None:
        return "\n## Critical path: no chunks completed\n"
    # Find decode duration + consumer recv_wait for this chunk.
    decode_us = None
    wait_us = None
    bootstrap_us = None
    isal_us = None
    for ev in events:
        if ev.ev == "chunk_decode_done" and ev.body.get("partition_idx") == chunk_idx:
            decode_us = ev.body.get("duration_us", 0)
        if ev.ev == "phase" and ev.body.get("phase") == "recv_wait" and ev.body.get("partition_idx") == chunk_idx:
            wait_us = ev.body.get("duration_us", 0)
        if ev.ev == "chunk_decode_split":
            # Per-chunk split events aren't keyed by partition_idx but by encoded_offset_bits.
            # Match the LAST one (chunks are processed in order).
            bootstrap_us = ev.body.get("bootstrap_us", bootstrap_us)
            isal_us = ev.body.get("isal_us", isal_us)
    out = ["\n## Critical path"]
    out.append(f"- Last-completed chunk (decided wall): **{chunk_idx}**")
    if decode_us is not None:
        out.append(f"- Its decode took: {decode_us/1000:.1f} ms")
    if bootstrap_us is not None and isal_us is not None:
        out.append(f"- Bootstrap (Rust marker decoder): {bootstrap_us/1000:.1f} ms")
        out.append(f"- ISA-L bulk:                    {isal_us/1000:.1f} ms")
    if wait_us is not None:
        out.append(f"- Consumer waited for its arrival: {wait_us/1000:.1f} ms")
        if decode_us is not None and wait_us > 0.5 * decode_us:
            out.append("- **Interpretation**: consumer was BLOCKED on this chunk's worker. "
                       "Decode time IS the bottleneck — speed up this chunk's path.")
        elif wait_us < 1000:
            out.append("- **Interpretation**: consumer-side work was the bottleneck "
                       "(consumer was still processing prior chunks when this one arrived).")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="GZIPPY_LOG_FILE path")
    p.add_argument("--width", type=int, default=120, help="ASCII chart width")
    p.add_argument("--output", help="markdown output (stdout if omitted)")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any thread's coverage < 99 percent")
    args = p.parse_args()

    events = parse_trace(Path(args.log))
    if not events:
        print("(no events)", file=sys.stderr)
        return 1
    tracks, t_min, t_max = build_tracks(events)
    wall_us = (t_max - t_min) // 1000

    out_lines: list[str] = []
    out_lines.append(f"# Parallel timeline — {args.log}\n")
    out_lines.append("```")
    out_lines.append(render_ascii(tracks, t_min, t_max, args.width))
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("Legend: `=` decoding, `_` pull-idle, `W` consumer wait, "
                     "`A` apply_window, `w` write/hand-off, `X` decode error, "
                     "`z` worker exit")
    coverage = compute_coverage(events)
    out_lines.append(render_summary(compute_summary(events), wall_us, coverage))
    out_lines.append(render_critical(events))
    md = "\n".join(out_lines)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)
    # Exit code: non-zero if coverage check failed (and --strict given).
    if args.strict:
        failed = [t for t, c in coverage.items() if not c["ok"]]
        if failed:
            print(f"COVERAGE FAILED on threads: {failed}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
