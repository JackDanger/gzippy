#!/usr/bin/env python3
"""parallel_sm_tail_metric.py — CONSERVATION-GATED per-thread busy/idle timeline
reducer for the GZIPPY_TIMELINE Chrome-trace, used to LOCATE where gzippy idles
~0.25 cores vs rapidgzip on silesia-T4 (STEP-1, kernel-converge-A).

WHY THIS WAS REPAIRED (the reconciliation it resolves)
------------------------------------------------------
The earlier version reported only `effcores = Σ(worker.decode_chunk SPAN) / wall`.
On silesia-T4 that is ~3.25/4. But STEP-0's perf `CPUs-utilized` for the SAME
decode is ~2.46/4. Those two cannot both be "cores busy": you cannot occupy 3.25
cores of decode SPANS while consuming only 2.46 cores of CPU. The decode_chunk
SPAN is WALL time — it includes any time the worker is parked/descheduled *inside*
the span (off-CPU stall) that perf's task-clock does NOT count. The ~0.8-core gap
between the SPAN lens and the CPU lens is the located idle signal.

This reducer therefore decomposes EVERY microsecond of EVERY thread's life into
three classes (no double-count: leaf/innermost-span attribution via a B/E stack):

  BUSY  — on-CPU work spans (decode_chunk self + its block_header/body children,
          apply_window, consumer book-keeping, writev, crc, coordination, ...).
  WAIT  — OFF-CPU blocking spans the trace explicitly marks:
            wait.future_recv          (consumer blocked on a not-ready chunk)
            wait.block_fetcher_get     (consumer blocked fetching head chunk)
              + child ttp.rx_recv_block (the actual recv; counted via its parent)
            pool.pick.wait             (worker parked on the task condvar)
            consumer.wait_replaced_markers
  GAP   — wall on a thread covered by NO span (begin/end truncation, or untraced
          off-CPU time the instrument does not name).

CONSERVATION (per-tid, EXACT by construction, then GATED):
  For every tid:  Σ(leaf BUSY) + Σ(leaf WAIT) + GAP  ==  (last_ts − first_ts).
  This always closes (GAP is the remainder) — the GATE is that GAP must be SMALL
  (the trace actually covers each thread's life) AND there are no end<start spans
  AND no worker decodes two chunks at once (mis-paired B/E). If GAP is large the
  busy/idle split is meaningless and we REFUSE to print a verdict.

CORE-BUDGET (the "Σ busy + Σ idle == wall × ncores" closure the plan asks for):
  budget        = drive_wall × T
  busy          = Σ BUSY over all tids
  located_idle  = Σ WAIT over all tids   (the idle we can NAME)
  unaccounted   = budget − busy − located_idle
                = spare cores never given work + per-thread GAP
  We print all four; the gate is that `busy` reproduces perf CPUs-utilized.

THE HONEST RECONCILIATION (Gate, when --perf-cpus is given):
  effcores_span = Σ(top-level worker.decode_chunk WALL) / drive_wall   (the OLD lens)
  effcores_cpu  = busy / drive_wall                                    (the NEW lens)
  If effcores_cpu ≈ --perf-cpus (perf task-clock/wall of the SAME run), the rig is
  honest and (effcores_span − effcores_cpu) is the IN-SPAN off-CPU stall — the
  located idle that lives INSIDE decode_chunk (distributed), distinct from the
  cross-chunk drain waits (wait.future_recv / pool.pick.wait).

NON-INERT cross-checks:
  * >0 worker.decode_chunk AND >0 wait.future_recv spans (the parallel consumer
    path actually fired);
  * --expected-chunks N  → distinct top-level decode_chunk spans == N
    (cross-check vs rg --verbose "Blocks Total Fetched" / ceil(comp/chunk)).

USAGE:
  parallel_sm_tail_metric.py TRACE.json [--expected-chunks N] [--perf-cpus X.XXX]
       [--gap-tol 0.08] [--json] [--label STR] [--sink /dev/null]
EXIT: 0 if every BLOCKING gate passes, 3 otherwise (the numbers do not exist).
"""
import argparse
import json
import math
import sys

TOL = 0.05  # 5% slack for float/scheduling jitter in span-window checks

# OFF-CPU blocking spans the trace explicitly marks. Everything NOT here (and not
# a pure umbrella of these) is treated as on-CPU BUSY work.
WAIT_NAMES = {
    "wait.future_recv",
    "wait.block_fetcher_get",
    "ttp.rx_recv_block",  # the blocking recv nested in wait.block_fetcher_get
    "ttp.take_prefetch",  # tiny; nested under the same wait
    "pool.pick.wait",     # worker parked on the task condvar (cross-chunk idle)
    "consumer.wait_replaced_markers",
}
# Sub-classes of located idle (by leaf name) for the localization verdict.
IDLE_CLASS = {
    "wait.future_recv": "consumer_cross_chunk (future_recv)",
    "wait.block_fetcher_get": "consumer_serial (block_fetcher_get)",
    "ttp.rx_recv_block": "consumer_serial (block_fetcher_get)",
    "ttp.take_prefetch": "consumer_serial (block_fetcher_get)",
    "pool.pick.wait": "worker_cross_chunk (pool.pick.wait)",
    "consumer.wait_replaced_markers": "consumer_marker_wait",
}


def is_wait(name):
    return name in WAIT_NAMES


def parse_events(path):
    """Lenient parse of the partial Chrome-trace array. Returns (events, bad)."""
    events = []
    bad = 0
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s == "[" or s == "]":
                continue
            if s.endswith(","):
                s = s[:-1]
            if not s.endswith("}"):
                bad += 1
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
    return events, bad


def per_tid_timeline(events_for_tid):
    """Stack machine over one tid's B/E events (already ts-sorted).

    Returns:
      leaf_excl: {name: microseconds where this name was the INNERMOST span}
      spans:     list of (name, start, end, args, has_decode_ancestor) for ALL
                 matched B/E pairs (used for top-level decode spans + per-chunk).
      anomalies: count of unmatched E / end<start.
      first_ts, last_ts: thread coverage bounds (over B/E/i events).
    """
    leaf_excl = {}
    spans = []
    anomalies = 0
    stack = []  # list of [name, start, args]
    last_change = None
    first_ts = None
    last_ts = None
    for e in events_for_tid:
        ph = e.get("ph")
        ts = float(e.get("ts", 0.0))
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        if ph not in ("B", "E"):
            # instant events: still bound coverage, do not change the stack
            continue
        # charge elapsed since last event to the current innermost span
        if stack and last_change is not None:
            dt = ts - last_change
            if dt > 0:
                leaf_excl[stack[-1][0]] = leaf_excl.get(stack[-1][0], 0.0) + dt
        last_change = ts
        if ph == "B":
            stack.append([e.get("name"), ts, e.get("args", {})])
        else:  # E — match nearest same-name from top
            idx = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == e.get("name"):
                    idx = i
                    break
            if idx is None:
                anomalies += 1
                continue
            name, start, args = stack[idx]
            has_dec = any(s[0] == "worker.decode_chunk" for s in stack[:idx])
            end = ts
            if end < start - 1.0:
                anomalies += 1
            spans.append((name, start, end, args, has_dec))
            # pop everything from idx up (tolerate skipped inner ends)
            del stack[idx:]
    return leaf_excl, spans, anomalies, (first_ts or 0.0), (last_ts or 0.0)


def overlap_free(intervals):
    iv = sorted(intervals)
    for i in range(1, len(iv)):
        if iv[i][0] + 1.0 < iv[i - 1][1]:  # >1us overlap
            return False
    return True


def _is_spec(args_obj):
    v = args_obj.get("speculative")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expected-chunks", type=int, default=None)
    ap.add_argument("--perf-cpus", type=float, default=None,
                    help="perf task-clock/wall (CPUs-utilized) of the SAME run; "
                         "gates the trace<->perf reconciliation")
    ap.add_argument("--perf-tol", type=float, default=0.12,
                    help="relative tolerance for effcores_cpu vs --perf-cpus")
    ap.add_argument("--gap-tol", type=float, default=0.10,
                    help="max per-thread untracked GAP fraction before refusing")
    ap.add_argument("--sink", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    events, bad_lines = parse_events(args.trace)

    fails = []
    warns = []

    # group by tid
    by_tid = {}
    for e in events:
        if e.get("ph") in ("B", "E", "i"):
            by_tid.setdefault(e.get("tid"), []).append(e)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda e: float(e.get("ts", 0.0)))

    all_ts = [float(e.get("ts", 0.0)) for e in events if e.get("ph") in ("B", "E", "i")]
    max_ts = max(all_ts, default=0.0)

    # ---- drive span (the wall) ----
    drive_begins = [e for e in events if e.get("name") == "drive" and e.get("ph") == "B"]
    if len(drive_begins) != 1:
        fails.append(f"CONSERVATION: expected exactly 1 'drive' begin, got {len(drive_begins)}")
        _emit_fail(fails, warns, bad_lines)
        return 3
    db = drive_begins[0]
    d_start = float(db.get("ts", 0.0))
    d_args = db.get("args", {})
    drive_ends = [e for e in events if e.get("name") == "drive" and e.get("ph") == "E"]
    if drive_ends:
        d_end = max(float(e.get("ts", 0.0)) for e in drive_ends)
    else:
        d_end = max_ts
        warns.append(f"drive span unclosed (per-thread flush truncation); "
                     f"drive_end recovered from max event ts={d_end:.1f}us")
    drive_wall = d_end - d_start
    T = int(d_args.get("parallelization", 0)) or None
    input_bytes = int(d_args.get("input_bytes", 0)) or None
    if drive_wall <= 0:
        fails.append(f"CONSERVATION: drive_wall <= 0 ({drive_wall:.1f}us)")

    # ---- per-tid leaf decomposition ----
    busy_total = 0.0
    wait_total = 0.0
    gap_total = 0.0
    life_total = 0.0
    n_active = 0
    idle_by_class = {}
    in_decode_stall = 0.0          # WAIT leaves nested under decode_chunk
    tid_rows = []
    all_decode_spans = []
    total_anom = 0
    worst_gap_frac = 0.0
    worst_gap_tid = None

    for tid, evs in by_tid.items():
        leaf, spans, anom, first_ts, last_ts = per_tid_timeline(evs)
        total_anom += anom
        all_decode_spans.extend(
            (nm, st, en, ar, hd) for (nm, st, en, ar, hd) in spans if nm == "worker.decode_chunk"
        )
        life = last_ts - first_ts
        covered = sum(leaf.values())
        busy = sum(v for k, v in leaf.items() if not is_wait(k))
        wait = sum(v for k, v in leaf.items() if is_wait(k))
        gap = max(0.0, life - covered)
        for k, v in leaf.items():
            if is_wait(k):
                idle_by_class[IDLE_CLASS.get(k, k)] = idle_by_class.get(IDLE_CLASS.get(k, k), 0.0) + v
        # in-decode off-CPU stall: WAIT spans whose ancestor includes decode_chunk
        for (nm, st, en, ar, hd) in spans:
            if hd and is_wait(nm):
                in_decode_stall += (en - st)
        busy_total += busy
        wait_total += wait
        gap_total += gap
        if life > 0.05 * drive_wall:
            life_total += life
            n_active += 1
        # role
        names = set(leaf.keys()) | {s[0] for s in spans}
        if "drive" in names or any(n.startswith("consumer.") for n in names):
            role = "consumer"
        elif "worker.decode_chunk" in names or "pool.pick.wait" in names:
            role = "worker"
        else:
            role = "other"
        gap_frac = (gap / life) if life > 0 else 0.0
        if role in ("worker", "consumer") and life > 0.05 * drive_wall and gap_frac > worst_gap_frac:
            worst_gap_frac = gap_frac
            worst_gap_tid = tid
        # non-overlap of decode_chunk on this worker tid
        dec_iv = [(s[1], s[2]) for s in spans if s[0] == "worker.decode_chunk"]
        overlap = (len(dec_iv) > 1) and (not overlap_free(dec_iv))
        if overlap:
            fails.append(f"CONSERVATION: decode_chunk spans overlap on tid {tid} (mis-paired B/E)")
        tid_rows.append({
            "tid": tid, "role": role,
            "life_ms": life / 1000.0, "busy_ms": busy / 1000.0,
            "wait_ms": wait / 1000.0, "gap_ms": gap / 1000.0,
            "gap_frac": gap_frac, "n_decode": len(dec_iv),
        })

    # ---- conservation gates ----
    if total_anom:
        warns.append(f"{total_anom} B/E anomalies (unmatched E or end<start; common at trace tail)")
    if worst_gap_frac > args.gap_tol:
        fails.append(
            f"CONSERVATION: tid {worst_gap_tid} has GAP frac {worst_gap_frac*100:.1f}% > "
            f"{args.gap_tol*100:.0f}% — trace does not cover the thread's life, "
            f"busy/idle split untrustworthy"
        )
    # THREAD-TIME CLOSURE: Σ(busy+wait+gap) over active threads == Σ life == wall×n_active
    # (exact by construction; assert it actually closes — catches a leaf-accounting bug).
    accounted = busy_total + wait_total + gap_total
    close_err = abs(accounted - life_total) / life_total if life_total > 0 else 1.0
    if close_err > 0.02:
        fails.append(
            f"CONSERVATION: Σ(busy+wait+gap)={accounted/1000:.1f}ms != Σlife={life_total/1000:.1f}ms "
            f"(err {close_err*100:.2f}%) — leaf-accounting does not close"
        )

    # ---- effcores ----
    sum_decode_span = sum((en - st) for (_, st, en, _, _) in all_decode_spans)
    effcores_span = sum_decode_span / drive_wall if drive_wall > 0 else float("nan")
    effcores_cpu = busy_total / drive_wall if drive_wall > 0 else float("nan")
    tail_idle = idle_by_class.get("consumer_cross_chunk (future_recv)", 0.0) / drive_wall \
        if drive_wall > 0 else float("nan")

    if T and effcores_span > T * (1 + TOL):
        fails.append(f"CONSERVATION: effcores_span {effcores_span:.3f} > T={T} (+{int(TOL*100)}%) "
                     f"=> nested-span double-count")

    # ---- NON-INERT ----
    n_decode = len(all_decode_spans)
    n_spec = sum(1 for s in all_decode_spans if _is_spec(s[3]))
    if n_decode == 0:
        fails.append("NON-INERT: zero worker.decode_chunk spans (trace did not fire / wrong env)")
    if wait_total <= 0:
        warns.append("NON-INERT: zero WAIT spans (consumer never blocked? T=1?)")
    if args.expected_chunks is not None and n_decode != args.expected_chunks:
        fails.append(f"NON-INERT: decode_chunk spans={n_decode} != --expected-chunks "
                     f"{args.expected_chunks} (trace disagrees with run)")

    # ---- RECONCILIATION (rig-honesty + oversubscription diagnostic) ----
    # The wall-span trace marks a thread BUSY for any time it is inside a CPU-work
    # span — INCLUDING time it is descheduled (off-CPU) inside that span. perf
    # task-clock counts only on-CPU time. So:
    #   * busy < perf  => the trace is MISSING CPU work (real rig bug) -> FAIL.
    #   * busy ≈ perf  => rig honest (no oversubscription) -> PASS.
    #   * busy > perf with n_active>T and tiny gaps => OVERSUBSCRIPTION: (busy-perf)
    #     cores are off-CPU deschedule inside busy spans (more threads than pinned
    #     cores). EXPECTED, not a rig bug; reported as deschedule_cores.
    reconcile = None
    if args.perf_cpus is not None and drive_wall > 0:
        rel = (effcores_cpu - args.perf_cpus) / args.perf_cpus
        oversub = (effcores_cpu > args.perf_cpus * (1 + args.perf_tol)
                   and (n_active or 0) > (T or 0))
        deschedule_cores = max(0.0, effcores_cpu - args.perf_cpus)
        ok = abs(rel) <= args.perf_tol or oversub
        reconcile = {
            "perf_cpus": args.perf_cpus,
            "effcores_cpu_from_trace": effcores_cpu,
            "rel_err": rel,
            "n_active_threads": n_active,
            "oversubscription": oversub,
            "deschedule_cores_offcpu_in_busy_spans": deschedule_cores if oversub else 0.0,
            "pass": ok,
        }
        if effcores_cpu < args.perf_cpus * (1 - args.perf_tol):
            fails.append(
                f"RECONCILE: effcores_cpu(trace)={effcores_cpu:.3f} < perf CPUs-utilized="
                f"{args.perf_cpus:.3f} — trace is MISSING CPU work (rig bug)"
            )
        elif not ok:
            fails.append(
                f"RECONCILE: effcores_cpu(trace)={effcores_cpu:.3f} vs perf={args.perf_cpus:.3f} "
                f"differ by {rel*100:.1f}% with n_active={n_active}<=T={T} "
                f"(not oversubscription) — trace does not reproduce perf"
            )

    # ---- per-chunk profile (order by start = dispatch order) ----
    dec_by_start = sorted(all_decode_spans, key=lambda s: s[1])
    chunk_times = [(en - st) for (_, st, en, _, _) in dec_by_start]
    decode_var, last_wave, order_inv = {}, {}, None
    per_chunk = []
    if chunk_times:
        mn, mx = min(chunk_times), max(chunk_times)
        mean = sum(chunk_times) / len(chunk_times)
        std = math.sqrt(sum((x - mean) ** 2 for x in chunk_times) / len(chunk_times))
        decode_var = {"min_us": mn, "max_us": mx, "mean_us": mean,
                      "max_over_min": (mx / mn) if mn > 0 else None,
                      "cv": (std / mean) if mean > 0 else None}
        if T:
            wave = max(1, math.ceil(len(chunk_times) / T))
            tail = chunk_times[-wave:]
            last_wave = {"wave_size": wave, "tail_mean_us": sum(tail) / len(tail),
                         "tail_max_us": max(tail), "global_mean_us": mean,
                         "tail_mean_over_global": (sum(tail) / len(tail) / mean) if mean > 0 else None}
        start_rank = {id(s): i for i, s in enumerate(dec_by_start)}
        by_end = sorted(all_decode_spans, key=lambda s: s[2])
        seq = [start_rank[id(s)] for s in by_end]
        order_inv = sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])
        for i, (nm, st, en, ar, hd) in enumerate(dec_by_start):
            per_chunk.append({
                "rank": i, "tid": None,
                "start_ms": (st - d_start) / 1000.0, "end_ms": (en - d_start) / 1000.0,
                "dur_ms": (en - st) / 1000.0,
                "spec": _is_spec(ar),
            })

    gate0_pass = not fails

    located_idle_cores = {k: v / drive_wall for k, v in idle_by_class.items()} if drive_wall > 0 else {}
    # thread-time budget = wall × n_active_threads (NOT × T: there are n_active
    # threads — T workers + 1 in-order consumer). busy+wait+gap fills this budget.
    budget = drive_wall * n_active if n_active else None
    unaccounted = (budget - busy_total - wait_total - gap_total) if budget else None

    result = {
        "label": args.label, "trace": args.trace, "sink": args.sink,
        "gate0_pass": gate0_pass, "fails": fails, "warns": warns,
        "bad_lines": bad_lines, "be_anomalies": total_anom,
        "T": T, "input_bytes": input_bytes,
        "n_active_threads": n_active,
        "drive_wall_ms": drive_wall / 1000.0,
        "n_decode_chunks": n_decode, "n_spec_decodes": n_spec,
        "spec_frac": (n_spec / n_decode) if n_decode else None,
        # the two lenses
        "effcores_span": effcores_span,
        "effcores_cpu": effcores_cpu,
        "span_minus_cpu_cores": (effcores_span - effcores_cpu),
        # conservation budget (wall × n_active_threads)
        "core_budget_threads": (budget / drive_wall) if budget else None,  # == n_active
        "busy_cores": effcores_cpu,
        "located_idle_cores_total": (wait_total / drive_wall) if drive_wall > 0 else None,
        "gap_cores": (gap_total / drive_wall) if drive_wall > 0 else None,
        "unaccounted_cores": (unaccounted / drive_wall) if (unaccounted is not None and drive_wall > 0) else None,
        "in_decode_marked_wait_cores": (in_decode_stall / drive_wall) if drive_wall > 0 else None,
        "located_idle_by_class_cores": located_idle_cores,
        "tail_idle": tail_idle,
        "reconcile": reconcile,
        "busy_ms": busy_total / 1000.0,
        "wait_ms": wait_total / 1000.0,
        "gap_ms": gap_total / 1000.0,
        "worst_gap_frac": worst_gap_frac, "worst_gap_tid": worst_gap_tid,
        "decode_var": decode_var, "last_wave": last_wave,
        "completion_order_inversions": order_inv,
        "tids": sorted(tid_rows, key=lambda r: r["tid"]),
        "per_chunk": per_chunk,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)
    return 0 if gate0_pass else 3


def _emit_fail(fails, warns, bad_lines):
    print("=== parallel_sm_tail_metric: GATE-0 FAIL (numbers do not exist) ===")
    for f in fails:
        print(f"  FAIL: {f}")
    for w in warns:
        print(f"  warn: {w}")
    print(f"  (bad_lines={bad_lines})")


def _pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def _f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def _print_human(r):
    status = "PASS" if r["gate0_pass"] else "FAIL"
    print(f"=== tail-metric [{r['label'] or r['trace']}]  GATE-0 {status} ===")
    for f in r["fails"]:
        print(f"  FAIL: {f}")
    for w in r["warns"]:
        print(f"  warn: {w}")
    T = r["T"]
    na = r["n_active_threads"]
    print(f"  T={T}  active_threads={na} (={T}+consumer)  drive_wall={r['drive_wall_ms']:.1f}ms  "
          f"chunks={r['n_decode_chunks']} (spec={r['n_spec_decodes']}, "
          f"spec_frac={_pct(r['spec_frac'])})")
    print("  --- two lenses (cores) ---")
    print(f"  effcores_SPAN (Σ decode wall / wall) = {_f(r['effcores_span'],3)}/{T}   [OLD lens]")
    print(f"  effcores_CPU  (Σ busy leaf / wall)   = {_f(r['effcores_cpu'],3)}/{T}   [NEW lens]")
    rc = r["reconcile"]
    if rc:
        vp = "PASS" if rc["pass"] else "FAIL"
        print(f"  RECONCILE vs perf CPUs-utilized {rc['perf_cpus']:.3f}: "
              f"effcores_CPU {rc['effcores_cpu_from_trace']:.3f}  "
              f"(rel {rc['rel_err']*100:+.1f}%) [{vp}]")
        if rc["oversubscription"]:
            print(f"    OVERSUBSCRIPTION: {rc['deschedule_cores_offcpu_in_busy_spans']:.3f} cores are "
                  f"OFF-CPU deschedule inside busy spans (n_active={rc['n_active_threads']} > T={T} "
                  f"pinned cores) — NOT span-localizable, NOT a rig bug.")
    print("  --- thread-time conservation (cores; budget = wall × active_threads) ---")
    print(f"  busy={_f(r['busy_cores'],3)}  located_idle={_f(r['located_idle_cores_total'],3)}  "
          f"gap={_f(r['gap_cores'],3)}  unaccounted={_f(r['unaccounted_cores'],3)}  "
          f"/ budget {_f(r['core_budget_threads'],2)}")
    print(f"  closure: busy_ms={r['busy_ms']:.1f}+wait_ms={r['wait_ms']:.1f}+gap_ms={r['gap_ms']:.1f}  "
          f"worst_tid_gap={_pct(r['worst_gap_frac'])} (tid {r['worst_gap_tid']})")
    print("  --- LOCATED idle by class (cores) ---")
    for k, v in sorted(r["located_idle_by_class_cores"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:42s} {v:.3f}")
    print(f"  in-decode MARKED-wait (off-CPU spans nested in decode_chunk) = "
          f"{_f(r['in_decode_marked_wait_cores'],3)} cores")
    dv = r["decode_var"]
    if dv:
        print(f"  per-chunk decode: min={dv['min_us']/1000:.1f}ms max={dv['max_us']/1000:.1f}ms "
              f"mean={dv['mean_us']/1000:.1f}ms  max/min={_f(dv['max_over_min'])}  cv={_f(dv['cv'])}")
    lw = r["last_wave"]
    if lw:
        print(f"  last-wave({lw['wave_size']}): mean={lw['tail_mean_us']/1000:.1f}ms "
              f"max={lw['tail_max_us']/1000:.1f}ms  tail/global={_f(lw['tail_mean_over_global'])}")
    print(f"  completion-order inversions vs dispatch: {r['completion_order_inversions']}")
    print("  --- per-tid (ms) ---")
    for t in r["tids"]:
        print(f"    tid {t['tid']:>3} {t['role']:8s} life={t['life_ms']:7.1f} busy={t['busy_ms']:7.1f} "
              f"wait={t['wait_ms']:7.1f} gap={t['gap_ms']:7.1f} ({_pct(t['gap_frac'])}) "
              f"n_decode={t['n_decode']}")


if __name__ == "__main__":
    sys.exit(main())
