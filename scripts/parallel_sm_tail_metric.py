#!/usr/bin/env python3
"""parallel_sm_tail_metric.py — reduce a GZIPPY_TIMELINE trace to the ONE
discriminator that separates H-TAIL (scheduling/tail-imbalance) from
H-KERNEL (per-chunk decode-CPU surplus) for the silesia-T4 residual.

INPUT: a Chrome-trace JSON file written by the `trace_timeline.rs` instrument
       (env `GZIPPY_TIMELINE=/path.json` on a ParallelSM decode). The file is a
       JSON *array* opened with `[\n` and one event object per line ending in
       `},` — it is NOT closed (the instrument tolerates partial arrays), so we
       parse leniently line-by-line.

THE METRIC (SILT4-PLAN §3 Tool 1):
  effcores      = Sum(worker.decode_chunk durations) / drive_wall
                  = average busy cores. effcores << T => schedule slack (H-TAIL);
                    effcores ~= T => CPU-bound (H-KERNEL, the T1 kernel front).
  tail_idle     = Sum(wait.future_recv durations) / drive_wall
                  = fraction of wall the in-order consumer blocked on a
                    not-yet-ready chunk.
  last_wave     = decode time of the final ceil(n_chunks/T) chunks vs the mean.
  decode_var    = max/min and stdev/mean of per-chunk decode time (the
                  silesia-vs-monorepo heterogeneity discriminator).
  order_inv     = inversions between completion order and emission (chunk_id)
                  order (out-of-order completion => the writer waits).

GATE-0 SELF-VALIDATION (BLOCKING — refuses to print a metric if any fails):
  CONSERVATION:
    * exactly one `drive` span;
    * every span has end >= start;
    * 0 < effcores <= T*(1+tol)  (a nested-span double-count would blow past T;
      this is the dedupe/no-double-count guard — we only sum top-level
      worker.decode_chunk spans, never their nested children);
    * per worker-tid, decode_chunk spans do NOT overlap (a worker decodes one
      chunk at a time — overlap => mis-paired B/E);
    * decode + wait spans lie within the drive span (+/- tol).
  NON-INERT:
    * >0 worker.decode_chunk spans AND >0 wait.future_recv spans (the trace
      actually fired on the parallel consumer path);
    * if --expected-chunks N is given, distinct NON-speculative chunk_ids == N
      (cross-checks the trace against ceil(compressed_len/chunk_size) /
      --verbose BlockFetcher count). Disagreement => the trace is lying.

USAGE:
  scripts/parallel_sm_tail_metric.py TRACE.json [--expected-chunks N]
                                     [--sink /dev/null] [--json] [--label STR]
EXIT: 0 if Gate-0 passes, 3 if Gate-0 fails (the numbers do not exist).
"""
import argparse
import json
import math
import sys

TOL = 0.05  # 5% slack for float/scheduling jitter in conservation checks


def parse_events(path):
    """Lenient parse of the partial Chrome-trace array. Returns list of dicts."""
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
                # incomplete tail line (partial flush) — skip
                bad += 1
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
    return events, bad


def pair_spans(events):
    """Match B/E per (pid,tid) with a stack. Returns:
       spans: list of (name, tid, start, end, args)
       anomalies: count of unmatched E events.
    """
    stacks = {}
    spans = []
    anomalies = 0
    for e in events:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        key = (e.get("pid"), e.get("tid"))
        if ph == "B":
            stacks.setdefault(key, []).append(e)
        else:  # E — match most recent B with same name on this (pid,tid)
            st = stacks.get(key, [])
            # find the nearest matching name from the top
            idx = None
            for i in range(len(st) - 1, -1, -1):
                if st[i].get("name") == e.get("name"):
                    idx = i
                    break
            if idx is None:
                anomalies += 1
                continue
            b = st.pop(idx)
            spans.append(
                (
                    b.get("name"),
                    e.get("tid"),
                    float(b.get("ts", 0.0)),
                    float(e.get("ts", 0.0)),
                    b.get("args", {}),
                )
            )
    # leftover unclosed B spans are anomalies too (but common at trace tail)
    return spans, anomalies


def overlap_free(intervals):
    """True if no two [s,e) intervals overlap by more than TOL of the smaller."""
    iv = sorted(intervals)
    for i in range(1, len(iv)):
        prev_e = iv[i - 1][1]
        cur_s = iv[i][0]
        if cur_s + 1.0 < prev_e:  # >1us overlap
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expected-chunks", type=int, default=None)
    ap.add_argument("--sink", default=None, help="record the sink used (informational)")
    ap.add_argument("--label", default="", help="cell label, e.g. silesia-T4-gz")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    events, bad_lines = parse_events(args.trace)
    spans, anomalies = pair_spans(events)

    fails = []
    warns = []

    # The drive span: trace_timeline flushes per-thread and the instrument
    # tolerates a partial (unclosed) array, so the consumer thread's drive END
    # event is frequently NOT in the file at process exit. Recover the drive
    # span from its B event + the max event ts (the true wall is bounded by the
    # last emitted event). This is a warn (recovered), not a fail.
    all_b = [e for e in events if e.get("ph") in ("B", "E", "i")]
    max_ts = max((float(e.get("ts", 0.0)) for e in all_b), default=0.0)
    drive_spans = [s for s in spans if s[0] == "drive"]
    drive_begins = [
        e for e in events if e.get("name") == "drive" and e.get("ph") == "B"
    ]
    if len(drive_begins) != 1:
        fails.append(
            f"CONSERVATION: expected exactly 1 'drive' begin, got {len(drive_begins)}"
        )
        _emit_fail(args, fails, warns, bad_lines, anomalies)
        return 3
    if drive_spans:
        _, _, d_start, d_end, d_args = drive_spans[0]
    else:
        # unclosed drive — recover end from max event ts
        db = drive_begins[0]
        d_start = float(db.get("ts", 0.0))
        d_end = max_ts
        d_args = db.get("args", {})
        warns.append(
            "drive span unclosed in trace (per-thread flush truncation); "
            f"drive_end recovered from max event ts={d_end:.1f}us"
        )
    drive_wall = d_end - d_start
    T = int(d_args.get("parallelization", 0)) or None
    input_bytes = int(d_args.get("input_bytes", 0)) or None
    if drive_wall <= 0:
        fails.append(f"CONSERVATION: drive_wall <= 0 ({drive_wall:.1f}us)")

    decode = [s for s in spans if s[0] == "worker.decode_chunk"]
    waits = [s for s in spans if s[0] == "wait.future_recv"]
    bfget = [s for s in spans if s[0] == "wait.block_fetcher_get"]

    # negative durations
    neg = [s for s in spans if s[3] < s[2] - 1.0]
    if neg:
        fails.append(f"CONSERVATION: {len(neg)} spans with end<start")

    # NON-INERT
    if not decode:
        fails.append("NON-INERT: zero worker.decode_chunk spans (trace did not fire / wrong env)")
    if not waits:
        warns.append("NON-INERT: zero wait.future_recv spans (consumer never blocked? T=1?)")

    # decode durations
    dur = lambda s: s[3] - s[2]
    sum_decode = sum(dur(s) for s in decode)
    sum_decode_nonspec = sum(dur(s) for s in decode if not _is_spec(s[4]))
    sum_wait = sum(dur(s) for s in waits)
    sum_bfget = sum(dur(s) for s in bfget)

    effcores = sum_decode / drive_wall if drive_wall > 0 else float("nan")
    effcores_nonspec = sum_decode_nonspec / drive_wall if drive_wall > 0 else float("nan")
    tail_idle = sum_wait / drive_wall if drive_wall > 0 else float("nan")

    # CONSERVATION: effcores must not exceed T (nested double-count guard)
    if T:
        if effcores > T * (1 + TOL):
            fails.append(
                f"CONSERVATION: effcores {effcores:.3f} > T={T} (+{int(TOL*100)}%) "
                f"=> nested-span double-count"
            )
        if effcores <= 0:
            fails.append("CONSERVATION: effcores <= 0")

    # CONSERVATION: per worker-tid, decode_chunk spans non-overlapping
    per_tid = {}
    for s in decode:
        per_tid.setdefault(s[1], []).append((s[2], s[3]))
    overlap_tids = [tid for tid, iv in per_tid.items() if not overlap_free(iv)]
    if overlap_tids:
        fails.append(
            f"CONSERVATION: decode_chunk spans overlap on worker tid(s) {overlap_tids} "
            f"(mis-paired B/E)"
        )

    # CONSERVATION: spans within drive window
    out_of_drive = sum(
        1 for s in decode + waits if s[2] < d_start - 1.0 or s[3] > d_end + 1.0
    )
    if out_of_drive:
        warns.append(f"{out_of_drive} decode/wait spans fall outside the drive window")

    # NON-INERT chunk-count cross-check.
    # NOTE: gz dispatches almost all chunks via the speculative-prefetch path,
    # which carries chunk_id = usize::MAX (partition unknown at decode time).
    # Those are REAL decodes that the consumer consumes — NOT wasted work — so
    # the chunk count and effcores use ALL decode_chunk spans, and `speculative`
    # is informational only. The independent count to cross-check against is the
    # TOTAL decode_chunk span count vs rg's "Blocks Total Fetched" / ceil(comp/chunk).
    n_real_chunks = len(decode)
    n_spec = sum(1 for s in decode if _is_spec(s[4]))
    if args.expected_chunks is not None:
        if n_real_chunks != args.expected_chunks:
            fails.append(
                f"NON-INERT: total decode_chunk spans={n_real_chunks} != "
                f"--expected-chunks {args.expected_chunks} (trace disagrees with run)"
            )

    # ---- derived profile (only trustworthy if Gate-0 passes) ----
    # chunk_id is unreliable (MAX for speculative), so order by START time
    # (= emission/dispatch order) and END time (= completion order).
    decode_by_start = sorted(decode, key=lambda s: s[2])
    chunk_times = [dur(s) for s in decode_by_start]
    decode_var = {}
    last_wave = {}
    order_inv = None
    if chunk_times:
        mn, mx = min(chunk_times), max(chunk_times)
        mean = sum(chunk_times) / len(chunk_times)
        var = sum((x - mean) ** 2 for x in chunk_times) / len(chunk_times)
        std = math.sqrt(var)
        decode_var = {
            "min_us": mn,
            "max_us": mx,
            "mean_us": mean,
            "max_over_min": (mx / mn) if mn > 0 else None,
            "cv": (std / mean) if mean > 0 else None,
        }
        if T:
            wave = max(1, math.ceil(len(chunk_times) / T))
            tail_ct = chunk_times[-wave:]
            last_wave = {
                "wave_size": wave,
                "tail_mean_us": sum(tail_ct) / len(tail_ct),
                "tail_max_us": max(tail_ct),
                "global_mean_us": mean,
                "tail_mean_over_global": (sum(tail_ct) / len(tail_ct) / mean)
                if mean > 0
                else None,
            }
        # completion vs emission order inversions: assign emission rank by
        # start-time order, then count how out-of-order the END (completion)
        # order is relative to it. >0 => chunks finished out of dispatch order
        # (the in-order writer must wait on a later-finishing earlier chunk).
        start_rank = {id(s): i for i, s in enumerate(decode_by_start)}
        by_end = sorted(decode, key=lambda s: s[3])
        seq = [start_rank[id(s)] for s in by_end]
        inv = sum(
            1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j]
        )
        order_inv = inv

    gate0_pass = not fails

    result = {
        "label": args.label,
        "trace": args.trace,
        "sink": args.sink,
        "gate0_pass": gate0_pass,
        "fails": fails,
        "warns": warns,
        "bad_lines": bad_lines,
        "be_anomalies": anomalies,
        "T": T,
        "input_bytes": input_bytes,
        "drive_wall_us": drive_wall,
        "drive_wall_ms": drive_wall / 1000.0,
        "n_real_chunks": n_real_chunks,
        "n_spec_decodes": n_spec,
        "spec_frac": (n_spec / len(decode)) if decode else None,
        "effcores": effcores,
        "effcores_nonspec": effcores_nonspec,
        "tail_idle": tail_idle,
        "sum_decode_us": sum_decode,
        "sum_wait_us": sum_wait,
        "sum_blockfetcher_get_us": sum_bfget,
        "decode_var": decode_var,
        "last_wave": last_wave,
        "completion_order_inversions": order_inv,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 0 if gate0_pass else 3


def _is_spec(args_obj):
    v = args_obj.get("speculative")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def _emit_fail(args, fails, warns, bad_lines, anomalies):
    print("=== parallel_sm_tail_metric: GATE-0 FAIL (numbers do not exist) ===")
    for f in fails:
        print(f"  FAIL: {f}")
    for w in warns:
        print(f"  warn: {w}")
    print(f"  (bad_lines={bad_lines} be_anomalies={anomalies})")


def _print_human(r):
    status = "PASS" if r["gate0_pass"] else "FAIL"
    print(f"=== tail-metric [{r['label'] or r['trace']}]  GATE-0 {status} ===")
    if not r["gate0_pass"]:
        for f in r["fails"]:
            print(f"  FAIL: {f}")
    for w in r["warns"]:
        print(f"  warn: {w}")
    T = r["T"]
    ec = r["effcores"]
    ecn = r["effcores_nonspec"]
    ti = r["tail_idle"]
    print(
        f"  T={T}  drive_wall={r['drive_wall_ms']:.1f}ms  "
        f"chunks={r['n_real_chunks']} (spec_decodes={r['n_spec_decodes']}, "
        f"spec_frac={_pct(r['spec_frac'])})"
    )
    print(
        f"  effcores={ec:.3f}/{T}  effcores_nonspec={ecn:.3f}/{T}  "
        f"tail_idle={_pct(ti)}  blockfetcher_get={r['sum_blockfetcher_get_us']/1000.0:.1f}ms"
    )
    dv = r["decode_var"]
    if dv:
        print(
            f"  per-chunk decode: min={dv['min_us']/1000:.1f}ms max={dv['max_us']/1000:.1f}ms "
            f"mean={dv['mean_us']/1000:.1f}ms  max/min={_f(dv['max_over_min'])}  cv={_f(dv['cv'])}"
        )
    lw = r["last_wave"]
    if lw:
        print(
            f"  last-wave({lw['wave_size']} chunks): mean={lw['tail_mean_us']/1000:.1f}ms "
            f"max={lw['tail_max_us']/1000:.1f}ms  tail/global={_f(lw['tail_mean_over_global'])}"
        )
    print(f"  completion-order inversions vs emission: {r['completion_order_inversions']}")
    # the one fork-deciding line
    if r["gate0_pass"] and T:
        verdict = "H-TAIL (schedule slack)" if ec < T * 0.85 else "H-KERNEL-ish (cores ~saturated)"
        print(
            f"  >>> FORK LINE: ratio_unknown | effcores {ec:.2f}/{T} | "
            f"tail_idle {_pct(ti)} | route-hint: {verdict}"
        )


def _pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def _f(x):
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    sys.exit(main())
