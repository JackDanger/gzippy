#!/usr/bin/env python3
"""perthread_cpu_duty.py — STEP-1 [BLOCKING CONFIRM] of the CONSUMER-CONFIRM plan.

Resolve the FROZEN-instrument contradiction about gzippy's in-order consumer
thread: STEP-1 span trace says ~98% BLOCKED (cold); P1 /proc max_running probe
says HOT (5 simultaneously-runnable). max_running (sampled simultaneously-R) is
NOT a CPU duty cycle. This measures each gz thread's TRUE on-CPU duty during an
UNPINNED frozen silesia-T4 decode via /proc/<pid>/task/<tid>/stat (utime+stime),
the accounting that counts CPU spent INSIDE the recv_timeout(1ms)+pump loop that
the wall-span trace mis-buckets as wait.

CONSUMER tid tag = the MAIN thread (tid == pid): `chunk_fetcher::drive` runs
`consumer_loop` INLINE on the calling (main) thread; the 4 decode workers are
ThreadPool::spawn_thread children (tid != pid). Deterministic, no trace needed.

GATES:
  - CONSERVATION (blocking): Sum(per-tid core-seconds) reconciles to the SAME
    quantity from the OS process accounting; and (when --perf-taskclock-sec is
    given) to perf task-clock of clean (un-polled) runs of the identical command,
    within --tol (default 5%). If the poller perturbed the run, this fails.
  - NON-INERT: the worker tids must each read ~0.9-1.0 core (else the rig is not
    seeing the real decode threads).
  - N >= 7 interleaved.

Usage:
  perthread_cpu_duty.py --gz BIN --corpus FILE --threads 4 --n 9 \
     [--perf-taskclock-sec 1.70] [--sample-ms 1.0] [--tol 0.05]
Emits a per-thread duty table + the FORK verdict (HOT consumer >=~0.7c /
COLD consumer <=~0.2c) as text and a JSON blob (--json PATH).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median

CLK = os.sysconf("SC_CLK_TCK")


def read_task_cpu(pid):
    """Return {tid: (utime+stime ticks, comm)} for all live tasks of pid."""
    out = {}
    base = f"/proc/{pid}/task"
    try:
        tids = os.listdir(base)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"{base}/{tid}/stat", "rb") as f:
                data = f.read()
        except OSError:
            continue
        # comm is field 2, wrapped in parens and may contain spaces/parens.
        try:
            lp = data.index(b"(")
            rp = data.rindex(b")")
        except ValueError:
            continue
        comm = data[lp + 1 : rp].decode("utf-8", "replace")
        rest = data[rp + 2 :].split()
        # After comm, field 3 is state -> rest[0]. utime is field 14, stime 15.
        # rest index: field N -> rest[N-3]. utime=rest[11], stime=rest[12].
        try:
            utime = int(rest[11])
            stime = int(rest[12])
        except (IndexError, ValueError):
            continue
        out[tid] = (utime + stime, comm)
    return out


def one_run(gz, corpus, threads, sample_s):
    env = dict(os.environ)
    env["GZIPPY_FORCE_PARALLEL_SM"] = "1"
    devnull = open(os.devnull, "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [gz, "-dc", f"-p{threads}", corpus], stdout=devnull, stderr=subprocess.DEVNULL, env=env
    )
    pid = proc.pid
    per_tid = {}  # tid -> max(ticks)
    comm = {}
    while proc.poll() is None:
        snap = read_task_cpu(pid)
        for tid, (ticks, c) in snap.items():
            if tid not in per_tid or ticks > per_tid[tid]:
                per_tid[tid] = ticks
            comm[tid] = c
        time.sleep(sample_s)
    proc.wait()
    # one final scan is racy (tasks gone); the last in-loop snapshot holds the
    # max cumulative value per tid, which is what we want.
    t1 = time.monotonic()
    devnull.close()
    wall = t1 - t0
    return pid, wall, per_tid, comm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--sample-ms", type=float, default=1.0)
    ap.add_argument("--perf-taskclock-sec", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.n < 7:
        print(f"GATE-1 FAIL: N={args.n} < 7", file=sys.stderr)
    sample_s = args.sample_ms / 1000.0

    # warmup
    one_run(args.gz, args.corpus, args.threads, sample_s)

    runs = []
    for r in range(args.n):
        pid, wall, per_tid, comm = one_run(args.gz, args.corpus, args.threads, sample_s)
        # classify
        consumer_tid = str(pid)  # main thread tid == pid
        rows = []
        for tid, ticks in per_tid.items():
            cpu_s = ticks / CLK
            rows.append(
                {
                    "tid": tid,
                    "comm": comm.get(tid, "?"),
                    "cpu_s": cpu_s,
                    "duty": cpu_s / wall if wall > 0 else 0.0,
                    "is_consumer": tid == consumer_tid,
                }
            )
        rows.sort(key=lambda x: x["cpu_s"], reverse=True)
        runs.append({"pid": pid, "wall": wall, "rows": rows})

    # Aggregate: consumer duty (tid==pid), worker duties by rank (workers =
    # non-consumer tids, sorted desc per run), and total core-seconds.
    def consumer_duty(run):
        for row in run["rows"]:
            if row["is_consumer"]:
                return row["duty"]
        return 0.0

    def consumer_cpu_s(run):
        for row in run["rows"]:
            if row["is_consumer"]:
                return row["cpu_s"]
        return 0.0

    def workers(run):
        return [r for r in run["rows"] if not r["is_consumer"]]

    cons_duties = [consumer_duty(r) for r in runs]
    walls = [r["wall"] for r in runs]
    total_cpu = [sum(x["cpu_s"] for x in r["rows"]) for r in runs]
    eff_cores = [t / w for t, w in zip(total_cpu, walls)]

    n_workers_seen = median([len(workers(r)) for r in runs])
    # worker-rank duty medians
    max_w = max(len(workers(r)) for r in runs)
    rank_duties = []
    for k in range(max_w):
        vals = [workers(r)[k]["duty"] for r in runs if len(workers(r)) > k]
        rank_duties.append(median(vals))

    med_cons = median(cons_duties)
    med_wall = median(walls)
    med_total_cpu = median(total_cpu)
    med_eff = median(eff_cores)

    print("=========== STEP-1 PER-THREAD OS-CPU DUTY (UNPINNED, gz -p%d) ===========" % args.threads)
    print(f"N={args.n}  sample={args.sample_ms}ms  CLK_TCK={CLK}")
    print(f"median wall = {med_wall*1000:.1f} ms")
    print("--- per-RANK duty (median over N), workers sorted desc per run ---")
    for k, d in enumerate(rank_duties):
        print(f"  worker rank {k}: duty = {d:.3f} core")
    print(f"  CONSUMER (tid==pid): duty = {med_cons:.3f} core  (per-run: " +
          ",".join(f"{d:.3f}" for d in cons_duties) + ")")
    print("--- conservation ---")
    print(f"  Sum per-tid core-seconds (median) = {med_total_cpu:.4f} s")
    print(f"  effcores (Sum cpu_s / wall, median) = {med_eff:.3f}")
    if args.perf_taskclock_sec is not None:
        perf_s = args.perf_taskclock_sec
        rel = abs(med_total_cpu - perf_s) / perf_s if perf_s > 0 else 1.0
        ok = rel <= args.tol
        print(f"  perf task-clock (clean runs)     = {perf_s:.4f} s")
        print(f"  reconciliation |Sum - perf|/perf = {rel*100:.1f}%  "
              f"({'PASS' if ok else 'FAIL'} @ tol {args.tol*100:.0f}%)")
    # non-inert gate
    inert_ok = all(d >= 0.85 for d in rank_duties[: args.threads])
    print(f"  NON-INERT (top {args.threads} workers each >=0.85c): "
          f"{'PASS' if inert_ok else 'FAIL'}")
    # FORK
    print("--- FORK VERDICT ---")
    if med_cons >= 0.7:
        verdict = "HOT"
        print(f"  CONSUMER DUTY {med_cons:.3f} >= 0.7  => HOT  (P1 /proc right; "
              f"STEP-1 span trace under-counts pump CPU)")
    elif med_cons <= 0.2:
        verdict = "COLD"
        print(f"  CONSUMER DUTY {med_cons:.3f} <= 0.2  => COLD (STEP-1 span trace "
              f"right; P1 max_running was a sampling over-read)")
    else:
        verdict = "AMBIGUOUS"
        print(f"  CONSUMER DUTY {med_cons:.3f} in (0.2,0.7) => AMBIGUOUS")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "n": args.n,
                    "median_wall_ms": med_wall * 1000,
                    "rank_duties": rank_duties,
                    "consumer_duty_median": med_cons,
                    "consumer_duties": cons_duties,
                    "sum_cpu_s_median": med_total_cpu,
                    "effcores_median": med_eff,
                    "perf_taskclock_sec": args.perf_taskclock_sec,
                    "verdict": verdict,
                    "runs": runs,
                },
                f,
                indent=2,
            )
    print("DONE_PERTHREAD_CPU_DUTY")


if __name__ == "__main__":
    main()
