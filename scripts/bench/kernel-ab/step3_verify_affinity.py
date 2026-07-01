#!/usr/bin/env python3
"""step3_verify_affinity.py — GATE-0 for STEP-3: confirm GZIPPY_PHYS_PIN=1 places
each gz thread (4 workers + consumer) on a DISTINCT PHYSICAL core. Launches gz
with the affinity env, samples each task's current processor (/proc stat field
39) over the run, maps logical->physical via thread_siblings_list, and reports
whether the observed physical cores are distinct (no SMT co-location).

Usage: step3_verify_affinity.py GZ CORP THREADS
"""
import os
import subprocess
import sys
import time


def siblings_rep(cpu):
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            body = f.read().strip()
    except OSError:
        return cpu
    ids = []
    for tok in body.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif tok:
            ids.append(int(tok))
    return min(ids) if ids else cpu


def proc_of(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    rp = data.rindex(b")")
    rest = data[rp + 2 :].split()
    try:
        return int(rest[36])  # field 39 = processor
    except (IndexError, ValueError):
        return None


def main():
    gz, corp, threads = sys.argv[1], sys.argv[2], sys.argv[3]
    env = dict(os.environ)
    env["GZIPPY_PHYS_PIN"] = "1"
    env["GZIPPY_FORCE_PARALLEL_SM"] = "1"
    devnull = open(os.devnull, "wb")
    proc = subprocess.Popen([gz, "-dc", f"-p{threads}", corp], stdout=devnull,
                            stderr=subprocess.DEVNULL, env=env)
    pid = proc.pid
    seen = {}  # tid -> set of cpus
    while proc.poll() is None:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            break
        for tid in tids:
            cpu = proc_of(pid, tid)
            if cpu is not None:
                seen.setdefault(tid, set()).add(cpu)
        time.sleep(0.001)
    proc.wait()
    devnull.close()

    print(f"pid={pid}  observed {len(seen)} tids")
    phys_used = {}
    for tid, cpus in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        # dominant cpu = the one most plausibly its pin (report full set)
        reps = {siblings_rep(c) for c in cpus}
        role = "CONSUMER(main)" if tid == str(pid) else "worker"
        print(f"  tid {tid:>7} {role:14s} cpus={sorted(cpus)} phys={sorted(reps)}")
        for r in reps:
            phys_used[r] = phys_used.get(r, 0) + 1
    # distinctness: each physical core should host at most one gz thread
    collisions = {p: c for p, c in phys_used.items() if c > 1}
    if collisions:
        print(f"  AFFINITY CHECK: SMT CO-LOCATION present on phys {collisions} — "
              f"(note: a thread may transiently show multiple cpus)")
    else:
        print(f"  AFFINITY CHECK: PASS — {len(phys_used)} distinct physical cores, "
              f"no SMT co-location")


if __name__ == "__main__":
    main()
