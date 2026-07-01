#!/usr/bin/env python3
"""pin_discriminator_placement.py — NON-INERT core-placement witness for the
PIN-DISCRIMINATOR 3-arm measurement.

Launches gzippy-native on a corpus at -pT, samples each task's CURRENT processor
(/proc/<pid>/task/<tid>/stat field 39) over the whole decode, reads each tid's
Cpus_allowed_list (/proc/<pid>/task/<tid>/status), and maps logical->physical via
/sys/.../topology/thread_siblings_list. PROVES where each arm's workers actually
land: distinct-physical vs SMT-co-located.

Usage:  pin_discriminator_placement.py <gz> <corp.gz> <threads> <arm-tag> [ENV=VAL ...]
  arm-tag is informational (A/B/C); ENV=VAL pairs set the gz environment
  (e.g. GZIPPY_NO_PIN=1 for arm B, GZIPPY_PHYS_PIN=1 for arm C, none for arm A).

Emits per-tid: observed cpus, dominant cpu, physical-core reps, allowed-list;
then a VERDICT line: number of distinct physical cores the workers occupied and
whether any physical core hosted >1 worker (SMT co-location).
"""
import os
import subprocess
import sys
import time


def parse_cpu_list(body):
    ids = []
    for tok in body.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(tok))
    return ids


def siblings_rep(cpu):
    """Smallest logical id in cpu's SMT sibling group = its physical-core key."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            ids = parse_cpu_list(f.read())
        return min(ids) if ids else cpu
    except OSError:
        return cpu


def proc_cpu(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    rest = data[data.rindex(b")") + 2:].split()
    try:
        return int(rest[36])  # field 39 (1-based) = processor; index 36 after comm
    except (IndexError, ValueError):
        return None


def allowed_list(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def main():
    gz, corp, threads, tag = sys.argv[1:5]
    env = dict(os.environ)
    env["GZIPPY_FORCE_PARALLEL_SM"] = "1"
    for kv in sys.argv[5:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    extra = " ".join(sys.argv[5:]) or "(default)"
    print(f"[ARM {tag}] env: {extra}")

    devnull = open(os.devnull, "wb")
    proc = subprocess.Popen([gz, "-dc", f"-p{threads}", corp], stdout=devnull,
                            stderr=subprocess.DEVNULL, env=env)
    pid = proc.pid
    seen = {}     # tid -> {cpu: count}
    allowed = {}  # tid -> Cpus_allowed_list
    while proc.poll() is None:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            break
        for tid in tids:
            cpu = proc_cpu(pid, tid)
            if cpu is not None:
                d = seen.setdefault(tid, {})
                d[cpu] = d.get(cpu, 0) + 1
                if tid not in allowed:
                    allowed[tid] = allowed_list(pid, tid)
        time.sleep(0.0005)
    proc.wait()
    devnull.close()

    print(f"  pid={pid}  observed {len(seen)} tids over the decode")
    # The N busiest tids (most samples) are the decode workers; main = consumer.
    worker_phys = {}  # physical-rep -> count of distinct workers landing there
    rows = []
    for tid, cpus in seen.items():
        total = sum(cpus.values())
        dominant = max(cpus, key=cpus.get)
        rows.append((total, tid, cpus, dominant))
    rows.sort(reverse=True)
    # classify: main thread (== pid) is consumer; the rest by sample-count.
    for total, tid, cpus, dominant in rows:
        reps = sorted({siblings_rep(c) for c in cpus})
        role = "CONSUMER(main)" if tid == str(pid) else "worker"
        al = allowed.get(tid, "?")
        cpu_hist = ",".join(f"{c}:{n}" for c, n in sorted(cpus.items()))
        print(f"    tid {tid:>7} {role:14s} dom_cpu={dominant:2d} "
              f"phys={reps} allowed=[{al}] cpus={{{cpu_hist}}}")
        if role == "worker":
            # attribute this worker to the physical core of its DOMINANT cpu
            worker_phys[siblings_rep(dominant)] = \
                worker_phys.get(siblings_rep(dominant), 0) + 1

    distinct = len(worker_phys)
    collisions = {p: c for p, c in worker_phys.items() if c > 1}
    nworkers = sum(worker_phys.values())
    if collisions:
        print(f"  [ARM {tag}] PLACEMENT VERDICT: SMT-CO-LOCATED — {nworkers} workers "
              f"on {distinct} distinct physical cores; collisions={collisions}")
    else:
        print(f"  [ARM {tag}] PLACEMENT VERDICT: DISTINCT-PHYSICAL — {nworkers} workers "
              f"on {distinct} distinct physical cores; no SMT co-location")
    # machine-readable summary line for the report
    print(f"PLACEMENT_SUMMARY tag={tag} workers={nworkers} distinct_phys={distinct} "
          f"collisions={'yes' if collisions else 'no'}")


if __name__ == "__main__":
    main()
