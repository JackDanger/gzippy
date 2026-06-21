#!/usr/bin/env python3
"""thread_topology_probe.py — count a decoder's ACTIVE thread topology.

Launches a decode (stdout -> /dev/null) and samples /proc/<pid>/task to report:
  - max total threads spawned (the process tree's thread count)
  - max simultaneously RUNNING threads (state R) — the "active core demand"
  - the per-thread comm names seen (so a separate in-order consumer/writer thread
    is visible by name)

This tests the OVERSUBSCRIPTION MECHANISM: gz spawns 4 decode workers + 1 in-order
consumer (= 5 active). Does rapidgzip ALSO spawn a separate consumer (=5) or fold
it into the worker pool (=4)? If rg folds (4), a 4-core pin is unfair to gz's 5.

Usage: thread_topology_probe.py <label> <bin> <args...>
   e.g. thread_topology_probe.py gz /path/gzippy -dc -p4 /root/silesia.gz
"""
import sys, os, subprocess, time, collections

label = sys.argv[1]
cmd = sys.argv[2:]

env = dict(os.environ)
env["GZIPPY_FORCE_PARALLEL_SM"] = "1"

devnull = open("/dev/null", "wb")
proc = subprocess.Popen(cmd, stdout=devnull, stderr=subprocess.DEVNULL, env=env)
pid = proc.pid

max_total = 0
max_running = 0
comms = collections.Counter()
samples = 0

taskdir = f"/proc/{pid}/task"
while proc.poll() is None:
    try:
        tids = os.listdir(taskdir)
    except (FileNotFoundError, ProcessLookupError):
        break
    total = len(tids)
    running = 0
    for tid in tids:
        try:
            with open(f"{taskdir}/{tid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)
            # state is the first token after the closing paren of comm
            state = fields[1].split()[0]
            if state in ("R",):
                running += 1
            with open(f"{taskdir}/{tid}/comm") as fh:
                comms[fh.read().strip()] += 1
        except (FileNotFoundError, ProcessLookupError, IndexError):
            continue
    max_total = max(max_total, total)
    max_running = max(max_running, running)
    samples += 1
    time.sleep(0.003)

proc.wait()
# top comm names by frequency of observation
top = ", ".join(f"{name}" for name, _ in comms.most_common(12))
print(f"[{label}] max_total_threads={max_total}  max_running(stateR)={max_running}  "
      f"samples={samples}")
print(f"[{label}] thread comms seen: {top}")
