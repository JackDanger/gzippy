#!/usr/bin/env python3
# quick placement probe: where do gz's threads run? arg1=gz arg2=corp arg3=threads
# env GZIPPY_PHYS_PIN passes through from caller.
import os, subprocess, sys, time

gz, corp, threads = sys.argv[1], sys.argv[2], sys.argv[3]
env = dict(os.environ)
env["GZIPPY_FORCE_PARALLEL_SM"] = "1"
p = subprocess.Popen([gz, "-dc", f"-p{threads}", corp],
                     stdout=open(os.devnull, "wb"), stderr=subprocess.DEVNULL, env=env)
pid = p.pid
seen = {}
while p.poll() is None:
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        break
    for t in tids:
        try:
            d = open(f"/proc/{pid}/task/{t}/stat", "rb").read()
        except OSError:
            continue
        rest = d[d.rindex(b")") + 2:].split()
        try:
            cpu = int(rest[36])
        except (IndexError, ValueError):
            continue
        seen.setdefault(t, set()).add(cpu)
    time.sleep(0.001)
p.wait()
for t, c in sorted(seen.items()):
    role = "MAIN" if t == str(pid) else "wk"
    print(f"  tid {t} {role} cpus={sorted(c)}")
