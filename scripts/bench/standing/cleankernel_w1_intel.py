#!/usr/bin/env python3
# W1-on-Intel: gz silesia clean-path (-p1) causal decomposition (wall, best-of-N).
# perf blocked in LXC -> wall time. /dev/null sink, sha-gated, byte-transparent knobs.
import os, subprocess, time, sys, hashlib

GZ = os.environ.get("GZ", "/dev/shm/cleankernel-target/release/gzippy")
CORP = os.environ.get("CORP", "/root/silesia.gz")
N = int(os.environ.get("N", "11"))
CAP = "/dev/shm/ck_sil_p1.cap"


def sha(env):
    e = dict(os.environ); e.update(env)
    out = subprocess.run([GZ, "-dc", "-p1", CORP], capture_output=True, env=e).stdout
    return hashlib.sha256(out).hexdigest()


def best(env, n=N):
    e = dict(os.environ); e.update(env)
    m = None
    with open(os.devnull, "wb") as dn:
        for _ in range(n):
            t = time.perf_counter()
            subprocess.run([GZ, "-dc", "-p1", CORP], stdout=dn,
                           stderr=subprocess.DEVNULL, env=e)
            d = (time.perf_counter() - t) * 1000
            if m is None or d < m:
                m = d
    return m


ref = hashlib.sha256(subprocess.run(f"zcat {CORP}", shell=True,
      capture_output=True).stdout).hexdigest()
# record for NODECODE
subprocess.run([GZ, "-dc", "-p1", CORP], stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL, env={**os.environ, "GZIPPY_ORACLE_RECORD": CAP})
print("nodecode sha==ref:", sha({"GZIPPY_ORACLE_NODECODE": CAP}) == ref)
print("slowdec byte-transparent:", sha({"GZIPPY_SLOW_DECODE": "100"}) == ref)
print("slowsto byte-transparent:", sha({"GZIPPY_SLOW_STORE": "100"}) == ref)

# warmup
best({}, 2)
b = best({})
nd = best({"GZIPPY_ORACLE_NODECODE": CAP})
sd = best({"GZIPPY_SLOW_DECODE": "100"})
ss = best({"GZIPPY_SLOW_STORE": "100"})
print(f"base       : {b:.1f} ms")
print(f"nodecode   : {nd:.1f} ms  (removes Huffman decode+bitread)")
print(f"slowdec100 : {sd:.1f} ms")
print(f"slowsto100 : {ss:.1f} ms")
print(f"[x86] NODECODE decode-wall share = {(b-nd)/b*100:.1f}%  (copy/store/rest = {nd/b*100:.1f}%)")
print(f"[x86] causal slow-inject: SLOW_DECODE +{sd-b:.0f}ms  SLOW_STORE +{ss-b:.0f}ms")
print(f"[x86] store/decode WALL-criticality ratio = {(ss-b)/(sd-b):.2f}  (>1 => COPY/STORE more wall-critical)")
