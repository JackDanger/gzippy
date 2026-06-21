#!/usr/bin/env python3
# rg_marker_added.py — measure rapidgzip ABSOLUTE instructions retired at -P1 and -P4
# via /usr/bin/time -l, interleaved best-of-N, /dev/null sink. Compute rg's
# marker-added instr/B = (instrP4 - instrP1)/bytes  (clean absolute subtraction,
# NO ratio*ratio / share*wall). This is the rg-side analog of gz's base4-base1 tax.
# MEASUREMENT only. macOS-aarch64 (rg has NO ISA-L here — scope caveat).
import subprocess, re, sys, statistics

RG = "/tmp/rgvenv/bin/rapidgzip"
CORPORA = {
    "silesia":  ("/tmp/silesia.gz",  211968000),
    "monorepo": ("/tmp/monorepo.gz",  50915328),
    "nasa":     ("/tmp/nasa.gz",     205242368),
    "squishy":  ("/tmp/squishy.gz",  400391411),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
THREADS = [1, 4]

instr_re = re.compile(r'^\s*(\d+)\s+instructions retired')

def run_once(path, t):
    # /usr/bin/time -l writes PMU to stderr; rg -dc to stdout -> /dev/null
    p = subprocess.run(
        ["/usr/bin/time", "-l", RG, "-dc", "-P", str(t), path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for line in p.stderr.decode().splitlines():
        m = instr_re.match(line)
        if m:
            return int(m.group(1))
    raise RuntimeError("no instr counter:\n" + p.stderr.decode())

# warmup
for c,(path,_) in CORPORA.items():
    for t in THREADS:
        run_once(path, t)

# interleaved best-of-N
samples = {(c,t): [] for c in CORPORA for t in THREADS}
for rep in range(N):
    for c,(path,_) in CORPORA.items():
        for t in THREADS:
            samples[(c,t)].append(run_once(path, t))

print(f"=== rapidgzip ABSOLUTE instructions retired (best-of-N={N}, /dev/null) ===")
print(f"{'corpus':9} {'rgP1 instr/B':>12} {'rgP4 instr/B':>12} {'rg mkr-added/B':>14} {'P1 spr%':>8} {'P4 spr%':>8}")
results = {}
for c,(path,nb) in CORPORA.items():
    p1 = min(samples[(c,1)]); p4 = min(samples[(c,4)])
    p1b = p1/nb; p4b = p4/nb
    spr1 = (max(samples[(c,1)])-p1)/p1*100
    spr4 = (max(samples[(c,4)])-p4)/p4*100
    added = p4b - p1b
    results[c] = (p1b, p4b, added)
    print(f"{c:9} {p1b:12.3f} {p4b:12.3f} {added:14.3f} {spr1:8.2f} {spr4:8.2f}")
print()
print("rg mkr-added/B = (rgP4 - rgP1) instr/B  [clean absolute subtraction; includes rg coordination]")
