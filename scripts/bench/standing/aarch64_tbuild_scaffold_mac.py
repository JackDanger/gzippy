#!/usr/bin/env python3
"""B-1 conservation closer (aarch64): apportion table-build + pipeline-scaffold.

Complements clean_core_decomp_mac.py (which isolated CRC[~0 / HW-converged] and
per-symbol-Huffman vs backref-copy). This closes conservation on the silesia
mixed anchor by sizing the two remaining buckets with byte-exact, non-inert,
deterministic-instruction perturbations (Gate-0):

  - TABLE-BUILD share: GZIPPY_TBUILD_MULT=N runs the per-block litlen LUT build N
    times into the SAME idempotent state. slope = (instr@M2 - instr@M1)/(M2-M1)
    per byte = ONE table-build's per-byte instruction cost (byte-transparent: the
    table the decode reads is byte-identical at any N).
  - PIPELINE-SCAFFOLD share: GZIPPY_THIN_T1_ORACLE swaps the parallel driver for a
    thin serial rolling-window pass (CRC/size still verified => byte-exact +
    non-inert). delta = full-pipeline instr - thin instr = the scaffold tax.

Whatever excess remains after CRC(~0) + copy + table-build + scaffold IS the
per-symbol Huffman-decode + bitreader core (the lever).

Gate-0: byte-exact every arm (sha == gzip -d); path=ParallelSM; thin banner fires;
TBUILD slope must be POSITIVE (non-inert) else table-build is reported inert.
SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch; deterministic-instr (load-immune).
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "gzippy")
N = int(os.environ.get("DECOMP_N", "9"))
GZIP = shutil.which("gzip")
TIME = "/usr/bin/time"
PATH = os.environ.get("CORPUS", "/tmp/silesia.gz")


def die(msg):
    print(f"### TBUILD_SCAFFOLD GATE FAILED: {msg} ###", file=sys.stderr)
    sys.exit(1)


def sha(cmd, env=None):
    full = dict(os.environ)
    if env:
        full.update(env)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=full)
    h = subprocess.run(["shasum", "-a", "256"], input=p.stdout, stdout=subprocess.PIPE)
    return h.stdout.split()[0].decode()


def measure(cmd, env=None):
    full = dict(os.environ)
    if env:
        full.update(env)
    with open(os.devnull, "wb") as dn:
        p = subprocess.run([TIME, "-l"] + cmd, stdout=dn, stderr=subprocess.PIPE, env=full)
    instr = cyc = None
    for line in p.stderr.decode().splitlines():
        s = line.split()
        if "instructions retired" in line:
            instr = int(s[0])
        elif "cycles elapsed" in line:
            cyc = int(s[0])
    if instr is None:
        die(f"no instr parse for {cmd}")
    return instr, cyc


def best(cmd, env=None):
    measure(cmd, env)
    samples = [measure(cmd, env) for _ in range(N)]
    cycs = sorted(c for _, c in samples)
    med = cycs[len(cycs) // 2]
    samples = [s for s in samples if s[1] <= 1.4 * med] or samples
    return min(i for i, _ in samples), min(c for _, c in samples)


def main():
    if not os.path.exists(BIN):
        die("no gzippy binary")
    ref = sha([GZIP, "-dc", PATH])
    ub = len(subprocess.run([GZIP, "-dc", PATH], stdout=subprocess.PIPE).stdout)

    # Gate-0 routing
    dbg = subprocess.run([BIN, "-dc", "-p1", PATH], env={**os.environ, "GZIPPY_DEBUG": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if b"path=ParallelSM" not in dbg.stderr:
        die("not ParallelSM")

    # Gate-0 byte-exact for every arm
    for label, env in [("normal", None),
                       ("tbuild2", {"GZIPPY_TBUILD_MULT": "2"}),
                       ("tbuild4", {"GZIPPY_TBUILD_MULT": "4"}),
                       ("thin", {"GZIPPY_THIN_T1_ORACLE": "1"})]:
        if sha([BIN, "-dc", "-p1", PATH], env=env) != ref:
            die(f"{label} not byte-exact")
    thin_fires = b"THIN_T1_ORACLE" in subprocess.run(
        [BIN, "-dc", "-p1", PATH], env={**os.environ, "GZIPPY_THIN_T1_ORACLE": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).stderr

    base_i, base_c = best([BIN, "-dc", "-p1", PATH])
    tb2_i, _ = best([BIN, "-dc", "-p1", PATH], env={"GZIPPY_TBUILD_MULT": "2"})
    tb4_i, _ = best([BIN, "-dc", "-p1", PATH], env={"GZIPPY_TBUILD_MULT": "4"})
    thin_i, thin_c = best([BIN, "-dc", "-p1", PATH], env={"GZIPPY_THIN_T1_ORACLE": "1"})

    # table-build per-byte = slope (instr per extra full rebuild) / ub
    slope_per_rebuild = (tb4_i - tb2_i) / 2.0  # (M4-M2)/(4-2)
    tbuild_per_b = slope_per_rebuild / ub
    scaffold_per_b = (base_i - thin_i) / ub

    print("\n== B-1 CONSERVATION CLOSER (aarch64): table-build + scaffold ==")
    print(f"   corpus={os.path.basename(PATH)} ub={ub} N={N} sink=/dev/null -p1")
    print(f"   base instr/B          = {base_i/ub:.4f}   (cyc/B {base_c/ub:.4f})")
    print(f"   TBUILD_MULT=2 instr/B = {tb2_i/ub:.4f}")
    print(f"   TBUILD_MULT=4 instr/B = {tb4_i/ub:.4f}")
    print(f"   thin-T1     instr/B   = {thin_i/ub:.4f}   (cyc/B {thin_c/ub:.4f})")
    print(f"   -> TABLE-BUILD per-byte  = {tbuild_per_b:.4f} instr/B  ({100*tbuild_per_b/(base_i/ub):.2f}% of gz)  [slope, non-inert={'YES' if slope_per_rebuild>0 else 'NO/INERT'}]")
    print(f"   -> PIPELINE-SCAFFOLD     = {scaffold_per_b:.4f} instr/B  ({100*scaffold_per_b/(base_i/ub):.2f}% of gz)  [full - thin]")
    print(f"   Gate-0: thin fires={bool(thin_fires)} | all arms byte-exact=PASS")
    print("   scope: macOS-aarch64 NOT-YET-LAW; deterministic-instr.\n")


if __name__ == "__main__":
    main()
