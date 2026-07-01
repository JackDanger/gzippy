#!/usr/bin/env python3
"""ENGINE-A CONVERGENCE instrument (macOS-aarch64) — STEP B-2.

Deterministic-instr screen + cyc/B for engine A (gzippy-native clean kernel)
vs libdeflate-gunzip (the frontier comparator) across the real frontier corpora
(silesia, monorepo, nasa) PLUS the two per-symbol/per-copy extremes
(decomp_literal, decomp_backref). Used to gate a byte-exact engine-A convergence
increment: KEEP iff byte-exact + instr/B drops toward libdeflate + the aarch64
T1 frontier gap shrinks, with no regression on the other corpora.

Gate-0 (LOUD-FAIL):
  - byte-exact: every gz arm sha == gzip -d sha == libdeflate sha.
  - routing:    GZIPPY_DEBUG=1 -> path=ParallelSM.
  - same sink:  /dev/null both arms.
Gate-1: best-of-N (instr min-of-N load-immune; cyc min-of-N HYPOTHESIS-tier),
  warm spread reported; reject E-core cyc outliers (>1.4x median).

SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch (AMD/Intel-asm-off owed).
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BIN = os.environ.get("GZ_BIN", os.path.join(ROOT, "target", "release", "gzippy"))
N = int(os.environ.get("DECOMP_N", "13"))

CORPORA = [
    ("decomp_literal", "/tmp/decomp_literal.bin.gz"),
    ("decomp_backref", "/tmp/decomp_backref.bin.gz"),
    ("silesia", "/tmp/silesia.gz"),
    ("monorepo", "/tmp/monorepo.gz"),
    ("nasa", "/tmp/nasa.gz"),
]

GZIP = shutil.which("gzip")
LIBD = shutil.which("libdeflate-gunzip")
TIME = "/usr/bin/time"


def die(msg):
    print(f"### ENGINEA_CONVERGE GATE FAILED: {msg} ###", file=sys.stderr)
    sys.exit(1)


def sha(cmd, env=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    h = subprocess.run(["shasum", "-a", "256"], input=p.stdout, stdout=subprocess.PIPE)
    return h.stdout.split()[0].decode()


def uncompressed_bytes(path):
    p = subprocess.run([GZIP, "-dc", path], stdout=subprocess.PIPE)
    return len(p.stdout)


def measure(cmd, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    with open(os.devnull, "wb") as dn:
        p = subprocess.run([TIME, "-l"] + cmd, stdout=dn, stderr=subprocess.PIPE, env=full_env)
    instr = cyc = None
    for line in p.stderr.decode().splitlines():
        s = line.split()
        if "instructions retired" in line:
            instr = int(s[0])
        elif "cycles elapsed" in line:
            cyc = int(s[0])
    if instr is None or cyc is None:
        die(f"could not parse time -l output for {cmd}")
    return instr, cyc


def best_of_n(cmd, env=None):
    measure(cmd, env)  # drop a cold warm-up
    samples = [measure(cmd, env) for _ in range(N)]
    cycs = sorted(c for _, c in samples)
    med = cycs[len(cycs) // 2]
    filt = [s for s in samples if s[1] <= 1.4 * med] or samples
    return (min(i for i, _ in filt), min(c for _, c in filt),
            (max(i for i, _ in filt) - min(i for i, _ in filt)),
            (max(c for _, c in filt) - min(c for _, c in filt)))


def main():
    if not os.path.exists(BIN):
        die(f"no gzippy at {BIN}")
    if not LIBD:
        die("libdeflate-gunzip not on PATH")

    route_corpus = next((p for _, p in CORPORA if os.path.exists(p)), None)
    dbg = subprocess.run([BIN, "-dc", "-p1", route_corpus],
                         env={**os.environ, "GZIPPY_DEBUG": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if b"path=ParallelSM" not in dbg.stderr:
        die("routing is not ParallelSM")

    rows = []
    for name, path in CORPORA:
        if not os.path.exists(path):
            print(f"   (skip {name}: {path} missing)")
            continue
        ub = uncompressed_bytes(path)
        ref = sha([GZIP, "-dc", path])
        if sha([BIN, "-dc", "-p1", path]) != ref:
            die(f"{name}: gzippy sha mismatch (not byte-exact)")
        if sha([LIBD, "-dc", path]) != ref:
            die(f"{name}: libdeflate sha mismatch")
        gz_i, gz_c, gz_isp, gz_csp = best_of_n([BIN, "-dc", "-p1", path])
        ld_i, ld_c, _, _ = best_of_n([LIBD, "-dc", path])
        rows.append(dict(name=name, ub=ub, gz_i=gz_i, gz_c=gz_c, ld_i=ld_i,
                         ld_c=ld_c, gz_isp=gz_isp, gz_csp=gz_csp))

    binsha = subprocess.run(['shasum', '-a', '256', BIN], stdout=subprocess.PIPE).stdout.split()[0].decode()[:12]
    print(f"\n== ENGINE-A CONVERGENCE (instr/B, cyc/B) bin={binsha} N={N} sink=/dev/null -p1 ==")
    hdr = (f"{'corpus':<16}{'gz instr/B':>12}{'ld instr/B':>12}{'i-ratio':>9}"
           f"{'gz cyc/B':>10}{'ld cyc/B':>10}{'c-ratio':>9}{'i-spr%':>8}{'c-spr%':>8}")
    print(hdr)
    for r in rows:
        gzib, ldib = r["gz_i"]/r["ub"], r["ld_i"]/r["ub"]
        gzcb, ldcb = r["gz_c"]/r["ub"], r["ld_c"]/r["ub"]
        isp = 100.0*r["gz_isp"]/r["gz_i"]
        csp = 100.0*r["gz_csp"]/r["gz_c"]
        print(f"{r['name']:<16}{gzib:>12.3f}{ldib:>12.3f}{gzib/ldib:>9.3f}"
              f"{gzcb:>10.3f}{ldcb:>10.3f}{gzcb/ldcb:>9.3f}{isp:>8.2f}{csp:>8.2f}")
    print("   i-ratio/c-ratio = gz/libdeflate (1.0 = parity; <1.0 = gz faster). "
          "instr load-immune; cyc HYPOTHESIS-tier (quiet box).")
    print("   SCOPE: macOS-aarch64 NOT-YET-LAW; AMD/Intel-asm-off owed.\n")


if __name__ == "__main__":
    main()
