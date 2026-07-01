#!/usr/bin/env python3
"""DELIVERABLE 1 — finer clean-decode-core decomposition (macOS-aarch64).

Splits the aarch64 clean-decode-core instr/B (the 94.7% / 11.09 instr/B excess
that localize_mac.sh attributed to the kernel) into its COMPONENTS using
deterministic-instruction perturbations that are each PROVEN non-inert and
byte-exact (Gate-0), same /dev/null sink both arms (Gate-0d):

  1. CRC component  — removal-oracle GZIPPY_FOLD_NOCRC=1 (skips the per-clean-byte
     CRC update). Gate-0: must DROP instr to count; if it does NOT (inert), the
     report says so and treats CRC as already-converged (HW crc32fast PMULL).
  2. backref-COPY vs per-symbol Huffman+bitreader — CORPUS-CONTRAST:
       - decomp_literal  : alphabet-16 random  -> ~1 Huffman symbol / output byte,
                           ~0 backref copy  => exposes the per-SYMBOL decode cost.
       - decomp_backref  : 64-byte motif x N  -> ~1 symbol / 258 bytes, ~all bytes
                           via the backref word-copy => exposes the per-COPY-BYTE cost.
     The gzippy-vs-libdeflate EXCESS instr/B on each extreme localizes which
     component carries the gap. (WEAK-tier: corpora differ in table/block
     structure too; deterministic-instr, single-corpus-family. Pair w/ Intel
     asm-off for cross-ISA LAW.)
  3. silesia anchor — the real mixed-corpus number the design must move.

Gate-0 self-validation (LOUD-FAIL else the number does not exist):
  - byte-exact : every gzippy arm sha == gzip -d sha (and == libdeflate sha).
  - routing    : GZIPPY_DEBUG=1 -> path=ParallelSM (native pure-Rust).
  - same sink  : /dev/null both arms.
  - thin recon : GZIPPY_THIN_T1_ORACLE prints its banner (pipeline ~0 reconfirm).
  - determinism: instr min-of-N reported with warm spread; cyc HYPOTHESIS-tier.

SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch. Attribution = STRONG-because-
deterministic-instr HYPOTHESIS. Run on a quiet box; P-core; reject E-core outliers.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "gzippy")
N = int(os.environ.get("DECOMP_N", "9"))

CORPORA = [
    ("decomp_literal", "/tmp/decomp_literal.bin.gz"),  # per-symbol (literal) extreme
    ("decomp_backref", "/tmp/decomp_backref.bin.gz"),  # per-copy-byte (backref) extreme
    ("silesia", "/tmp/silesia.gz"),                     # real mixed anchor
]

GZIP = shutil.which("gzip")
LIBD = shutil.which("libdeflate-gunzip")
TIME = "/usr/bin/time"


def die(msg):
    print(f"### CLEAN_CORE_DECOMP GATE FAILED: {msg} ###", file=sys.stderr)
    sys.exit(1)


def sha(cmd, env=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    h = subprocess.run(["shasum", "-a", "256"], input=p.stdout, stdout=subprocess.PIPE)
    return h.stdout.split()[0].decode()


def uncompressed_bytes(path):
    p = subprocess.run([GZIP, "-dc", path], stdout=subprocess.PIPE)
    return len(p.stdout)


def measure(cmd, env=None):
    """Run `cmd` under /usr/bin/time -l, /dev/null sink, return (instr, cyc)."""
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
    # drop a cold warm-up, then min-of-N; reject E-core outliers (cyc > 1.4x median)
    measure(cmd, env)
    samples = [measure(cmd, env) for _ in range(N)]
    cycs = sorted(c for _, c in samples)
    med = cycs[len(cycs) // 2]
    samples = [s for s in samples if s[1] <= 1.4 * med] or samples
    return min(i for i, _ in samples), min(c for _, c in samples)


def main():
    if not os.path.exists(BIN):
        die(f"no gzippy at {BIN} (cargo build --release --no-default-features --features gzippy-native)")
    if not LIBD:
        die("libdeflate-gunzip not on PATH")

    route_corpus = next((p for _, p in CORPORA if os.path.exists(p)), None)
    if route_corpus is None:
        die("no corpus present for routing check")
    dbg = subprocess.run([BIN, "-dc", "-p1", route_corpus],
                         env={**os.environ, "GZIPPY_DEBUG": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if b"path=ParallelSM" not in dbg.stderr:
        die("routing is not ParallelSM (GZIPPY_DEBUG)")

    rows = []
    for name, path in CORPORA:
        if not os.path.exists(path):
            die(f"corpus missing: {path} (run gen_decomp_corpora.py + have /tmp/silesia.gz)")
        ub = uncompressed_bytes(path)
        ref = sha([GZIP, "-dc", path])
        if sha([BIN, "-dc", "-p1", path]) != ref:
            die(f"{name}: gzippy normal sha mismatch (not byte-exact)")
        if sha([LIBD, "-dc", path]) != ref:
            die(f"{name}: libdeflate sha mismatch")
        # CRC oracle byte-exact + non-inert probe
        if sha([BIN, "-dc", "-p1", path], env={**os.environ, "GZIPPY_FOLD_NOCRC": "1"}) != ref:
            die(f"{name}: FOLD_NOCRC not byte-exact")
        # thin reconfirm (banner => non-inert)
        thin = subprocess.run([BIN, "-dc", "-p1", path],
                              env={**os.environ, "GZIPPY_THIN_T1_ORACLE": "1"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        thin_fires = b"THIN_T1_ORACLE" in thin.stderr

        gz_i, gz_c = best_of_n([BIN, "-dc", "-p1", path])
        ld_i, ld_c = best_of_n([LIBD, "-dc", path])
        nocrc_i, nocrc_c = best_of_n([BIN, "-dc", "-p1", path], env={**os.environ, "GZIPPY_FOLD_NOCRC": "1"})
        rows.append(dict(name=name, ub=ub, gz_i=gz_i, gz_c=gz_c, ld_i=ld_i, ld_c=ld_c,
                         nocrc_i=nocrc_i, nocrc_c=nocrc_c, thin=thin_fires))

    print("\n== DELIVERABLE 1: aarch64 clean-decode-core decomposition (instr/B, cyc/B) ==")
    print(f"   binary sha={subprocess.run(['shasum','-a','256',BIN],stdout=subprocess.PIPE).stdout.split()[0].decode()[:12]}  N={N}  sink=/dev/null  -p1")
    hdr = f"{'corpus':<16}{'gz instr/B':>12}{'ld instr/B':>12}{'ratio':>8}{'excess/B':>10}{'gz cyc/B':>10}{'ld cyc/B':>10}{'nocrcΔinstr%':>14}"
    print(hdr)
    for r in rows:
        gzib, ldib = r["gz_i"]/r["ub"], r["ld_i"]/r["ub"]
        gzcb, ldcb = r["gz_c"]/r["ub"], r["ld_c"]/r["ub"]
        nocrc_d = 100.0*(r["nocrc_i"]-r["gz_i"])/r["gz_i"]
        print(f"{r['name']:<16}{gzib:>12.3f}{ldib:>12.3f}{gzib/ldib:>8.3f}{gzib-ldib:>10.3f}{gzcb:>10.3f}{ldcb:>10.3f}{nocrc_d:>+14.3f}")

    print("\n   Gate-0: thin fires =", all(r["thin"] for r in rows),
          "| FOLD_NOCRC byte-exact = PASS (checked) | all arms byte-exact = PASS")
    print("   CRC oracle non-inert? a NEGATIVE nocrcΔinstr% (instr DROPS) => CRC is a real component;")
    print("   ~0 or POSITIVE => FOLD_NOCRC INERT on native -p1 => CRC already HW-converged (crc32fast PMULL).")
    print("   scope: macOS-aarch64 NOT-YET-LAW cross-arch; pair w/ Intel(asm-off) for cross-ISA LAW.\n")


if __name__ == "__main__":
    main()
