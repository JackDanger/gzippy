#!/usr/bin/env python3
"""B-1 x86 T1 frontier-gap apportionment (run ON the FROZEN Intel guest).

Apportions gz-native (asm run_contig kernel) T1 single-core gap vs igzip into:
  - CRC            : GZIPPY_FOLD_NOCRC=1 removal-oracle (x86 crc32fast = PCLMUL HW)
  - table-build    : GZIPPY_TBUILD_MULT slope (2 vs 4 full idempotent rebuilds)
  - pipeline-scaff : GZIPPY_THIN_T1_ORACLE (thin serial driver, SAME decode kernel)
  - clean-decode   : remainder = run_contig asm kernel + decode_clean_into_contig
                     wrapper (the bucket the +30% must live in if it's the kernel)

Gate-0: byte-exact every gz arm (sha==gzip -d); path=ParallelSM; thin banner fires;
TBUILD slope POSITIVE (non-inert) else inert. /dev/null sink ALL arms (incl igzip).
Gate-1: perf stat min-of-N (cyc load-sensitive -> FROZEN box; instr load-immune).
SCOPE: Intel x86_64 FROZEN, NOT-YET-LAW (AMD owed). cyc/B is the wall proxy here.
"""
import os
import subprocess
import sys

BIN = os.environ.get("GZBIN", "/mnt/internal/gz-head/target/release/gzippy")
IGZIP = "/usr/bin/igzip"
GZIP = "/usr/bin/gzip"
N = int(os.environ.get("N", "11"))
CORPORA = [("silesia", "/root/silesia.gz"),
           ("monorepo", "/root/monorepo.gz"),
           ("nasa", "/root/nasa.gz")]


def die(m):
    print(f"### X86_APPORTION GATE FAILED: {m} ###", file=sys.stderr)
    sys.exit(1)


def sha(cmd, env=None):
    full = dict(os.environ)
    if env:
        full.update(env)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=full)
    return subprocess.run(["sha256sum"], input=p.stdout,
                          stdout=subprocess.PIPE).stdout.split()[0].decode()


def perf_once(cmd, env=None):
    full = dict(os.environ)
    if env:
        full.update(env)
    with open(os.devnull, "wb") as dn:
        p = subprocess.run(["perf", "stat", "-x,", "-e", "instructions:u,cycles:u"] + cmd,
                           stdout=dn, stderr=subprocess.PIPE, env=full)
    instr = cyc = None
    for line in p.stderr.decode().splitlines():
        f = line.split(",")
        if len(f) < 3 or f[0] in ("<not counted>", "<not supported>", ""):
            continue
        if "instructions:u" in f[2]:
            instr = int(f[0])
        elif "cycles:u" in f[2]:
            cyc = int(f[0])
    if instr is None or cyc is None:
        die(f"perf parse fail: {cmd}\n{p.stderr.decode()[:400]}")
    return instr, cyc


def best(cmd, env=None):
    perf_once(cmd, env)  # warm
    s = [perf_once(cmd, env) for _ in range(N)]
    cycs = sorted(c for _, c in s)
    med = cycs[len(cycs) // 2]
    s = [x for x in s if x[1] <= 1.4 * med] or s
    return min(i for i, _ in s), min(c for _, c in s)


def main():
    if not os.path.exists(BIN):
        die(f"no gzippy at {BIN}")
    rows = []
    for name, path in CORPORA:
        if not os.path.exists(path):
            print(f"  skip {name}: missing {path}", file=sys.stderr)
            continue
        ub = len(subprocess.run([GZIP, "-dc", path], stdout=subprocess.PIPE).stdout)
        ref = sha([GZIP, "-dc", path])
        dbg = subprocess.run([BIN, "-dc", "-p1", path],
                             env={**os.environ, "GZIPPY_DEBUG": "1"},
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if b"path=ParallelSM" not in dbg.stderr:
            die(f"{name}: not ParallelSM")
        for label, env in [("norm", None), ("nocrc", {"GZIPPY_FOLD_NOCRC": "1"}),
                           ("tb2", {"GZIPPY_TBUILD_MULT": "2"}),
                           ("tb4", {"GZIPPY_TBUILD_MULT": "4"}),
                           ("thin", {"GZIPPY_THIN_T1_ORACLE": "1"})]:
            if sha([BIN, "-dc", "-p1", path], env=env) != ref:
                die(f"{name}/{label}: not byte-exact")
        if sha([IGZIP, "-dc", path]) != ref:
            die(f"{name}: igzip sha mismatch")
        thin_fires = b"THIN_T1_ORACLE" in subprocess.run(
            [BIN, "-dc", "-p1", path],
            env={**os.environ, "GZIPPY_THIN_T1_ORACLE": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).stderr

        gz_i, gz_c = best([BIN, "-dc", "-p1", path])
        ig_i, ig_c = best([IGZIP, "-dc", path])
        nc_i, _ = best([BIN, "-dc", "-p1", path], env={"GZIPPY_FOLD_NOCRC": "1"})
        t2_i, _ = best([BIN, "-dc", "-p1", path], env={"GZIPPY_TBUILD_MULT": "2"})
        t4_i, _ = best([BIN, "-dc", "-p1", path], env={"GZIPPY_TBUILD_MULT": "4"})
        th_i, th_c = best([BIN, "-dc", "-p1", path], env={"GZIPPY_THIN_T1_ORACLE": "1"})
        rows.append(dict(name=name, ub=ub, gz_i=gz_i, gz_c=gz_c, ig_i=ig_i, ig_c=ig_c,
                         nc_i=nc_i, t2_i=t2_i, t4_i=t4_i, th_i=th_i, th_c=th_c,
                         thin=thin_fires))

    print("\n== B-1 x86 T1 FRONTIER-GAP APPORTIONMENT (gz asm run_contig vs igzip) ==")
    print(f"   bin={BIN}")
    print(f"   N={N} sink=/dev/null -p1  (FROZEN-box cyc; instr load-immune)")
    h = (f"{'corpus':<10}{'gz cyc/B':>10}{'ig cyc/B':>10}{'cyc gap%':>9}"
         f"{'gz i/B':>9}{'ig i/B':>9}{'i gap%':>8}"
         f"{'CRC%':>7}{'tbuild/B':>9}{'tb%':>6}{'scaff%':>8}{'kernel%':>9}")
    print(h)
    for r in rows:
        ub = r["ub"]
        gzc, igc = r["gz_c"]/ub, r["ig_c"]/ub
        gzi, igi = r["gz_i"]/ub, r["ig_i"]/ub
        cyc_gap = 100*(gzc-igc)/igc
        i_gap = 100*(gzi-igi)/igi
        crc_pct = 100*(r["gz_i"]-r["nc_i"])/r["gz_i"]          # instr dropped by nocrc
        tb_per_rebuild = (r["t4_i"]-r["t2_i"])/2.0
        tb_per_b = tb_per_rebuild/ub
        tb_pct = 100*tb_per_b/gzi
        scaff_pct = 100*(r["gz_i"]-r["th_i"])/r["gz_i"]
        kernel_pct = 100 - crc_pct - tb_pct - scaff_pct
        print(f"{r['name']:<10}{gzc:>10.3f}{igc:>10.3f}{cyc_gap:>+9.1f}"
              f"{gzi:>9.3f}{igi:>9.3f}{i_gap:>+8.1f}"
              f"{crc_pct:>+7.2f}{tb_per_b:>9.4f}{tb_pct:>6.2f}{scaff_pct:>+8.2f}{kernel_pct:>9.2f}")
    print("\n   CRC%/scaff%: instr-share REMOVED by the oracle (POSITIVE=real, ~0=inert/converged).")
    print("   tbuild/B: ONE table-build's per-byte instr (slope); tb% its share of gz.")
    print("   kernel% = 100 - CRC - tbuild - scaffold = run_contig asm + decode_clean_into_contig wrapper.")
    print("   Gate-0: thin fires =", all(r["thin"] for r in rows), "| all gz arms byte-exact = PASS")
    print("   scope: Intel x86_64 FROZEN, NOT-YET-LAW (AMD/Zen2 owed).\n")


if __name__ == "__main__":
    main()
