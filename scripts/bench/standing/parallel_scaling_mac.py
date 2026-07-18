#!/usr/bin/env python3
"""parallel_scaling_mac.py — fine-wall parallel-scaling disentangler (macOS aarch64).

Answers, deterministically on a QUIET mac, the standing-matrix question:
  - Does gzippy-native ParallelSM scale MONOTONICALLY (T2>T1>...) on a quiet box?
  - How much of any T2<T1 regression is the T1-special-path (1 MiB inline vs
    4 MiB pipeline) vs genuine pipeline fixed-overhead vs box contention?

Measurement primitives (Gate-1 fine timing — NOT the 10 ms /usr/bin/time `real`):
  - WALL: time.perf_counter() around each subprocess (microsecond resolution),
    interleaved best-of-N, report min (floor; noise is additive) + median + spread.
  - CPU:  /usr/bin/time -l -> instructions retired (deterministic ~0.04%) and
    cycles elapsed; reported per decompressed byte (instr/B, cyc/B).
  - STARTUP: a near-empty .gz decoded at the same -pN measures process + thread-pool
    spawn fixed cost; reported and subtracted to give decode-only wall.

Gate-0 (enforced by the caller standing wrapper / asserted here):
  byte-exact sha (checked once per corpus), path=ParallelSM, /dev/null both arms,
  GZIPPY_CHUNK_KIB perturbation proven non-inert via 'Total Fetched' chunk count.

Usage:
  parallel_scaling_mac.py --mode scaling   --out /tmp/scal.csv
  parallel_scaling_mac.py --mode chunkfix  --out /tmp/chunk.csv
"""
import argparse, os, re, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BIN = os.path.join(ROOT, "target/release/gzippy")
DEVNULL = open(os.devnull, "wb")

CORPORA = {
    # name: (path, label, real?)
    "silesia3": ("/tmp/silesia3.gz", "silesia x3 (real, 635MB single-member)", True),
    "silesia":  ("/tmp/silesia.gz",  "silesia (real, 212MB single-member)", True),
    "big2":     ("/tmp/big2.gz",     "big2 (SYNTHETIC highly-redundant, 554MB)", False),
}
TINY = "/tmp/_scal_tiny.gz"  # near-empty, for startup measurement

def decompressed_bytes(path):
    p = subprocess.run(["gzip", "-dc", path], stdout=subprocess.PIPE, stderr=DEVNULL)
    return len(p.stdout)

def sha(path_or_cmd_bytes):
    pass

def gz_sha(path, env=None, threads=1):
    p1 = subprocess.run([BIN, "-dc", "-p%d" % threads, path], stdout=subprocess.PIPE,
                        stderr=DEVNULL, env={**os.environ, **(env or {})})
    import hashlib
    return hashlib.sha256(p1.stdout).hexdigest()

def ref_sha(path):
    import hashlib
    p = subprocess.run(["gzip", "-dc", path], stdout=subprocess.PIPE, stderr=DEVNULL)
    return hashlib.sha256(p.stdout).hexdigest()

def run_wall(threads, env_extra, path):
    env = {**os.environ, **(env_extra or {})}
    args = [BIN, "-dc", "-p%d" % threads, path]
    t0 = time.perf_counter()
    r = subprocess.run(args, stdout=DEVNULL, stderr=DEVNULL, env=env)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError("gzippy rc=%d t=%d env=%s" % (r.returncode, threads, env_extra))
    return dt

def run_timel(threads, env_extra, path):
    env = {**os.environ, **(env_extra or {})}
    args = ["/usr/bin/time", "-l", BIN, "-dc", "-p%d" % threads, path]
    r = subprocess.run(args, stdout=DEVNULL, stderr=subprocess.PIPE, env=env)
    txt = r.stderr.decode()
    mi = re.search(r"(\d+)\s+instructions retired", txt)
    mc = re.search(r"(\d+)\s+cycles elapsed", txt)
    if not (mi and mc):
        raise RuntimeError("time -l parse fail:\n" + txt)
    return int(mi.group(1)), int(mc.group(1))

def chunk_count(threads, env_extra, path):
    env = {**os.environ, **(env_extra or {}), "GZIPPY_VERBOSE": "1"}
    r = subprocess.run([BIN, "-dc", "-p%d" % threads, path], stdout=DEVNULL,
                       stderr=subprocess.PIPE, env=env)
    m = re.search(r"Total Fetched\s*:\s*(\d+)", r.stderr.decode())
    return int(m.group(1)) if m else -1

def summarize(samples):
    s = sorted(samples)
    mn = s[0]
    med = statistics.median(s)
    # inter-run spread as a fraction of the median (Gate-1 spread)
    p10 = s[max(0, int(0.1 * (len(s) - 1)))]
    p90 = s[int(0.9 * (len(s) - 1))]
    spread = (p90 - p10) / med if med else 0.0
    return mn, med, spread

def measure(configs, N):
    """configs: list of dict(label, threads, env, path). Returns dict label->results."""
    res = {c["label"]: {"wall": [], "instr": [], "cyc": [], **c} for c in configs}
    # warmup (untimed) once each
    for c in configs:
        run_wall(c["threads"], c["env"], c["path"])
    # interleaved wall
    for rep in range(N):
        for c in configs:
            res[c["label"]]["wall"].append(run_wall(c["threads"], c["env"], c["path"]))
    # interleaved cpu (separate pass; /usr/bin/time perturbs wall so keep apart)
    for rep in range(N):
        for c in configs:
            ins, cyc = run_timel(c["threads"], c["env"], c["path"])
            res[c["label"]]["instr"].append(ins)
            res[c["label"]]["cyc"].append(cyc)
    return res

def make_tiny():
    import gzip
    with gzip.open(TINY, "wb") as f:
        f.write(b"x")

def gate0(paths):
    print("== GATE-0: build-flavor + path + byte-exact + non-inert ==")
    dbg = subprocess.run([BIN, "-dc", "-p1", paths[0]], stdout=DEVNULL,
                         stderr=subprocess.PIPE, env={**os.environ, "GZIPPY_DEBUG": "1"}).stderr.decode()
    assert "build-flavor=parallel-sm+pure" in dbg, "build-flavor != pure"
    assert "path=ParallelSM" in dbg, "path != ParallelSM"
    print("   PASS build-flavor=parallel-sm+pure path=ParallelSM")
    for p in paths:
        r = ref_sha(p)
        g = gz_sha(p, threads=4)
        assert r == g, "SHA MISMATCH %s" % p
        print("   PASS sha %s == gzip -d (-p4)" % os.path.basename(p))

def report(title, res, byte_map, baseline_label=None):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    hdr = "%-34s %8s %8s %7s %9s %8s %8s" % (
        "config", "wall_ms", "med_ms", "spr%", "MB/s", "cyc/B", "instr/B")
    print(hdr)
    rows = {}
    for label, r in res.items():
        nb = byte_map[label]
        wmn, wmed, wspr = summarize(r["wall"])
        imn = min(r["instr"]); cmn = min(r["cyc"])
        mbps = nb / wmn / 1e6
        cycB = cmn / nb; insB = imn / nb
        rows[label] = dict(wmn=wmn, wmed=wmed, wspr=wspr, mbps=mbps, cycB=cycB, insB=insB, nb=nb)
        print("%-34s %8.1f %8.1f %6.1f%% %9.1f %8.3f %8.3f" % (
            label, wmn * 1e3, wmed * 1e3, wspr * 100, mbps, cycB, insB))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["scaling", "chunkfix"])
    ap.add_argument("-N", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    make_tiny()
    if args.mode == "scaling":
        corps = ["silesia3", "big2"]
        paths = [CORPORA[c][0] for c in corps]
        gate0(paths)
        for cname in corps:
            path, clabel, real = CORPORA[cname]
            nb = decompressed_bytes(path)
            configs = []
            for t in (1, 2, 4, 8):
                configs.append(dict(label="%s T%d" % (cname, t), threads=t, env={}, path=path))
            # startup configs (tiny input at each T)
            for t in (1, 2, 4, 8):
                configs.append(dict(label="startup T%d" % t, threads=t, env={}, path=TINY))
            res = measure(configs, args.N)
            byte_map = {}
            for c in configs:
                byte_map[c["label"]] = nb if c["path"] != TINY else 1
            print("\n### CORPUS: %s — %s ###" % (cname, clabel))
            print("   chunk counts:", {("T%d" % t): chunk_count(t, {}, path) for t in (1, 2, 4, 8)})
            rows = report("SCALING %s (raw end-to-end wall)" % cname, res, byte_map)
            # startup-subtracted decode-only
            print("\n  decode-only (raw wall minus measured startup at same -pN):")
            print("  %-10s %9s %9s %9s %9s %9s" % ("T", "startup_ms", "decode_ms", "MB/s", "speedup", "cyc/B"))
            base = None
            for t in (1, 2, 4, 8):
                lab = "%s T%d" % (cname, t)
                stp = rows["startup T%d" % t]["wmn"]
                dec = rows[lab]["wmn"] - stp
                mbps = nb / dec / 1e6
                if t == 1:
                    base = dec
                spd = base / dec
                print("  T%-9d %9.1f %9.1f %9.1f %8.2fx %9.3f" % (
                    t, stp * 1e3, dec * 1e3, mbps, spd, rows[lab]["cycB"]))

    else:  # chunkfix
        cname = "silesia3"
        path = CORPORA[cname][0]
        gate0([path])
        nb = decompressed_bytes(path)
        print("\n### CHUNK-FIXED A/B on %s ###" % cname)
        # prove non-inert
        cc = {
            "T1@1MiB": chunk_count(1, {"GZIPPY_CHUNK_KIB": "1024"}, path),
            "T1@4MiB": chunk_count(1, {"GZIPPY_CHUNK_KIB": "4096"}, path),
            "T2@1MiB": chunk_count(2, {"GZIPPY_CHUNK_KIB": "1024"}, path),
            "T2@4MiB": chunk_count(2, {"GZIPPY_CHUNK_KIB": "4096"}, path),
        }
        print("   non-inert chunk counts (Total Fetched):", cc)
        configs = [
            dict(label="T1 default(1MiB)", threads=1, env={}, path=path),
            dict(label="T2 default(4MiB)", threads=2, env={}, path=path),
            dict(label="T1 @1MiB", threads=1, env={"GZIPPY_CHUNK_KIB": "1024"}, path=path),
            dict(label="T2 @1MiB", threads=2, env={"GZIPPY_CHUNK_KIB": "1024"}, path=path),
            dict(label="T1 @4MiB", threads=1, env={"GZIPPY_CHUNK_KIB": "4096"}, path=path),
            dict(label="T2 @4MiB", threads=2, env={"GZIPPY_CHUNK_KIB": "4096"}, path=path),
        ]
        res = measure(configs, args.N)
        byte_map = {c["label"]: nb for c in configs}
        rows = report("CHUNK-FIXED A/B (silesia3)", res, byte_map)
        print("\n  --- disentangle (cyc/B = deterministic CPU primitive) ---")
        def d(a, b, key):
            return rows[a][key] - rows[b][key]
        print("  T2<T1 at DEFAULT paths (T1=1MiB inline vs T2=4MiB pipeline):")
        print("    wall  T1 %.1fms -> T2 %.1fms  (%.2fx)" % (
            rows["T1 default(1MiB)"]["wmn"]*1e3, rows["T2 default(4MiB)"]["wmn"]*1e3,
            rows["T2 default(4MiB)"]["wmn"]/rows["T1 default(1MiB)"]["wmn"]))
        print("    cyc/B T1 %.3f -> T2 %.3f  (Δ=%.3f)" % (
            rows["T1 default(1MiB)"]["cycB"], rows["T2 default(4MiB)"]["cycB"],
            d("T2 default(4MiB)", "T1 default(1MiB)", "cycB")))
        print("  At SAME 1MiB chunk (isolates pipeline-vs-inline, no chunk-size confound):")
        print("    cyc/B T1@1MiB %.3f -> T2@1MiB %.3f  Δ=%.3f  (pipeline fixed overhead)" % (
            rows["T1 @1MiB"]["cycB"], rows["T2 @1MiB"]["cycB"], d("T2 @1MiB", "T1 @1MiB", "cycB")))
        print("  At SAME 4MiB chunk:")
        print("    cyc/B T1@4MiB %.3f -> T2@4MiB %.3f  Δ=%.3f  (pipeline fixed overhead)" % (
            rows["T1 @4MiB"]["cycB"], rows["T2 @4MiB"]["cycB"], d("T2 @4MiB", "T1 @4MiB", "cycB")))
        print("  chunk-size effect within T1 (inline path, 1MiB vs 4MiB):")
        print("    cyc/B T1@1MiB %.3f vs T1@4MiB %.3f  Δ=%.3f" % (
            rows["T1 @1MiB"]["cycB"], rows["T1 @4MiB"]["cycB"], d("T1 @4MiB", "T1 @1MiB", "cycB")))

    print("\n  binary sha:", subprocess.run(["shasum", "-a", "256", BIN],
          stdout=subprocess.PIPE).stdout.decode().split()[0])
    print("  load avg:", os.getloadavg())

if __name__ == "__main__":
    main()
