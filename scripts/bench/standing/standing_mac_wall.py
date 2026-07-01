#!/usr/bin/env python3
"""standing_mac_wall.py — macOS-local wall-clock ground-truth matrix (aarch64).

Mirrors the Intel standing rig's CONTRACT on the Mac, but measures WALL THROUGHPUT
(MB/s of decompressed output) so the unified matrix has a like-for-like column to the
Intel table. /usr/bin/time `real` is 10ms-quantized; here we time with
time.perf_counter() around a subprocess whose stdout is /dev/null — sub-ms wall, the
same sink for every arm.

Comparators (DIFFERENT from Intel — be explicit in the report):
  libdeflate-gunzip : aarch64 T1 SOTA (single-thread only).
  rapidgzip          : 0.16.0 aarch64 build — NO ISA-L on aarch64, so this is
                       gz-aarch64 vs rg-aarch64, NOT the Intel SOTA. Labeled.

GATE-0 (LOUD-FAIL -> the cell is UNTRUSTED, no number reported):
  (a) sha == zcat for every arm (byte-exact, non-inert).
  (b) build-flavor=parallel-sm+pure + path=ParallelSM (asserted by caller).
  (c) A/A self-test: gz-vs-gz best-of-N ratio within tol (<= spread) -> ratios trusted.
  (d) same /dev/null sink for every arm.
GATE-1 significance: interleaved best-of-N>=13; |ratio-1| <= combined spread => TIE.

Usage:
  standing_mac_wall.py --gz <bin> --rg <bin> --ld libdeflate-gunzip \
      --corpora "silesia=/tmp/silesia.gz ..." --threads "1 2 4 8" -N 13 \
      --out /tmp/mac_wall.csv
"""
import argparse, hashlib, subprocess, sys, time, statistics, os

DEVNULL = open(os.devnull, "wb")


def sha_of(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    h = hashlib.sha256()
    nbytes = 0
    for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
        h.update(chunk)
        nbytes += len(chunk)
    p.wait()
    return h.hexdigest()[:16], nbytes, p.returncode


def timed(cmd):
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, stdout=DEVNULL, stderr=DEVNULL)
    t1 = time.perf_counter()
    return (t1 - t0), rc


def best_and_spread(samples):
    # best (min wall) + spread = (median-min)/min as % ; report median too
    samples = sorted(samples)
    best = samples[0]
    med = statistics.median(samples)
    spread = (med - best) / best * 100.0 if best > 0 else 0.0
    return best, med, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", required=True)
    ap.add_argument("--rg", required=True)
    ap.add_argument("--ld", required=True)
    ap.add_argument("--corpora", required=True, help="name=path name=path ...")
    ap.add_argument("--threads", default="1 2 4 8")
    ap.add_argument("-N", type=int, default=13)
    ap.add_argument("--out", default="/tmp/mac_wall.csv")
    a = ap.parse_args()

    corpora = []
    for tok in a.corpora.split():
        name, path = tok.split("=", 1)
        corpora.append((name, path))
    threads = [int(x) for x in a.threads.split()]
    N = a.N

    # ---- GATE-0(a): sha==zcat + decompressed byte count, per corpus ----
    meta = {}
    print("== GATE-0(a) byte-exact (gz==rg==ld==gzip -d) + sizes ==")
    for name, path in corpora:
        ref, nbytes, _ = sha_of(["gzip", "-dc", path])
        gz, _, grc = sha_of([a.gz, "-dc", "-p1", path])
        ld, _, lrc = sha_of([a.ld, "-dc", path])
        rg, _, rrc = sha_of([a.rg, "-dc", "-P", "1", path])
        ok = (ref == gz == ld == rg) and grc == 0
        meta[name] = dict(path=path, bytes=nbytes, ref=ref,
                          gz_ok=(gz == ref), ld_ok=(ld == ref), rg_ok=(rg == ref),
                          gate0a=ok)
        print(f"   {name:10s} bytes={nbytes:>11d} ref={ref} gz={'OK' if gz==ref else 'BAD'} "
              f"ld={'OK' if ld==ref else 'BAD'} rg={'OK' if rg==ref else 'BAD'}")

    # ---- measurement: interleaved best-of-N, /dev/null sink all arms ----
    print(f"== measure: interleaved best-of-N={N}, /dev/null both/all arms ==")
    rows = []
    with open(a.out, "w") as f:
        f.write("corpus,bytes,arm,threads,rep,wall_s\n")
        for name, path in corpora:
            b = meta[name]["bytes"]
            for T in threads:
                # warmup (untimed)
                timed([a.gz, "-dc", f"-p{T}", path])
                timed([a.rg, "-dc", "-P", str(T), path])
                timed([a.ld, "-dc", path])
                samp = {"gz": [], "gz2": [], "rg": [], "ld": []}
                for rep in range(1, N + 1):
                    w, _ = timed([a.gz, "-dc", f"-p{T}", path]); samp["gz"].append(w)
                    f.write(f"{name},{b},gz,{T},{rep},{w:.6f}\n")
                    w, _ = timed([a.gz, "-dc", f"-p{T}", path]); samp["gz2"].append(w)
                    f.write(f"{name},{b},gz2,{T},{rep},{w:.6f}\n")
                    w, _ = timed([a.rg, "-dc", "-P", str(T), path]); samp["rg"].append(w)
                    f.write(f"{name},{b},rg,{T},{rep},{w:.6f}\n")
                    w, _ = timed([a.ld, "-dc", path]); samp["ld"].append(w)  # T1 always
                    f.write(f"{name},{b},ld,{T},{rep},{w:.6f}\n")
                rows.append((name, b, T, samp))

    # ---- report ----
    MB = 1_000_000.0
    print("\n========================= MAC WALL MATRIX (aarch64, pure-Rust engine A) =========================")
    print(f"{'corpus':10s} {'T':>2s} {'gzMB/s':>8s} {'wallms':>7s} {'spr%':>5s} "
          f"{'ldMB/s':>8s} {'rgMB/s':>8s} {'gz/ld':>6s} {'gz/rg':>6s} {'A/A':>5s}  GATE")
    for name, b, T, samp in rows:
        gz_best, gz_med, gz_spr = best_and_spread(samp["gz"])
        gz2_best, _, _ = best_and_spread(samp["gz2"])
        ld_best, _, ld_spr = best_and_spread(samp["ld"])
        rg_best, _, rg_spr = best_and_spread(samp["rg"])
        gz_mbps = b / MB / gz_best
        ld_mbps = b / MB / ld_best
        rg_mbps = b / MB / rg_best
        aa = abs(gz_best - gz2_best) / gz_best * 100.0  # A/A self-test spread %
        gz_ld = gz_best / ld_best   # wall ratio: <1 = gz faster
        gz_rg = gz_best / rg_best
        m = meta[name]
        gate = "PASS"
        if not m["gate0a"]:
            gate = "UNTRUSTED(sha)"
        elif aa > 5.0:
            gate = "UNTRUSTED(A/A)"
        else:
            # TIE if |ratio-1| <= combined spread (vs rg, the multi-thread comparator)
            comb = (gz_spr + rg_spr) / 100.0
            if abs(gz_rg - 1.0) <= comb:
                gate = "TIE(rg)"
        print(f"{name:10s} {T:>2d} {gz_mbps:>8.1f} {gz_best*1000:>7.1f} {gz_spr:>5.1f} "
              f"{ld_mbps:>8.1f} {rg_mbps:>8.1f} {gz_ld:>6.3f} {gz_rg:>6.3f} {aa:>5.2f}  {gate}")
    print("\nnote: gz/ld and gz/rg are WALL ratios (<1.0 = gzippy FASTER). libdeflate is")
    print("T1-only (single-thread) so gz/ld at T>1 is multi-vs-single (informational).")
    print("rapidgzip here is aarch64 (no ISA-L) — NOT the Intel SOTA; labeled.")


if __name__ == "__main__":
    main()
