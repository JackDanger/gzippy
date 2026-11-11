#!/usr/bin/env python3
# FLAG SPOT-CENSUS — final-tree census section 4 (arms within A only).
# Grid mirrors the flipsafe 416-cell census at reduced volume (20 members):
#   SKIP:  nearopt-fixedpoint-skip ON vs the arm-A default build
#          20 members x {L11,L12} x {-p2,-p4}  (the -p2 rows are route-inert)
#   SAFE:  near-opt-parallel-flush + -flipsafe ON vs the arm-A default build
#          20 members x L9..L12 x {-p2,-p4}
# Assertion per cell: ON stream sha == OFF stream sha (the default build is the
# same binary the byte-matrix already 3-leg verified at these cells). Every ON
# stream additionally gets one pigz decode leg. Mismatches are captured and the
# section stops for that feature.
import os
import subprocess
import sys
import concurrent.futures as cf

ROOT = "/Users/jackdanger/www/gzippy"
RUN = f"{ROOT}/.review-tmp/wC-final/run"
OFF_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-a/release/gzippy"
SKIP_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-skip/release/gzippy"
SAFE_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-safe/release/gzippy"
PIGZ = f"{ROOT}/vendor/pigz/pigz"
TMP = f"{RUN}/tmprows"

import hashlib


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(binpath, member, level, threads, outpath):
    with open(member, "rb") as fin, open(outpath, "wb") as fout:
        r = subprocess.run([binpath, "--level", str(level), "-p", str(threads),
                            "-c"], stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"encode failed: {r.stderr.decode()[:200]}")


def pigz_ok(stream, in_sha):
    r = subprocess.run(["bash", "-c",
                        f"'{PIGZ}' -dc < '{stream}' | shasum -a 256"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().split()[0] == in_sha


def cell(argt):
    feature, name, path, in_sha, level, threads = argt
    tag = f"{feature}_{name}_L{level}_T{threads}"
    rowf = f"{TMP}/{tag}.row"
    if os.path.exists(rowf):
        return tag, None
    sb = f"{TMP}/{tag}.on.gz"
    try:
        encode(SKIP_BIN if feature == "skip" else SAFE_BIN, path, level,
               threads, sb)
        sha_on = sha_file(sb)
        so = f"{TMP}/{tag}.off.gz"
        encode(OFF_BIN, path, level, threads, so)
        sha_off = sha_file(so)
        eq = sha_on == sha_off
        rt = pigz_ok(sb, in_sha)
        verdict = "OK" if (eq and rt) else "FAIL"
        row = "\t".join([feature, name, str(level), str(threads),
                         sha_off, str(os.path.getsize(so)), sha_on,
                         str(os.path.getsize(sb)),
                         "eq" if eq else "DIFF", "pigz-ok" if rt else "pigz-FAIL",
                         verdict])
        tmp = rowf + ".part"
        with open(tmp, "w") as f:
            f.write(row + "\n")
        os.replace(tmp, rowf)
        os.remove(sb)
        os.remove(so)
        return tag, row
    except Exception as exc:
        return tag, "\t".join([feature, name, str(level), str(threads),
                               "EXC:" + str(exc)[:150], "", "", "", "", "",
                               "FAIL"])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    spot = []
    with open(f"{RUN}/members_spot.tsv") as f:
        for line in f:
            name, path, nbytes, in_sha = line.rstrip("\n").split("\t")
            spot.append((name, path, int(nbytes), in_sha))
    jobs = []
    if which in ("all", "skip"):
        for name, path, _n, in_sha in spot:
            for level in (11, 12):
                for t in (2, 4):
                    jobs.append(("skip", name, path, in_sha, level, t))
    if which in ("all", "safe"):
        for name, path, _n, in_sha in spot:
            for level in (9, 10, 11, 12):
                for t in (2, 4):
                    jobs.append(("safe", name, path, in_sha, level, t))
    print(f"jobs: {len(jobs)}", flush=True)
    fails = 0
    done = 0
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for tag, row in pool.map(cell, jobs):
            done += 1
            if row is None:
                continue
            if row.split("\t")[-1] == "FAIL":
                fails += 1
                print(f"[FAIL] {row}", flush=True)
            if done % 40 == 0:
                print(f"[progress] {done}/{len(jobs)} fails={fails}", flush=True)
    rows = []
    for fn in sorted(os.listdir(TMP)):
        if fn.endswith(".row"):
            data = open(f"{TMP}/{fn}").read().strip()
            if data.split("\t")[0] in ("skip", "safe"):
                rows.append(data)
    with open(f"{RUN}/flag_census.tsv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"DONE cells={len(rows)} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
