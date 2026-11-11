#!/usr/bin/env python3
# BYTE MATRIX — final-tree census section 2 (wave-B recipe, 30 members).
# Cells: 30 members x L1..L12 x {T1,T2,T4} = 1,080 per arm; each cell encodes
# with arm A (main b1502a07) AND arm B (06859145), sha256 + sizes both; verdict
# OK iff the shas are EQUAL (the trio is default-off => expect identity) and the
# A stream roundtrips byte-exactly through THREE decoders (Apple gzip,
# vendored pigz 2.8, python3 zlib member-chain). On a sha mismatch both
# streams are decoded and the cell captured for the FAIL report.
# Box load is irrelevant for byte results (deterministic outputs).
import hashlib
import os
import subprocess
import sys
import concurrent.futures as cf

ROOT = "/Users/jackdanger/www/gzippy"
RUN = f"{ROOT}/.review-tmp/wC-final/run"
A_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-a/release/gzippy"
B_BIN = f"{ROOT}/.review-tmp/wC-final/tgt-b/release/gzippy"
PYDEC = f"{RUN}/decode_sha.py"
PIGZ = f"{ROOT}/vendor/pigz/pigz"
TMP = f"{RUN}/tmprows"
os.makedirs(TMP, exist_ok=True)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, stdin=None, check=True):
    r = subprocess.run(cmd, stdin=stdin, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed {cmd[:3]}: {r.stderr.decode()[:200]}")
    return r


def encode(binpath, member, level, threads, outpath):
    with open(member, "rb") as fin, open(outpath, "wb") as fout:
        r = subprocess.run([binpath, "--level", str(level), "-p", str(threads),
                            "-c"], stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(
            f"encode failed {os.path.basename(binpath)} {level} {threads} "
            f"on {os.path.basename(member)}: {r.stderr.decode()[:200]}")


def decode_legs(stream, in_sha):
    """Three independent decoders; returns list of per-leg OK/FAIL."""
    legs = []
    for name, cmd in (
        ("apple-gzip", f"/usr/bin/gzip -dc < '{stream}' | shasum -a 256"),
        ("pigz-2.8", f"'{PIGZ}' -dc < '{stream}' | shasum -a 256"),
    ):
        r = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
        got = r.stdout.decode().split()[0] if r.stdout else "ERR"
        legs.append(1 if got == in_sha else 0)
    r = subprocess.run(["python3", PYDEC, stream, in_sha],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    legs.append(1 if r.returncode == 0 else 0)
    return legs


def cell(job):
    name, path, _bytes, in_sha, level, threads = job
    tag = f"{name}_L{level}_T{threads}"
    rowf = f"{TMP}/{tag}.row"
    if os.path.exists(rowf):
        return tag, None  # already done (resume)
    sa = f"{TMP}/{tag}.a.gz"
    try:
        encode(A_BIN, path, level, threads, sa)
        sha_a = sha_file(sa)
        size_a = os.path.getsize(sa)
        # arm B: encode straight into a hasher via temp file (cheap, no store)
        sb = f"{TMP}/{tag}.b.gz"
        encode(B_BIN, path, level, threads, sb)
        sha_b = sha_file(sb)
        size_b = os.path.getsize(sb)
        eq = sha_a == sha_b
        legs = decode_legs(sa, in_sha)
        legs_b = decode_legs(sb, in_sha) if not eq else legs
        ok = eq and sum(legs) == 3 and sum(legs_b) == 3
        verdict = "OK" if ok else "FAIL"
        row = "\t".join([name, str(level), str(threads), in_sha,
                         sha_a, str(size_a), sha_b, str(size_b),
                         "eq" if eq else "DIFF",
                         "".join(str(x) for x in legs),
                         "".join(str(x) for x in legs_b), verdict])
        tmp = rowf + ".part"
        with open(tmp, "w") as f:
            f.write(row + "\n")
        os.replace(tmp, rowf)
        os.remove(sa)
        os.remove(sb)
        return tag, row
    except Exception as exc:  # cell failure never kills the census runner
        return tag, "\t".join([name, str(level), str(threads), "EXC:" +
                               str(exc)[:160], "", "", "", "", "", "", "",
                               "FAIL"])


def main():
    members = []
    with open(f"{RUN}/members.tsv") as f:
        for line in f:
            name, path, nbytes, in_sha = line.rstrip("\n").split("\t")
            if name == "EXC":
                continue
            members.append((name, path, int(nbytes), in_sha))
    jobs = []
    for name, path, nbytes, in_sha in members:
        for level in range(1, 13):
            for threads in (1, 2, 4):
                jobs.append((name, path, nbytes, in_sha, level, threads))
    print(f"jobs: {len(jobs)}", flush=True)
    done = 0
    fails = 0
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for tag, row in pool.map(cell, jobs):
            done += 1
            if row is None:
                continue
            if row.split("\t")[-1] == "FAIL":
                fails += 1
                print(f"[FAIL] {row}", flush=True)
            if done % 50 == 0:
                print(f"[progress] {done}/{len(jobs)} fails={fails}", flush=True)
    # collapse rows into the matrix tsv
    rows = []
    for fn in sorted(os.listdir(TMP)):
        if fn.endswith(".row"):
            with open(f"{TMP}/{fn}") as f:
                rows.append(f.read().strip())
    with open(f"{RUN}/byte_matrix.tsv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"DONE cells={len(rows)} fails={fails}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
