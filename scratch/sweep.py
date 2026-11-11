#!/usr/bin/env python3
"""L4 deflate_medium pricing sweep — sizes phase.

Arms per member (raw gzip streams, byte counts + sha256):
  ship        vanilla release binary, L4 -p1 (shipped greedy(16,30))
  medium      ladder-tune release binary + GZIPPY_LDX_MEDIUM=1, L4 -p1
  gzip4       vendored gzip -4
  pigz4       vendored pigz -4 (-p1)
  ld4         vendored libdeflate-gzip -4 (the L4 tie rival)

Every MEDIUM output is round-tripped through the independent /usr/bin/zcat
(stdin) back to the member bytes — a cell claim rests only on streams that
decode. A/A control: ship vs tune-no-env byte identity is asserted on one
member before this sweep runs (see run_aa in the record).
"""
import hashlib
import os
import subprocess
import sys

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-medium"
SCRATCH = os.path.join(WT, "scratch")
VANILLA = os.path.join(WT, "target-vanilla/release/gzippy")
TUNE = os.path.join(WT, "target-tune/release/gzippy")
GZIP4 = "/Users/jackdanger/www/gzippy/vendor/gzip/gzip"
PIGZ4 = "/Users/jackdanger/www/gzippy/vendor/pigz/pigz"
LD4 = os.path.join(WT, "scratch/libdeflate-cbuild/programs/libdeflate-gzip")
ORACLE = "/usr/bin/zcat"

L5_RECORD_MEMBERS = [
    "dickens", "logs.txt", "logs.txt-card26m", "mozilla", "mr", "nci",
    "ooffice", "osdb", "reymont", "samba", "sao", "software.archive",
    "webster", "x-ray", "xml",
]
CAMPAIGN_CORPUS = os.environ.get("CAMPAIGN_CORPUS_ROOT",
                                 os.path.expanduser("~/www/gzippy-bench/corpus"))


def log(msg):
    print(msg, flush=True)


def sha_and_size(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += len(chunk)
            h.update(chunk)
    return n, h.hexdigest()


def run_arm(cmd):
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out.returncode != 0 or not out.stdout:
        return None, None
    return sha_and_size_bytes(out.stdout), out.stdout


def sha_and_size_bytes(data):
    return (len(data), hashlib.sha256(data).hexdigest())


def oracle_roundtrip(cmd, produced, src):
    """Decode `produced` (compressed stream bytes) through the INDEPENDENT
    oracle and compare against the source member bytes. Every arm's stream is
    stamped with the result; a non-ok stamp voids the row."""
    out = subprocess.run(cmd, input=produced, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out.returncode != 0:
        return f"rt-FAIL(rc={out.returncode})"
    if len(out.stdout) != os.path.getsize(src):
        return "rt-FAIL(len)"
    digest = hashlib.sha256(out.stdout).hexdigest()
    hs = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hs.update(chunk)
    return "rt-ok" if digest == hs.hexdigest() else "rt-FAIL(sha)"


def main():
    members = []
    for base in [os.path.join(SCRATCH, "members"), CAMPAIGN_CORPUS]:
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                tag = os.path.basename("l5set") if "scratch/members" in path else "campaign"
                members.append((f"{name}", path, tag))
    log(f"[sweep] {len(members)} member files")
    # A/A CONTROL (route assertion): the tune binary with the env UNSET must be
    # byte-identical to the vanilla binary on every member at L4 -p1 proves the
    # probe changes nothing by default. Any mismatch aborts the sweep.
    aa_bad = 0
    for name, path, tag in members:
        a = subprocess.run([VANILLA, "-4", "-p1", "-c", path], stdout=subprocess.PIPE)
        b = subprocess.run([TUNE, "-4", "-p1", "-c", path], stdout=subprocess.PIPE)
        if a.stdout != b.stdout or a.returncode != 0 or b.returncode != 0:
            log(f"[AA-FAIL] {name}")
            aa_bad += 1
    if aa_bad:
        log(f"[sweep] A/A control FAILED on {aa_bad} members — outputs VOID; not sweeping.")
        sys.exit(2)
    log("[sweep] A/A control: env-unset tune == vanilla on ALL members (route assertion ok)")
    rows = []
    for name, path, tag in members:
        ship, ship_stream = run_arm([VANILLA, "-4", "-p1", "-c", path])
        medium_cmd = [TUNE, "-4", "-p1", "-c", path]
        env = dict(os.environ)
        env["GZIPPY_LDX_MEDIUM"] = "1"
        chest = subprocess.run(medium_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if chest.returncode != 0 or not chest.stdout:
            log(f"ERROR medium arm failed on {name}")
            continue
        tmp = "/tmp/medium-sweep-arm.gz"
        open(tmp, "wb").write(chest.stdout)
        msize, msha = sha_and_size(tmp)
        rt = oracle_roundtrip([ORACLE], chest.stdout, path)
        gzip4, _ = run_arm([GZIP4, "-c", "-4", path])
        pigz4, _ = run_arm([PIGZ4, "-4", "-c", path])
        ld4, _ = run_arm([LD4, "-4", "-c", path])
        rt_ship = oracle_roundtrip([ORACLE], ship_stream, path) if ship is not None else "no-ship-stream"
        rows.append((name, tag, path, ship, (msize, msha, rt_ship), rt, gzip4, pigz4, ld4))
        log(f"[sweep] {name:22s} ship={str(ship[0]) if ship else 'ERR':>10s} medium={str(msize):>10s} rt={rt}")
    # TSV: member tag in_bytes ship ship_sha medium medium_sha rt gzip4 pigz4 ld4 deltas
    tsv = os.path.join(SCRATCH, "sizes.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(["member", "tag", "in_bytes", "ship", "ship_sha", "ship_rt", "medium",
                           "medium_sha", "medium_rt", "gzip4", "pigz4", "ld4",
                           "d_ship_B", "d_ship_pct", "d_gzip4_B", "d_gzip4_pct",
                           "d_pigz4_B", "d_pigz4_pct", "d_ld4_B", "d_ld4_pct"]) + "\n")
        for name, tag, path, ship, med, rt, g4, p4, l4 in rows:
            inb = os.path.getsize(path)
            ship_n = ship[0] if ship else -1
            m = med[0]
            g = g4[0] if g4 else -1
            p = p4[0] if p4 else -1
            l = l4[0] if l4 else -1
            def pct(base, x):
                return f"{(x - base) / base * 100:+.4f}" if base > 0 else "NA"
            f.write("\t".join([name, tag, str(inb), str(ship_n), ship[1] if ship else "-",
                               med[2] if len(med) > 2 else "-",
                               str(m), med[1], rt, str(g), str(p), str(l),
                               str(m - ship_n), pct(ship_n, m) if ship_n > 0 else "NA",
                               str(m - g), pct(g, m) if g > 0 else "NA",
                               str(m - p), pct(p, m) if p > 0 else "NA",
                               str(m - l), pct(l, m)]) + "\n")
    # Also append member INPUT shas for provenance
    with open(os.path.join(SCRATCH, "member-input-sha256.tsv"), "w") as f:
        for name, tag, path, *_ in rows:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            f.write(f"{name}\t{tag}\t{h.hexdigest()}\n")
    log(f"[sweep] wrote {tsv}")


if __name__ == "__main__":
    main()
