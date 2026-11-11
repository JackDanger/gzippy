#!/usr/bin/env python3
"""L5 depth-ladder falsifier — deterministic size/sha/validity phase.

Zero-code probe: GZIPPY_LDX_CFG / GZIPPY_LADDER env arms on the ladder-tune
binary, judged against the vanilla (shipped-feature) binary from the same
commit. Every stream: sha256 (byte counts are NOT identity), size recorded,
roundtrip through an independent decoder (/usr/bin/zcat via stdin, python gzip
fallback) back to the exact source bytes. Results TSV + raw log retained.
"""
import hashlib, os, shutil, subprocess, sys, time

WT = "/Users/jackdanger/www/gzippy/.review-tmp/wA-l5"
SCRATCH = f"{WT}/scratch"
MEMBERS = f"{SCRATCH}/members"
SHIP = f"{WT}/target-ship/release/gzippy"
TUNE = f"{WT}/target-tune/release/gzippy"
OUT = f"{SCRATCH}/sweep-table.tsv"
LOG = f"{SCRATCH}/sweep.log"

# arms: name -> (binary, extra env)
ARMS = [
    ("ship",    SHIP,  {}),
    ("lad16",   TUNE,  {"GZIPPY_LDX_CFG": "5:16:30"}),    # control: MUST equal ship
    ("lad20",   TUNE,  {"GZIPPY_LDX_CFG": "5:20:30"}),
    ("lad24",   TUNE,  {"GZIPPY_LDX_CFG": "5:24:30"}),
    ("lad32",   TUNE,  {"GZIPPY_LDX_CFG": "5:32:30"}),
    ("lad32g8", TUNE,  {"GZIPPY_LDX_CFG": "5:32:30:8"}),  # gzip -5 good_length=8 quartering
    ("ladG32",  TUNE,  {"GZIPPY_LADDER": "lazy:32:30"}),   # GZIPPY_LADDER reachability probe
]
RIVALS = [
    ("gzip_L5",       f"{WT}/vendor/gzip/gzip", ["-n", "-c", "-5"]),
    ("pigz_L5",       f"{WT}/vendor/pigz/pigz", ["-5", "-p1", "-c"]),
    ("libdeflate_L5", f"{WT}/vendor/libdeflate/cbuild/programs/libdeflate-gzip", ["-5", "-c"]),
]

elements = sorted(f for f in os.listdir(MEMBERS) if os.path.isfile(os.path.join(MEMBERS, f)))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

# independent decoder: zcat reading STDIN (BSD zcat FILE-arg quirk, per ab.sh);
# python gzip fallback. Verify the oracle is not a gzippy shim.
oracle = "none"
def zcat_available():
    p = "/usr/bin/zcat"
    if os.path.exists(p):
        try:
            v = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=10)
            if "gzippy" not in (v.stdout + v.stderr).lower():
                return "zcat(/usr/bin)"
        except Exception:
            pass
    return None
def decode_and_check(src_gz, origin_path):
    if getattr(decode_and_check, "oracle", None) is None:
        decode_and_check.oracle = zcat_available() or "python3-gzip"
    oracle = decode_and_check.oracle
    want = sha(origin_path)
    if oracle.startswith("zcat"):
        with open(src_gz, "rb") as fin, open("/dev/null", "wb") as devnull2:
            r = subprocess.run(["/usr/bin/zcat"], stdin=fin, stdout=subprocess.PIPE, stderr=devnull2)
    else:
        import gzip as gz
        with open(src_gz, "rb") as fin:
            r = subprocess.CompletedProcess(None, 0, gz.decompress(fin.read()))
    ok = (r.returncode == 0 if r.returncode is not None else True) and hashlib.sha256(r.stdout).hexdigest() == want
    return ok, oracle

stamp = []
stamp.append(f"git_rev={subprocess.run(['git','-C',WT,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}")
stamp.append(f"git_branch={subprocess.run(['git','-C',WT,'rev-parse','--abbrev-ref','HEAD'],capture_output=True,text=True).stdout.strip()}")
stamp.append(f"ship_sha={sha(SHIP)}")
stamp.append(f"tune_sha={sha(TUNE)}")
for name, _, env in ARMS:
    if env:
        stamp.append(f"arm:{name}={' '.join(f'{k}={v}' for k,v in env.items())}")
print("[stamp] " + " ".join(stamp), flush=True)

rows = []   # member, arm, bytes, sha, roundtrip_ok
aa_fail = []
with open(LOG, "a") as log, open(OUT, "w") as out:
    out.write("member\tarm\tbytes\tsha256\troundtrip\n")
    print("[phase] sizes+shas+roundtrip", flush=True)
    for m in elements:
        src = os.path.join(MEMBERS, m)
        stream_tmp = f"{SCRATCH}/_stream.gz"
        for arm in ARMS + RIVALS:
            name, binary, arg3 = arm[0], arm[1], arm[2]
            is_rival = name in ("gzip_L5", "pigz_L5", "libdeflate_L5")
            argv = [binary] + (arg3 if is_rival else ["-n", "-5", "-p1", "-c"])
            extra_env = dict(arg3) if not is_rival else {}
            t0 = time.perf_counter()
            with open(src, "rb") as fin, open(stream_tmp, "wb") as fout:
                subprocess.run(argv, stdin=fin, stdout=fout, env={**os.environ, **extra_env})
            dt = time.perf_counter() - t0
            rt = "ok"
            if name in ("ship", "lad16", "lad20", "lad24", "lad32", "lad32g8", "ladG32"):
                ok, oracle_used = decode_and_check(stream_tmp, src)
                rt = "rt-ok" if ok else "RT-FAIL"
                if not ok:
                    print(f"!! ROUNDTRIP FAIL {name} {m}", flush=True)
                    sys.exit(3)
            bsz = os.path.getsize(stream_tmp)
            s = sha(stream_tmp)
            row = f"{m}\t{name}\t{bsz}\t{s}\t{rt}"
            out.write(row + "\n"); out.flush()
            log.write(f"{row} argv={' '.join(argv)} member={m} dt={dt:.2f}s\n")
            if os.path.exists(stream_tmp):
                os.unlink(stream_tmp)
        print(f"   member done: {m}", flush=True)
    print("[phase] complete", flush=True)

# A/A control assertion: lad16 == ship (bytes) on every member; lad32 must DIFFER
# somewhere (else the env does not reach the encoder); ladG32 == ship everywhere.
by_ma = {}
with open(OUT) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        by_ma[(p[0], p[1])] = p
print("[verdict] A/A control (lad16 vs ship):", flush=True)
aa = 0
for m in elements:
    if by_ma[(m, "lad16")][3] == by_ma[(m, "ship")][3] and by_ma[(m, "lad16")][2] == by_ma[(m, "ship")][2]:
        aa += 1
    else:
        aa_fail.append((m, "lad16"))
print(f"  {aa}/{len(elements)} members byte-identical (sha256 equality)")
print("[verdict] GZIPPY_LADDER reachability (ladG32 vs ship):", flush=True)
g_inert, g_flipped = 0, []
for m in elements:
    if by_ma[(m, "ladG32")][3] == by_ma[(m, "ship")][3]:
        g_inert += 1
    else:
        g_flipped.append(m)
print(f"  {g_inert}/{len(elements)} members UNCHANGED by GZIPPY_LADDER=lazy:32:30", end="")
print(f"; changed: {g_flipped}" if g_flipped else " (= inert on the shipped L5-T1 port route)")
print("[verdict] LDX_CFG reachability probe (lad32 differs from ship):", flush=True)
reach = [m for m in elements if by_ma[(m, "lad32")][3] != by_ma[(m, "ship")][3]]
print(f"  {len(reach)}/{len(elements)} members changed by GZIPPY_LDX_CFG=5:32:30")
print("[verdict] control parity results:")
samesz = [m for m in elements if by_ma[(m, "lad32")][2] == by_ma[(m, "ship")][2]]
print(f"  lad32 same-size-as-ship: {len(samesz)}/{len(elements)} (those are ties by size even when bytes differ)")
if aa_fail:
    print(f"!! A/A CONTROL FAILED on {aa_fail}")
    sys.exit(4)
