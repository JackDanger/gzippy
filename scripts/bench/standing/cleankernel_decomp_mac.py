#!/usr/bin/env python3
"""cleankernel_decomp_mac.py — DISPROOF W1: decompose gzippy's CLEAN-path (-p1)
decode cost on the QUIET mac into components (Huffman-decode-arithmetic+bitread
vs copy/store vs CRC) using byte-transparent removal oracles + causal slow-inject.

This is gz's OWN component breakdown (aarch64, NO ISA-L). It answers WHICH gz
clean-path component is the candidate cost — NOT whether ISA-L beats gz there
(that needs Intel/AMD). Deterministic primitive = instructions retired
(/usr/bin/time -l), warm, ~0.05% spread.

ARMS (all -p1 = pure clean in-order decode, NO markers):
  base        : full clean decode (Huffman + bitread + copy/store + CRC)
  nodecode    : GZIPPY_ORACLE_NODECODE=<cap> — replay removes Huffman decode +
                bit reads + LUT builds, KEEPS the stores. base-nodecode = the
                Huffman-decode-arithmetic+bitread cost (CAVEAT: replay loop
                substitutes per-op overhead -> this UNDER-states pure decode).
                byte-EXACT on a replay hit (sha-gated, hits/miss reported).
  nocrc       : GZIPPY_FOLD_NOCRC=1 — removes per-byte CRC. base-nocrc = CRC cost.
  slowdec50/100 : GZIPPY_SLOW_DECODE — causal inject at Huffman-decode events only.
  slowsto50/100 : GZIPPY_SLOW_STORE  — causal inject at store/copy events only.
                  slope d(instr)/dF * proportional => event-count of each site;
                  slope d(cyc)/dF => causal wall-criticality of that site.

Gate-0: byte-exact gz==gzip per corpus (nocrc/slow* exempt where noted);
path=ParallelSM build=parallel-sm+pure; /dev/null sink; interleaved best-of-N.
"""
import os, subprocess, sys, statistics, tempfile, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
B = os.path.join(ROOT, "target", "release", "gzippy")
N = int(os.environ.get("N", "7"))
CORPORA = os.environ.get("CORPORA", "silesia squishy").split()


def resolve(c):
    for p in (c, f"/tmp/{c}.gz", os.path.join(ROOT, "benchmark_data", f"{c}.gz")):
        if os.path.isfile(p):
            return p
    sys.exit(f"corpus not found: {c}")


def ref_sha(p):
    out = subprocess.run(f"gzip -dc {p}", shell=True, capture_output=True).stdout
    return hashlib.sha256(out).hexdigest()


def decoded_bytes(p):
    # uncompressed size via gzip -l (ISIZE)
    r = subprocess.run(["gzip", "-l", p], capture_output=True, text=True).stdout.splitlines()
    return int(r[1].split()[1])


def run_instr_cyc(env, args, p):
    """Return (instr, cyc) from /usr/bin/time -l, stdout->/dev/null."""
    e = dict(os.environ)
    e.update(env)
    with tempfile.NamedTemporaryFile("r", delete=False) as tf:
        tfn = tf.name
    with open(os.devnull, "wb") as dn, open(tfn, "wb") as errf:
        subprocess.run(["/usr/bin/time", "-l", B, "-dc"] + args + [p],
                       stdout=dn, stderr=errf, env=e)
    instr = cyc = 0
    with open(tfn, "r", errors="replace") as f:
        for line in f:
            if "instructions retired" in line:
                instr = int(line.split()[0])
            elif "cycles elapsed" in line:
                cyc = int(line.split()[0])
    os.unlink(tfn)
    return instr, cyc


def sha_of(env, args, p):
    e = dict(os.environ)
    e.update(env)
    out = subprocess.run([B, "-dc"] + args + [p], capture_output=True, env=e).stdout
    return hashlib.sha256(out).hexdigest()


def main():
    if not os.path.isfile(B):
        sys.exit(f"no binary {B}")
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    # Gate-0(b): path assertion
    dbg = subprocess.run([B, "-dc", "-p1", resolve(CORPORA[0])],
                         env={**os.environ, "GZIPPY_DEBUG": "1"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         text=True, errors="replace").stderr
    assert "path=ParallelSM" in dbg and "build-flavor=parallel-sm+pure" in dbg, dbg
    print(f"GATE-0 PASS path=ParallelSM build=parallel-sm+pure git={git}")

    for corp in CORPORA:
        p = resolve(corp)
        ub = decoded_bytes(p)
        ref = ref_sha(p)
        cap = f"/tmp/cleankernel_{corp}.cap"
        # record pass for NODECODE (timing not measured)
        subprocess.run([B, "-dc", "-p1", p], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       env={**os.environ, "GZIPPY_ORACLE_RECORD": cap})
        # replay non-inert proof
        rep = subprocess.run([B, "-dc", "-p1", p], stdout=subprocess.DEVNULL,
                             env={**os.environ, "GZIPPY_ORACLE_NODECODE": cap},
                             stderr=subprocess.PIPE, text=True, errors="replace").stderr
        hitline = [l for l in rep.splitlines() if "replay: hits" in l]
        arms = {
            "base":      ({}, ["-p1"]),
            "nodecode":  ({"GZIPPY_ORACLE_NODECODE": cap}, ["-p1"]),
            "nocrc":     ({"GZIPPY_FOLD_NOCRC": "1"}, ["-p1"]),
            "slowdec50": ({"GZIPPY_SLOW_DECODE": "50"}, ["-p1"]),
            "slowdec100":({"GZIPPY_SLOW_DECODE": "100"}, ["-p1"]),
            "slowsto50": ({"GZIPPY_SLOW_STORE": "50"}, ["-p1"]),
            "slowsto100":({"GZIPPY_SLOW_STORE": "100"}, ["-p1"]),
        }
        # sha gate (byte-exact arms: base, nodecode; slow*/nocrc byte-transparent too)
        sha_base = sha_of({}, ["-p1"], p)
        assert sha_base == ref, f"{corp} base sha mismatch"
        sha_nd = sha_of({"GZIPPY_ORACLE_NODECODE": cap}, ["-p1"], p)
        assert sha_nd == ref, f"{corp} nodecode sha mismatch (replay not byte-exact)"
        # slow knobs are byte-transparent -> sha must still match
        for k in ("slowdec100", "slowsto100"):
            assert sha_of(arms[k][0], arms[k][1], p) == ref, f"{corp} {k} not byte-transparent"
        # nocrc: bytes correct but exits nonzero / sha still equals
        # (FOLD_NOCRC keeps decoded bytes, only skips verify) -> check
        sha_nocrc = sha_of({"GZIPPY_FOLD_NOCRC": "1"}, ["-p1"], p)

        samples = {a: {"instr": [], "cyc": []} for a in arms}
        # warmup
        for a, (env, args) in arms.items():
            run_instr_cyc(env, args, p)
        for rep_i in range(N):
            for a, (env, args) in arms.items():
                i, c = run_instr_cyc(env, args, p)
                samples[a]["instr"].append(i)
                samples[a]["cyc"].append(c)

        def best(a, k):
            return min(samples[a][k])

        def med(a, k):
            return statistics.median(samples[a][k])

        def spread(a, k):
            v = samples[a][k]
            return (max(v) - min(v)) / min(v) * 100

        print(f"\n===== {corp}  (uncompressed {ub/1e6:.1f} MB) =====")
        print(f"  NODECODE non-inert: {hitline[0] if hitline else 'NO HITLINE'}; "
              f"nodecode sha==ref:{sha_nd==ref} slow*byte-transparent:OK nocrc sha==ref:{sha_nocrc==ref}")
        print(f"  {'arm':<11} {'instr(M)':>11} {'spr%':>5} {'cyc(M)':>11} {'instr/B':>8} {'cyc/B':>7}")
        for a in arms:
            bi, bc = best(a, "instr"), best(a, "cyc")
            print(f"  {a:<11} {bi/1e6:>11.1f} {spread(a,'instr'):>5.2f} {bc/1e6:>11.1f} "
                  f"{bi/ub:>8.3f} {bc/ub:>7.3f}")
        bi = best("base", "instr")
        bc = best("base", "cyc")
        nd = best("nodecode", "instr")
        ndc = best("nodecode", "cyc")
        nc = best("nocrc", "instr")
        # component shares (instr)
        decode_arith = bi - nd            # Huffman decode + bitread (replay-caveat)
        crc = bi - nc                     # CRC
        copy_store_rest = nd              # everything kept by replay (stores+loop+coord)
        print(f"\n  --- W1 CLEAN-PATH INSTR DECOMPOSITION (gz own, no ISA-L) ---")
        print(f"  base total instr      : {bi/1e6:>9.1f} M  ({bi/ub:.3f} instr/B)")
        print(f"  Huffman-decode+bitread: {decode_arith/1e6:>9.1f} M  = {decode_arith/bi*100:5.2f}% "
              f"(base-nodecode; replay-overhead caveat -> UNDERSTATES decode)")
        print(f"  CRC                    : {crc/1e6:>9.1f} M  = {crc/bi*100:5.2f}% (base-nocrc, clean)")
        print(f"  copy/store+loop+coord  : {copy_store_rest/1e6:>9.1f} M  = {copy_store_rest/bi*100:5.2f}% "
              f"(nodecode residual; INCLUDES replay loop overhead)")
        # cyc shares (wall proxy at T1 serial)
        print(f"  [cyc] base {bc/1e6:.1f}M  nodecode {ndc/1e6:.1f}M  "
              f"decode-cyc-share {(bc-ndc)/bc*100:.2f}%")
        # causal slow-inject slopes (instr -> event counts; cyc -> wall criticality)
        d50i = best("slowdec50", "instr") - bi
        d100i = best("slowdec100", "instr") - bi
        s50i = best("slowsto50", "instr") - bi
        s100i = best("slowsto100", "instr") - bi
        d50c = best("slowdec50", "cyc") - bc
        d100c = best("slowdec100", "cyc") - bc
        s50c = best("slowsto50", "cyc") - bc
        s100c = best("slowsto100", "cyc") - bc
        print(f"\n  --- W1 CAUSAL SLOW-INJECT (Gate-2; uniform spin/event) ---")
        print(f"  SLOW_DECODE  +instr: F50={d50i/1e6:.0f}M F100={d100i/1e6:.0f}M  "
              f"+cyc: F50={d50c/1e6:.0f}M F100={d100c/1e6:.0f}M")
        print(f"  SLOW_STORE   +instr: F50={s50i/1e6:.0f}M F100={s100i/1e6:.0f}M  "
              f"+cyc: F50={s50c/1e6:.0f}M F100={s100c/1e6:.0f}M")
        # spin=BASE_SPIN(22)*F/100 per event; injected instr per event ~ const ->
        # +instr ratio decode/store ~ n_decode_events/n_store_events
        if s100i:
            print(f"  decode/store event-weighted ratio (F100 instr): {d100i/s100i:.2f} "
                  f"(>1 => more decode events; ~1 => comparable)")
        if s100c:
            print(f"  decode/store WALL-criticality ratio (F100 cyc): {d100c/s100c:.2f} "
                  f"(>1 => decode site more wall-critical)")


if __name__ == "__main__":
    main()
