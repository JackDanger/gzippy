#!/usr/bin/env python3
"""STEP-0.5 — aarch64 ENGINE-A (flat) vs ENGINE-B (two-level) clean-decode A/B.

Drives `examples/kernel_ab_aarch64` to answer the dominant CLEAN-KERNEL-DESIGN
question: does the EXISTING flat libdeflate-style decoder (engine A,
`decode_huffman_libdeflate_style`) BEAT the production two-level contig clean path
(engine B, `Block::decode_clean_into_contig`) on gzippy's OWN primitives, and is
the gap TABLE-WIDTH (flat wins big on the per-symbol extreme) or the
UNPACK/refill CADENCE?

DETERMINISTIC instruction primitive: `/usr/bin/time -l` ("instructions retired",
"cycles elapsed"). To isolate the KERNEL loop from all fixed process/setup cost,
each arm is run at reps=R and reps=2R and the MARGINAL per-rep cost is
   per_rep = (metric(2R) - metric(R)) / R
which exactly cancels build_real_block + binary startup + table-build (identical
in both invocations). instr/B = per_rep_instr / bytes_per_rep.

Gate-0 (the example LOUD-FAILs internally else the number does not exist):
  byte-exact engine A == engine B == flate2 oracle; same body bit; FLAT_DECODE_CALLS
  advanced by exactly reps (engine A NON-INERT on aarch64); /dev/null both arms.

SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch. instr = STRONG-because-deterministic
HYPOTHESIS; cyc = HYPOTHESIS-tier (reported min-of-N with spread).
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXE = os.path.join(ROOT, "target", "release", "examples", "kernel_ab_aarch64")
TIME = "/usr/bin/time"
N = int(os.environ.get("KAB_N", "7"))
R = int(os.environ.get("KAB_R", "2000"))

# (name, plaintext_path, offset, slice). The two synthetic extremes are
# decompressed from their .gz on demand.
CORPORA = [
    ("webster", "/tmp/silesia_x/silesia/webster", 1_000_000, 512 * 1024),
    ("mozilla", "/tmp/silesia_x/silesia/mozilla", 1_000_000, 512 * 1024),
    ("lit_extreme", "/tmp/decomp_literal.bin", 200_000, 512 * 1024),
    ("backref_extreme", "/tmp/decomp_backref.bin", 0, 90 * 1024),
]


def die(msg):
    print(f"### KERNEL_AB_AARCH64 GATE FAILED: {msg} ###", file=sys.stderr)
    sys.exit(1)


def ensure_plain(path, gz):
    if os.path.exists(path):
        return
    if not os.path.exists(gz):
        die(f"missing {path} and {gz}")
    with open(path, "wb") as f:
        subprocess.run(["gzip", "-dc", gz], stdout=f, check=True)


def measure(arm, corpus, offset, slc, reps, name):
    """Run one invocation under /usr/bin/time -l, /dev/null sink -> (instr, cyc)."""
    cmd = [TIME, "-l", EXE, "--arm", arm, "--reps", str(reps),
           "--corpus", corpus, "--offset", str(offset), "--slice", str(slc),
           "--name", name]
    with open(os.devnull, "wb") as dn:
        p = subprocess.run(cmd, stdout=dn, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"{name}/{arm}: example exited {p.returncode}: {p.stderr.decode()[-400:]}")
    instr = cyc = None
    for line in p.stderr.decode().splitlines():
        s = line.split()
        if "instructions retired" in line:
            instr = int(s[0])
        elif "cycles elapsed" in line:
            cyc = int(s[0])
    if instr is None or cyc is None:
        die(f"{name}/{arm}: could not parse time -l")
    return instr, cyc


def best(arm, corpus, offset, slc, reps, name):
    measure(arm, corpus, offset, slc, reps, name)  # drop cold
    samples = [measure(arm, corpus, offset, slc, reps, name) for _ in range(N)]
    cycs = sorted(c for _, c in samples)
    med = cycs[len(cycs) // 2]
    kept = [s for s in samples if s[1] <= 1.4 * med] or samples
    return min(i for i, _ in kept), min(c for _, c in kept), cycs


def bytes_per_rep(corpus, offset, slc, name):
    cmd = [EXE, "--arm", "both", "--reps", "1", "--corpus", corpus,
           "--offset", str(offset), "--slice", str(slc), "--name", name]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"{name}: gate-0 both run failed: {p.stderr.decode()[-400:]}")
    for line in p.stdout.decode().splitlines():
        if line.startswith("RESULT-BOTH"):
            for tok in line.split():
                if tok.startswith("bytes_per_rep="):
                    return int(tok.split("=")[1])
    die(f"{name}: no bytes_per_rep")


def per_rep(arm, corpus, offset, slc, name):
    i_r, c_r, _ = best(arm, corpus, offset, slc, R, name)
    i_2r, c_2r, cyc_spread = best(arm, corpus, offset, slc, 2 * R, name)
    instr_pr = (i_2r - i_r) / R
    cyc_pr = (c_2r - c_r) / R
    # cyc spread of the 2R point as a fraction (for the TIE gate)
    lo, hi = cyc_spread[0], cyc_spread[-1]
    spread = (hi - lo) / lo if lo else 0.0
    return instr_pr, cyc_pr, spread


def main():
    if not os.path.exists(EXE):
        die(f"no example at {EXE} (cargo build --release --no-default-features "
            f"--features gzippy-native --example kernel_ab_aarch64)")
    ensure_plain("/tmp/decomp_literal.bin", "/tmp/decomp_literal.bin.gz")
    ensure_plain("/tmp/decomp_backref.bin", "/tmp/decomp_backref.bin.gz")

    print("\n== STEP-0.5 aarch64 A/B: engine A (flat) vs engine B (two-level) ==")
    print(f"   N={N}  R={R} (per-rep = (metric(2R)-metric(R))/R)  sink=/dev/null  pure-Rust both arms")
    hdr = (f"{'corpus':<16}{'B/rep':>8}{'A instr/B':>11}{'B instr/B':>11}{'B/A':>7}"
           f"{'Aexcess':>9}{'A cyc/B':>9}{'B cyc/B':>9}{'cycB/A':>8}")
    print(hdr)
    rows = []
    for name, path, off, slc in CORPORA:
        gz = path + ".gz" if not path.endswith(".bin") else None
        if not os.path.exists(path):
            die(f"corpus missing: {path}")
        bpr = bytes_per_rep(path, off, slc, name)
        a_i, a_c, a_sp = per_rep("a", path, off, slc, name)
        b_i, b_c, b_sp = per_rep("b", path, off, slc, name)
        aib, bib = a_i / bpr, b_i / bpr
        acb, bcb = a_c / bpr, b_c / bpr
        rows.append((name, bpr, aib, bib, acb, bcb, max(a_sp, b_sp)))
        print(f"{name:<16}{bpr:>8}{aib:>11.3f}{bib:>11.3f}{bib/aib:>7.3f}"
              f"{aib - bib:>9.3f}{acb:>9.3f}{bcb:>9.3f}{bcb/acb:>8.3f}")

    print("\n   READ: B/A > 1  => engine B (two-level) spends MORE instr/B than engine A (flat)")
    print("        => flat WINS by (B/A - 1). 'Aexcess' = engine-A excess vs B (negative = A cheaper).")
    print("   Discriminator: if flat's win is LARGE on lit_extreme (per-symbol) => TABLE-WIDTH/per-symbol")
    print("        structure is the lever; if SMALL there but the win is in match/cadence => CADENCE/unpack.")
    print("   scope: macOS-aarch64 NOT-YET-LAW; pair w/ Intel(asm-off) for cross-ISA LAW.\n")


if __name__ == "__main__":
    main()
