#!/usr/bin/env python3
"""NIGHT37 — kernel REMOVAL-ORACLE cyc/byte analyzer (paired, interleaved).

Reads perf-stat CSV per (corpus, arm, rep) from OUTDIR. Arms:
  A1, A2 = BASE  (GZIPPY_ISAL_ENGINE_ORACLE=0 -> native clean decode = run_contig
                  + its per-block table-build + native clean glue)
  B      = ORACLE (GZIPPY_ISAL_ENGINE_ORACLE=1 -> igzip clean decode = the _04
                  family via isal_inflate + igzip's table-build + igzip glue)

Both arms are the SAME binary at T1 (-p1), pinned to one P-core, /dev/null sink,
interleaved per rep (A1,A2,B). Per-rep pairing of A1[r] vs B[r] is therefore valid.

The DROP = (BASE - ORACLE) cyc/byte = the ABSOLUTE on-wall production share of the
WHOLE native clean-decode engine (kernel + table-build + clean glue). Because _04
is SOTA, this is an UPPER BOUND on what converging the native clean decode can buy.
It is NOT a floor and NOT a kernel-only number (it also folds in table-build+glue);
decompose against NIGHT35's table-build share (~0.11-0.14 cyc/B silesia).

GATES:
  0(b) self-test  A2 vs A1 paired median ratio ~ 1.000 (CI brackets 0 delta)
  0(d) freq stable: max achieved-GHz spread small
  0(e) LLC-miss rate reported (mem-bound confounder)
  1    paired bootstrap 95% CI on (BASE-ORACLE) cyc/B, N reps.

Usage: _kernoracle_analyze.py OUTDIR REF_GAP_SILESIA REF_GAP_MONO corpus1 [corpus2 ...]
  REF_GAP_* = the NIGHT36 +cyc/B gap vs igzip for that corpus (denominator).
"""
import sys, os, glob, statistics, random

OUT = sys.argv[1]
REF_GAP = {"silesia": float(sys.argv[2]), "monorepo": float(sys.argv[3])}
CORPORA = sys.argv[4:]

EVMAP = {
    "instructions": "ins", "cycles": "cyc", "branches": "br",
    "branch-misses": "bmiss", "cache-references": "llcref",
    "cache-misses": "llcmiss", "task-clock": "tclk",
}

def parse_csv(path):
    d = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                val, _u, ev = parts[0], parts[1], parts[2]
                if val in ("<not supported>", "<not counted>", ""):
                    continue
                for key, short in EVMAP.items():
                    if key in ev:
                        try:
                            d[short] = float(val)
                        except ValueError:
                            pass
                        break
    except FileNotFoundError:
        return None
    return d

def load_series(corp, arm, nreps):
    """Return per-rep dict list ordered by rep (None if missing)."""
    out = []
    for r in range(1, nreps + 1):
        p = os.path.join(OUT, f"{corp}.{arm}.{r}.csv")
        d = parse_csv(p)
        out.append(d if (d and "cyc" in d) else None)
    return out

bytes_map = {}
with open(os.path.join(OUT, "bytes.txt")) as fh:
    for line in fh:
        c, b = line.split()
        bytes_map[c] = int(b)

# how many reps?
def detect_nreps(corp):
    n = 0
    for p in glob.glob(os.path.join(OUT, f"{corp}.B.*.csv")):
        try:
            n = max(n, int(p.rsplit(".", 2)[1]))
        except Exception:
            pass
    return n

def spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    return (max(vals) - min(vals)) / min(vals)

def boot_ci(diffs, iters=20000):
    """95% bootstrap CI of the mean of paired diffs."""
    if len(diffs) < 2:
        return (0.0, 0.0)
    means = []
    n = len(diffs)
    for _ in range(iters):
        s = sum(diffs[random.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)

random.seed(1337)
print("==================== NIGHT37 KERNEL REMOVAL-ORACLE (BASE=run_contig vs ORACLE=igzip _04) ====================")
for corp in CORPORA:
    nb = bytes_map.get(corp)
    if not nb:
        continue
    nreps = detect_nreps(corp)
    A1 = load_series(corp, "A1", nreps)
    A2 = load_series(corp, "A2", nreps)
    B = load_series(corp, "B", nreps)
    print(f"\n---- {corp}  raw_bytes={nb:,}  nreps={nreps} ----")

    def arm_cpb(series):
        return [(d["cyc"] / nb) if d else None for d in series]
    a1c, a2c, bc = arm_cpb(A1), arm_cpb(A2), arm_cpb(B)

    def summ(name, cpb, series):
        vals = [v for v in cpb if v is not None]
        if not vals:
            print(f"  {name}: NO DATA"); return
        ipb = statistics.median([d["ins"] / nb for d in series if d])
        ipc = statistics.median([d["ins"] / d["cyc"] for d in series if d])
        ghz = [d["cyc"] / (d["tclk"] * 1e6) for d in series if d and d.get("tclk")]
        llc = [100.0 * d["llcmiss"] / d["llcref"] for d in series if d and d.get("llcref")]
        print(f"  {name:<5} n={len(vals):>2} cyc/B med={statistics.median(vals):.4f} "
              f"min={min(vals):.4f} spread={100*spread(vals):.2f}%  instr/B={ipb:.4f} "
              f"IPC={ipc:.3f} GHz={statistics.mean(ghz) if ghz else 0:.3f} "
              f"LLCmiss%={statistics.median(llc) if llc else 0:.2f}")
    summ("BASE", a1c, A1)
    summ("BASEt", a2c, A2)
    summ("ORACL", bc, B)

    # GATE0(b) self-test: paired A2-A1
    self_d = [a2c[i] - a1c[i] for i in range(nreps) if a1c[i] is not None and a2c[i] is not None]
    if self_d:
        lo, hi = boot_ci(self_d)
        med = statistics.median(self_d)
        ok = lo <= 0.0 <= hi
        print(f"  GATE0(b) self-test BASEt-BASE: med={med:+.4f} cyc/B CI[{lo:+.4f},{hi:+.4f}] "
              f"{'PASS(brackets 0)' if ok else 'WARN-RIG-BIASED'}")

    # GATE0(d) freq stability across all arms
    ghz_all = []
    for series in (A1, A2, B):
        ghz_all += [d["cyc"] / (d["tclk"] * 1e6) for d in series if d and d.get("tclk")]
    if ghz_all:
        gsp = (max(ghz_all) - min(ghz_all)) / min(ghz_all)
        print(f"  GATE0(d) freq-stable: GHz spread {100*gsp:.2f}% {'PASS' if gsp <= 0.06 else 'WARN-JITTER'}")

    # GATE1 A/B paired: BASE - ORACLE (positive = ORACLE faster = the DROP)
    diffs = [a1c[i] - bc[i] for i in range(nreps) if a1c[i] is not None and bc[i] is not None]
    if diffs:
        med = statistics.median(diffs)
        mean = statistics.mean(diffs)
        lo, hi = boot_ci(diffs)
        npos = sum(1 for d in diffs if d > 0)
        gap = REF_GAP.get(corp)
        frac = (med / gap * 100.0) if gap else 0.0
        sig = "CI-DISJOINT-FROM-0" if (lo > 0 or hi < 0) else "CI-INCLUDES-0(TIE)"
        print(f"  GATE1 DROP (BASE-ORACLE): med={med:+.4f} mean={mean:+.4f} cyc/B "
              f"CI[{lo:+.4f},{hi:+.4f}]  n={len(diffs)} pos={npos}/{len(diffs)}  {sig}")
        if gap:
            flo, fhi = lo / gap * 100.0, hi / gap * 100.0
            print(f"        vs NIGHT36 igzip gap {gap:+.3f} cyc/B  ->  DROP = {frac:.1f}% of the gap "
                  f"(CI [{flo:.1f}%,{fhi:.1f}%])")
print("\nNOTE: DROP folds kernel + native-table-build + native-clean-glue. Decompose with")
print("NIGHT35 table-build share (~0.11-0.14 cyc/B silesia). NOT a floor/irreducible claim.")
print("Intel i7-13700T LXC, single P-core, T1 -p1. NOT-YET-LAW; AMD/Zen2 OWED.")
