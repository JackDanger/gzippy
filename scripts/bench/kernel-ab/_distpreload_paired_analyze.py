#!/usr/bin/env python3
"""Paired-statistics analyzer for the dist-preload cyc/byte A/B.

REPLACES the min-of-N + Δ-vs-spread estimator (which an advisor downgraded:
min-of-N biases toward whichever arm caught the cleanest bandwidth window, and
"Δ vs arm-level spread" is not a significance test). Because the arms are
INTERLEAVED per rep (A1,A2,B,A1,A2,B,...), rep r of each arm sees ~the same
instantaneous box state -> the samples are PAIRED. We therefore estimate the
effect as the per-rep PAIRED difference and test it properly:

  - estimator  : median of per-rep paired diffs  (B_r - A1_r), in cyc/byte
  - CI         : paired bootstrap 95% CI of that median (resample reps w/ repl.)
  - p-value    : Wilcoxon signed-rank (two-sided, normal approx w/ tie + cc)
  - self-test  : same machinery on (A2_r - A1_r) -> median ~= 0, CI includes 0

Pure stdlib (no scipy). Also writes per-rep raw cycle samples to OUTDIR/<tag>.samples.tsv
so the stats are reproducible off-box.

Usage: _distpreload_paired_analyze.py OUTDIR [--tag LABEL] corpus1 [corpus2 ...]
"""
import sys, os, glob, math, random, statistics

random.seed(20260618)  # reproducible bootstrap

args = sys.argv[1:]
TAG = ""
if "--tag" in args:
    i = args.index("--tag")
    TAG = args[i + 1]
    del args[i:i + 2]
OUT = args[0]
CORPORA = args[1:]
BOOT = 20000

EVMAP = {
    "instructions": "ins",
    "cycles": "cyc",
    "branches": "br",
    "branch-misses": "bmiss",
    "cache-references": "llcref",
    "cache-misses": "llcmiss",
    "task-clock": "tclk",  # ms
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
                val, _unit, ev = parts[0], parts[1], parts[2]
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


def load_by_rep(corp, arm):
    """Return {rep:int -> counterdict} for a given corpus+arm."""
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, f"{corp}.{arm}.*.csv"))):
        base = os.path.basename(p)
        # corp.arm.REP.csv
        try:
            rep = int(base.split(".")[-2])
        except ValueError:
            continue
        d = parse_csv(p)
        if d and "cyc" in d and "ins" in d:
            out[rep] = d
    return out


def normal_sf(z):
    """Survival function of standard normal: P(Z>z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank p (normal approx, tie + continuity corr).
    Returns (W_plus, z, p, n_nonzero)."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return (0.0, 0.0, 1.0, 0)
    absd = sorted((abs(d), d) for d in nz)
    # average ranks for ties on |d|
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    Wp = sum(ranks[k] for k in range(n) if absd[k][1] > 0)
    mean = n * (n + 1) / 4.0
    # variance with tie correction
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # tie correction term
    from collections import Counter
    tie_groups = Counter(round(absd[k][0], 12) for k in range(n))
    tcorr = sum(t**3 - t for t in tie_groups.values()) / 48.0
    var -= tcorr
    if var <= 0:
        return (Wp, 0.0, 1.0, n)
    # continuity correction toward the mean
    if Wp > mean:
        z = (Wp - mean - 0.5) / math.sqrt(var)
    elif Wp < mean:
        z = (Wp - mean + 0.5) / math.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * normal_sf(abs(z))
    p = min(1.0, p)
    return (Wp, z, p, n)


def bootstrap_ci_median(diffs, iters=BOOT, alpha=0.05):
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    meds = []
    for _ in range(iters):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int((alpha / 2) * iters)]
    hi = meds[int((1 - alpha / 2) * iters) - 1]
    return (statistics.median(diffs), lo, hi)


bytes_map = {}
with open(os.path.join(OUT, "bytes.txt")) as fh:
    for line in fh:
        c, b = line.split()
        bytes_map[c] = int(b)

label = f"[{TAG}] " if TAG else ""
print(f"\n############ PAIRED-STAT ANALYSIS {label}############")
summary = []
for corp in CORPORA:
    nb = bytes_map.get(corp)
    if not nb:
        continue
    A1 = load_by_rep(corp, "A1")
    A2 = load_by_rep(corp, "A2")
    B = load_by_rep(corp, "B")
    reps = sorted(set(A1) & set(B))
    reps_self = sorted(set(A1) & set(A2))
    if not reps:
        print(f"\n== {corp}: NO PAIRED DATA ==")
        continue

    print(f"\n==================== {corp}  (raw_bytes={nb:,}, N_pairs={len(reps)}) ====================")

    # persist raw per-rep cycle samples
    spath = os.path.join(OUT, f"{corp}.{TAG or 'run'}.samples.tsv")
    with open(spath, "w") as fh:
        fh.write("rep\tA1_cyc\tA2_cyc\tB_cyc\tA1_ins\tB_ins\tA1_bmiss\tB_bmiss\tA1_br\tB_br\tA1_llcmiss\tB_llcmiss\tA1_llcref\tB_llcref\tA1_tclk\tB_tclk\n")
        for r in sorted(set(A1) | set(A2) | set(B)):
            a1 = A1.get(r, {}); a2 = A2.get(r, {}); b = B.get(r, {})
            fh.write("\t".join(str(x) for x in [
                r, a1.get("cyc", ""), a2.get("cyc", ""), b.get("cyc", ""),
                a1.get("ins", ""), b.get("ins", ""),
                a1.get("bmiss", ""), b.get("bmiss", ""),
                a1.get("br", ""), b.get("br", ""),
                a1.get("llcmiss", ""), b.get("llcmiss", ""),
                a1.get("llcref", ""), b.get("llcref", ""),
                a1.get("tclk", ""), b.get("tclk", ""),
            ]) + "\n")

    # ---- paired cyc/byte diffs (B - A1) ----
    dcpb = [(B[r]["cyc"] - A1[r]["cyc"]) / nb for r in reps]
    med, lo, hi = bootstrap_ci_median(dcpb)
    Wp, z, p, nnz = wilcoxon_signed_rank(dcpb)

    # arm medians for context
    a1_cpb = statistics.median([A1[r]["cyc"] / nb for r in reps])
    b_cpb = statistics.median([B[r]["cyc"] / nb for r in reps])
    pct = 100.0 * med / a1_cpb if a1_cpb else 0.0

    # IPC / instr-byte / branch-miss paired deltas (medians)
    d_ipc = statistics.median([B[r]["ins"] / B[r]["cyc"] - A1[r]["ins"] / A1[r]["cyc"] for r in reps])
    d_ipb = statistics.median([(B[r]["ins"] - A1[r]["ins"]) / nb for r in reps])
    d_bmr = statistics.median([
        (100.0 * B[r].get("bmiss", 0) / B[r]["br"] if B[r].get("br") else 0) -
        (100.0 * A1[r].get("bmiss", 0) / A1[r]["br"] if A1[r].get("br") else 0)
        for r in reps])

    # LLC-miss rate (confounder)
    llc = statistics.median([
        100.0 * A1[r].get("llcmiss", 0) / A1[r]["llcref"] if A1[r].get("llcref") else 0
        for r in reps])
    # GHz stability
    ghz = []
    for r in reps:
        for dd in (A1[r], B[r]):
            if dd.get("tclk"):
                ghz.append(dd["cyc"] / (dd["tclk"] * 1e6))
    ghz_spread = (max(ghz) - min(ghz)) / min(ghz) if len(ghz) > 1 else 0.0

    # ---- self-test (A2 - A1) ----
    if reps_self:
        dself = [(A2[r]["cyc"] - A1[r]["cyc"]) / nb for r in reps_self]
        smed, slo, shi = bootstrap_ci_median(dself)
        sWp, sz, sp, _ = wilcoxon_signed_rank(dself)
        self_ok = (slo <= 0 <= shi)
        sa1 = statistics.median([A1[r]["cyc"] / nb for r in reps_self])
        spct = 100.0 * smed / sa1 if sa1 else 0.0
    else:
        smed = slo = shi = sp = spct = 0.0
        self_ok = False

    ci_excludes_0 = (lo > 0) or (hi < 0)
    sig = (p < 0.01) and ci_excludes_0

    print(f"  arm medians: A1={a1_cpb:.4f} cyc/byte   B={b_cpb:.4f} cyc/byte")
    print(f"  GATE0(d) GHz spread = {100*ghz_spread:.3f}%  {'PASS' if ghz_spread<=0.05 else 'WARN'}")
    print(f"  GATE0(e) LLC-miss rate (A1, median) = {llc:.1f}%  (confounder; controlled by stressor run)")
    print(f"  GATE0(b) SELF-TEST (A2-A1): median Δ={smed:+.5f} cyc/byte ({spct:+.3f}%)  "
          f"95%CI=[{slo:+.5f},{shi:+.5f}]  Wilcoxon p={sp:.3f}  -> {'PASS (CI includes 0)' if self_ok else 'FAIL (CI excludes 0 -> rig biased)'}")
    print(f"  PAIRED Δ (B-A1): median={med:+.5f} cyc/byte ({pct:+.3f}%)  "
          f"95%CI=[{lo:+.5f},{hi:+.5f}]")
    print(f"     Wilcoxon signed-rank: W+={Wp:.0f}  z={z:+.3f}  p={p:.4g}  (n_nonzero={nnz})")
    print(f"     ΔIPC(med)={d_ipc:+.4f}  Δinstr/byte(med)={d_ipb:+.4f}  Δbranch-miss%%(med)={d_bmr:+.4f}")
    print(f"     CI excludes 0: {ci_excludes_0}   p<0.01: {p<0.01}   -> {'SIGNIF (B faster)' if sig and med<0 else ('SIGNIF (B slower)' if sig and med>0 else 'NOT SIGNIF (WASH/TIE)')}")
    print(f"  raw samples -> {spath}")

    summary.append((corp, med, lo, hi, p, sig, "faster" if med < 0 else "slower", self_ok, llc))

print(f"\n############ SUMMARY {label}(T1 inner-kernel, Intel i7-13700T LXC) ############")
print(f"  {'corpus':<10} {'medΔcyc/B':>11} {'95%CI':>24} {'Wilcox-p':>10} {'self-ok':>8} {'LLC%':>6} {'verdict':>16}")
for corp, med, lo, hi, p, sig, dirn, self_ok, llc in summary:
    v = (f"SIGNIF-{dirn}" if sig else "WASH/TIE")
    print(f"  {corp:<10} {med:>+11.5f} {f'[{lo:+.5f},{hi:+.5f}]':>24} {p:>10.4g} {str(self_ok):>8} {llc:>6.1f} {v:>16}")
