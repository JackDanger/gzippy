#!/usr/bin/env python3
"""Verdict analyzer for kernel_gate.sh — KEEP / TIE / REVERT per corpus.

Direct descendant of _distpreload_paired_analyze.py (the advisor-approved paired
estimator): the arms are INTERLEAVED per rep (A1,A2,B,A1,A2,B,...) so rep r of
each arm sees ~the same instantaneous box state -> the samples are PAIRED.

  arm A = BASELINE sha   (A1 = the compared baseline, A2 = a 2nd baseline run)
  arm B = CANDIDATE sha

Per corpus it reports:
  - arm medians (cyc/byte + wall ms)
  - GATE0(d) GHz spread, GATE0(e) LLC-miss%% (confounders)
  - GATE0(b) A/A SELF-TEST (A2-A1): median ~= 0 & CI includes 0  (else cell UNTRUSTED)
  - INTER-RUN SPREAD of the baseline arm (cyc/byte, %% of median) — the Gate-1 noise floor
  - PAIRED Δ (B-A1): median cyc/byte + paired-bootstrap 95%% CI + Wilcoxon signed-rank p
  - PAIRED Δ wall (B-A1): median ms (secondary, production-wall)
  - VERDICT:
        KEEP    B significantly FASTER (Wilcoxon p<0.01 AND CI excludes 0 AND |Δ|>spread)
        REVERT  B significantly SLOWER (same gate, other sign)
        TIE     not significant, or |Δ| <= inter-run spread (Gate-1: Δ<spread => TIE)
    A cell whose A/A self-test FAILS or whose GHz spread is wild is stamped UNTRUSTED
    (the loaded box did not hold still; the verdict does not exist for that cell).

Pure stdlib (no scipy). Writes per-rep raw samples to OUTDIR/<corp>.samples.tsv.

Usage: _kernel_gate_analyze.py OUTDIR [--tag LABEL] [--threads T] corpus1 [corpus2 ...]
"""
import sys, os, glob, math, random, statistics

random.seed(20260620)  # reproducible bootstrap

args = sys.argv[1:]
TAG = ""
THREADS = "?"
if "--tag" in args:
    i = args.index("--tag"); TAG = args[i + 1]; del args[i:i + 2]
if "--threads" in args:
    i = args.index("--threads"); THREADS = args[i + 1]; del args[i:i + 2]
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
    "task-clock": "tclk",      # ms
    "duration_time": "wall",   # ns
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
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, f"{corp}.{arm}.*.csv"))):
        base = os.path.basename(p)
        try:
            rep = int(base.split(".")[-2])
        except ValueError:
            continue
        d = parse_csv(p)
        if d and "cyc" in d:
            out[rep] = d
    return out


def normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(diffs):
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return (0.0, 0.0, 1.0, 0)
    absd = sorted((abs(d), d) for d in nz)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    Wp = sum(ranks[k] for k in range(n) if absd[k][1] > 0)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    from collections import Counter
    tie_groups = Counter(round(absd[k][0], 12) for k in range(n))
    tcorr = sum(t**3 - t for t in tie_groups.values()) / 48.0
    var -= tcorr
    if var <= 0:
        return (Wp, 0.0, 1.0, n)
    if Wp > mean:
        z = (Wp - mean - 0.5) / math.sqrt(var)
    elif Wp < mean:
        z = (Wp - mean + 0.5) / math.sqrt(var)
    else:
        z = 0.0
    p = min(1.0, 2.0 * normal_sf(abs(z)))
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
print(f"\n############ KERNEL-GATE VERDICT {label}(T{THREADS}, paired) ############")
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
        summary.append((corp, 0, 0, 0, 1.0, 0, "UNTRUSTED", False))
        continue

    print(f"\n==================== {corp}  (raw_bytes={nb:,}, N_pairs={len(reps)}) ====================")

    spath = os.path.join(OUT, f"{corp}.{TAG or 'run'}.samples.tsv")
    with open(spath, "w") as fh:
        fh.write("rep\tA1_cyc\tA2_cyc\tB_cyc\tA1_ins\tB_ins\tA1_wall_ns\tB_wall_ns\tA1_llcmiss\tA1_llcref\tA1_tclk\tB_tclk\n")
        for r in sorted(set(A1) | set(A2) | set(B)):
            a1 = A1.get(r, {}); a2 = A2.get(r, {}); b = B.get(r, {})
            fh.write("\t".join(str(x) for x in [
                r, a1.get("cyc", ""), a2.get("cyc", ""), b.get("cyc", ""),
                a1.get("ins", ""), b.get("ins", ""),
                a1.get("wall", ""), b.get("wall", ""),
                a1.get("llcmiss", ""), a1.get("llcref", ""),
                a1.get("tclk", ""), b.get("tclk", ""),
            ]) + "\n")

    # ---- paired cyc/byte diffs (B - A1) ----
    dcpb = [(B[r]["cyc"] - A1[r]["cyc"]) / nb for r in reps]
    med, lo, hi = bootstrap_ci_median(dcpb)
    Wp, z, p, nnz = wilcoxon_signed_rank(dcpb)

    a1_cpb = statistics.median([A1[r]["cyc"] / nb for r in reps])
    b_cpb = statistics.median([B[r]["cyc"] / nb for r in reps])
    pct = 100.0 * med / a1_cpb if a1_cpb else 0.0

    # ---- INTER-RUN SPREAD of the baseline arm (the Gate-1 noise floor) ----
    a1_series = sorted(A1[r]["cyc"] / nb for r in reps)
    spread_abs = a1_series[-1] - a1_series[0]           # cyc/byte max-min
    spread_pct = 100.0 * spread_abs / a1_cpb if a1_cpb else 0.0

    # ---- wall (production-wall) paired Δ, ms ----
    have_wall = all("wall" in A1[r] and "wall" in B[r] for r in reps)
    if have_wall:
        dwall_ms = [(B[r]["wall"] - A1[r]["wall"]) / 1e6 for r in reps]
        wmed, wlo, whi = bootstrap_ci_median(dwall_ms)
        a1_wall_ms = statistics.median([A1[r]["wall"] / 1e6 for r in reps])
        wpct = 100.0 * wmed / a1_wall_ms if a1_wall_ms else 0.0
    else:
        wmed = wlo = whi = a1_wall_ms = wpct = 0.0

    # IPC / instr-byte
    have_ins = all("ins" in A1[r] and "ins" in B[r] for r in reps)
    if have_ins:
        d_ipc = statistics.median([B[r]["ins"] / B[r]["cyc"] - A1[r]["ins"] / A1[r]["cyc"] for r in reps])
        d_ipb = statistics.median([(B[r]["ins"] - A1[r]["ins"]) / nb for r in reps])
    else:
        d_ipc = d_ipb = 0.0

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
    ghz_ok = ghz_spread <= 0.05

    # ---- A/A self-test (A2 - A1) ----
    if reps_self:
        dself = [(A2[r]["cyc"] - A1[r]["cyc"]) / nb for r in reps_self]
        smed, slo, shi = bootstrap_ci_median(dself)
        _, _, sp, _ = wilcoxon_signed_rank(dself)
        self_ok = (slo <= 0 <= shi)
        sa1 = statistics.median([A1[r]["cyc"] / nb for r in reps_self])
        spct = 100.0 * smed / sa1 if sa1 else 0.0
    else:
        smed = slo = shi = sp = spct = 0.0
        self_ok = False

    ci_excludes_0 = (lo > 0) or (hi < 0)
    beats_spread = abs(med) > spread_abs
    signif = (p < 0.01) and ci_excludes_0 and beats_spread
    trusted = self_ok and ghz_ok

    if not trusted:
        verdict = "UNTRUSTED"
    elif signif and med < 0:
        verdict = "KEEP"
    elif signif and med > 0:
        verdict = "REVERT"
    else:
        verdict = "TIE"

    print(f"  arm medians: A1={a1_cpb:.4f} cyc/B  B={b_cpb:.4f} cyc/B" +
          (f"   |  wall A1={a1_wall_ms:.2f} ms" if have_wall else ""))
    print(f"  GATE0(d) GHz spread = {100*ghz_spread:.3f}%  {'PASS' if ghz_ok else 'WARN/UNTRUSTED'}")
    print(f"  GATE0(e) LLC-miss rate (A1, median) = {llc:.1f}%  (confounder)")
    print(f"  GATE0(b) A/A SELF-TEST (A2-A1): median Δ={smed:+.5f} cyc/B ({spct:+.3f}%)  "
          f"95%CI=[{slo:+.5f},{shi:+.5f}]  -> {'PASS (CI includes 0)' if self_ok else 'FAIL (rig biased -> UNTRUSTED)'}")
    print(f"  GATE-1 INTER-RUN SPREAD (A1, max-min) = {spread_abs:.5f} cyc/B ({spread_pct:.3f}%)  "
          f"[Δ must exceed this]")
    print(f"  PAIRED Δ (B-A1): median={med:+.5f} cyc/B ({pct:+.3f}%)  95%CI=[{lo:+.5f},{hi:+.5f}]")
    print(f"     Wilcoxon p={p:.4g}  CI-excl-0={ci_excludes_0}  |Δ|>spread={beats_spread}  (n={nnz})")
    if have_wall:
        print(f"     wall Δ (B-A1): median={wmed:+.3f} ms ({wpct:+.3f}%)  95%CI=[{wlo:+.3f},{whi:+.3f}]")
    if have_ins:
        print(f"     ΔIPC(med)={d_ipc:+.4f}  Δinstr/B(med)={d_ipb:+.4f}")
    print(f"  >>> VERDICT[{corp}] = {verdict}")

    summary.append((corp, med, lo, hi, p, spread_abs, verdict, trusted))

print(f"\n############ KERNEL-GATE SUMMARY {label}(Intel-class arch, NOT-YET-LAW until AMD) ############")
print(f"  {'corpus':<10} {'medΔcyc/B':>11} {'95%CI':>24} {'spread':>9} {'Wilcox-p':>10} {'verdict':>10}")
verdicts = []
for corp, med, lo, hi, p, spread, verdict, trusted in summary:
    print(f"  {corp:<10} {med:>+11.5f} {f'[{lo:+.5f},{hi:+.5f}]':>24} {spread:>9.5f} {p:>10.4g} {verdict:>10}")
    verdicts.append(verdict)

# overall: REVERT if any cell REVERTs (a regression anywhere blocks); else KEEP if
# any KEEP and no REVERT; else TIE; UNTRUSTED dominates if a trusted verdict is absent.
if "UNTRUSTED" in verdicts and not any(v in ("KEEP", "REVERT", "TIE") for v in verdicts):
    overall = "UNTRUSTED"
elif "REVERT" in verdicts:
    overall = "REVERT"
elif "KEEP" in verdicts:
    overall = "KEEP"
else:
    overall = "TIE" if verdicts else "UNTRUSTED"
print(f"\n  OVERALL VERDICT = {overall}   (per-corpus: {', '.join(verdicts)})")
print(f"  (KEEP=candidate faster & significant; REVERT=slower & significant; TIE=Δ<spread or not signif)")
