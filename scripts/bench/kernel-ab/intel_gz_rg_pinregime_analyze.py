#!/usr/bin/env python3
"""Analyze intel_gz_rg_pinregime_guest.sh output: the PARALLEL-PARITY verdict.

Per regime (UNPIN, PIN5, PIN4) and arm (gzA,gzB,rgA,rgB), reads perf-stat CSV and
prints: gz/rg wall ratio (best-of-N + median), cyc/B both, CPUs-utilized both,
util-deficit, GHz-stability, A/A self-tests, and conservation
(wall_ratio ~= cyc/B_ratio x util-deficit).

Usage: intel_gz_rg_pinregime_analyze.py OUTDIR OUTBYTES
"""
import sys, os, glob, statistics, math

OUT = sys.argv[1]
OUTBYTES = float(sys.argv[2])
REGIMES = ["UNPIN", "PIN5", "PIN4"]

EVMAP = {
    "instructions": "ins", "cycles": "cyc", "LLC-load-misses": "llc",
    "task-clock": "tclk", "duration_time": "wall",
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

def load(tag, arm):
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{tag}.{arm}.*.csv"))):
        d = parse_csv(p)
        if d and "cyc" in d and "wall" in d and "tclk" in d:
            runs.append(d)
    return runs

def spread(vals):
    if not vals or min(vals) == 0:
        return float("nan")
    return (max(vals) - min(vals)) / min(vals) * 100.0

def summarize(runs):
    cyc = [r["cyc"] for r in runs]
    wall = [r["wall"] for r in runs]       # ns
    tclk = [r["tclk"] for r in runs]       # ms
    bi = min(range(len(runs)), key=lambda i: wall[i])  # best = min wall
    s = {}
    s["n"] = len(runs)
    s["cycB_best"] = runs[bi]["cyc"] / OUTBYTES
    s["cycB_mean"] = statistics.mean(cyc) / OUTBYTES
    ghz = [c / (t / 1000.0) / 1e9 for c, t in zip(cyc, tclk)]
    s["ghz_mean"] = statistics.mean(ghz)
    s["ghz_spread"] = spread(ghz)
    s["wall_best_ms"] = min(wall) / 1e6
    s["wall_med_ms"] = statistics.median(wall) / 1e6
    s["wall_mean_ms"] = statistics.mean(wall) / 1e6
    s["wall_spread_ms"] = (max(wall) - min(wall)) / 1e6
    s["tclk_mean_ms"] = statistics.mean(tclk)
    # CPUs-utilized = task-clock / wall (effective cores kept busy), per-run then mean
    utils = [t / (w / 1e6) for t, w in zip(tclk, wall)]
    s["util_mean"] = statistics.mean(utils)
    return s

def report_regime(tag):
    gzA = load(tag, "gzA"); gzB = load(tag, "gzB")
    rgA = load(tag, "rgA"); rgB = load(tag, "rgB")
    if not (gzA and rgA):
        print(f"\n### {tag}: MISSING DATA (gzA={len(gzA)} rgA={len(rgA)}) — VOID")
        return None
    g = summarize(gzA); r = summarize(rgA)
    gB = summarize(gzB) if gzB else None
    rB = summarize(rgB) if rgB else None
    wall_best_ratio = g["wall_best_ms"] / r["wall_best_ms"]
    wall_med_ratio = g["wall_med_ms"] / r["wall_med_ms"]
    cycB_ratio = g["cycB_best"] / r["cycB_best"]
    util_def = r["util_mean"] / g["util_mean"]

    print(f"\n===================== REGIME {tag} (N={g['n']}) =====================")
    print(f"{'metric':<22}{'gz(native)':>14}{'rapidgzip':>14}{'gz/rg':>10}")
    def row(label, gv, rv, fmt="{:.4f}", ratio=True):
        rr = (gv / rv) if (ratio and rv) else float('nan')
        print(f"{label:<22}{fmt.format(gv):>14}{fmt.format(rv):>14}" + (f"{rr:>10.4f}" if ratio else f"{'':>10}"))
    row("wall ms (best)", g["wall_best_ms"], r["wall_best_ms"], fmt="{:.1f}")
    row("wall ms (med)",  g["wall_med_ms"],  r["wall_med_ms"],  fmt="{:.1f}")
    row("cyc/B (best)",   g["cycB_best"],    r["cycB_best"])
    row("CPUs-utilized",  g["util_mean"],    r["util_mean"], fmt="{:.3f}")
    row("GHz (mean)",     g["ghz_mean"],     r["ghz_mean"])
    print(f"-- Gate-0 (regime {tag}) --")
    print(f"   wall spread (max-min ms): gz {g['wall_spread_ms']:.1f}  rg {r['wall_spread_ms']:.1f}")
    print(f"   GHz spread:               gz {g['ghz_spread']:.3f}%  rg {r['ghz_spread']:.3f}%  "
          f"(gate <=1%): {'PASS' if g['ghz_spread']<=1.0 and r['ghz_spread']<=1.0 else 'WARN'}")
    if gB and rB:
        aa_gz = g["wall_best_ms"] / gB["wall_best_ms"]
        aa_rg = r["wall_best_ms"] / rB["wall_best_ms"]
        ok = abs(aa_gz - 1) <= 0.02 and abs(aa_rg - 1) <= 0.02
        print(f"   A/A wall ratio:           gz {aa_gz:.4f}  rg {aa_rg:.4f}  "
              f"(license <=1.02): {'PASS' if ok else 'WARN'}")
    ghz_gap = abs(g["ghz_mean"] - r["ghz_mean"]) / r["ghz_mean"] * 100
    print(f"   GHz fairness gz-vs-rg:    {ghz_gap:.2f}% apart ({'fair' if ghz_gap<=2 else 'CAUTION'})")
    # conservation
    print(f"-- {tag} VERDICT NUMBERS --")
    print(f"   wall ratio gz/rg (best) = {wall_best_ratio:.4f}  ({(wall_best_ratio-1)*100:+.1f}%)")
    print(f"   wall ratio gz/rg (med)  = {wall_med_ratio:.4f}  ({(wall_med_ratio-1)*100:+.1f}%)")
    print(f"   cyc/B ratio gz/rg       = {cycB_ratio:.4f}  ({(cycB_ratio-1)*100:+.1f}%)")
    print(f"   util gz={g['util_mean']:.3f}  rg={r['util_mean']:.3f}  util-deficit(rg/gz)={util_def:.4f} ({(util_def-1)*100:+.1f}%)")
    print(f"   CONSERVATION: wall_ratio {wall_best_ratio:.4f} ~= cyc/B {cycB_ratio:.4f} x util-deficit {util_def:.4f} = {cycB_ratio*util_def:.4f}")
    return {"tag": tag, "wall_best_ratio": wall_best_ratio, "wall_med_ratio": wall_med_ratio,
            "cycB_ratio": cycB_ratio, "util_def": util_def, "g": g, "r": r}

def main():
    print("\n#################### PIN-REGIME PARITY ANALYSIS ####################")
    print(f"output_bytes = {OUTBYTES:.0f}")
    res = {}
    for tag in REGIMES:
        res[tag] = report_regime(tag)
    print("\n==================== SUMMARY (gz/rg wall) ====================")
    print(f"{'regime':<10}{'gz_best':>10}{'rg_best':>10}{'gz/rg best':>12}{'gz/rg med':>12}{'gz_util':>9}{'rg_util':>9}")
    for tag in REGIMES:
        x = res[tag]
        if not x:
            print(f"{tag:<10}  MISSING")
            continue
        print(f"{tag:<10}{x['g']['wall_best_ms']:>10.1f}{x['r']['wall_best_ms']:>10.1f}"
              f"{x['wall_best_ratio']:>12.4f}{x['wall_med_ratio']:>12.4f}"
              f"{x['g']['util_mean']:>9.3f}{x['r']['util_mean']:>9.3f}")
    # the verdict guidance
    u = res.get("UNPIN")
    if u:
        rb = u["wall_best_ratio"]
        print("\n>>> UNPINNED (production) gz/rg best =", f"{rb:.4f}", f"({(rb-1)*100:+.1f}%)")
        if rb <= 1.03:
            print("    => PARALLEL AT-PARITY CONFIRMED (gated frozen-Intel) — the pinned +14-17% was the 4-core oversubscription artifact.")
        elif rb >= 1.10:
            print("    => REAL LOSS beyond pinning — route to the located idle (pool.pick.wait / heterogeneous tail / consumer topology) for a Gate-2 perturbation.")
        else:
            print("    => INTERMEDIATE (1.03..1.10) — partial pin artifact; report both, recommend Gate-2 on the residual.")
    # does the deficit close at >=5 cores?
    if res.get("PIN5") and res.get("PIN4"):
        print(f"\n>>> deficit at PIN5 best={res['PIN5']['wall_best_ratio']:.4f} vs PIN4 best={res['PIN4']['wall_best_ratio']:.4f} "
              f"(addressable-vs-irreducible: closes at >=5 cores?)")

if __name__ == "__main__":
    main()
