#!/usr/bin/env python3
"""Analyze intel_gz_rg_cycbyte_guest.sh output: the silesia-T4 AXIS verdict.

Reads perf-stat CSV per (tier in {T1,T4}, arm in {gzA,gzB,rgA,rgB}, rep) from
OUTDIR and prints, per tier:
  - per-tool ABSOLUTE instr/B (load-immune; mean +- inter-run spread),
    cyc/B (best-of-N = min cycles run, least-interrupted), IPC, GHz (+spread),
    LLC-load-miss/B, wall ms (best-of-N), task-clock ms.
  - GATE-0: instructions inter-run spread <0.5% (the load-immune primitive IS
    stable); GHz spread <=1%; gz-vs-gz and rg-vs-rg A/A wall ratio ~1.0.
  - THE AXIS CALL: gz instr/B vs rg instr/B vs the cyc/B gap and IPC.

Usage: intel_gz_rg_cycbyte_analyze.py OUTDIR OUTBYTES
"""
import sys, os, glob, statistics

OUT = sys.argv[1]
OUTBYTES = float(sys.argv[2])

EVMAP = {
    "instructions": "ins",
    "cycles": "cyc",
    "LLC-load-misses": "llc",
    "task-clock": "tclk",      # milliseconds
    "duration_time": "wall",   # nanoseconds
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

def load(tier, arm):
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{tier}.{arm}.*.csv"))):
        d = parse_csv(p)
        if d and "cyc" in d and "ins" in d and "wall" in d:
            runs.append(d)
    return runs

def spread(vals):
    if not vals or min(vals) == 0:
        return float("nan")
    return (max(vals) - min(vals)) / min(vals) * 100.0

def summarize(runs):
    """Return dict of summary metrics for a list of per-run dicts."""
    ins = [r["ins"] for r in runs]
    cyc = [r["cyc"] for r in runs]
    wall = [r["wall"] for r in runs]                 # ns
    tclk = [r["tclk"] for r in runs]                 # ms
    llc = [r.get("llc", 0.0) for r in runs]
    # best-of-N anchor = the min-cycles run (least interrupted)
    bi = min(range(len(runs)), key=lambda i: cyc[i])
    best = runs[bi]
    s = {}
    s["n"] = len(runs)
    # instr/B: load-immune -> report mean + spread
    s["insB_mean"] = statistics.mean(ins) / OUTBYTES
    s["insB_spread"] = spread(ins)
    # cyc/B: best-of-N (min cycles)
    s["cycB_best"] = best["cyc"] / OUTBYTES
    s["cycB_mean"] = statistics.mean(cyc) / OUTBYTES
    # IPC from the best run (and mean)
    s["ipc_best"] = best["ins"] / best["cyc"]
    s["ipc_mean"] = statistics.mean(ins) / statistics.mean(cyc)
    # GHz per run = cycles / (task-clock_sec); report mean + spread
    ghz = [c / (t / 1000.0) / 1e9 for c, t in zip(cyc, tclk)]
    s["ghz_mean"] = statistics.mean(ghz)
    s["ghz_spread"] = spread(ghz)
    # LLC-load-miss / byte (best run)
    s["llcB"] = best.get("llc", 0.0) / OUTBYTES
    # wall best-of-N (ms)
    s["wall_best_ms"] = min(wall) / 1e6
    s["wall_med_ms"] = statistics.median(wall) / 1e6
    s["wall_mean_ms"] = statistics.mean(wall) / 1e6
    s["wall_spread"] = spread(wall)
    s["tclk_best_ms"] = best["tclk"]
    s["tclk_mean_ms"] = statistics.mean(tclk)
    # CPUs-utilized = task-clock / wall (effective cores kept busy). The RATIO of
    # this between tools (A/A-licensed) is the parallel-efficiency signal at T4.
    s["util_mean"] = s["tclk_mean_ms"] / s["wall_mean_ms"]
    return s

def report_tier(tier):
    gzA = load(tier, "gzA"); gzB = load(tier, "gzB")
    rgA = load(tier, "rgA"); rgB = load(tier, "rgB")
    if not (gzA and rgA):
        print(f"\n### {tier}: MISSING DATA (gzA={len(gzA)} rgA={len(rgA)}) — VOID")
        return None
    g = summarize(gzA); r = summarize(rgA)
    # A/A self-tests on wall (best-of-N), licenses the loaded-box ratio
    gB = summarize(gzB) if gzB else None
    rB = summarize(rgB) if rgB else None
    print(f"\n========================= {tier} =========================")
    print(f"{'metric':<22}{'gz(native)':>16}{'rapidgzip':>16}{'gz/rg':>10}")
    def row(label, gv, rv, fmt="{:.4f}", ratio=True):
        rr = (gv / rv) if (ratio and rv) else float('nan')
        print(f"{label:<22}{fmt.format(gv):>16}{fmt.format(rv):>16}"
              + (f"{rr:>10.4f}" if ratio else f"{'':>10}"))
    row("instr/B (mean)", g["insB_mean"], r["insB_mean"])
    row("cyc/B (best-of-N)", g["cycB_best"], r["cycB_best"])
    row("cyc/B (mean)", g["cycB_mean"], r["cycB_mean"])
    row("IPC (best)", g["ipc_best"], r["ipc_best"])
    row("IPC (mean)", g["ipc_mean"], r["ipc_mean"])
    row("GHz (mean)", g["ghz_mean"], r["ghz_mean"])
    row("LLC-miss/B", g["llcB"], r["llcB"])
    row("wall ms (best)", g["wall_best_ms"], r["wall_best_ms"], fmt="{:.1f}")
    row("wall ms (med)", g["wall_med_ms"], r["wall_med_ms"], fmt="{:.1f}")
    row("task-clock ms", g["tclk_mean_ms"], r["tclk_mean_ms"], fmt="{:.1f}")
    row("CPUs-utilized", g["util_mean"], r["util_mean"], fmt="{:.3f}")
    # ---- Gate-0 stability ----
    print(f"-- Gate-0 stability --")
    print(f"   instr inter-run spread:  gz {g['insB_spread']:.3f}%  rg {r['insB_spread']:.3f}%  "
          f"(LOAD-IMMUNE PRIMITIVE; gate <0.5%): "
          f"{'PASS' if g['insB_spread']<0.5 and r['insB_spread']<0.5 else 'WARN'}")
    print(f"   GHz spread:              gz {g['ghz_spread']:.3f}%  rg {r['ghz_spread']:.3f}%  "
          f"(gate <=1%): {'PASS' if g['ghz_spread']<=1.0 and r['ghz_spread']<=1.0 else 'WARN'}")
    print(f"   wall spread:             gz {g['wall_spread']:.1f}%   rg {r['wall_spread']:.1f}%")
    if gB and rB:
        aa_gz = g["wall_best_ms"] / gB["wall_best_ms"]
        aa_rg = r["wall_best_ms"] / rB["wall_best_ms"]
        print(f"   A/A wall ratio:          gz {aa_gz:.4f}  rg {aa_rg:.4f}  "
              f"(license <=1.02): {'PASS' if abs(aa_gz-1)<=0.02 and abs(aa_rg-1)<=0.02 else 'WARN'}")
    # GHz fairness between tools
    ghz_gap = abs(g["ghz_mean"] - r["ghz_mean"]) / r["ghz_mean"] * 100
    print(f"   GHz fairness gz-vs-rg:   {ghz_gap:.2f}% apart "
          f"({'fair' if ghz_gap<=2 else 'CAUTION — freq differs'})")
    # ---- the per-tier axis numbers ----
    insB_ratio = g["insB_mean"] / r["insB_mean"]
    cycB_ratio = g["cycB_best"] / r["cycB_best"]
    ipc_ratio  = g["ipc_best"] / r["ipc_best"]
    wall_ratio = g["wall_best_ms"] / r["wall_best_ms"]
    print(f"-- {tier} AXIS NUMBERS --")
    print(f"   instr/B ratio gz/rg = {insB_ratio:.4f}  ({'+' if insB_ratio>=1 else ''}{(insB_ratio-1)*100:.1f}%)  [LOAD-IMMUNE]")
    print(f"   cyc/B   ratio gz/rg = {cycB_ratio:.4f}  ({'+' if cycB_ratio>=1 else ''}{(cycB_ratio-1)*100:.1f}%)")
    print(f"   IPC     ratio gz/rg = {ipc_ratio:.4f}  ({'+' if ipc_ratio>=1 else ''}{(ipc_ratio-1)*100:.1f}%)")
    print(f"   wall    ratio gz/rg = {wall_ratio:.4f}  ({'+' if wall_ratio>=1 else ''}{(wall_ratio-1)*100:.1f}%)  (A/A-licensed)")
    return {"insB_ratio": insB_ratio, "cycB_ratio": cycB_ratio,
            "ipc_ratio": ipc_ratio, "wall_ratio": wall_ratio,
            "g": g, "r": r}

def axis_verdict(tag, res):
    if not res:
        return
    insR, cycR, ipcR = res["insB_ratio"], res["cycB_ratio"], res["ipc_ratio"]
    # cyc/B gap accounted for by instr (cyc_ratio = ins_ratio / ipc_ratio):
    # if ins_ratio ~ cyc_ratio (and ipc ~ 1 or favors gz), it's INSTRUCTION-COUNT.
    print(f"\n>>> {tag} AXIS:")
    print(f"    cyc/B gap = {(cycR-1)*100:+.1f}%   instr/B gap = {(insR-1)*100:+.1f}%   IPC gap = {(ipcR-1)*100:+.1f}%")
    # WALL decomposition: wall_ratio = (Sigma-cyc/B ratio) x (utilization deficit).
    # util_deficit = rg_util / gz_util  (gz keeps fewer cores busy => >1 => inflates wall).
    g = res["g"]; r = res["r"]
    util_def = r["util_mean"] / g["util_mean"]
    wallR = res["wall_ratio"]
    print(f"    util: gz={g['util_mean']:.3f} cores  rg={r['util_mean']:.3f} cores  "
          f"util-deficit(rg/gz)={util_def:.4f} ({(util_def-1)*100:+.1f}%)")
    print(f"    WALL decomposition: wall_ratio {wallR:.4f} ~= cyc/B_ratio {cycR:.4f} "
          f"x util-deficit {util_def:.4f} (= {cycR*util_def:.4f})")
    if abs(wallR - 1) >= 0.05:
        comp_work = (cycR - 1) / (wallR - 1) * 100
        comp_util = (util_def - 1) / (wallR - 1) * 100
        print(f"    WALL gap share (approx): per-byte CPU-work {comp_work:.0f}% | parallel-utilization {comp_util:.0f}%")
    # decomposition of the cyc/B ratio into instr and (1/ipc) contributions (logs).
    # UNSTABLE when the cyc/B gap is near zero (log denominator -> 0); suppress then.
    import math
    if cycR > 0 and insR > 0 and ipcR > 0 and abs(cycR - 1) >= 0.02:
        L = math.log(cycR)
        share_ins = math.log(insR) / L * 100
        share_ipc = -math.log(ipcR) / L * 100
        print(f"    cyc/B gap decomposition: instructions {share_ins:.0f}% | IPC(1/ipc) {share_ipc:.0f}%")
    else:
        print(f"    (cyc/B gap |{(cycR-1)*100:.1f}%| too small for a stable share decomposition; "
              f"read instr/B and IPC directly)")
    if insR >= 1.05 and ipcR >= 0.98:
        print("    => INSTRUCTION-COUNT AXIS (gz executes more instr/B at comparable/better IPC)")
        print("       -> silesia gap is apportionable/portable -> recommend STEP 2 (per-phase localize)")
    elif insR <= 1.03 and cycR >= 1.05 and ipcR < 0.98:
        print("    => IPC / ISA-L-CODEGEN AXIS (gz ~equal instr/B but lower IPC)")
        print("       -> deep-kernel codegen lever (poor ROI) -> recommend BANK-vs-fund")
    else:
        print("    => MIXED / inconclusive — see numbers (instr and IPC both contribute)")

def main():
    print("\n#################### AXIS ANALYSIS ####################")
    print(f"output_bytes = {OUTBYTES:.0f}")
    t1 = report_tier("T1")
    t4 = report_tier("T4")
    axis_verdict("T1", t1)
    axis_verdict("T4", t4)
    if t1 and t4:
        agree = (
            (t1["insB_ratio"] >= 1.05) == (t4["insB_ratio"] >= 1.05)
        )
        print(f"\n>>> T1-vs-T4 AGREEMENT on instr-axis sign: {'YES' if agree else 'NO — they disagree'}")

if __name__ == "__main__":
    main()
