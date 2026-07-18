#!/usr/bin/env python3
"""mac_rg_gap_report.py — reduce mac_rg_gap.sh samples to the gz-vs-rapidgzip GAP MAP.

Reads samples.csv (corpus,bytes,arm,threads,rep,instr,cyc,real).

PRIMARY metric: instructions retired (deterministic ~0.04% warm) -> gz/rg total-work
ratio. SECONDARY: cycles elapsed. TERTIARY: real wall (coarse 10ms clock on mac).

Best-of-N = min per (corpus,arm,threads). Spread = (max-min)/min within the arm =
the A/A self-test (instr spread >0.5% -> instr UNTRUSTED for that cell; cyc >3% ->
cyc UNTRUSTED).

VERDICT (per cell, instr-primary): ratio = gz_instr/rg_instr.
  rg-WINS  if ratio > 1 + (gz_spread+rg_spread)   (gz does more work; Δ > spread)
  gz-WINS  if ratio < 1 - (gz_spread+rg_spread)
  TIE      otherwise.

BIFURCATION (#2 structural cause, deterministic multiplicative decomposition of the
total-instr ratio):
  gz_instr(T)/rg_instr(T)
     = [gz_instr(T1)/rg_instr(T1)]  ×  [gz_instr(T)/gz_instr(T1)] / [rg_instr(T)/rg_instr(T1)]
     =       KERNEL                 ×              PIPELINE-tax differential
  KERNEL          = the single-thread total-work gap (decode kernel dominates).
  PIPELINE-tax    = how much MORE coordination instruction-work gz's parallel
                    pipeline adds going T1->T than rg's does.
  Fork: PIPELINE-tax > 1  => gz pipeline ABOVE rg's => CONVERGENCE opportunity
        (faithful structural port can close it).
        PIPELINE-tax ~= 1 => pipeline at parity; the cell gap is the KERNEL carried
        up => FUNDAMENTAL to the (aarch64 pure-Rust) decode kernel, not the pipeline.
"""
import sys, csv, collections

INSTR_SPREAD_MAX = 0.005   # 0.5%
CYC_SPREAD_MAX   = 0.03    # 3%

def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("NO SAMPLES"); return 1

    # group: (corpus, arm, threads) -> lists
    G = collections.defaultdict(lambda: {"instr": [], "cyc": [], "real": [], "bytes": 0})
    corpora, threads_set = [], set()
    for r in rows:
        k = (r["corpus"], r["arm"], int(r["threads"]))
        G[k]["instr"].append(int(r["instr"]))
        G[k]["cyc"].append(int(r["cyc"]))
        G[k]["real"].append(float(r["real"]))
        G[k]["bytes"] = int(r["bytes"])
        if r["corpus"] not in corpora:
            corpora.append(r["corpus"])
        threads_set.add(int(r["threads"]))
    threads = sorted(threads_set)

    def stat(corpus, arm, t):
        g = G.get((corpus, arm, t))
        if not g or not g["instr"]:
            return None
        def mm(xs):
            lo = min(xs); hi = max(xs)
            return lo, (hi - lo) / lo if lo else 0.0
        i_min, i_sp = mm(g["instr"])
        c_min, c_sp = mm(g["cyc"])
        r_min = min(g["real"])
        return dict(instr=i_min, instr_sp=i_sp, cyc=c_min, cyc_sp=c_sp,
                    real=r_min, bytes=g["bytes"])

    print()
    print("=" * 100)
    print("gz-vs-rapidgzip GAP MAP  (macOS-aarch64, quiet box)   ratio = gzippy/rapidgzip   <1 = gz wins")
    print("  PRIMARY = instructions retired (deterministic).  cyc secondary, wall(real) coarse-tertiary.")
    print("=" * 100)
    hdr = f"{'corpus':<10}{'T':>3} | {'gz instr/B':>11}{'rg instr/B':>11}{'instr↕':>9} | {'I-ratio':>8}{'C-ratio':>8}{'W-ratio':>8} | {'VERDICT':<9}"
    print(hdr)
    print("-" * len(hdr))

    cell = {}   # (corpus,t) -> dict with ratios + verdict
    rg_wins = []
    for corpus in corpora:
        for t in threads:
            gz = stat(corpus, "gzippy", t)
            rg = stat(corpus, "rapidgzip", t)
            if not gz or not rg:
                continue
            B = gz["bytes"]
            gz_ipb = gz["instr"] / B
            rg_ipb = rg["instr"] / B
            i_ratio = gz["instr"] / rg["instr"]
            c_ratio = gz["cyc"] / rg["cyc"]
            w_ratio = (gz["real"] / rg["real"]) if rg["real"] > 0 else float("nan")
            comb_sp = gz["instr_sp"] + rg["instr_sp"]
            instr_trust = (gz["instr_sp"] <= INSTR_SPREAD_MAX and rg["instr_sp"] <= INSTR_SPREAD_MAX)
            # verdict (instr-primary)
            if i_ratio > 1 + comb_sp:
                verdict = "rg-WINS"
            elif i_ratio < 1 - comb_sp:
                verdict = "gz-WINS"
            else:
                verdict = "TIE"
            note = "" if instr_trust else "  [instr UNTRUSTED: spread>0.5%]"
            print(f"{corpus:<10}{t:>3} | {gz_ipb:>11.2f}{rg_ipb:>11.2f}{gz['instr_sp']*100:>8.2f}% | "
                  f"{i_ratio:>8.3f}{c_ratio:>8.3f}{w_ratio:>8.3f} | {verdict:<9}{note}")
            cell[(corpus, t)] = dict(i_ratio=i_ratio, c_ratio=c_ratio, w_ratio=w_ratio,
                                     verdict=verdict, gz=gz, rg=rg)
            if verdict == "rg-WINS":
                rg_wins.append((corpus, t))
        print("-" * len(hdr))

    # ---- BIFURCATION for ALL T>1 cells (the transferable structural signal) ----
    # NOTE: on aarch64 the ABSOLUTE instr gap REVERSES (rg has no ISA-L here, falls
    # to a portable inflate ~3x heavier), so most cells read "gz-WINS" on instr.
    # That absolute verdict does NOT transfer to x86 (rg uses ISA-L there). The
    # TRANSFERABLE deterministic signal is the T1->T instruction INFLATION
    # (pipeline coordination work), which is ~arch-independent. So decompose EVERY
    # T>1 cell, not just rg-WINS.
    print()
    print("=" * 100)
    print("#2 BIFURCATION — deterministic instr decomposition, ALL T>1 cells  (KERNEL × PIPELINE-tax)")
    print("  (absolute instr verdict reverses on aarch64 [rg lacks ISA-L]; PIPELINE-tax inflation is the transferable signal)")
    print("=" * 100)
    bh = (f"{'corpus':<10}{'T':>3} | {'cell I-ratio':>12} | {'KERNEL(T1)':>11} | "
          f"{'gz infl':>8}{'rg infl':>8}{'PIPE-tax':>9} | {'fork':<28}{'verdict':>9}")
    print(bh)
    print("-" * len(bh))
    for corpus in corpora:
        for t in threads:
            if t == 1:
                continue
            if (corpus, t) not in cell:
                continue
            c1g = stat(corpus, "gzippy", 1)
            c1r = stat(corpus, "rapidgzip", 1)
            if not c1g or not c1r:
                continue
            kernel = c1g["instr"] / c1r["instr"]
            gz_infl = cell[(corpus, t)]["gz"]["instr"] / c1g["instr"]
            rg_infl = cell[(corpus, t)]["rg"]["instr"] / c1r["instr"]
            pipe_tax = gz_infl / rg_infl
            ir = cell[(corpus, t)]["i_ratio"]
            vrd = cell[(corpus, t)]["verdict"]
            if pipe_tax > 1.02:
                fork = "CONVERGENCE (gz pipe>rg)"
            elif pipe_tax < 0.98:
                fork = "gz pipe<rg"
            else:
                fork = "pipe at parity"
            print(f"{corpus:<10}{t:>3} | {ir:>12.3f} | {kernel:>11.3f} | "
                  f"{gz_infl:>8.3f}{rg_infl:>8.3f}{pipe_tax:>9.3f} | {fork:<28}{vrd:>9}")
        print("-" * len(bh))
    print()
    print("  rg-WINS cells (absolute instr, gz does more total work, Δ>spread): "
          + (", ".join(f"{c}/T{t}" for (c, t) in rg_wins) if rg_wins else "NONE"))
    print()
    print("  KERNEL = gz/rg single-thread total-work ratio. On aarch64 this reverses (<1) because rg has no")
    print("           ISA-L kernel here -> does NOT transfer; the x86 ISA-L kernel gap is OWED to a quiet Intel window.")
    print("  PIPE-tax = (gz T1->T instr inflation) / (rg T1->T instr inflation) — the parallel coordination overhead")
    print("           differential. ~ARCH-INDEPENDENT, so THIS transfers cross-arch.")
    print("    PIPE-tax > 1 (CONVERGENCE) => gz's pipeline emits MORE coordination instr-work than rg's per added")
    print("           thread -> a faithful structural port of rg's pipeline can close it (live convergence target).")
    print("    PIPE-tax ~= 1 (parity)     => pipeline is at parity; remaining gap is the kernel carried up the")
    print("           thread count, i.e. FUNDAMENTAL to the decode kernel, not a pipeline defect.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
