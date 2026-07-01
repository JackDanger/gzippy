#!/usr/bin/env python3
"""Analyze _distpreload_cycbyte_guest.sh output: best-of-N cyc/byte A/B.

Reads perf-stat CSV per (corpus, arm, rep) from OUTDIR and prints, per corpus:
  - per-arm best-of-N (min cycles) cyc/byte, instr/byte, IPC, branch-miss-rate,
    LLC-miss-rate, GHz, task-clock-ms, and inter-run spread.
  - GATE0(b) self-test ratio  A2.min/A1.min  (want 1.000 +- spread).
  - GATE0(e) LLC-miss rate (memory-bound confounder).
  - A/B verdict: ratio B.min/A1.min, Delta cyc/byte vs spread.

Usage: _distpreload_cycbyte_analyze.py OUTDIR corpus1 [corpus2 ...]
"""
import sys, os, glob, statistics

OUT = sys.argv[1]
CORPORA = sys.argv[2:]

EVMAP = {
    "instructions": "ins",
    "cycles": "cyc",
    "branches": "br",
    "branch-misses": "bmiss",
    "cache-references": "llcref",
    "cache-misses": "llcmiss",
    "task-clock": "tclk",  # milliseconds
}

def parse_csv(path):
    """perf stat -x, lines: value,unit,event,runtime,pct,... -> {short:float}"""
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

def load(corp, arm):
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{corp}.{arm}.*.csv"))):
        d = parse_csv(p)
        if d and "cyc" in d and "ins" in d:
            runs.append(d)
    return runs

bytes_map = {}
with open(os.path.join(OUT, "bytes.txt")) as fh:
    for line in fh:
        c, b = line.split()
        bytes_map[c] = int(b)

def spread(vals):
    """relative spread = (max-min)/min as a fraction."""
    if len(vals) < 2:
        return 0.0
    return (max(vals) - min(vals)) / min(vals)

verdicts = []
for corp in CORPORA:
    nb = bytes_map.get(corp)
    if not nb:
        continue
    print(f"\n==================== {corp}  (raw_bytes={nb:,}) ====================")
    arms = {}
    for arm in ("A1", "A2", "B"):
        runs = load(corp, arm)
        if not runs:
            print(f"  {arm}: NO DATA")
            continue
        cycs = [r["cyc"] for r in runs]
        best = min(cycs)
        bi = cycs.index(best)
        r = runs[bi]  # the best-of-N run's full counter set
        ghz_list = [rr["cyc"] / (rr["tclk"] * 1e6) for rr in runs if rr.get("tclk")]
        arms[arm] = {
            "n": len(runs),
            "cyc_min": best,
            "cyc_spread": spread(cycs),
            "ins": r["ins"],
            "br": r.get("br", 0),
            "bmiss": r.get("bmiss", 0),
            "llcref": r.get("llcref", 0),
            "llcmiss": r.get("llcmiss", 0),
            "tclk": r.get("tclk", 0),
            "ghz_mean": statistics.mean(ghz_list) if ghz_list else 0,
            "ghz_spread": spread(ghz_list) if ghz_list else 0,
        }
    print(f"  {'arm':<4} {'N':>3} {'cyc/byte':>10} {'spread%':>8} {'instr/byte':>11} "
          f"{'IPC':>6} {'bmiss%br':>9} {'LLCmiss%':>9} {'GHz':>6} {'tclk_ms':>8}")
    for arm in ("A1", "A2", "B"):
        a = arms.get(arm)
        if not a:
            continue
        cpb = a["cyc_min"] / nb
        ipb = a["ins"] / nb
        ipc = a["ins"] / a["cyc_min"]
        bmr = 100.0 * a["bmiss"] / a["br"] if a["br"] else 0.0
        llcm = 100.0 * a["llcmiss"] / a["llcref"] if a["llcref"] else 0.0
        print(f"  {arm:<4} {a['n']:>3} {cpb:>10.4f} {100*a['cyc_spread']:>8.2f} "
              f"{ipb:>11.4f} {ipc:>6.3f} {bmr:>9.3f} {llcm:>9.3f} "
              f"{a['ghz_mean']:>6.3f} {a['tclk']:>8.1f}")

    # ---- GATE0(d) freq-stability + GATE0(e) memory-bound ----
    ghz_sp = max((a["ghz_spread"] for a in arms.values()), default=0)
    print(f"  GATE0(d) freq-stable: max achieved-GHz spread = {100*ghz_sp:.2f}%  "
          f"{'PASS' if ghz_sp <= 0.05 else 'WARN-FREQ-JITTER'}")
    llc = max((100.0 * a["llcmiss"] / a["llcref"] if a["llcref"] else 0 for a in arms.values()), default=0)
    print(f"  GATE0(e) mem-bound:  max LLC-miss rate = {llc:.1f}%  (confounder note; "
          f"arbiter is cyc-spread/self-test, not absolute LLC%)")

    # ---- GATE0(b) self-test + A/B verdict ----
    if "A1" in arms and "A2" in arms:
        self_ratio = arms["A2"]["cyc_min"] / arms["A1"]["cyc_min"]
        sp = max(arms["A1"]["cyc_spread"], arms["A2"]["cyc_spread"])
        ok = abs(self_ratio - 1.0) <= sp
        print(f"  GATE0(b) self-test  A2/A1 = {self_ratio:.4f}  (spread {100*sp:.2f}%)  "
              f"{'PASS' if ok else 'FAIL-RIG-NOISY'}")
    if "A1" in arms and "B" in arms:
        a1 = arms["A1"]; b = arms["B"]
        ratio = b["cyc_min"] / a1["cyc_min"]              # <1 => B faster
        dcpb = (b["cyc_min"] - a1["cyc_min"]) / nb         # cyc/byte delta (B-A)
        sp = max(a1["cyc_spread"], b["cyc_spread"])
        dpct = 100.0 * (ratio - 1.0)
        a1_ipc = a1["ins"] / a1["cyc_min"]; b_ipc = b["ins"] / b["cyc_min"]
        # significance: |Δ%| vs spread%
        if abs(dpct) <= 100 * sp:
            verdict = "TIE/WASH (|Δ| <= spread)"
        elif ratio < 1.0:
            verdict = "WIN for B (dist-preload faster)"
        else:
            verdict = "REGRESSION for B (dist-preload slower)"
        print(f"  A/B  B/A1 = {ratio:.4f}  Δ={dpct:+.2f}%  Δcyc/byte={dcpb:+.4f}  "
              f"spread={100*sp:.2f}%  ΔIPC={b_ipc-a1_ipc:+.4f}")
        print(f"  VERDICT[{corp}]: {verdict}")
        verdicts.append((corp, verdict, dpct, 100 * sp))

print("\n==================== OVERALL (gated-HYPOTHESIS, Intel i7-13700T, T1 inner-kernel) ====================")
for corp, v, d, s in verdicts:
    print(f"  {corp:<12} Δ={d:+.2f}%  spread={s:.2f}%  -> {v}")
