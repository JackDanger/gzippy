#!/usr/bin/env python3
"""STEP-0 EMISSION AXIS analyzer.

Reads the CSV from emission_axis_guest.sh. Isolates kernel-LOOP-only counts via
the difference method: per round, per arm, Δ = (count at 2R) - (count at R), over
Δbytes = bytes(2R)-bytes(R). That cancels the one-time setup/compress/warm.

Reports per arm (gz=A, igzip _04=B): instr/B, core-cyc/B, IPC, LLC-load-miss/B,
achieved GHz, with median + spread. Then Q1 axis verdict (instr-count vs IPC) and
the gated self-tests (A/A split-half, GHz-stability).
"""
import csv, sys, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/emission_axis.csv"

rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        if r["round"].startswith("#"):
            continue
        rows.append(r)

# index: (round, arm, reps) -> dict of numeric fields
idx = {}
reps_set = defaultdict(set)
for r in rows:
    rnd = int(r["round"]); arm = r["arm"]; reps = int(r["reps"])
    idx[(rnd, arm, reps)] = {
        "bytes": float(r["bytes"]),
        "cyc": float(r["cycles"]),
        "ins": float(r["instructions"]),
        "llc": float(r["llc_load_misses"]),
        "tclk_ms": float(r["task_clock_ms"]),
        "rdtsc_cpb": float(r["rdtsc_cpb"]),
    }
    reps_set[arm].add(reps)

def med_spread(xs):
    xs = sorted(xs)
    m = st.median(xs)
    spread = (max(xs) - min(xs)) / m * 100 if m else 0.0
    return m, spread

per_arm = {}
for arm in ("a", "b"):
    reps = sorted(reps_set[arm])
    if len(reps) != 2:
        print(f"FATAL: arm {arm} needs exactly 2 rep levels, got {reps}")
        sys.exit(2)
    R, R2 = reps
    rounds = sorted({k[0] for k in idx if k[1] == arm})
    insB, cycB, ipc, llcB, ghz, rdtsc = [], [], [], [], [], []
    for rnd in rounds:
        lo = idx.get((rnd, arm, R)); hi = idx.get((rnd, arm, R2))
        if not lo or not hi:
            continue
        dbytes = hi["bytes"] - lo["bytes"]
        dins = hi["ins"] - lo["ins"]
        dcyc = hi["cyc"] - lo["cyc"]
        dllc = hi["llc"] - lo["llc"]
        dtclk = (hi["tclk_ms"] - lo["tclk_ms"]) / 1e3  # seconds
        if dbytes <= 0 or dcyc <= 0:
            continue
        insB.append(dins / dbytes)
        cycB.append(dcyc / dbytes)
        ipc.append(dins / dcyc)
        llcB.append(dllc / dbytes)
        ghz.append(dcyc / dtclk / 1e9)
        rdtsc.append((lo["rdtsc_cpb"] + hi["rdtsc_cpb"]) / 2.0)
    per_arm[arm] = {
        "n": len(insB), "R": R, "R2": R2,
        "insB": insB, "cycB": cycB, "ipc": ipc,
        "llcB": llcB, "ghz": ghz, "rdtsc": rdtsc,
    }

def split_half_ratio(xs):
    # A/A self-test: median of first half vs second half, ratio ~1.0
    if len(xs) < 4:
        return float("nan")
    h = len(xs) // 2
    a = st.median(xs[:h]); b = st.median(xs[h:])
    return a / b if b else float("nan")

print(f"=== STEP-0 EMISSION AXIS  (loop-only, difference method, file={path}) ===")
labels = {"a": "ARM A  gz clean-path emission (decode_clean_into_contig)",
          "b": "ARM B  igzip _04 (decode_huffman_code_block_stateless_04)"}
summ = {}
for arm in ("a", "b"):
    d = per_arm[arm]
    insB_m, insB_s = med_spread(d["insB"])
    cycB_m, cycB_s = med_spread(d["cycB"])
    ipc_m, ipc_s = med_spread(d["ipc"])
    llcB_m, llcB_s = med_spread(d["llcB"])
    ghz_m, ghz_s = med_spread(d["ghz"])
    rd_m, rd_s = med_spread(d["rdtsc"])
    summ[arm] = dict(insB=insB_m, cycB=cycB_m, ipc=ipc_m, llcB=llcB_m, ghz=ghz_m, rdtsc=rd_m)
    print(f"\n{labels[arm]}   (n={d['n']} rounds, R={d['R']}/{d['R2']})")
    print(f"  instr/B        = {insB_m:8.4f}   (spread {insB_s:.2f}%)   [load-immune]")
    print(f"  core-cyc/B     = {cycB_m:8.4f}   (spread {cycB_s:.2f}%)")
    print(f"  IPC            = {ipc_m:8.4f}   (spread {ipc_s:.2f}%)")
    print(f"  LLC-load-miss/B= {llcB_m:8.5f}   (spread {llcB_s:.2f}%)")
    print(f"  achieved GHz   = {ghz_m:8.4f}   (spread {ghz_s:.2f}%)   [GHz-stability gate <~0.5-1%]")
    print(f"  rdtsc cyc/B    = {rd_m:8.4f}   (spread {rd_s:.2f}%)   [independent loop-only cross-check]")
    print(f"  A/A split-half ratios: instr={split_half_ratio(d['insB']):.4f} cyc={split_half_ratio(d['cycB']):.4f} ipc={split_half_ratio(d['ipc']):.4f}")

A, B = summ["a"], summ["b"]
print("\n=== Q1 — AXIS ===")
ins_ratio = A["insB"] / B["insB"]
cyc_ratio = A["cycB"] / B["cycB"]
ipc_ratio = A["ipc"] / B["ipc"]
print(f"  instr/B   gz/_04 = {ins_ratio:6.4f}   (gz {A['insB']:.3f} vs _04 {B['insB']:.3f}, Δ={A['insB']-B['insB']:+.3f} instr/B)")
print(f"  core-cyc/B gz/_04 = {cyc_ratio:6.4f}   (gz {A['cycB']:.3f} vs _04 {B['cycB']:.3f}, Δ={A['cycB']-B['cycB']:+.3f} cyc/B)")
print(f"  IPC       gz/_04 = {ipc_ratio:6.4f}   (gz {A['ipc']:.3f} vs _04 {B['ipc']:.3f})")
print(f"  rdtsc cyc/B gz/_04 = {A['rdtsc']/B['rdtsc']:6.4f}   (cross-check of core-cyc/B)")
# decompose the cyc/B gap: cyc/B = instr/B / IPC. ln(cyc ratio) = ln(instr ratio) - ln(ipc ratio)
import math
g_cyc = math.log(cyc_ratio)
g_ins = math.log(ins_ratio)
g_ipc = -math.log(ipc_ratio)
print(f"\n  cyc/B gap decomposition (multiplicative, log-shares of the gz/_04 cyc gap):")
if abs(g_cyc) > 1e-9:
    print(f"    from MORE INSTRUCTIONS : {g_ins/g_cyc*100:5.1f}%   (gz executes {(ins_ratio-1)*100:+.1f}% instr/B)")
    print(f"    from LOWER IPC         : {g_ipc/g_cyc*100:5.1f}%   (gz IPC is {(ipc_ratio-1)*100:+.1f}% vs _04)")
if ins_ratio > 1.03 and ipc_ratio >= 0.99:
    verdict = "INSTRUCTION-COUNT (gz executes more instr/B at comparable-or-higher IPC) -> gate axis = instr/B"
elif ins_ratio <= 1.03 and ipc_ratio < 0.99:
    verdict = "IPC/THROUGHPUT (instr/B ~equal, gz IPC lower) -> gate axis = cyc/IPC, NOT instr"
elif ins_ratio > 1.03 and ipc_ratio < 0.99:
    verdict = "MIXED (gz both more instr AND lower IPC) -> dominant share above decides primary gate axis"
else:
    verdict = "NEAR-TIE in this isolated slice (instr/B ~equal, IPC ~equal) -> isolated emission gap is SMALL"
print(f"\n  Q1 VERDICT: {verdict}")
