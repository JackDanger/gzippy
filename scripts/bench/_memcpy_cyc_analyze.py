#!/usr/bin/env python3
# Analyze interleaved cyc/instr CSV: cyc/byte, instr/byte, IPC base vs after,
# gz/rg cyc/byte ratio before/after. Reports median + spread (CoV%).
import csv, sys, statistics as st

SIZES = {"silesia": 211968000, "monorepo": 50915328}
path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/memcpy_cyc.csv"

rows = list(csv.DictReader(open(path)))
# group: (corpus, arm) -> list of (cyc, ins)
g = {}
for r in rows:
    k = (r["corpus"], r["arm"])
    g.setdefault(k, []).append((int(r["cycles"]), int(r["instructions"])))

def med(xs): return st.median(xs)
def cov(xs):
    m = st.mean(xs)
    return 100.0 * st.pstdev(xs) / m if m else 0.0

def summ(corpus, arm):
    cyc = [c for c, i in g[(corpus, arm)]]
    ins = [i for c, i in g[(corpus, arm)]]
    n = len(cyc)
    sz = SIZES[corpus]
    return {
        "n": n,
        "cyc_med": med(cyc), "cyc_cov": cov(cyc),
        "ins_med": med(ins), "ins_cov": cov(ins),
        "cyc_per_byte": med(cyc) / sz,
        "ins_per_byte": med(ins) / sz,
        "ipc": med(ins) / med(cyc),
    }

print(f"{'corpus':9} {'arm':6} {'N':>3} {'cyc/byte':>9} {'covC%':>6} {'ins/byte':>9} {'covI%':>6} {'IPC':>6}")
for corpus in ["silesia", "monorepo"]:
    arms = {}
    for arm in ["base", "after", "rg"]:
        if (corpus, arm) not in g:
            continue
        s = summ(corpus, arm)
        arms[arm] = s
        print(f"{corpus:9} {arm:6} {s['n']:>3} {s['cyc_per_byte']:>9.4f} {s['cyc_cov']:>6.2f} "
              f"{s['ins_per_byte']:>9.4f} {s['ins_cov']:>6.2f} {s['ipc']:>6.3f}")
    b, a = arms.get("base"), arms.get("after")
    if b and a:
        dc = (a["cyc_per_byte"] - b["cyc_per_byte"]) / b["cyc_per_byte"] * 100
        di = (a["ins_per_byte"] - b["ins_per_byte"]) / b["ins_per_byte"] * 100
        dipc = (a["ipc"] - b["ipc"]) / b["ipc"] * 100
        print(f"  -> {corpus} DELTA after vs base:  cyc/byte {dc:+.2f}%   instr/byte {di:+.2f}%   IPC {dipc:+.2f}%")
        print(f"     spread context: base covC={b['cyc_cov']:.2f}% after covC={a['cyc_cov']:.2f}%  "
              f"(cyc Δ {'>' if abs(dc) > max(b['cyc_cov'],a['cyc_cov']) else '<='} spread)")
    rg = arms.get("rg")
    if rg and b and a:
        rb = b["cyc_per_byte"] / rg["cyc_per_byte"]
        ra = a["cyc_per_byte"] / rg["cyc_per_byte"]
        print(f"     gz/rg cyc/byte ratio:  BASE {rb:.4f}x  ->  AFTER {ra:.4f}x   (rg cyc/byte={rg['cyc_per_byte']:.4f}, covC={rg['cyc_cov']:.2f}%)")
    print()
