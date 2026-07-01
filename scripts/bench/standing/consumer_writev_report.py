#!/usr/bin/env python3
"""consumer_writev_report.py — analyze the 3-arm consumer-writev verdict matrix.

Reads /dev/shm/cwv-art/<corp>.T<T>.<arm>.<rep>.csv (perf stat -x, output) and
prints, per thread-count: each arm's wall (best-of-N + median + stdev spread),
gz/rg ratio, cyc/B; the A/A and rg/rg self-tests; and the FORK math:
  - how much of the gz/rg gap arm B (removal ceiling) closes
  - how much arm C (byte-exact fix) closes
  - TIE gate: |Delta(A-B)| < inter-run spread => B FLAT => STOP.
"""
import sys, os, glob, statistics, re

def parse_csv(path):
    """Return dict of event->value from a perf -x, file."""
    vals = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            raw, unit, ev = parts[0], parts[1], parts[2]
            try:
                v = float(raw)
            except ValueError:
                continue
            vals[ev] = v
    return vals

def load(outdir):
    data = {}  # (T, arm) -> list of dict
    for p in glob.glob(os.path.join(outdir, "*.csv")):
        b = os.path.basename(p)
        m = re.match(r".+\.T(\d+)\.([A-Za-z0-9]+)\.(\d+)\.csv$", b)
        if not m:
            continue
        T = int(m.group(1)); arm = m.group(2)
        v = parse_csv(p)
        if "duration_time" not in v:
            continue
        data.setdefault((T, arm), []).append(v)
    return data

def ms(ns): return ns / 1e6

def summarize(recs):
    walls = sorted(ms(r["duration_time"]) for r in recs)
    cycs = [r.get("cpu_core/cycles/", r.get("cycles", 0.0)) for r in recs]
    best = walls[0]
    med = statistics.median(walls)
    sd = statistics.pstdev(walls) if len(walls) > 1 else 0.0
    medcyc = statistics.median(cycs)
    return dict(best=best, med=med, sd=sd, medcyc=medcyc, n=len(walls), walls=walls)

def main():
    outdir = sys.argv[1]
    meta = {}
    mp = os.path.join(outdir, "meta.txt")
    if os.path.exists(mp):
        for line in open(mp):
            kv = line.split(None, 1)
            if len(kv) == 2:
                meta[kv[0]] = kv[1].strip()
    bytes_ = float(meta.get("bytes", 0)) or 1.0
    data = load(outdir)
    Ts = sorted({T for (T, _) in data})
    print(f"== CONSUMER-WRITEV VERDICT  sha={meta.get('sha','?')[:12]} corp={meta.get('corp','?')} "
          f"bytes={int(bytes_)} N={meta.get('n','?')} no_turbo={meta.get('no_turbo','?')} gov={meta.get('gov','?')} ==")
    print(f"   load_start={meta.get('load_start','?')}")
    ARMS = ["A", "A2", "B", "C", "RG", "RG2"]
    for T in Ts:
        print(f"\n--- silesia T{T} ---")
        S = {}
        for arm in ARMS:
            if (T, arm) in data:
                S[arm] = summarize(data[(T, arm)])
        # header
        print(f"   {'arm':<5}{'best(ms)':>10}{'med(ms)':>10}{'sd(ms)':>9}{'cyc/B':>9}{'n':>4}  {'gz/rg(best)':>12} {'gz/rg(med)':>11}")
        rg_best = S["RG"]["best"]; rg_med = S["RG"]["med"]
        for arm in ARMS:
            if arm not in S:
                continue
            s = S[arm]
            rb = s["best"]/rg_best
            rm = s["med"]/rg_med
            cb = s["medcyc"]/bytes_
            print(f"   {arm:<5}{s['best']:>10.2f}{s['med']:>10.2f}{s['sd']:>9.2f}{cb:>9.3f}{s['n']:>4}  {rb:>12.4f} {rm:>11.4f}")
        # self-tests
        aa_best = S["A"]["best"]/S["A2"]["best"]; aa_med = S["A"]["med"]/S["A2"]["med"]
        rr_best = S["RG"]["best"]/S["RG2"]["best"]; rr_med = S["RG"]["med"]/S["RG2"]["med"]
        print(f"   self-test A/A  best={aa_best:.4f} med={aa_med:.4f}   (want ~1.0)")
        print(f"   self-test RG/RG best={rr_best:.4f} med={rr_med:.4f}  (want ~1.0)")
        # FORK math — use BOTH best-of-N and median for robustness.
        for metric, key in (("best-of-N", "best"), ("median", "med")):
            a = S["A"][key]; b = S["B"][key]; c = S["C"][key]; rg = S["RG"][key]
            gap_abs = a - rg            # ms the gz arm is behind rg
            gap_pct = (a/rg - 1.0)*100
            d_ab = a - b                # ceiling: ms removed by skipping writev
            d_ac = a - c                # real fix: ms saved by overlap
            # pooled inter-run spread (sd of A and the compared arm)
            sp_b = (S["A"]["sd"] + S["B"]["sd"])
            sp_c = (S["A"]["sd"] + S["C"]["sd"])
            b_close = (d_ab/gap_abs*100) if gap_abs > 0 else float('nan')
            c_close = (d_ac/gap_abs*100) if gap_abs > 0 else float('nan')
            tie_b = "TIE(flat)" if abs(d_ab) < sp_b else "MOVES"
            tie_c = "TIE(flat)" if abs(d_ac) < sp_c else "MOVES"
            print(f"   [{metric}] gap A vs rg = {gap_abs:+.2f}ms ({gap_pct:+.1f}%)")
            print(f"      B ceiling: A-B = {d_ab:+.2f}ms  spread(A+B)={sp_b:.2f}ms  -> {tie_b}; closes {b_close:.0f}% of gap")
            print(f"      C fix    : A-C = {d_ac:+.2f}ms  spread(A+C)={sp_c:.2f}ms  -> {tie_c}; closes {c_close:.0f}% of gap")

if __name__ == "__main__":
    main()
