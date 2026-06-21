#!/usr/bin/env python3
"""pin_discriminator_report.py — analyze the PIN-DISCRIMINATOR 3-arm matrix.

Reads perf `-x,` CSVs named <corp>.T<T>.<arm>.<r>.csv in OUT, for arms:
  A   = HEAD default pin (with_pinning_for_capacity = SMT-packing)
  A2  = A self-test replicate
  B   = GZIPPY_NO_PIN=1 (empty map, OS schedules — faithful to rg BlockFetcher:185)
  B2  = B self-test replicate
  C   = GZIPPY_PHYS_PIN=1 (distinct-physical diagnostic)
  C2  = C self-test replicate
  RG  = rapidgzip
  RG2 = rg self-test replicate

Per cell prints per-arm best/median wall + cyc/B + effcores + GHz, the A/A
self-tests, inter-run spread, and the headline ratios A/rg B/rg C/rg with a
TIE/WIN/LOSS verdict gated on Delta vs inter-run spread (Gate-1).
"""
import glob
import os
import sys
from statistics import median, pstdev


def parse_csv(path):
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 3:
                    continue
                raw, _unit, ev = p[0], p[1], p[2]
                try:
                    vals[ev] = float(raw)
                except ValueError:
                    continue
    except OSError:
        return None
    return vals


def collect(out, corp, T, arm, outbytes):
    walls, cycb, eff, ghz = [], [], [], []
    pat = os.path.join(out, f"{corp}.T{T}.{arm}.*.csv")
    for path in sorted(glob.glob(pat)):
        v = parse_csv(path)
        if not v:
            continue
        dur_ns = v.get("duration_time")
        cyc = next((v[k] for k in v if "cycles" in k), None)
        tc_ms = v.get("task-clock")
        if dur_ns is None or cyc is None:
            continue
        wall_s = dur_ns / 1e9
        walls.append(wall_s * 1000.0)
        cycb.append(cyc / outbytes)
        if tc_ms is not None and wall_s > 0:
            eff.append((tc_ms / 1000.0) / wall_s)
            ghz.append(cyc / (tc_ms / 1000.0) / 1e9)
    return walls, cycb, eff, ghz


def merge(out, corp, T, a, b, outbytes):
    wa, ca, ea, ga = collect(out, corp, T, a, outbytes)
    wb, cb, eb, gb = collect(out, corp, T, b, outbytes)
    return wa + wb, ca + cb, ea + eb, ga + gb


def summ(name, walls, cycb, eff, ghz):
    if not walls:
        print(f"    {name:7s}: NO DATA")
        return None
    b = min(walls)
    m = median(walls)
    spread = (max(walls) - min(walls))
    cb = median(cycb)
    ef = median(eff) if eff else float("nan")
    gh = median(ghz) if ghz else float("nan")
    print(f"    {name:7s}: best {b:7.1f} ms  med {m:7.1f} ms  spread {spread:6.1f} ms  "
          f"cyc/B {cb:6.3f}  effcores {ef:5.3f}  GHz {gh:5.3f}  (n={len(walls)})")
    return {"best": b, "med": m, "spread": spread, "cycb": cb, "eff": ef,
            "ghz": gh, "walls": walls}


def verdict(num, den, label):
    """Gate-1 significance: compare best-wall ratio with a TIE band derived from
    the larger arm's inter-run spread (as a fraction of its best)."""
    if not num or not den:
        return
    r_best = num["best"] / den["best"]
    r_med = num["med"] / den["med"]
    r_cyc = num["cycb"] / den["cycb"]
    # spread band: relative spread of each arm, take the max as the tie threshold
    band = max(num["spread"] / num["best"], den["spread"] / den["best"])
    delta = abs(r_best - 1.0)
    if delta <= band:
        tag = "TIE"
    elif r_best > 1.0:
        tag = "LOSS(gz slower)"
    else:
        tag = "WIN(gz faster)"
    print(f"    {label:10s} best {r_best:.4f}  med {r_med:.4f}  cyc/B {r_cyc:.4f}  "
          f"| Delta={delta:.3f} band={band:.3f} => {tag}")


def main():
    out = sys.argv[1]
    # meta: corp -> outbytes
    bytes_map = {}
    cells = []
    meta = os.path.join(out, "meta.txt")
    if os.path.exists(meta):
        with open(meta) as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[0] == "bytes":
                    bytes_map[p[1]] = int(p[2])
                if len(p) >= 3 and p[0] == "cell":
                    cells.append((p[1], p[2]))
    print(f"=== PIN-DISCRIMINATOR 3-arm report  (cells={len(cells)}) ===")
    print(f"    A=HEAD-pin  B=NO_PIN(rg-faithful)  C=PHYS_PIN(diag)  RG=rapidgzip")
    for corp, T in cells:
        ob = bytes_map.get(corp)
        if not ob:
            continue
        print(f"\n--- {corp} T{T}  (outbytes={ob}) ---")
        A = summ("A", *merge(out, corp, T, "A", "A2", ob))
        B = summ("B", *merge(out, corp, T, "B", "B2", ob))
        C = summ("C", *merge(out, corp, T, "C", "C2", ob))
        RG = summ("RG", *merge(out, corp, T, "RG", "RG2", ob))
        # A/A self-tests
        print("    -- A/A self-tests (license <=1.03) --")
        for nm, a, b in [("A", "A", "A2"), ("B", "B", "B2"),
                         ("C", "C", "C2"), ("RG", "RG", "RG2")]:
            wa, _, _, _ = collect(out, corp, T, a, ob)
            wb, _, _, _ = collect(out, corp, T, b, ob)
            if wa and wb:
                r = min(wa) / min(wb)
                ok = "PASS" if 0.97 <= r <= 1.03 else "CHECK"
                print(f"       {nm} A/A best ratio = {r:.4f} ({ok})")
        print("    -- HEADLINE RATIOS (Gate-1 TIE band = max relative spread) --")
        verdict(A, RG, "A/rg")
        verdict(B, RG, "B/rg")
        verdict(C, RG, "C/rg")
        verdict(B, C, "B/C")
    print("\nDONE_PIN_DISCRIMINATOR_ANALYZE")


if __name__ == "__main__":
    main()
