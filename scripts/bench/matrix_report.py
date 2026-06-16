#!/usr/bin/env python3
"""matrix_report.py — render the (binary,archive,T) matrix from a decide.sh
artifact dir. Reads ONLY the deterministic per-cell sample files produced by
_decide_guest.sh (wall_gz.txt / wall_rg.txt and the pcpu_*.txt sidecars).
This is RENDERING of captured data, not a measurement: the authoritative wall
ratios are also produced by `fulcrum decide` over the same artifacts; this script
adds the P (avg-busy-CPUs) column the analyzer does not surface.

Usage: matrix_report.py <label> <artifact_dir> [<label2> <artifact_dir2> ...]
"""
import sys, os, re, statistics

ORDER = ["silesia", "monorepo", "nasa", "squishy_realdata"]
TS = [1, 2, 4, 7]

def samples(p):
    if not os.path.exists(p):
        return []
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(float(ln))
        except ValueError:
            pass
    return out

def med(xs):
    return statistics.median(xs) if xs else float("nan")

def cell_dir(art, corpus, t):
    return os.path.join(art, f"cell_{corpus}_T{t}")

def manifest_kv(art):
    kv = {}
    mf = os.path.join(art, "manifest.txt")
    if os.path.exists(mf):
        for ln in open(mf):
            if "=" in ln:
                k, v = ln.strip().split("=", 1)
                kv.setdefault(k, v)
    return kv

def main():
    args = sys.argv[1:]
    pairs = list(zip(args[0::2], args[1::2]))
    hdr = ("binary", "archive", "T", "gz_ms", "rg_ms", "gz/rg",
           "P_gz", "P_rg", "gz_spr%", "rg_spr%", "N", "sha")
    w = [16, 16, 3, 9, 9, 6, 5, 5, 7, 7, 3, 4]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for label, art in pairs:
        kv = manifest_kv(art)
        binsha = kv.get("bin_sha", "?")[:12]
        for corpus in ORDER:
            for t in TS:
                d = cell_dir(art, corpus, t)
                gz = samples(os.path.join(d, "wall_gz.txt"))
                rg = samples(os.path.join(d, "wall_rg.txt"))
                if not gz or not rg:
                    continue
                gz_min, rg_min = min(gz), min(rg)
                gz_ms, rg_ms = gz_min * 1000, rg_min * 1000
                ratio = gz_min / rg_min if rg_min else float("nan")
                gzspr = (max(gz) - min(gz)) / min(gz) * 100 if min(gz) else 0
                rgspr = (max(rg) - min(rg)) / min(rg) * 100 if min(rg) else 0
                pgz = samples(os.path.join(d, "pcpu_gz.txt"))
                prg = samples(os.path.join(d, "pcpu_rg.txt"))
                Pgz = med(pgz) / 100 if pgz else float("nan")
                Prg = med(prg) / 100 if prg else float("nan")
                n = min(len(gz), len(rg))
                row = (
                    label.ljust(w[0]), corpus.ljust(w[1]), str(t).ljust(w[2]),
                    f"{gz_ms:.1f}".ljust(w[3]), f"{rg_ms:.1f}".ljust(w[4]),
                    f"{ratio:.3f}".ljust(w[5]),
                    (f"{Pgz:.2f}" if Pgz == Pgz else "NA").ljust(w[6]),
                    (f"{Prg:.2f}" if Prg == Prg else "NA").ljust(w[7]),
                    f"{gzspr:.1f}".ljust(w[8]), f"{rgspr:.1f}".ljust(w[9]),
                    str(n).ljust(w[10]), "OK".ljust(w[11]),
                )
                print("  ".join(row))
        print(f"# {label}: bin_sha={binsha} freeze={kv.get('freeze_state','?')} "
              f"gov={kv.get('governor','?')} no_turbo={kv.get('no_turbo','?')} "
              f"quiet={kv.get('quiet_state','?')} host={kv.get('host_cpu_model','?')}")
    print()
    # deficit ranking (gz/rg) across all rows
    rows = []
    for label, art in pairs:
        for corpus in ORDER:
            for t in TS:
                d = cell_dir(art, corpus, t)
                gz = samples(os.path.join(d, "wall_gz.txt"))
                rg = samples(os.path.join(d, "wall_rg.txt"))
                if gz and rg and min(rg):
                    rows.append((min(gz) / min(rg), label, corpus, t))
    rows.sort(reverse=True)
    print("## gz/rg deficit ranking (worst gzippy first; >1 = gzippy slower):")
    for r in rows:
        print(f"  {r[0]:.3f}  {r[1]:<14} {r[2]:<16} T{r[3]}")

if __name__ == "__main__":
    main()
