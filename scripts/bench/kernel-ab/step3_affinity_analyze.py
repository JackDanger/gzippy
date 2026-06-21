#!/usr/bin/env python3
"""step3_affinity_analyze.py — analyze the STEP-3 internal physical-core affinity
probe. Reads perf `-x,` CSVs named <arm>.<r>.csv in OUT for arms:
  gzbase  (gz, GZIPPY_PHYS_PIN unset = default get_core_ids cycling pin)
  gzphys  (gz, GZIPPY_PHYS_PIN=1 = workers+consumer on distinct physical cores)
  rg      (rapidgzip reference)
Emits per-arm best/median wall + cyc/B + effcores + GHz, and the ratios
gzbase/rg, gzphys/rg, gzphys/gzbase, with A/A self-tests (gzbaseA/gzbaseB etc.).
"""
import glob
import os
import sys
from statistics import median


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
                    v = float(raw)
                except ValueError:
                    continue
                vals[ev] = v
    except OSError:
        return None
    return vals


def collect(out, arm, outbytes):
    walls, cycb, eff, ghz = [], [], [], []
    for path in sorted(glob.glob(os.path.join(out, f"{arm}.*.csv"))):
        v = parse_csv(path)
        if not v:
            continue
        dur_ns = v.get("duration_time")
        cyc = None
        for k in v:
            if k.endswith("cycles") or k == "cycles":
                cyc = v[k]
        ins = None
        for k in v:
            if k.endswith("instructions") or k == "instructions":
                ins = v[k]
        tc_ms = v.get("task-clock")
        if dur_ns is None or cyc is None:
            continue
        wall_s = dur_ns / 1e9
        walls.append(wall_s * 1000.0)  # ms
        cycb.append(cyc / outbytes)
        if tc_ms is not None and wall_s > 0:
            eff.append((tc_ms / 1000.0) / wall_s)
            ghz.append(cyc / (tc_ms / 1000.0) / 1e9)
    return walls, cycb, eff, ghz


def summ(name, walls, cycb, eff, ghz):
    if not walls:
        print(f"  {name}: NO DATA")
        return None
    b = min(walls)
    m = median(walls)
    cb = median(cycb)
    ef = median(eff) if eff else float("nan")
    gh = median(ghz) if ghz else float("nan")
    print(f"  {name:8s}: best {b:7.1f} ms  med {m:7.1f} ms  cyc/B {cb:6.3f}  "
          f"effcores {ef:5.3f}  GHz {gh:5.3f}  (n={len(walls)})")
    return {"best": b, "med": m, "cycb": cb, "eff": ef, "ghz": gh}


def main():
    out = sys.argv[1]
    outbytes = int(sys.argv[2])
    print(f"=== STEP-3 affinity analysis (outbytes={outbytes}) ===")
    arms = {}
    for arm in ["gzbaseA", "gzbaseB", "gzphysA", "gzphysB", "rgA", "rgB"]:
        arms[arm] = summ(arm, *collect(out, arm, outbytes))

    # merged arms for the headline ratios
    def merge(a, b):
        wa, ca, ea, ga = collect(out, a, outbytes)
        wb, cb_, eb, gb = collect(out, b, outbytes)
        return wa + wb, ca + cb_, ea + eb, ga + gb

    print("--- merged ---")
    gzbase = summ("gzbase", *merge("gzbaseA", "gzbaseB"))
    gzphys = summ("gzphys", *merge("gzphysA", "gzphysB"))
    rg = summ("rg", *merge("rgA", "rgB"))

    print("--- A/A self-tests (license <=1.02) ---")
    for nm, a, b in [("gz_base", "gzbaseA", "gzbaseB"),
                     ("gz_phys", "gzphysA", "gzphysB"),
                     ("rg", "rgA", "rgB")]:
        if arms.get(a) and arms.get(b):
            r = arms[a]["best"] / arms[b]["best"]
            print(f"  {nm} A/A best ratio = {r:.4f} "
                  f"({'PASS' if 0.98 <= r <= 1.02 else 'CHECK'})")

    print("--- HEADLINE RATIOS ---")
    if gzbase and rg:
        print(f"  gzbase/rg  best {gzbase['best']/rg['best']:.4f}  "
              f"med {gzbase['med']/rg['med']:.4f}  cyc/B {gzbase['cycb']/rg['cycb']:.4f}")
    if gzphys and rg:
        print(f"  gzphys/rg  best {gzphys['best']/rg['best']:.4f}  "
              f"med {gzphys['med']/rg['med']:.4f}  cyc/B {gzphys['cycb']/rg['cycb']:.4f}")
    if gzphys and gzbase:
        print(f"  gzphys/gzbase best {gzphys['best']/gzbase['best']:.4f}  "
              f"cyc/B {gzphys['cycb']/gzbase['cycb']:.4f}  "
              f"(<1 = affinity helped)")
    print("DONE_STEP3_ANALYZE")


if __name__ == "__main__":
    main()
