#!/usr/bin/env python3
"""localize_mac_report.py — aarch64 T1 gap decomposition (instr + cyc).

Reads the CSV produced by localize_mac.sh and splits gzippy-native's -p1
instr/B (and cyc/B) into three regions, vs libdeflate:

  region                what it is                                     how measured
  --------------------- ---------------------------------------------- -----------------------
  pipeline scaffold     parallel chunk lifecycle: speculation, marker  normal - thin
                        machinery, apply_window, replace_markers       (both byte-exact)
  table-build (litlen)  per-block litlen LUT build                     GZIPPY_TBUILD_MULT slope
  clean decode core     Huffman kernel (run_contig_ref_biased) +       thin - table-build
                        backref copy + bit reader + CRC + I/O

libdeflate is itself a clean decode core (no scaffold, inline table build), so
(clean-core - libdeflate) is the kernel/copy/bitread CODEGEN gap = the NEON
candidate. Estimator = best-of-N (min). instr is deterministic (STRONG
attribution); cyc carries a trust label.

ATTRIBUTION = HYPOTHESIS-tier, but with deterministic instr counts it is a
STRONG attribution. macOS-aarch64, NOT-YET-LAW cross-arch.

CSV columns: corpus,bytes,arm,mult,rep,instr,cyc
  arm in {libdeflate, normal, thin, tbuild}; mult>=1 only for tbuild else 0.

Usage:  localize_mac_report.py <samples.csv>
"""
import csv
import sys
from collections import defaultdict


def best(vals):
    return min(vals) if vals else float("nan")


def lsq_slope(xs, ys):
    """Least-squares slope of y vs x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def main():
    if len(sys.argv) != 2:
        print("usage: localize_mac_report.py <samples.csv>", file=sys.stderr)
        sys.exit(2)

    rows = []
    with open(sys.argv[1], newline="") as fh:
        for r in csv.DictReader(fh):
            r["bytes"] = int(r["bytes"]); r["mult"] = int(r["mult"])
            r["rep"] = int(r["rep"]); r["instr"] = int(r["instr"]); r["cyc"] = int(r["cyc"])
            rows.append(r)
    if not rows:
        print("localize_mac_report: NO SAMPLES — FAIL", file=sys.stderr); sys.exit(1)

    corpora = sorted({r["corpus"] for r in rows})
    suspect = False

    print()
    print("=" * 96)
    print("  AARCH64 T1 GAP LOCALIZATION (Apple M1 Pro) — gzippy-native -p1 decomposition vs libdeflate")
    print("  STRONG-attribution-HYPOTHESIS (deterministic instr counts) | macOS-aarch64 NOT-YET-LAW")
    print("=" * 96)

    for corpus in corpora:
        cr = [r for r in rows if r["corpus"] == corpus]
        bytes_ = cr[0]["bytes"]

        def pb(arm, metric, mult=0):
            vals = [r[metric] for r in cr if r["arm"] == arm and (mult == 0 or r["mult"] == mult)]
            return best(vals) / bytes_ if vals else float("nan")

        L_i = pb("libdeflate", "instr"); L_c = pb("libdeflate", "cyc")
        G_i = pb("normal", "instr");     G_c = pb("normal", "cyc")
        T_i = pb("thin", "instr");       T_c = pb("thin", "cyc")

        # table-build (litlen) slope: instr/B (and cyc/B) per extra build, from tbuild mult sweep.
        mults = sorted({r["mult"] for r in cr if r["arm"] == "tbuild"})
        tb_i = [pb("tbuild", "instr", m) for m in mults]
        tb_c = [pb("tbuild", "cyc", m) for m in mults]
        TB_i = lsq_slope([float(m) for m in mults], tb_i) if len(mults) >= 2 else float("nan")
        TB_c = lsq_slope([float(m) for m in mults], tb_c) if len(mults) >= 2 else float("nan")

        # regions
        scaff_i = G_i - T_i; scaff_c = G_c - T_c
        core_i = T_i - TB_i; core_c = T_c - TB_c
        core_excess_i = core_i - L_i; core_excess_c = core_c - L_c
        gap_i = G_i - L_i; gap_c = G_c - L_c

        # tbuild non-inert: must increase with mult
        tb_fires = len(tb_i) >= 2 and tb_i[-1] > tb_i[0]
        if not tb_fires:
            suspect = True

        print()
        print(f"  ── {corpus}  ({bytes_:,} B decompressed) ──")
        print(f"     whole-program  gzippy normal -p1 : {G_i:8.3f} instr/B   {G_c:7.3f} cyc/B")
        print(f"     reference      libdeflate        : {L_i:8.3f} instr/B   {L_c:7.3f} cyc/B")
        print(f"     ratio gz/libd                    : {G_i/L_i:8.3f}x        {G_c/L_c:7.3f}x")
        print(f"     ── region decomposition of gzippy's {G_i:.3f} instr/B ──")
        print(f"       pipeline scaffold (normal-thin) : {scaff_i:8.3f} instr/B  ({pct(scaff_i,G_i)})   {scaff_c:7.3f} cyc/B")
        print(f"       table-build litlen (tbuild slope): {TB_i:8.3f} instr/B  ({pct(TB_i,G_i)})   {TB_c:7.3f} cyc/B  [lower bound: dist-build excluded]")
        print(f"       clean decode core (thin-tbuild) : {core_i:8.3f} instr/B  ({pct(core_i,G_i)})   {core_c:7.3f} cyc/B")
        print(f"     ── decomposition of the GAP over libdeflate ({gap_i:.3f} instr/B = {G_i/L_i:.2f}x) ──")
        print(f"       scaffold                        : {scaff_i:8.3f} instr/B  ({pct(scaff_i,gap_i)} of gap)")
        print(f"       table-build (litlen, lower bnd) : {TB_i:8.3f} instr/B  ({pct(TB_i,gap_i)} of gap)")
        print(f"       clean-core excess vs libdeflate : {core_excess_i:8.3f} instr/B  ({pct(core_excess_i,gap_i)} of gap)   {core_excess_c:7.3f} cyc/B")
        print(f"     tbuild non-inert (instr rises w/ mult): {'PASS' if tb_fires else 'FAIL'}   "
              f"(mult{mults[0]}={tb_i[0]:.3f} .. mult{mults[-1]}={tb_i[-1]:.3f} instr/B)")

    print()
    print("=" * 96)
    print("  VERDICT: see clean-core-excess vs scaffold share above.")
    print("    clean-core-excess dominant  -> KERNEL-bound (aarch64): NEON copy/Huffman kernel + table-build/bitread convergence (also helps Intel pure-Rust).")
    print("    scaffold dominant           -> STRUCTURAL: parallel pipeline convergence (also helps Intel).")
    print("=" * 96)
    sys.exit(1 if suspect else 0)


def pct(part, whole):
    if whole == 0 or whole != whole:
        return "  n/a"
    return f"{part/whole*100:5.1f}%"


if __name__ == "__main__":
    main()
