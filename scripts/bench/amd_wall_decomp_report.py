#!/usr/bin/env python3
"""AMD WALL-DECOMP report: per-cell median region partition for gz & rg, the
per-region {gz,rg,delta} table, and the THREE conservation residuals.

Currency: perf `cycles` total per arm (conserved total); region cyc = thread-summed
exclusive TSC (frozen box => TSC ~= core cycles). R_OTHER = perf_total - sum(regions).

resid_i   = |R_OTHER_gz| / gz_perf
resid_ii  = |R_OTHER_rg| / rg_perf
resid_iii = |gap - sum(gz_r - rg_r)| / |gap|   (== |R_OTHER_gz - R_OTHER_rg| / |gap|)
"""
import sys, csv
from statistics import median

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/amd-wall-decomp/raw.csv"
rows = list(csv.DictReader(open(path)))

def med(rs, k):
    vs = [float(r[k]) for r in rs if r[k] not in ("", None)]
    return median(vs) if vs else 0.0

def spread(rs, k):
    vs = [float(r[k]) for r in rs if r[k] not in ("", None)]
    if len(vs) < 2: return 0.0
    return (max(vs) - min(vs)) / median(vs) * 100.0

cells = {}
for r in rows:
    cells.setdefault((r["corpus"], r["th"]), []).append(r)

print(f"# AMD WALL-DECOMP — gz-vs-rg 3-region partition + 3 conservation residuals")
print(f"# source: {path}  (perf cycles total; region cyc = thread-summed exclusive TSC, frozen box)\n")

for (corpus, th), rs in sorted(cells.items()):
    n = len(rs)
    gp = med(rs, "gz_perf"); rp = med(rs, "rg_perf")
    gw = med(rs, "gz_w"); gm = med(rs, "gz_m"); go = med(rs, "gz_o")
    rw = med(rs, "rg_w"); rm = med(rs, "rg_m"); ro = med(rs, "rg_o")
    ap = med(rs, "gzaa_perf"); aw = med(rs, "gzaa_w")
    gov = max(float(r["gz_ov"] or 0) for r in rs)
    rov = max(float(r["rg_ov"] or 0) for r in rs)
    gcs = med(rs, "gz_cs"); rcs = med(rs, "rg_cs"); gmig = med(rs, "gz_mig"); rmig = med(rs, "rg_mig")

    gz_other = gp - (gw + gm + go)
    rg_other = rp - (rw + rm + ro)
    gap = gp - rp
    sum_dr = (gw - rw) + (gm - rm) + (go - ro)

    resid_i = abs(gz_other) / gp * 100 if gp else 0
    resid_ii = abs(rg_other) / rp * 100 if rp else 0
    resid_iii = abs(gap - sum_dr) / abs(gap) * 100 if gap else 0
    resid_iii_tot = abs(gap - sum_dr) / gp * 100 if gp else 0  # vs total (floor-robust)
    gap_pct = abs(gap) / rp * 100 if rp else 0
    ATTRIB_FLOOR = 3.0  # gaps below this % of total are near-ties; (iii)/gap is uninformative

    # A/A: gzAA worker vs gz worker, and gzAA perf vs gz perf
    aa_perf = abs(ap - gp) / gp * 100 if gp else 0
    aa_w = abs(aw - gw) / gw * 100 if gw else 0

    print(f"## {corpus} T{th}  (N={n})")
    print(f"   gz_perf={gp:,.0f}  rg_perf={rp:,.0f}  gap=gz-rg={gap:,.0f} ({gap/rp*100:+.1f}%)")
    print(f"   spread: gz_perf={spread(rs,'gz_perf'):.1f}% rg_perf={spread(rs,'rg_perf'):.1f}%  A/A: perf={aa_perf:.2f}% worker={aa_w:.2f}%")
    print(f"   sched: ctx-sw gz={gcs:.0f} rg={rcs:.0f}  migrations gz={gmig:.0f} rg={rmig:.0f}")
    print(f"   {'region':<10}{'gz_cyc':>16}{'rg_cyc':>16}{'delta(gz-rg)':>16}{'%ofgap':>9}")
    for nm, g, rr in [("WORKER", gw, rw), ("MARKERPP", gm, rm), ("OUTPUT", go, ro), ("OTHER", gz_other, rg_other)]:
        d = g - rr
        pct = d / gap * 100 if gap else 0
        print(f"   {nm:<10}{g:>16,.0f}{rr:>16,.0f}{d:>16,.0f}{pct:>8.1f}%")
    print(f"   -- CONSERVATION --")
    print(f"   (i)   R_OTHER_gz = {gz_other:,.0f}  resid_i  = {resid_i:.2f}%  {'PASS' if resid_i<5 else 'FAIL'}")
    print(f"   (ii)  R_OTHER_rg = {rg_other:,.0f}  resid_ii = {resid_ii:.2f}%  {'PASS' if resid_ii<5 else 'FAIL'}")
    if gap_pct < ATTRIB_FLOOR:
        iii_verdict = f"N/A (gap {gap_pct:.1f}% < {ATTRIB_FLOOR}% attribution floor: near-TIE, not a loss cell)"
    else:
        iii_verdict = "PASS" if resid_iii < 5 else f"resid={resid_iii:.1f}%/gap ({resid_iii_tot:.2f}% of total)"
    print(f"   (iii) gap - sum(D_r) = {gap-sum_dr:,.0f}  = {resid_iii_tot:.2f}% of total | {resid_iii:.1f}% of gap -> {iii_verdict}")
    print(f"   OVERLAP: gz={gov:.0f} rg={rov:.0f}  {'PASS' if gov==0 and rov==0 else 'FAIL (regions not exclusive!)'}")
    print()
