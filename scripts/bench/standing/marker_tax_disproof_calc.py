#!/usr/bin/env python3
# Assemble the disproof comparison from the two gated runs.
import statistics

BYTES = {"silesia":211968000,"monorepo":50915328,"nasa":205242368,"squishy":400391411}
# gz from mac_pipeline_components.sh N=7 (this session), instructions in millions
gz = {  # base1, base4, noprefetch4  (M instr)
 "silesia": (2099.4,4012.9,2192.2),
 "monorepo":(348.5,1098.6,373.0),
 "nasa":    (701.6,2761.3,802.6),
 "squishy": (3816.8,6865.8,4076.7),
}
# rg from rg_marker_added.py N=7 (this session), instr/B at P1,P4
rg = {
 "silesia": (26.939,31.977),
 "monorepo":(22.793,34.703),
 "nasa":    (13.934,24.087),
 "squishy": (26.637,31.240),
}
# rg --verbose replaced-marker fraction (from PIPELINE-TAX-LOCALIZE D1.3/D1.4)
markerfrac = {"silesia":34.5,"monorepo":80.9,"nasa":89.8,"squishy":31.6}
# Intel gz/rg wall ratio @T4 (STANDING-MATRIX-2026-06-20); squishy not on Intel
intel_t4 = {"silesia":1.161,"monorepo":1.002,"nasa":0.960}

print(f"{'corpus':9} {'mkr%':>5} | {'gzT1/B':>7} {'gzT4/B':>7} {'rgT1/B':>7} {'rgT4/B':>7} | {'gz_infl':>7} {'rg_infl':>7} {'PIPEtax':>7} | {'gz_tax/B':>8} {'gz_mkrMach/B':>12} {'rg_add/B':>8} | {'gz-rg gap/B':>11}")
rows=[]
for c in ["nasa","monorepo","silesia","squishy"]:  # by DESCENDING marker fraction
    b1,b4,npf = (x*1e6 for x in gz[c])
    nb = BYTES[c]
    gzT1 = b1/nb; gzT4 = b4/nb
    gz_tax = (b4-b1)/nb            # base4-base1 incl coordination (apples to rg T4-T1)
    gz_mkr = (b4-npf)/nb           # causal marker-only (coordination removed)
    rgT1,rgT4 = rg[c]
    rg_add = rgT4-rgT1
    gz_infl = b4/b1; rg_infl = rgT4/rgT1
    pipetax = gz_infl/rg_infl
    gap = gz_tax - rg_add
    rows.append((markerfrac[c], gap, intel_t4.get(c)))
    print(f"{c:9} {markerfrac[c]:5.1f} | {gzT1:7.2f} {gzT4:7.2f} {rgT1:7.2f} {rgT4:7.2f} | {gz_infl:7.2f} {rg_infl:7.2f} {pipetax:7.2f} | {gz_tax:8.2f} {gz_mkr:12.2f} {rg_add:8.2f} | {gap:+11.2f}")

print("\n--- CORRELATION: gz-vs-rg gap vs marker fraction (claim => POSITIVE) ---")
mf=[r[0] for r in rows]; gap=[r[1] for r in rows]
def pearson(x,y):
    mx=statistics.mean(x);my=statistics.mean(y)
    num=sum((a-mx)*(b-my) for a,b in zip(x,y))
    den=(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))**0.5
    return num/den
print(f"mac absolute (gz_tax-rg_add) instr/B vs marker%: Pearson r = {pearson(mf,gap):+.3f}  (N=4)")
intel=[(r[0],r[2]) for r in rows if r[2] is not None]
print(f"Intel gz/rg wall@T4 vs marker%: ", end="")
ix=[a for a,_ in intel]; iy=[b for _,b in intel]
print(f"Pearson r = {pearson(ix,iy):+.3f}  (N=3)  points={intel}")
print("\nclaim predicts r>0 (loss grows with marker fraction); measured r<0 on BOTH => ANTI-correlation")
