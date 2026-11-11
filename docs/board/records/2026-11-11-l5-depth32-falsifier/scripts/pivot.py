#!/usr/bin/env python3
"""Pivot the falsifier sweep TSV into the receipt table (sizes, deltas vs ship)."""
rows, members = {}, []
for line in open("/Users/jackdanger/www/gzippy/.review-tmp/wA-l5/scratch/sweep-table.tsv"):
    p = line.rstrip("\n").split("\t")
    if p[0] == "member":
        continue
    m, arm, sz, sha, rt = p[0], p[1], int(p[2]), p[3], p[4]
    rows.update({(m, arm): (sz, sha)})
    if m not in members:
        members.append(m)
order = ["ship", "lad16", "lad20", "lad24", "lad32", "lad32g8", "ladG32", "gzip_L5", "pigz_L5", "libdeflate_L5"]

print("RAW SIZES (B):")
print(f"{'member':18s}" + "".join(f"{a:>12s}" for a in order))
for m in sorted(members):
    print(f"{m:18s}" + "".join(f"{rows[(m, a)][0]:>12,d}" for a in order))

print("\nDELTAS vs ship (B and % of ship):  ship = shipped L5-T1 Lazy(16,30)")
print(f"{'member':18s}" + "".join(f"{a:>18s}" for a in ["lad20", "lad24", "lad32", "lad32g8", "gzip_L5", "pigz_L5", "libdeflate_L5"]))
agg = {a: 0 for a in ["lad20", "lad24", "lad32", "lad32g8"]}
worse = {a: 0 for a in ["lad32", "lad32g8", "lad24", "lad20"]}
for m in sorted(members):
    ship = rows[(m, "ship")][0]
    cells = []
    for a in ["lad20", "lad24", "lad32", "lad32g8", "gzip_L5", "pigz_L5", "libdeflate_L5"]:
        d = rows[(m, a)][0] - ship
        cells.append(f"{d:+,d} ({d/ship*100:+.4f}%)")
        if a in agg:
            agg[a] += d
        if a in worse and d > 0:
            worse[a] += 1
    print(f"{m:18s}" + "".join(f"{c:>18s}" for c in cells))
print("\nAGGREGATE deltas vs ship:", {k: f"{v:+,d} B" for k, v in agg.items()})
print("members made BIGGER by arm:", {k: f"{v}/15" for k, v in worse.items()})

sha16 = [(m, rows[(m, "lad16")][1] == rows[(m, "ship")][1]) for m in members]
print("\nA/A control (lad16 vs ship) sha byte-identity:", f"{sum(1 for _, ok in sha16 if ok)}/15")
flips32g8 = [(m, rows[(m, "lad32g8")][0] - rows[(m, "ship")][0]) for m in members]
print("\nbest-of-depth-arms per member (min over lad20/lad24/lad32/lad32g8):")
for m in sorted(members):
    best = min(rows[(m, a)][0] for a in ["lad20", "lad24", "lad32", "lad32g8"])
    print(f"  {m:18s} ship={rows[(m,'ship')][0]:>10,d}  best-arm={best:>10,d}  delta={best-rows[(m,'ship')][0]:+d} ({(best-rows[(m,'ship')][0])/rows[(m,'ship')][0]*100:+.4f}%)")
