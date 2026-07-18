#!/usr/bin/env python3
"""build_unified_matrix.py — fold the Intel standing REPORT.txt + meta.txt into a
clean markdown matrix (adds the MB/s column the deliverable wants). Pure parse +
arithmetic over already-gated numbers; no re-measurement. MB/s = decomp_bytes /
(gz_wall_s) / 1e6.
"""
import sys, re, os

art = sys.argv[1]
meta_bytes = {}
with open(os.path.join(art, "meta.txt")) as f:
    for line in f:
        t = line.split()
        if len(t) == 3 and t[0] == "bytes":
            meta_bytes[t[1]] = int(t[2])

paths = {}
pf = os.path.join(art, "paths.txt")
if os.path.exists(pf):
    for line in open(pf):
        t = line.split()
        if len(t) == 2:
            paths[t[0]] = t[1]

# parse REPORT.txt: blocks of cell lines
rep = open(os.path.join(art, "REPORT.txt")).read()
cells = {}  # (corp,T) -> dict
cur = None
for line in rep.splitlines():
    m = re.match(r"^(\S+(?:\s\S+)?)\s+T(\d+)\s+(gz|rg|ig)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*(.*)$", line)
    # the corpus token can contain a trailing 'T<n>' glued (pure_stored_100mb T1gz) — handle separately
    if not m:
        m2 = re.match(r"^(\S+) T(\d+)(gz|rg|ig)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*(.*)$", line)
        if m2:
            corp, T, tool, cyc, wall, spr, rrg, rig, ghz, ghzs, llc, verd = m2.groups()
        else:
            mst = re.search(r"self-test A/A: rg\|gz = ([\d.]+)% / ([\d.]+)%.*=> (OK|UNTRUSTED)", line)
            if mst and cur:
                cells[cur]["aa_rg"], cells[cur]["aa_gz"], cells[cur]["aa"] = mst.group(1), mst.group(2), mst.group(3)
            continue
    else:
        corp, T, tool, cyc, wall, spr, rrg, rig, ghz, ghzs, llc, verd = m.groups()
    corp = corp.strip()
    key = (corp, int(T))
    cells.setdefault(key, {})
    cells[key][tool] = dict(cyc=float(cyc), wall=float(wall), spr=float(spr),
                            rrg=float(rrg), rig=float(rig), verd=verd.strip())
    if tool == "gz":
        cur = key

order = ["silesia", "monorepo", "nasa", "model", "squishy", "bignasa", "weights",
         "storedmix", "storedheavy", "pure_stored_100mb"]
def corpkey(k):
    c = k[0]
    return (order.index(c) if c in order else 99, k[1])

print("| corpus | path | T | gz MB/s | gz wall ms | spr% | cyc/B | gz/rg | gz/ig | A/A gz | GATE / verdict |")
print("|--------|------|---|---------|-----------|------|-------|-------|-------|--------|----------------|")
for key in sorted(cells, key=corpkey):
    c = cells[key]
    if "gz" not in c:
        continue
    corp, T = key
    g = c["gz"]
    nb = meta_bytes.get(corp.replace(" ", ""), meta_bytes.get(corp, 0))
    mbps = nb / (g["wall"] / 1000.0) / 1e6 if g["wall"] else 0
    aa = c.get("aa_gz", "?")
    gate = c.get("aa", "OK")
    verd = g["verd"]
    p = paths.get(corp, "?")
    disp = "pure_stored" if corp.startswith("pure_stored") else corp
    print(f"| {disp} | {p} | {T} | {mbps:.1f} | {g['wall']:.1f} | {g['spr']:.1f} | "
          f"{g['cyc']:.2f} | {g['rrg']:.3f} | {g['rig']:.3f} | {aa}% | {verd} |")
