#!/usr/bin/env python3
"""Assemble the fulcrum excess artifact from amd_excess_raw.csv.
Matched per-region denominators (cursor-agent reviewed):
  R_WORKER  : bytes = total decompressed output (both arms; region decodes whole output)
  R_MARKERPP: bytes = data-with-markers u16 element count (gz_m_b == rg_m_b, byte-identical)
  R_OUTPUT  : bytes = total decompressed output (both arms write whole output to /dev/null)
metric=cycle; per-region instructions not capturable via rdtsc spans -> 0 (cycle verdict ignores).
"""
import csv, json, sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/amd_excess_raw.csv"
LOSS = "silesia"; CONTROL = "nasa"

rows = list(csv.DictReader(open(CSV)))

def f(r, k): return float(r[k])

def samples(rows_for_corpus, arm, region):
    out = []
    for r in rows_for_corpus:
        tot = f(r, "total_out")
        if region == "R_WORKER":
            cyc = f(r, "gz_w_cyc" if arm == "gz" else "rg_w_cyc"); b = tot
        elif region == "R_MARKERPP":
            cyc = f(r, "gz_m_cyc" if arm == "gz" else "rg_m_cyc")
            b = f(r, "gz_m_b" if arm == "gz" else "rg_m_b")
        elif region == "R_OUTPUT":
            cyc = f(r, "gz_o_cyc" if arm == "gz" else "rg_o_cyc"); b = tot
        out.append({"cycles": cyc, "instructions": 0.0, "bytes": b})
    return out

loss_rows = [r for r in rows if r["corpus"] == LOSS]
ctrl_rows = [r for r in rows if r["corpus"] == CONTROL]

regions = []
for rg in ("R_WORKER", "R_MARKERPP", "R_OUTPUT"):
    regions.append({
        "label": rg,
        "loss": {"gz": samples(loss_rows, "gz", rg), "rg": samples(loss_rows, "rg", rg)},
        "control": {"gz": samples(ctrl_rows, "gz", rg), "rg": samples(ctrl_rows, "rg", rg)},
    })

art = {
    "metric": "cycle", "epsilon": 0.05,
    "loss_corpus": "silesia", "control_corpus": "nasa",
    "arch": "amd-zen2", "cross_arch_replicated": False,
    "gz_sha": "1b7473f8ff8d", "rg_sha": "b407b3d56849",
    "regions": regions,
}
json.dump(art, open("/tmp/amd_excess_artifact.json", "w"), indent=1)
print("wrote /tmp/amd_excess_artifact.json with", len(regions), "regions, N(loss)=",
      len(loss_rows), "N(ctrl)=", len(ctrl_rows))

# quick median cyc/byte sanity print
import statistics as st
for rg in regions:
    def med(arm, corp):
        s = rg[corp][arm]; return st.median([x["cycles"]/x["bytes"] for x in s])
    lg, lr = med("gz","loss"), med("rg","loss")
    cg, cr = med("gz","control"), med("rg","control")
    print(f"{rg['label']:<12} loss gz/rg={lg:.4f}/{lr:.4f}={lg/lr:.3f}  ctrl gz/rg={cg:.4f}/{cr:.4f}={cg/cr:.3f}  recov={lg-lr:+.4f}")
