#!/usr/bin/env python3
"""Assemble the fulcrum excess artifact for the R_WORKER SUB-decomposition.

MATCHED regions (the only ones honest gz-vs-rg given rg-WITH_ISAL fuses Huffman
table-build INTO its ISA-L readStream decode, while gz separates read_header):
  DECODE_TOTAL = R_TABLE + R_DECODE  (ALL decode incl table build + clean-CRC, both
                 tools). gz = pure-Rust read_header + Block::read/contig; rg = deflate
                 bootstrap + ISA-L readStream (tables fused) + append.
  RING_OTHER   = R_WORKER - (R_TABLE + R_DECODE)  (chunk setup / fold-drain / finalize
                 vs rg collect).
Denominator = total decoded output (identical gz/rg per corpus → ratio is the honest
absolute-cycle ratio). loss=silesia, control=nasa. metric=cycle.

The gz-internal R_TABLE-vs-R_DECODE split is reported separately (diagnostic) — it is
NOT a matched fulcrum region because rg fuses tables into ISA-L decode.
"""
import csv, json, sys, statistics as st

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/amd_subdecomp_raw.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/amd_subdecomp_artifact.json"
rows = list(csv.DictReader(open(CSV)))

def f(r, k): return float(r[k])

def samples(rows_for_corpus, arm, region):
    out = []
    for r in rows_for_corpus:
        tot = f(r, "total_out")
        if region == "DECODE_TOTAL":
            cyc = (f(r, "gz_t")+f(r, "gz_d")) if arm == "gz" else (f(r, "rg_t")+f(r, "rg_d"))
        elif region == "RING_OTHER":
            cyc = (f(r, "gz_w")-f(r, "gz_t")-f(r, "gz_d")) if arm == "gz" \
                  else (f(r, "rg_w")-f(r, "rg_t")-f(r, "rg_d"))
        out.append({"cycles": cyc, "instructions": 0.0, "bytes": tot})
    return out

loss = [r for r in rows if r["corpus"] == "silesia"]
ctrl = [r for r in rows if r["corpus"] == "nasa"]

regions = []
for rg in ("DECODE_TOTAL", "RING_OTHER"):
    regions.append({
        "label": rg,
        "loss":    {"gz": samples(loss, "gz", rg), "rg": samples(loss, "rg", rg)},
        "control": {"gz": samples(ctrl, "gz", rg), "rg": samples(ctrl, "rg", rg)},
    })

art = {
    "metric": "cycle", "epsilon": 0.05,
    "loss_corpus": "silesia", "control_corpus": "nasa",
    "arch": "amd-zen2", "cross_arch_replicated": False,
    "gz_sha": "39acc213+subdecomp", "rg_sha": "rg-subregion-patched",
    "regions": regions,
}
json.dump(art, open(OUT, "w"), indent=1)
print("wrote", OUT, "regions:", [r["label"] for r in regions], "N(loss)=", len(loss), "N(ctrl)=", len(ctrl))

# ── diagnostic medians ───────────────────────────────────────────────────────
def med(rows_, expr): return st.median([expr(r) for r in rows_])
print("\n── median cyc/byte (loss=silesia, control=nasa) ──")
for rg in regions:
    def m(arm, corp):
        s = rg[corp][arm]; return st.median([x["cycles"]/x["bytes"] for x in s])
    lg, lr = m("gz","loss"), m("rg","loss"); cg, cr = m("gz","control"), m("rg","control")
    print(f"{rg['label']:<13} loss gz/rg={lg:.4f}/{lr:.4f}={lg/lr:.3f}  ctrl gz/rg={cg:.4f}/{cr:.4f}={cg/cr:.3f}  recov(loss gz-rg)={lg-lr:+.4f}")

print("\n── gz-internal sub-split (DIAGNOSTIC; rg fuses tables so not matched) ──")
for corp, rs in (("silesia(loss)", loss), ("nasa(ctrl)", ctrl)):
    tot = med(rs, lambda r: f(r,"total_out"))
    gt = med(rs, lambda r: f(r,"gz_t")/f(r,"total_out"))
    gd = med(rs, lambda r: f(r,"gz_d")/f(r,"total_out"))
    gr = med(rs, lambda r: (f(r,"gz_w")-f(r,"gz_t")-f(r,"gz_d"))/f(r,"total_out"))
    gw = med(rs, lambda r: f(r,"gz_w")/f(r,"total_out"))
    hdr = med(rs, lambda r: f(r,"gz_tn"))
    print(f"  gz {corp:<14} R_TABLE={gt:.4f}  R_DECODE_body={gd:.4f}  ring={gr:.4f}  R_WORKER={gw:.4f} cyc/B  (hdr_calls={hdr:.0f}; table={100*gt/gw:.1f}% of worker)")

# ── A/A sink-law check ───────────────────────────────────────────────────────
aa = med(loss, lambda r: abs(f(r,"gzaa_w")-f(r,"gz_w"))/f(r,"gz_w"))
print(f"\nA/A (gz vs gzAA R_WORKER, silesia) median rel-diff = {100*aa:.2f}%  (SINK-LAW; want <few%)")
