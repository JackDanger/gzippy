#!/usr/bin/env python3
"""Build per-corpus fulcrum optgate artifacts from samples.csv.

samples.csv columns: corp,arm,rep,cycles,instructions,minorfaults,elapsed_s,bytes
arms: mono (AFTER, streaming monolith) / thin (BASE, thin-T1) / igzip (rg comparator).

Emits plans/t1-monolith-finish-data/optgate-<corp>.json (OptGateInput shape).
"""
import csv, json, sys, os

SRC = os.path.join(os.path.dirname(__file__), "samples.csv")
# 16-char sha prefixes (mono==thin==igzip==zcat verified byte-exact on the box).
SHA = {
    "silesia": "028bd002c89c9a90",
    "nasa": "96551161b5bdcaac",
    "monorepo": "0dd50d07b0147211",
    "squishy": "4ee30c1571688ca4",
}
PROCS_RUNNING = 2.0  # observed nr_running on the box (cpu4-pinned cell); k=1, slack=1 -> ceiling 2

rows = {}
with open(SRC) as f:
    for r in csv.reader(f):
        if len(r) < 8:
            continue
        corp, arm, rep, cyc, ins, flt, el, by = r[:8]
        rows.setdefault(corp, {}).setdefault(arm, []).append(
            {"cycles": float(cyc), "instructions": float(ins),
             "bytes": float(by), "procs_running": PROCS_RUNNING}
        )

def arm(label, samples, sha=None):
    a = {"label": label, "samples": samples}
    if sha is not None:
        a["sha"] = sha
    return a

made = []
for corp, arms in rows.items():
    if not all(k in arms for k in ("mono", "thin", "igzip")):
        print(f"skip {corp}: missing arm(s) {set(arms)}", file=sys.stderr)
        continue
    sha = SHA.get(corp, "unknown")
    art = {
        "base": arm("thin-T1", arms["thin"], sha),
        "after": arm("streaming-monolith", arms["mono"], sha),
        "rg": arm("igzip", arms["igzip"]),
        "reference_sha": sha,
        "clean_base": arm("thin-T1", arms["thin"], sha),
        "clean_after": arm("streaming-monolith", arms["mono"], sha),
        "k": 1.0,
        "clean_k": 1.0,
        "arch": "intel-i7-13700T",
        "cross_arch_replicated": False,
        "base_commit": "thin-T1(GZIPPY_NO_MONOLITH=1)@43d2ae21",
        "after_commit": "streaming-monolith@43d2ae21",
    }
    out = os.path.join(os.path.dirname(__file__), f"optgate-{corp}.json")
    with open(out, "w") as f:
        json.dump(art, f, indent=2)
    n = {k: len(v) for k, v in arms.items()}
    made.append((corp, out, n))
    print(f"wrote {out}  N={n}")

if not made:
    sys.exit("no artifacts built")
