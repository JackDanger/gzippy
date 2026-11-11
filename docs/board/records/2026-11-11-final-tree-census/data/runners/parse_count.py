#!/usr/bin/env python3
# Parse ANATOMY_COUNTERS={json} stderr lines from the instrument builds into
# a tsv: tag, member, level, p, reconcile, and the counters the fp-settle /
# fixedpoint reconciliation needs.
import re
import sys

KEYS = [
    "nearopt_pass_entries",
    "set_from_codes_calls",
    "cost_sheet_equal_flags",
    "nearopt_finite_skip_would_save",
    "cost_sheet_equal_at_last_pass",
    "near_opt_only_literals_blocks",
    "near_opt_flip_safe_settles",
]


def main(argv):
    cells_path, outdir = argv[1], argv[2]
    print("tag\tmember\tlevel\tp\treconcile\tcounters")
    for line in open(cells_path):
        if not line.strip():
            continue
        tag, _binpath, name, level, threads = line.rstrip("\n").split("\t")
        text = open(f"{outdir}/{tag}.stderr", "r", errors="replace").read()
        m = re.search(r"ANATOMY_COUNTERS=(\{.*\})", text)
        if not m:
            print(f"{tag}\t{name}\t{level}\t{threads}\tNOCOUNTERS\t")
            continue
        rec = m.group(1)
        vals = {}
        for k in KEYS:
            km = re.search(r'"%s":(-?\d+)' % re.escape(k), rec)
            vals[k] = km.group(1) if km else "absent"
        ro = re.search(r"ANATOMY_RECONCILE=(\w+)", text)
        recon = ro.group(1) if ro else "?"
        print(f"{tag}\t{name}\t{level}\t{threads}\t{recon}\t" +
              ";".join(f"{k}={vals[k]}" for k in KEYS))


if __name__ == "__main__":
    main(sys.argv)
