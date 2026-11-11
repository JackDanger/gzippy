#!/usr/bin/env python3
"""Generate the README census table + verdict stanzas from census.tsv."""
import sys
from collections import defaultdict

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "probe/census/census.tsv"
    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        tag, lev, t, so, sho, ss, shs, v, settles, _yes, flips = f[:11]
        rows.append((tag, lev, t, so, sho, ss, shs, v, settles, flips))
    n = len(rows)
    ok = sum(1 for r in rows if r[7] == "OK")
    diffs = [(r[0], r[4][:12], r[6][:12]) for r in rows if r[7] != "OK"]
    total_settles = sum(int(r[8]) for r in rows if r[8])
    total_flips = sum(int(r[9]) for r in rows if r[9])
    per_member = defaultdict(lambda: [0, 0])  # cells, OK
    for r in rows:
        m = r[0].split("_L")[0]
        per_member[m][0] += 1
        if r[7] == "OK":
            per_member[m][1] += 1
    print(f"**cells={n} (one sha row per member×level×-p, OFF-vs-SAFE compared) "
          f"OK={ok} DIFF={n-ok}**\n")
    print("Waits: the SAFE arm's settle counter fired across the census on "
          f"{total_settles} block boundaries; the serial-decision space "
          f"carries {total_flips} only-literals blocks (identical on both "
          "arms — the design's exactness corroboration).\n")
    if diffs:
        print("DIFF cells (the run's falsifier rows):")
        for d in diffs[:20]:
            print(f"  {d[0]} off={d[1]} safe={d[2]}")
    print()
    print("| member | cells | OK |")
    print("|---|---:|---:|")
    for m in sorted(per_member):
        c, o = per_member[m]
        print(f"| {m} | {c} | {o} |")

if __name__ == "__main__":
    main()
