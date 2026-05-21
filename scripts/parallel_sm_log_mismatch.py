#!/usr/bin/env python3
"""For each partition, correlate the speculative start (from
speculative_submit) with the authoritative start (from
authoritative_submit) to show the per-partition mismatch distance.

Output: one line per partition with speculative_start, authoritative_start,
offset (auth - spec, in bits and bytes), and which side moved.

Usage:
    python3 scripts/parallel_sm_log_mismatch.py /path/to/log.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()

    by_partition = defaultdict(dict)
    boundaries = {}
    for line in Path(args.log).read_bytes().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ek = ev.get("ev")
        pi = ev.get("partition_idx")
        if pi is None:
            continue
        if ek == "boundary_done":
            boundaries[pi] = ev.get("found_bit")
        elif ek == "speculative_submit":
            by_partition[pi]["spec_start"] = ev["start_bit"]
        elif ek == "authoritative_submit":
            by_partition[pi]["auth_start"] = ev["expected_start"]
        elif ek == "decode_ok":
            by_partition[pi].setdefault("decoded", []).append(
                {
                    "path": ev.get("path"),
                    "start": ev["start_bit"],
                    "end": ev["end_bit"],
                    "decoded": ev["decoded"],
                    "preemptive": ev.get("preemptive"),
                }
            )
        elif ek == "consume_done":
            by_partition[pi]["consumed_end"] = ev["end_bit"]

    print(f"{'idx':>4} {'spec_start':>12} {'auth_start':>12} {'offset_bits':>12} {'offset_kib':>10} {'path':>5}")
    offsets = []
    for pi in sorted(by_partition.keys()):
        row = by_partition[pi]
        spec = row.get("spec_start")
        auth = row.get("auth_start")
        path = "?"
        for d in row.get("decoded", []):
            if d["path"]:
                path = d["path"]
                break
        if spec is None or auth is None:
            print(f"{pi:>4} {spec if spec else 'None':>12} {auth if auth else 'None':>12}  (one missing)")
            continue
        off = auth - spec
        offsets.append(off)
        print(
            f"{pi:>4} {spec:>12} {auth:>12} {off:>12} {off / 8 / 1024:>10.2f} {path:>5}"
        )

    if offsets:
        print()
        print(f"mismatch offset stats (bits):")
        print(f"  count={len(offsets)}")
        print(f"  mean={statistics.mean(offsets):.0f}")
        print(f"  median={statistics.median(offsets):.0f}")
        print(f"  min={min(offsets)}, max={max(offsets)}")
        print(f"  abs_mean={statistics.mean(abs(o) for o in offsets):.0f}")
        print(f"  abs_median={statistics.median(abs(o) for o in offsets):.0f}")
        within_1kib = sum(1 for o in offsets if abs(o) <= 8192)
        within_8kib = sum(1 for o in offsets if abs(o) <= 65536)
        within_64kib = sum(1 for o in offsets if abs(o) <= 524288)
        print(
            f"  |offset| ≤ 1 KiB: {within_1kib}/{len(offsets)} ({100 * within_1kib / len(offsets):.0f}%)"
        )
        print(
            f"  |offset| ≤ 8 KiB: {within_8kib}/{len(offsets)} ({100 * within_8kib / len(offsets):.0f}%)"
        )
        print(
            f"  |offset| ≤ 64 KiB: {within_64kib}/{len(offsets)} ({100 * within_64kib / len(offsets):.0f}%)"
        )


if __name__ == "__main__":
    main()
