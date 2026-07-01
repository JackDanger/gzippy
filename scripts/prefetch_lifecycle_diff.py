#!/usr/bin/env python3
"""Diff gzippy vs rapidgzip prefetch-lifecycle traces.

Reads the Chrome-trace JSON emitted with the instrumentation in
block_fetcher.rs::prefetch_new_blocks + the matching vendor-side patches
in scripts/rapidgzip_trace_patch/patch_vendor.sh, then reports:

  * Total prefetch_new_blocks invocations (and per-call submitted distribution)
  * Total coord.prefetch_emit count per side
  * Partition-aligned vs sub-partition emit split (only gzippy has args)
  * Skip-reason counts (gzippy side only — vendor patch doesn't tag skips)
  * Prefetch USEFUL rate: hits at cache.get_outcome / total emits
  * Prefetch WASTE: discard_unused.summary.unused_count

Usage:
  python3 scripts/prefetch_lifecycle_diff.py /tmp/gzippy.json /tmp/rapidgzip.json
"""
import json
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path) as f:
        s = f.read().strip()
    if s.startswith("[") and not s.endswith("]"):
        s = s.rstrip(",\n") + "\n]"
    elif s.endswith(","):
        s = s.rstrip(",\n") + "\n]"
    if not s.startswith("["):
        s = "[" + s
    return json.loads(s)


def summarize(path, label):
    events = load(path)
    pf_calls = 0
    pf_emits = 0
    skip_reasons = Counter()
    submitted_dist = Counter()
    candidates_dist = Counter()
    emit_offsets = []
    offset_eq_partition_true = 0
    offset_eq_partition_false = 0
    cache_outcomes = Counter()
    discard_unused = 0
    outcomes_with_submitted = []

    for e in events:
        name = e.get("name", "")
        args = e.get("args") or {}
        ph = e.get("ph", "")
        if name == "coord.prefetch_call" and ph == "B":
            pf_calls += 1
        elif name == "coord.prefetch_emit" and ph == "B":
            pf_emits += 1
            off = args.get("offset")
            if off is not None:
                emit_offsets.append(int(off) if not isinstance(off, str) else int(off, 0))
            eq = args.get("offset_eq_partition")
            if eq is True or eq == "true":
                offset_eq_partition_true += 1
            elif eq is False or eq == "false":
                offset_eq_partition_false += 1
        elif name == "coord.prefetch_skip":
            skip_reasons[args.get("reason", "?")] += 1
        elif name == "coord.prefetch_strategy":
            cand = args.get("candidates")
            if cand is not None:
                candidates_dist[int(cand)] += 1
        elif name == "coord.prefetch_call.outcome":
            sub = args.get("submitted")
            if sub is not None:
                try:
                    sub_i = int(sub)
                    submitted_dist[sub_i] += 1
                    outcomes_with_submitted.append(sub_i)
                except (ValueError, TypeError):
                    pass
        elif name == "cache.get_outcome":
            cache_outcomes[args.get("source", "?")] += 1
        elif name == "cache.discard_unused.summary":
            try:
                discard_unused += int(args.get("unused_count", 0))
            except (ValueError, TypeError):
                pass

    return {
        "label": label,
        "pf_calls": pf_calls,
        "pf_emits": pf_emits,
        "skip_reasons": skip_reasons,
        "submitted_dist": submitted_dist,
        "candidates_dist": candidates_dist,
        "emit_offsets": emit_offsets,
        "eq_part_true": offset_eq_partition_true,
        "eq_part_false": offset_eq_partition_false,
        "cache_outcomes": cache_outcomes,
        "discard_unused": discard_unused,
        "outcomes_with_submitted": outcomes_with_submitted,
    }


def fmt_dist(c, k=10):
    if not c:
        return "(empty)"
    items = sorted(c.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{k}:{v}" for k, v in items)


def print_one(s):
    print(f"\n=== {s['label']} ===")
    print(f"prefetch_new_blocks calls : {s['pf_calls']}")
    print(f"coord.prefetch_emit total : {s['pf_emits']}")
    if s["eq_part_true"] or s["eq_part_false"]:
        total = s["eq_part_true"] + s["eq_part_false"]
        pct_t = (s["eq_part_true"] / total * 100) if total else 0
        print(
            f"  partition-aligned       : {s['eq_part_true']:>4} ({pct_t:>5.1f}%)"
        )
        print(
            f"  sub-partition (extra)   : {s['eq_part_false']:>4} ({100 - pct_t:>5.1f}%)"
        )
    print(f"submitted per call dist   : {fmt_dist(s['submitted_dist'])}")
    print(f"candidates per call dist  : {fmt_dist(s['candidates_dist'])}")
    if s["outcomes_with_submitted"]:
        nums = [n for n in s["outcomes_with_submitted"] if n >= 0]
        if nums:
            print(
                f"submitted sum             : {sum(nums)}  mean={sum(nums)/len(nums):.2f}"
            )
    print(f"skip reasons              : {fmt_dist(s['skip_reasons'])}")
    print(f"cache.get_outcome         : {fmt_dist(s['cache_outcomes'])}")
    if s["pf_emits"]:
        hits = s["cache_outcomes"].get("prefetch", 0)
        useful_pct = hits / s["pf_emits"] * 100
        print(
            f"useful rate (hits/emits)  : {hits}/{s['pf_emits']} = {useful_pct:.1f}%"
        )
    print(f"discard_unused total      : {s['discard_unused']}")


def print_diff(g, r):
    print("\n=== DIFF (gzippy vs rapidgzip) ===")
    g_emits = g["pf_emits"]
    r_emits = r["pf_emits"]
    delta = g_emits - r_emits
    ratio = (g_emits / r_emits) if r_emits else float("inf")
    print(f"coord.prefetch_emit       : {g_emits} vs {r_emits}  (delta={delta:+d}, ratio={ratio:.2f}x)")
    print(f"prefetch_new_blocks calls : {g['pf_calls']} vs {r['pf_calls']}")
    g_useful = g["cache_outcomes"].get("prefetch", 0)
    r_useful = r["cache_outcomes"].get("prefetch", 0)
    print(f"prefetch hits             : {g_useful} vs {r_useful}")
    print(f"wasted prefetches (g)     : {g_emits - g_useful} (= emits - hits)")
    print(f"wasted prefetches (r)     : {r_emits - r_useful}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: prefetch_lifecycle_diff.py <gzippy.json> <rapidgzip.json>", file=sys.stderr)
        sys.exit(1)
    g = summarize(sys.argv[1], "gzippy")
    r = summarize(sys.argv[2], "rapidgzip")
    print_one(g)
    print_one(r)
    print_diff(g, r)
