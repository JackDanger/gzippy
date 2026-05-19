#!/usr/bin/env python3
"""
profile_diff.py — machine-readable flamegraph diff between gzippy and rapidgzip.

Input: two Brendan-Gregg folded-stack files (`frame1;frame2;... count` per line)
       or per-tool profile.json files (samply's Firefox profiler format).

Output: a single JSON document with:
  - per-tool sample counts and total
  - frames classified into vendor-agnostic bands
  - per-band gzippy-vs-rapidgzip delta (percentage points)
  - L1 distance between the two band distributions (single number, lower = closer)
  - verdict: "closer" / "further" / "first run"

Bands (ordered most-specific-first; first match wins per frame):
  isal_inflate         frames running ISA-L's inflate machinery
  memcpy_refill        memcpy / __memmove in refill / output paths
  allocator            malloc / free / arena / tcache / mmap-for-alloc
  marker_apply         gzippy MapMarkers / apply_window / replace_markers
  subchunk_boundary    append_block_boundary / appendDeflateBlockBoundary
  crc32                crc32 / CRC32Calculator
  thread_pool          ThreadPool / spawn / parker / condvar / queue
  file_io              read / pread / lseek / fstat
  bit_reader           BitReader / refill / inflatePrime
  decode_orchestration GzipChunkFetcher / ParallelGzipReader / chunk_fetcher::drive
  other                everything else

Usage:
  scripts/profile_diff.py \\
      --gzippy-folded   target/profile/gzippy.folded \\
      --rapidgzip-folded target/profile/rapidgzip.folded \\
      [--out JSON_PATH] \\
      [--baseline PRIOR_DIFF_JSON]   # to compute delta vs the last diff

Or from samply JSON (auto-folds):
  scripts/profile_diff.py \\
      --gzippy-samply    target/profile/gzippy.profile.json \\
      --rapidgzip-samply target/profile/rapidgzip.profile.json \\
      [--out JSON_PATH]

Exit code: 0 always (this is a measurement, not a gate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# ── Band classification ────────────────────────────────────────────────────

# Ordered list of (band_name, [regex_patterns]). First match wins.
# Patterns are matched case-insensitively against the FULL stack line, not just
# the leaf frame, because some allocator activity lives mid-stack.
BANDS: list[tuple[str, list[re.Pattern]]] = [
    (
        "isal_inflate",
        [
            re.compile(r"isal_inflate", re.IGNORECASE),
            re.compile(r"decode_huffman", re.IGNORECASE),
            re.compile(r"\bdecode_lz77_block\b", re.IGNORECASE),
        ],
    ),
    (
        "memcpy_refill",
        [
            re.compile(r"\b(?:__)?(?:memcpy|memmove|bcopy)\b", re.IGNORECASE),
            re.compile(r"refillBuffer|refill_buffer", re.IGNORECASE),
            re.compile(r"copy_nonoverlapping", re.IGNORECASE),
        ],
    ),
    (
        "allocator",
        [
            re.compile(r"_int_(?:malloc|free|realloc)", re.IGNORECASE),
            re.compile(r"tcache_(?:get|put)", re.IGNORECASE),
            re.compile(r"\b(?:malloc|free|calloc|realloc)\b", re.IGNORECASE),
            re.compile(r"alloc::raw_vec|alloc::alloc|Box::new", re.IGNORECASE),
            re.compile(r"je_malloc|rp_malloc|mi_malloc", re.IGNORECASE),
            re.compile(r"sys_alloc|posix_memalign", re.IGNORECASE),
        ],
    ),
    (
        "marker_apply",
        [
            re.compile(r"apply_window|replace_markers|MapMarkers", re.IGNORECASE),
            re.compile(r"data_with_markers", re.IGNORECASE),
        ],
    ),
    (
        "subchunk_boundary",
        [
            re.compile(r"append_block_boundary", re.IGNORECASE),
            re.compile(r"appendDeflateBlockBoundary", re.IGNORECASE),
            re.compile(r"subchunk", re.IGNORECASE),
        ],
    ),
    (
        "crc32",
        [
            re.compile(r"\bcrc32\b", re.IGNORECASE),
            re.compile(r"CRC32Calculator", re.IGNORECASE),
        ],
    ),
    (
        "thread_pool",
        [
            re.compile(r"ThreadPool|thread_pool", re.IGNORECASE),
            re.compile(r"std::sys::pal::.*::thread", re.IGNORECASE),
            re.compile(r"parker|park_timeout|unpark", re.IGNORECASE),
            re.compile(r"condvar|cv_wait|cv_signal", re.IGNORECASE),
            re.compile(r"crossbeam|mpsc", re.IGNORECASE),
        ],
    ),
    (
        "bit_reader",
        [
            re.compile(r"BitReader|bit_reader", re.IGNORECASE),
            re.compile(r"inflatePrime|inflate_prime", re.IGNORECASE),
        ],
    ),
    (
        "decode_orchestration",
        [
            re.compile(r"GzipChunkFetcher|chunk_fetcher", re.IGNORECASE),
            re.compile(r"ParallelGzipReader|parallel_gzip_reader", re.IGNORECASE),
            re.compile(r"decodeChunkWithRapidgzip|decode_chunk_with_window", re.IGNORECASE),
            re.compile(r"decodeChunkWithInflateWrapper|decode_chunk_with_inflate_wrapper", re.IGNORECASE),
            re.compile(r"BlockFetcher|block_fetcher", re.IGNORECASE),
            re.compile(r"WindowMap|window_map", re.IGNORECASE),
        ],
    ),
    (
        "file_io",
        [
            re.compile(r"\b(?:__)?(?:read|pread|lseek|fstat|open|close)\b", re.IGNORECASE),
            re.compile(r"\bmmap\b|\bmunmap\b", re.IGNORECASE),
            re.compile(r"std::fs::|std::io::|BufReader|BufWriter", re.IGNORECASE),
        ],
    ),
]


def classify(stack_line: str) -> str:
    """Return the first band whose patterns match anywhere in the stack line."""
    for band, patterns in BANDS:
        for p in patterns:
            if p.search(stack_line):
                return band
    return "other"


# ── Folded-stack parsing ──────────────────────────────────────────────────


def parse_folded(path: Path) -> dict[str, int]:
    """Parse a Brendan-Gregg folded-stack file. Returns {full_stack: count}."""
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            # Format: "frame1;frame2;... count"
            # Split from the right because frame names can theoretically contain
            # spaces, but the count is always the last whitespace-separated token.
            try:
                stack_part, count_part = line.rsplit(None, 1)
                out[stack_part] = out.get(stack_part, 0) + int(count_part)
            except ValueError:
                continue
    return out


def samply_to_folded(path: Path) -> dict[str, int]:
    """
    Convert a samply (Firefox profiler) JSON profile to {full_stack: count}.

    samply's format: profile.threads[].samples[] is a SamplesTable with a
    `stack` column referencing profile.threads[].stackTable, which references
    profile.threads[].frameTable, which references profile.threads[].stringArray.

    We walk every sample, resolve its full stack as `frame_leaf;...;frame_root`
    strings, and accumulate counts.
    """
    with path.open("r", encoding="utf-8") as f:
        prof = json.load(f)
    out: dict[str, int] = defaultdict(int)
    for thread in prof.get("threads", []):
        strings: list[str] = thread.get("stringArray", []) or thread.get("stringTable", [])
        frame_table = thread.get("frameTable", {})
        frame_locs = frame_table.get("location", []) or frame_table.get("func", [])
        stack_table = thread.get("stackTable", {})
        stack_frames = stack_table.get("frame", [])
        stack_prefixes = stack_table.get("prefix", [])

        def stack_to_str(stack_idx: int | None) -> str:
            parts: list[str] = []
            while stack_idx is not None and stack_idx >= 0:
                fi = stack_frames[stack_idx] if stack_idx < len(stack_frames) else None
                if fi is not None and 0 <= fi < len(frame_locs):
                    si = frame_locs[fi]
                    if si is not None and 0 <= si < len(strings):
                        parts.append(strings[si])
                stack_idx = stack_prefixes[stack_idx] if stack_idx < len(stack_prefixes) else None
            return ";".join(reversed(parts)) if parts else "[no stack]"

        samples = thread.get("samples", {})
        sample_stacks = samples.get("stack", [])
        for stack_idx in sample_stacks:
            key = stack_to_str(stack_idx) if stack_idx is not None else "[no stack]"
            out[key] += 1
    return dict(out)


# ── Band aggregation ──────────────────────────────────────────────────────


def aggregate_bands(stacks: dict[str, int]) -> tuple[int, dict[str, int]]:
    """Return (total_samples, {band: samples})."""
    band_counts: dict[str, int] = defaultdict(int)
    total = 0
    for stack, count in stacks.items():
        band = classify(stack)
        band_counts[band] += count
        total += count
    return total, dict(band_counts)


def pct(numer: int, denom: int) -> float:
    return (100.0 * numer / denom) if denom > 0 else 0.0


# ── Diff computation ──────────────────────────────────────────────────────


def make_diff(
    gzippy_bands: dict[str, int],
    gzippy_total: int,
    rapidgzip_bands: dict[str, int],
    rapidgzip_total: int,
) -> dict:
    all_bands: list[str] = [b for b, _ in BANDS] + ["other"]
    per_band: dict[str, dict[str, float]] = {}
    l1_distance = 0.0
    for band in all_bands:
        gz_pct = pct(gzippy_bands.get(band, 0), gzippy_total)
        rg_pct = pct(rapidgzip_bands.get(band, 0), rapidgzip_total)
        delta = gz_pct - rg_pct
        per_band[band] = {
            "gzippy_pct": round(gz_pct, 2),
            "rapidgzip_pct": round(rg_pct, 2),
            "delta_pp": round(delta, 2),
            "gzippy_samples": int(gzippy_bands.get(band, 0)),
            "rapidgzip_samples": int(rapidgzip_bands.get(band, 0)),
        }
        l1_distance += abs(delta)
    return {
        "by_band": per_band,
        "l1_distance_pp": round(l1_distance, 2),
        "interpretation": (
            "L1 distance is the sum of per-band |gzippy_pct - rapidgzip_pct|. "
            "0 = identical band distribution. Lower is closer."
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def load_for_tool(args, prefix: str) -> tuple[str, dict[str, int]]:
    folded = getattr(args, f"{prefix}_folded")
    samply = getattr(args, f"{prefix}_samply")
    if folded and samply:
        sys.exit(f"--{prefix}-folded and --{prefix}-samply are mutually exclusive")
    if folded:
        return ("folded", parse_folded(Path(folded)))
    if samply:
        return ("samply", samply_to_folded(Path(samply)))
    sys.exit(f"must specify --{prefix}-folded or --{prefix}-samply")


def write_md_summary(out_md: Path, diff_doc: dict) -> None:
    lines: list[str] = []
    lines.append("# Flamegraph band diff: gzippy vs rapidgzip\n")
    lines.append(f"- Source: `{diff_doc.get('fixture', 'unknown')}`")
    lines.append(f"- gzippy samples: {diff_doc['gzippy']['total_samples']}")
    lines.append(f"- rapidgzip samples: {diff_doc['rapidgzip']['total_samples']}")
    l1 = diff_doc["diff"]["l1_distance_pp"]
    lines.append(f"- **L1 band distance: {l1} pp** (0 = identical distribution)")
    delta = diff_doc.get("baseline_delta")
    if delta is not None:
        sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        lines.append(f"- vs prior diff: {sign} {abs(delta)} pp")
    lines.append("")
    lines.append("| Band | gzippy % | rapidgzip % | Δ pp |")
    lines.append("|---|---:|---:|---:|")
    bands_sorted = sorted(
        diff_doc["diff"]["by_band"].items(),
        key=lambda kv: -abs(kv[1]["delta_pp"]),
    )
    for band, vals in bands_sorted:
        lines.append(
            f"| {band} | {vals['gzippy_pct']:.2f} | {vals['rapidgzip_pct']:.2f} | "
            f"{vals['delta_pp']:+.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gzippy-folded")
    ap.add_argument("--gzippy-samply")
    ap.add_argument("--rapidgzip-folded")
    ap.add_argument("--rapidgzip-samply")
    ap.add_argument("--out", help="Output JSON path. Defaults to stdout if omitted.")
    ap.add_argument("--out-md", help="Optional Markdown summary path.")
    ap.add_argument("--fixture", default="", help="Free-form fixture description.")
    ap.add_argument(
        "--baseline",
        help="Path to a prior diff JSON. If given, includes baseline_delta_pp in output.",
    )
    args = ap.parse_args()

    gz_src, gz_stacks = load_for_tool(args, "gzippy")
    rg_src, rg_stacks = load_for_tool(args, "rapidgzip")

    gz_total, gz_bands = aggregate_bands(gz_stacks)
    rg_total, rg_bands = aggregate_bands(rg_stacks)

    if gz_total == 0:
        print("WARNING: gzippy profile has 0 samples", file=sys.stderr)
    if rg_total == 0:
        print("WARNING: rapidgzip profile has 0 samples", file=sys.stderr)

    diff = make_diff(gz_bands, gz_total, rg_bands, rg_total)

    doc = {
        "schema_version": 1,
        "fixture": args.fixture,
        "gzippy": {
            "source": gz_src,
            "total_samples": int(gz_total),
            "bands": {b: {"samples": int(v), "pct": round(pct(v, gz_total), 2)}
                      for b, v in gz_bands.items()},
        },
        "rapidgzip": {
            "source": rg_src,
            "total_samples": int(rg_total),
            "bands": {b: {"samples": int(v), "pct": round(pct(v, rg_total), 2)}
                      for b, v in rg_bands.items()},
        },
        "diff": diff,
    }

    if args.baseline:
        try:
            prior = json.loads(Path(args.baseline).read_text())
            prior_l1 = float(prior["diff"]["l1_distance_pp"])
            doc["baseline_delta"] = round(diff["l1_distance_pp"] - prior_l1, 2)
            doc["baseline_path"] = args.baseline
        except (KeyError, ValueError, FileNotFoundError) as e:
            print(f"WARNING: baseline read failed: {e}", file=sys.stderr)

    text = json.dumps(doc, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)

    if args.out_md:
        write_md_summary(Path(args.out_md), doc)
        print(f"wrote {args.out_md}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
