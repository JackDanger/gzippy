#!/usr/bin/env python3
"""
Multi-format gzippy-vs-rapidgzip comparison + 100%-coverage profiling.

Runs bench_stats.py on every gzip-family archive format both tools can
decode, then captures perf-record + flamegraphs for each. Output answers
"how does gzippy compare to rapidgzip across the full surface, and where
is gzippy's time actually going?"

Archive formats covered (per CLAUDE.md scope):
- single-member gzip      (gzippy + rapidgzip)
- multi-member gzip       (gzippy + rapidgzip)
- BGZF                    (gzippy + rapidgzip)
- gzippy-parallel "GZ"    (gzippy only — rapidgzip cannot read this format)

For each format:
1. Generate the compressed fixture from `--source` (deterministic).
2. Verify both tools decode it byte-for-byte (the gzippy-parallel format
   skips the rapidgzip leg with a NOTE in the report, since rapidgzip
   cannot read it; the format is in gzippy scope per CLAUDE.md).
3. Run bench_stats.py with --trials N for statistical CIs.
4. (Linux only, with perf) Run profile_compare.sh for flamegraphs.

Output:
    $OUT_DIR/format_NAME/                 # one dir per format
        fixture.gz                        # the compressed input
        bench.json + bench.md             # bench_stats.py output
        profile/gzippy.flamegraph.svg     # if --profile and Linux
        profile/rapidgzip.flamegraph.svg
        profile/perfstat_diff.md
    $OUT_DIR/summary.md                   # cross-format comparison table

Usage:
    bench_all_formats.py \\
        --source path/to/raw.bin              # single source decompressed file
        --gzippy target/release/gzippy \\
        --rapidgzip vendor/.../rapidgzip \\
        --bgzip /usr/bin/bgzip \\             # optional, for BGZF fixture
        --threads 16 \\
        --trials 20 \\
        --out-dir target/tooling/bench-all-TIMESTAMP \\
        [--profile]                            # Linux only — capture perf

Designed to be run on neurotic via:
    scripts/remote_bench.sh --cmd "python3 scripts/bench_all_formats.py ..."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command with sensible defaults and verbose error output."""
    print(f"$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=True, **kwargs)


def make_single_member(source: Path, dst: Path, gzippy: Path) -> None:
    """Plain gzippy compress to a single-member archive (no `-i` flag)."""
    with open(source, "rb") as src, open(dst, "wb") as out:
        run([str(gzippy), "-c"], stdin=src, stdout=out)


def make_multi_member(source: Path, dst: Path, parts: int = 8) -> None:
    """Split source into N chunks, compress each separately, concatenate.

    Produces a real multi-member gzip stream — each part is its own
    self-contained gzip member. rapidgzip routes this through
    `decode_multi_member_*`; gzippy routes through `bgzf::decompress_multi_member_parallel`
    (T>1) or `decompress_multi_member_sequential` (T=1) per CLAUDE.md.
    """
    src_bytes = source.read_bytes()
    chunk_sz = (len(src_bytes) + parts - 1) // parts
    with open(dst, "wb") as out:
        for i in range(parts):
            chunk = src_bytes[i * chunk_sz : (i + 1) * chunk_sz]
            if not chunk:
                break
            # Use gzip -c so the output is a vanilla gzip member with no
            # gzippy-specific FEXTRA.
            proc = subprocess.run(
                ["gzip", "-c"], input=chunk, capture_output=True, check=True
            )
            out.write(proc.stdout)


def make_bgzf(source: Path, dst: Path, bgzip: Optional[Path]) -> bool:
    """BGZF via `bgzip` (htslib). Returns False if bgzip is unavailable."""
    if bgzip is None or not bgzip.exists():
        return False
    with open(source, "rb") as src, open(dst, "wb") as out:
        run([str(bgzip), "-c"], stdin=src, stdout=out)
    return True


def make_gzippy_parallel(source: Path, dst: Path, gzippy: Path) -> None:
    """gzippy `-i` (independent blocks → "GZ" FEXTRA subfield).
    Only gzippy can decode this format. Included to measure gzippy's
    own parallel-decompressable wire format.
    """
    with open(source, "rb") as src, open(dst, "wb") as out:
        run([str(gzippy), "-c", "-i"], stdin=src, stdout=out)


def run_bench(
    label: str,
    compressed: Path,
    original: Path,
    gzippy: Path,
    rapidgzip: Optional[Path],
    threads: int,
    trials: int,
    out_dir: Path,
) -> dict:
    """Invoke bench_stats.py for one format. Returns parsed JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "bench.json"
    md_out = out_dir / "bench.md"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "bench_stats.py"),
        "--compressed", str(compressed),
        "--original", str(original),
        "--gzippy", str(gzippy),
        "--threads", str(threads),
        "--trials", str(trials),
        "--output", str(json_out),
        "--md", str(md_out),
    ]
    if rapidgzip is not None:
        cmd += ["--rapidgzip", str(rapidgzip)]
    run(cmd)
    return json.loads(json_out.read_text())


def maybe_profile(
    label: str,
    compressed: Path,
    gzippy: Path,
    rapidgzip: Optional[Path],
    threads: int,
    out_dir: Path,
) -> bool:
    """Run profile_compare.sh if available. Returns True if profile ran.

    Linux + perf required. Falls back silently on macOS.
    """
    if sys.platform != "linux":
        return False
    if rapidgzip is None:
        return False
    script = REPO / "scripts" / "profile_compare.sh"
    if not script.exists():
        return False
    profile_dir = out_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(script),
        "--compressed", str(compressed),
        "--gzippy", str(gzippy),
        "--rapidgzip", str(rapidgzip),
        "--threads", str(threads),
        "--out-dir", str(profile_dir),
    ]
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARN: profile_compare.sh failed for {label}: {e}", file=sys.stderr)
        return False


def summary_row(label: str, bench: dict) -> str:
    """One markdown table row summarizing a format's gzippy-vs-rapidgzip result.
    Matches bench_stats.py output schema: top-level `summary` keyed by tool
    with `throughput_mbps` dict; top-level `ratios` keyed by tool with
    `ratio_to_rapidgzip_throughput` dict.
    """
    summary = bench.get("summary", {})
    g = summary.get("gzippy", {}).get("throughput_mbps", {})
    r = summary.get("rapidgzip", {}).get("throughput_mbps", {})
    g_mbps = g.get("median")
    r_mbps = r.get("median") if r else None

    ratios = bench.get("ratios", {}).get("gzippy", {})
    rt = ratios.get("ratio_to_rapidgzip_throughput", {})
    median_ratio = rt.get("median")
    ci_lo = rt.get("ci95_lo")
    ci_hi = rt.get("ci95_hi")

    if r_mbps is not None and median_ratio is not None:
        return (
            f"| {label} | {g_mbps:.0f} MB/s | {r_mbps:.0f} MB/s | "
            f"{median_ratio:.2f}× | [{ci_lo:.2f}, {ci_hi:.2f}] |"
        )
    if g_mbps is not None:
        return f"| {label} | {g_mbps:.0f} MB/s | (no rapidgzip) | — | — |"
    return f"| {label} | (no data) | (no data) | — | — |"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, type=Path,
                   help="Decompressed source file to compress in each format")
    p.add_argument("--gzippy", required=True, type=Path)
    p.add_argument("--rapidgzip", type=Path,
                   help="Path to rapidgzip; skipped if absent")
    p.add_argument("--bgzip", type=Path, default=Path("/usr/bin/bgzip"),
                   help="bgzip tool for BGZF fixture (skipped if absent)")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--profile", action="store_true",
                   help="Capture perf flamegraphs (Linux + perf only)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["single", "multi", "bgzf", "gz-subfield"],
                   help="Formats to skip")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.source.exists():
        print(f"ERROR: source file missing: {args.source}", file=sys.stderr)
        return 2

    formats: list[tuple[str, str, callable, bool]] = []
    # (slug, display, factory, rapidgzip-decodes?)
    if "single" not in args.skip:
        formats.append((
            "single-member",
            "Single-member gzip",
            lambda dst: make_single_member(args.source, dst, args.gzippy),
            True,
        ))
    if "multi" not in args.skip:
        formats.append((
            "multi-member",
            "Multi-member gzip (8 parts)",
            lambda dst: make_multi_member(args.source, dst),
            True,
        ))
    if "bgzf" not in args.skip:
        if args.bgzip and args.bgzip.exists():
            formats.append((
                "bgzf",
                "BGZF (bgzip)",
                lambda dst: make_bgzf(args.source, dst, args.bgzip) or None,
                True,
            ))
        else:
            print("NOTE: bgzip not found, skipping BGZF fixture", file=sys.stderr)
    if "gz-subfield" not in args.skip:
        formats.append((
            "gz-subfield",
            "gzippy-parallel 'GZ' subfield",
            lambda dst: make_gzippy_parallel(args.source, dst, args.gzippy),
            False,  # rapidgzip cannot read this format
        ))

    results: list[tuple[str, dict, bool]] = []

    for slug, display, factory, rapidgzip_decodes in formats:
        print(f"\n=== {display} ===", file=sys.stderr)
        fmt_dir = out_dir / f"format_{slug}"
        fmt_dir.mkdir(parents=True, exist_ok=True)
        fixture = fmt_dir / "fixture.gz"
        factory(fixture)

        if not fixture.exists() or fixture.stat().st_size == 0:
            print(f"  SKIP — fixture not produced", file=sys.stderr)
            continue
        print(f"  fixture: {fixture} ({fixture.stat().st_size:,} bytes)", file=sys.stderr)

        rapidgzip_for_bench = args.rapidgzip if rapidgzip_decodes else None
        bench = run_bench(
            display, fixture, args.source,
            args.gzippy, rapidgzip_for_bench,
            args.threads, args.trials, fmt_dir,
        )

        profiled = False
        if args.profile:
            profiled = maybe_profile(
                display, fixture, args.gzippy, rapidgzip_for_bench,
                args.threads, fmt_dir,
            )

        results.append((display, bench, profiled))

    # Cross-format summary.
    summary_md = out_dir / "summary.md"
    lines: list[str] = []
    lines.append(f"# gzippy vs rapidgzip — all-format comparison\n")
    lines.append(f"Source: `{args.source}` ({args.source.stat().st_size:,} bytes)\n")
    lines.append(f"Threads: {args.threads}  Trials: {args.trials}\n")
    lines.append("")
    lines.append("| Format | gzippy (median) | rapidgzip (median) | speedup | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for display, bench, _ in results:
        lines.append(summary_row(display, bench))
    lines.append("")
    if args.profile:
        lines.append("## Flamegraphs\n")
        for display, _, profiled in results:
            slug = display.replace(" ", "_")
            if profiled:
                lines.append(
                    f"- **{display}**: `format_{slug}/profile/gzippy.flamegraph.svg`, "
                    f"`format_{slug}/profile/rapidgzip.flamegraph.svg`"
                )
            else:
                lines.append(f"- **{display}**: not profiled (Linux+perf+rapidgzip required)")
    summary_md.write_text("\n".join(lines) + "\n")

    print(f"\n=== done ===\n  summary: {summary_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
