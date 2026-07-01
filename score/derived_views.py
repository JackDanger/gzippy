#!/usr/bin/env python3
"""
score/derived_views.py — derive self-speedup, absolute wall table, and regime analysis
from the committed score/ cell files.

Reads all score/<arch>/<tN>/<corpus>.md files, extracts wall_ms from the YAML
builds block, and produces:
  (a) Self-speedup S(t) = wall(t1) / wall(tN) per tool per corpus per arch
  (b) Absolute wall_ms table annotated with decompressed file size
  (c) Regime annotation (IO-bound / compute-bound / overhead-dominated)

Writes output to stdout (for pasting into DERIVED.md) or --out FILE.

Usage:
    python3 score/derived_views.py [--out score/DERIVED.md] [--score-root score/]
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus metadata: compressed size (bytes), decompressed size (bytes), regime
# Source: `stat` + `python3 -c "import struct,gzip; ..."` on solvency 2026-06-13
# Regime classification:
#   compute-bound:       DEFLATE work is the bottleneck; scales well with T
#   IO-bound:            disk read/write is the bottleneck (stored or near-stored blocks)
#   overhead-dominated:  fixed startup / header overhead significant vs decode time
# NOTE: "model" is LLM weights (binary floats) → ratio=1.27x but still compute-bound;
#       verified by rg T1→T8 scaling of 4.1x. storedheavy is truly stored blocks (1.0x ratio).
# ---------------------------------------------------------------------------
CORPUS_META = {
    "bignasa":     {"compressed_mb": 74.8, "decompressed_mb": 783.1,
                    "regime": "compute-bound"},
    "model":       {"compressed_mb": 203.4, "decompressed_mb": 256.6,
                    "regime": "compute-bound (lightly-compressed binary)"},
    "monorepo":    {"compressed_mb": 9.4,  "decompressed_mb": 48.6,
                    "regime": "overhead-dominated (small file)"},
    "silesia":     {"compressed_mb": 65.1, "decompressed_mb": 202.1,
                    "regime": "compute-bound"},
    "storedheavy": {"compressed_mb": 95.4, "decompressed_mb": 95.4,
                    "regime": "IO-bound (stored blocks, 1.0x ratio)"},
}


def regime(corpus):
    return CORPUS_META.get(corpus, {}).get("regime", "unknown")


THREAD_NUMS = [1, 4, 8, 12, 16]
TOOLS = ["rapidgzip-native", "gzippy-native", "gzippy-isal"]
TOOL_SHORT = {"rapidgzip-native": "rg", "gzippy-native": "native", "gzippy-isal": "isal"}


def parse_cell(path: Path):
    """Return dict with keys threads, corpus, arch_os, and builds={tool: wall_ms}."""
    text = path.read_text()

    # Extract YAML block between ```yaml ... ```
    m = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        return None

    yaml_text = m.group(1)

    def field(key):
        mm = re.search(rf"^{key}:\s*(.+)$", yaml_text, re.MULTILINE)
        return mm.group(1).strip() if mm else None

    try:
        threads = int(field("threads"))
    except (TypeError, ValueError):
        return None

    corpus = field("corpus")
    arch_os = field("arch_os")

    # Extract per-build wall_ms values
    # Builds block structure:
    #   builds:
    #     rapidgzip-native:
    #       wall_ms: 247
    #     gzippy-native:
    #       wall_ms: 334
    #     gzippy-isal:
    #       wall_ms: 249
    builds = {}
    builds_m = re.search(r"^builds:\n(.*?)(?=^\w|\Z)", yaml_text, re.DOTALL | re.MULTILINE)
    if builds_m:
        builds_block = builds_m.group(1)
        current_tool = None
        for line in builds_block.splitlines():
            tool_m = re.match(r"^\s{2}(\S.*?):\s*$", line)
            if tool_m:
                current_tool = tool_m.group(1)
                builds[current_tool] = {}
            elif current_tool:
                wm = re.match(r"^\s+wall_ms:\s*(\d+)", line)
                if wm:
                    builds[current_tool]["wall_ms"] = int(wm.group(1))

    return {
        "threads": threads,
        "corpus": corpus,
        "arch_os": arch_os,
        "builds": {t: b.get("wall_ms") for t, b in builds.items()},
    }


def load_all_cells(score_root: Path):
    """Load all cell files, return list of parsed dicts."""
    cells = []
    for md in sorted(score_root.glob("**/t*/*.md")):
        if md.name in ("README.md", "SCHEMA.md", "DERIVED.md", "CELL-TEMPLATE.md"):
            continue
        if "archive" in md.parts:
            continue
        parsed = parse_cell(md)
        if parsed:
            cells.append(parsed)
    return cells


def build_matrix(cells):
    """Organize cells into matrix[arch][corpus][t] = {tool: wall_ms}."""
    matrix = {}
    for c in cells:
        arch = c["arch_os"]
        corpus = c["corpus"]
        t = c["threads"]
        matrix.setdefault(arch, {}).setdefault(corpus, {})[t] = c["builds"]
    return matrix


def fmt_ratio(r):
    if r is None:
        return "  n/a"
    return f"{r:5.2f}x"


def render(matrix, out):
    archs = sorted(matrix.keys())
    all_corpora = sorted({corpus for arch_data in matrix.values() for corpus in arch_data})

    out.write("# score/DERIVED — derived matrix views\n\n")
    out.write("Generated by `python3 score/derived_views.py`. Do not edit by hand; regenerate.\n")
    out.write("Source: all `score/<arch>/<tN>/<corpus>.md` cell files.\n\n")

    # -----------------------------------------------------------------------
    # (b) Corpus size + regime table
    # -----------------------------------------------------------------------
    out.write("## Corpus sizes and regime\n\n")
    out.write("| corpus | compressed | decompressed | ratio | regime |\n")
    out.write("|--------|-----------|-------------|-------|--------|\n")
    for c in all_corpora:
        m = CORPUS_META.get(c, {})
        comp = m.get("compressed_mb", 0)
        decomp = m.get("decompressed_mb", 0)
        ratio = decomp / comp if comp > 0 else 0
        r = regime(c)
        out.write(f"| {c} | {comp:.0f} MiB | {decomp:.0f} MiB | {ratio:.1f}x | {r} |\n")
    out.write("\n")

    for arch in archs:
        arch_data = matrix[arch]
        out.write(f"## Arch: {arch}\n\n")

        # -------------------------------------------------------------------
        # (b) Absolute wall_ms table
        # -------------------------------------------------------------------
        out.write("### (b) Absolute wall_ms\n\n")

        # Header
        header_tools = [("rg", "rapidgzip-native"), ("native", "gzippy-native"), ("isal", "gzippy-isal")]
        thead = "| corpus | tN |"
        for short, _ in header_tools:
            thead += f" {short}_ms |"
        out.write(thead + "\n")
        out.write("|--------|-----|" + "--------|" * len(header_tools) + "\n")

        for corpus in sorted(arch_data.keys()):
            for t in THREAD_NUMS:
                builds = arch_data[corpus].get(t, {})
                row = f"| {corpus} | t{t} |"
                for short, tool in header_tools:
                    w = builds.get(tool)
                    row += f" {w if w else 'n/a':>6} |"
                out.write(row + "\n")

        out.write("\n")

        # -------------------------------------------------------------------
        # (a) Self-speedup S(tN) = wall(t1) / wall(tN)
        # -------------------------------------------------------------------
        out.write("### (a) Self-speedup S(tN) = wall(t1) / wall(tN)\n\n")
        out.write("A value > 1.0 means tN is faster than t1. Interpret as: how well does this tool scale?\n\n")

        out.write("| corpus | regime | tool | S(t4) | S(t8) | S(t12) | S(t16) |\n")
        out.write("|--------|--------|------|-------|-------|--------|--------|\n")

        for corpus in sorted(arch_data.keys()):
            tdata = arch_data[corpus]
            t1_builds = tdata.get(1, {})
            r = regime(corpus)

            for tool, short in [
                ("rapidgzip-native", "rg"),
                ("gzippy-native", "native"),
                ("gzippy-isal", "isal"),
            ]:
                t1_ms = t1_builds.get(tool)
                row = f"| {corpus} | {r.split('(')[0].strip()} | {short} |"
                for tN in [4, 8, 12, 16]:
                    tN_builds = tdata.get(tN, {})
                    tN_ms = tN_builds.get(tool)
                    if t1_ms and tN_ms and tN_ms > 0:
                        s = t1_ms / tN_ms
                        row += f" {s:5.2f} |"
                    else:
                        row += "   n/a |"
                out.write(row + "\n")

        out.write("\n")

        # -------------------------------------------------------------------
        # Key observations per corpus
        # -------------------------------------------------------------------
        out.write("### Key observations (self-speedup)\n\n")
        for corpus in sorted(arch_data.keys()):
            tdata = arch_data[corpus]
            t1_builds = tdata.get(1, {})
            t8_builds = tdata.get(8, {})

            rg_t1 = t1_builds.get("rapidgzip-native")
            rg_t8 = t8_builds.get("rapidgzip-native")
            nat_t1 = t1_builds.get("gzippy-native")
            nat_t8 = t8_builds.get("gzippy-native")

            if rg_t1 and rg_t8 and nat_t1 and nat_t8:
                rg_scale = rg_t1 / rg_t8
                nat_scale = nat_t1 / nat_t8
                t1_ratio = rg_t1 / nat_t1  # rg/native (how much faster rg is at t1)
                t8_ratio = rg_t8 / nat_t8
                r = regime(corpus)
                out.write(f"- **{corpus}** ({r}): ")
                out.write(f"rg scales {rg_scale:.2f}x T1→T8, native scales {nat_scale:.2f}x. ")
                out.write(f"Ratio rg/native: T1={t1_ratio:.2f} T8={t8_ratio:.2f}. ")
                if abs(rg_scale - nat_scale) < 0.15:
                    out.write("Both tools SCALE SIMILARLY → gap is per-unit-of-work, not parallelism.\n")
                elif nat_scale > rg_scale + 0.15:
                    out.write("native SCALES BETTER → gzippy parallelizes more effectively.\n")
                else:
                    out.write("rg scales better → gzippy has higher serial fraction.\n")

        out.write("\n")

    # -----------------------------------------------------------------------
    # Cross-arch summary
    # -----------------------------------------------------------------------
    out.write("## Cross-arch summary: parity cells\n\n")
    out.write("Cell verdict (from SCORE lines): PASS = ratio>=0.99 vs rapidgzip-native.\n\n")

    # Read SCORE lines
    out.write("| arch | corpus | tN | native | isal |\n")
    out.write("|------|--------|-----|--------|------|\n")

    for arch in archs:
        arch_data = matrix[arch]
        for corpus in sorted(arch_data.keys()):
            for t in THREAD_NUMS:
                builds = arch_data[corpus].get(t, {})
                rg_ms = builds.get("rapidgzip-native")
                nat_ms = builds.get("gzippy-native")
                isal_ms = builds.get("gzippy-isal")

                def verdict(tool_ms):
                    if rg_ms is None or tool_ms is None or tool_ms == 0:
                        return "n/a"
                    r = rg_ms / tool_ms
                    return f"{r:.2f} {'PASS' if r >= 0.99 else 'FAIL'}"

                out.write(f"| {arch} | {corpus} | t{t} | {verdict(nat_ms)} | {verdict(isal_ms)} |\n")

    out.write("\n")
    out.write("_Note: ratio = rapidgzip-native wall / gzippy wall; >=0.99 = PASS._\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-root", default="score", help="path to score/ directory")
    ap.add_argument("--out", default=None, help="output file (default: stdout)")
    args = ap.parse_args()

    score_root = Path(args.score_root)
    if not score_root.exists():
        print(f"ERROR: score root not found: {score_root}", file=sys.stderr)
        sys.exit(1)

    cells = load_all_cells(score_root)
    if not cells:
        print(f"ERROR: no cell files found under {score_root}", file=sys.stderr)
        sys.exit(1)

    matrix = build_matrix(cells)

    if args.out:
        with open(args.out, "w") as f:
            render(matrix, f)
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        render(matrix, sys.stdout)


if __name__ == "__main__":
    main()
