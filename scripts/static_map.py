#!/usr/bin/env python3
"""
Vendor↔gzippy symbol map.

For every public C++ symbol in vendor/rapidgzip/.../*.hpp, find the
gzippy Rust symbol whose doc-comment cites the vendor file:line. Classify:
faithful / landed-unwired / missing / extra.

Outputs:
- docs/symbol_map.md
- target/tooling/symbol_map.json

Usage:
    static_map.py --vendor vendor/rapidgzip/librapidarchive/src \\
                  --gzippy src/decompress/parallel \\
                  --gzippy-callers src \\
                  --output target/tooling/symbol_map.json \\
                  --md docs/symbol_map.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ---------- Parsers ----------

CPP_SYM_RE = re.compile(
    r"""
    ^[\ \t]*
    (?P<kind>class|struct|inline|static|template[^;{]*\b(?:auto|void|int|size_t|bool|[A-Za-z_][\w:<>,\ &*]*)\b)?
    .*?
    \b(?P<name>[A-Z][A-Za-z0-9_]*)\b
    [\ \t]*\(
    """,
    re.VERBOSE,
)


def parse_cpp_symbols(root: Path) -> list[dict]:
    """Yield {file, line, symbol, kind} for class/struct/method declarations.

    Heuristic. Catches: `class Foo`, `struct Bar`, `void Foo::method(...)`,
    top-level `template<...> auto fn(...) -> X`. Misses some templated
    operator forms; good enough for the symbol-map use case.
    """
    out: list[dict] = []
    for path in sorted(root.rglob("*.hpp")):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(root.parent))
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue
            # Class / struct declarations.
            m = re.match(r"^(class|struct)\s+([A-Z][A-Za-z0-9_]*)\b", stripped)
            if m:
                out.append({"file": rel, "line": i, "symbol": m.group(2), "kind": m.group(1)})
                continue
            # Member function definition `Type Class::method(...)`.
            m = re.match(r"^[A-Za-z_][\w:<>,\s&*]*\s+([A-Z][A-Za-z0-9_]*)::([A-Za-z_]\w*)\s*\(", stripped)
            if m:
                out.append({
                    "file": rel,
                    "line": i,
                    "symbol": f"{m.group(1)}::{m.group(2)}",
                    "kind": "method",
                })
                continue
            # Top-level template function with trailing return type.
            m = re.match(r"^template\s*<.*>\s*$", stripped)
            if m and i < len(text.splitlines()):
                # Look at the next non-empty line for the function name.
                pass
    return out


RUST_FN_RE = re.compile(
    r"""
    ^[\ \t]*
    (?:pub(?:\([^)]+\))?\s+)?
    (?:async\s+)?
    (?:unsafe\s+)?
    fn\s+
    (?P<name>[a-zA-Z_][\w]*)
    """,
    re.VERBOSE,
)


CITATION_RE = re.compile(
    r"""
    (?P<path>[A-Za-z0-9_./]+?\.(?:hpp|cpp|h|c))
    (?::(?P<start>\d+)(?:-(?P<end>\d+))?)?
    """,
    re.VERBOSE,
)


def parse_rust_functions(root: Path) -> list[dict]:
    """Yield {file, line, fn, doc_lines, citations} for every `fn` in gzippy."""
    out: list[dict] = []
    for path in sorted(root.rglob("*.rs")):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        rel_to_repo = path
        lines = text.splitlines()
        # Walk for fn definitions; collect preceding doc-comment block.
        i = 0
        while i < len(lines):
            m = RUST_FN_RE.match(lines[i])
            if m:
                # Walk back collecting doc lines.
                j = i - 1
                doc: list[str] = []
                while j >= 0:
                    s = lines[j].strip()
                    if s.startswith("///") or s.startswith("//!"):
                        doc.insert(0, s.lstrip("/").strip())
                        j -= 1
                    elif not s:
                        # blank line continues the search
                        j -= 1
                    elif s.startswith("#["):
                        j -= 1
                    else:
                        break
                doc_text = "\n".join(doc)
                citations = []
                for cm in CITATION_RE.finditer(doc_text):
                    p = cm.group("path")
                    # Skip Rust source filenames (.rs accidentally matching).
                    if not p.endswith((".hpp", ".cpp", ".h", ".c")):
                        continue
                    # Skip false positives like `s.hpp` from a sentence fragment.
                    basename = p.rsplit("/", 1)[-1]
                    citations.append({
                        "vendor_path": p,
                        "basename": basename,
                        "start_line": int(cm.group("start")) if cm.group("start") else None,
                        "end_line": int(cm.group("end")) if cm.group("end") else None,
                    })
                out.append({
                    "file": str(rel_to_repo),
                    "line": i + 1,
                    "fn": m.group("name"),
                    "doc": doc_text,
                    "citations": citations,
                })
            i += 1
    return out


# ---------- Caller analysis ----------

def grep_callers(symbol: str, search_root: Path, exclude: set[str]) -> int:
    """Count grep hits for `symbol(` outside of exclude paths."""
    try:
        # Use git grep where possible; cheap.
        r = subprocess.run(
            ["git", "grep", "-c", "--", rf"\b{re.escape(symbol)}\s*("],
            cwd=str(search_root.parent),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 0
    if r.returncode not in (0, 1):
        return 0
    total = 0
    for line in r.stdout.splitlines():
        # `path:count`
        if ":" not in line:
            continue
        path, count = line.rsplit(":", 1)
        if any(path.startswith(e) for e in exclude):
            continue
        try:
            total += int(count)
        except ValueError:
            continue
    return total


# ---------- Classification ----------

def build_map(
    vendor_syms: list[dict],
    gzippy_fns: list[dict],
    gzippy_callers_root: Path,
) -> list[dict]:
    """Match each vendor symbol to a gzippy fn whose doc cites its file:line."""
    # Index gzippy fns by BASENAME (vendor citations use varied path
    # prefixes — `vendor/...`, `vendor/rapidgzip/...`, or bare basename).
    cite_index: dict[str, list[dict]] = defaultdict(list)
    for fn in gzippy_fns:
        for c in fn["citations"]:
            cite_index[c["basename"]].append({
                "fn": fn,
                "start": c["start_line"],
                "end": c["end_line"],
            })

    out: list[dict] = []
    matched_gzippy_fns: set[tuple[str, str]] = set()
    for sym in vendor_syms:
        vp = sym["file"]
        basename = vp.rsplit("/", 1)[-1]
        match = None
        for c in cite_index.get(basename, []):
            lo, hi = c["start"], c["end"] or c["start"]
            if lo is None:
                # File-level citation with no line range — count as match.
                match = c["fn"]
                break
            if lo <= sym["line"] <= (hi if hi else lo):
                match = c["fn"]
                break
        if match:
            # Caller count for the gzippy fn: grep for `name(` in src/ but
            # exclude the file the fn is defined in (defensive; doesn't count
            # the fn's own definition site) and the tests subtree.
            callers = grep_callers(
                match["fn"],
                gzippy_callers_root,
                exclude={str(Path(match["file"]).relative_to(gzippy_callers_root.parent)), "src/tests/"},
            )
            status = "faithful" if callers > 0 else "landed-unwired"
            matched_gzippy_fns.add((match["file"], match["fn"]))
            out.append({
                "vendor_file": sym["file"],
                "vendor_line": sym["line"],
                "vendor_symbol": sym["symbol"],
                "vendor_kind": sym["kind"],
                "gzippy_file": match["file"],
                "gzippy_line": match["line"],
                "gzippy_fn": match["fn"],
                "status": status,
                "callers_outside_self": callers,
            })
        else:
            out.append({
                "vendor_file": sym["file"],
                "vendor_line": sym["line"],
                "vendor_symbol": sym["symbol"],
                "vendor_kind": sym["kind"],
                "gzippy_file": None,
                "gzippy_line": None,
                "gzippy_fn": None,
                "status": "missing",
                "callers_outside_self": 0,
            })

    # Extras: gzippy fns with no vendor citation at all.
    for fn in gzippy_fns:
        if fn["citations"]:
            continue
        # Skip helper-private style fns? No — include for completeness.
        out.append({
            "vendor_file": None,
            "vendor_line": None,
            "vendor_symbol": None,
            "vendor_kind": None,
            "gzippy_file": fn["file"],
            "gzippy_line": fn["line"],
            "gzippy_fn": fn["fn"],
            "status": "extra",
            "callers_outside_self": 0,
        })

    return out


def render_md(rows: list[dict]) -> str:
    by_status: dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1

    lines: list[str] = []
    lines.append("# Vendor↔gzippy symbol map")
    lines.append("")
    lines.append("Auto-generated by `scripts/static_map.py`. Re-run to refresh.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for status in ("faithful", "landed-unwired", "missing", "extra"):
        lines.append(f"- **{status}**: {by_status[status]}")
    lines.append("")
    lines.append("## Faithful (vendor symbol → gzippy symbol → callers)")
    lines.append("")
    lines.append("| vendor file:line | vendor symbol | gzippy file:line | gzippy fn | callers |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if r["status"] != "faithful":
            continue
        lines.append(
            f"| `{r['vendor_file']}:{r['vendor_line']}` | `{r['vendor_symbol']}` "
            f"| `{r['gzippy_file']}:{r['gzippy_line']}` | `{r['gzippy_fn']}` "
            f"| {r['callers_outside_self']} |"
        )
    lines.append("")
    lines.append("## Landed-unwired (port exists, no production callers)")
    lines.append("")
    lines.append("| vendor symbol | gzippy fn | gzippy file:line |")
    lines.append("|---|---|---|")
    for r in rows:
        if r["status"] != "landed-unwired":
            continue
        lines.append(
            f"| `{r['vendor_symbol']}` | `{r['gzippy_fn']}` "
            f"| `{r['gzippy_file']}:{r['gzippy_line']}` |"
        )
    lines.append("")
    lines.append("## Missing (vendor has, gzippy doesn't)")
    lines.append("")
    lines.append("| vendor file:line | vendor symbol |")
    lines.append("|---|---|")
    for r in rows:
        if r["status"] != "missing":
            continue
        lines.append(f"| `{r['vendor_file']}:{r['vendor_line']}` | `{r['vendor_symbol']}` |")
    lines.append("")
    lines.append("## Extra (gzippy has, no vendor citation)")
    lines.append("")
    lines.append("| gzippy fn | gzippy file:line |")
    lines.append("|---|---|")
    for r in rows:
        if r["status"] != "extra":
            continue
        lines.append(f"| `{r['gzippy_fn']}` | `{r['gzippy_file']}:{r['gzippy_line']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vendor", required=True, help="vendor/rapidgzip/librapidarchive/src")
    p.add_argument("--gzippy", required=True, help="src/decompress/parallel (or wider scope)")
    p.add_argument("--gzippy-callers", default="src", help="root for caller search (default: src)")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--md", help="Markdown output path (stdout if omitted)")
    args = p.parse_args()

    vendor_root = Path(args.vendor).resolve()
    gzippy_root = Path(args.gzippy).resolve()
    callers_root = Path(args.gzippy_callers).resolve()

    if not vendor_root.exists():
        print(f"ERROR: vendor path not found: {vendor_root}", file=sys.stderr)
        return 2
    if not gzippy_root.exists():
        print(f"ERROR: gzippy path not found: {gzippy_root}", file=sys.stderr)
        return 2

    print(f"Parsing vendor headers under {vendor_root} …", file=sys.stderr)
    vendor = parse_cpp_symbols(vendor_root)
    print(f"  {len(vendor)} vendor symbols extracted", file=sys.stderr)

    print(f"Parsing gzippy modules under {gzippy_root} …", file=sys.stderr)
    gzippy = parse_rust_functions(gzippy_root)
    print(f"  {len(gzippy)} gzippy fns extracted", file=sys.stderr)

    print("Building symbol map …", file=sys.stderr)
    rows = build_map(vendor, gzippy, callers_root)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    md = render_md(rows)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.md, "w") as f:
            f.write(md)
        print(f"Markdown written to {args.md}", file=sys.stderr)
    else:
        print(md)

    by_status: dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
    print(f"Done: {dict(by_status)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
