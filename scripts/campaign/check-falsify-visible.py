#!/usr/bin/env python3
"""Assert every in-code falsification record is VISIBLE to the machinery that reads them.

WHY THIS EXISTS. On 2026-07-30 a session's central meta-fix was "the in-code FALSIFY note
is the record, because the `commit-msg` hook and `fulcrum candidates` both scan CODE, and
markdown does not fail closed." The falsification records that session then wrote were
headed `FALSIFIED <date>` — and `"FALSIFY" in "FALSIFIED"` is **False**; the seventh
character differs (Y vs I). `grep -q "FALSIFY"` therefore matched none of them.

Two whole files, `pipelined.rs` and `block_split.rs`, ended up carrying falsifications
that were invisible to both mechanisms — including the one guarding against re-sweeping
five chunk-grid shapes, the single most expensive search of the session.

The checker and the data it checks were written in the same session and never run against
each other. That is the general defect this file closes: **an enforcement mechanism and
the records it enforces against must be tested together, mechanically, not by eye.**

Run: scripts/campaign/check-falsify-visible.py [--fix]
Exit 0 = every record is visible. Exit 5 = at least one is not.
"""
import pathlib
import re
import sys

# The literal the `commit-msg` hook and `fulcrum candidates` search for. If either ever
# changes its pattern, change it HERE too and this check will prove the tree still matches.
CANONICAL = "FALSIFY"
# A RECORD is a comment that LEADS with the marker — that is what both mechanisms look
# for and what a reader treats as binding. A mid-sentence mention ("see the FALSIFIED note
# above") is prose pointing AT a record, not a record, so it is not flagged.
RECORD = re.compile(r"^\s*(?://|///|//!)\s*(?:\*\*)?FALSIF(?:IED|IES)\b")

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"


def main() -> int:
    bad = []
    for path in sorted(ROOT.rglob("*.rs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if RECORD.match(line) and CANONICAL not in line:
                bad.append((path, n, line.strip()[:100]))

    if not bad:
        print(f"FALSIFY-VISIBILITY=OK (every record in {ROOT} carries the canonical token)")
        return 0

    print(
        f"FALSIFY-VISIBILITY=FAIL — {len(bad)} line(s) name a falsification but do NOT\n"
        f"contain the literal '{CANONICAL}', so neither the commit-msg hook nor\n"
        f"`fulcrum candidates` can see them. A record the machinery cannot read is not a\n"
        f"record.\n",
        file=sys.stderr,
    )
    for path, n, line in bad:
        print(f"  {path.relative_to(ROOT.parent)}:{n}: {line}", file=sys.stderr)
    print(
        f"\nFix: make the line contain '{CANONICAL}' — e.g. 'FALSIFY <date> (FALSIFIED)'.",
        file=sys.stderr,
    )
    return 5


if __name__ == "__main__":
    sys.exit(main())
