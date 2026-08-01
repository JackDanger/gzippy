#!/usr/bin/env python3
"""Print a one-line index of every FALSIFY record in `src/`.

WHY THIS EXISTS
---------------
Hard stop #2 says a FALSIFY note is BINDING. `scripts/commit-msg` enforces that
for CODE — it blocks a commit whose diff lands within 40 lines of a record
unless the message carries `REOPEN:`/`RESTORE:`. Nothing enforced it for PROSE,
and on 2026-08-01 that gap cost a full lever cycle: a census re-derived the
"109 T4 cells have ZERO headroom" fact that was ALREADY written verbatim inside
`huffman/fast.rs`'s FALSIFY record, and then prescribed that record's own
already-measured lever (exact package-merge as a costed dual candidate) into
CLAUDE.md STEP 2 as the route forward. It had to be retracted in the next commit.

A prose-matching hook would fire on every doc commit, because the docs discuss
these mechanisms constantly. The real cost was never enforcement — it was that
CHECKING was expensive: the records live in 12 files and run to 80 lines each,
so "what has already been tried?" meant reading them all. This makes that
question cost two seconds, which is the only version of the check that actually
gets run.

    make falsified            # every record, one line each
    make falsified Q=huffman  # only records whose file or text matches Q

Matches the STEM `FALSIF(Y|IED|IES)`, for the reason `scripts/commit-msg`
documents: a 2026-07-30 audit found the session's most important records written
as "FALSIFIED <date>", which `grep -q FALSIFY` does not match. Two whole files
were invisible to both the hook and `fulcrum candidates`.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
STEM = re.compile(r"FALSIF(?:Y|IED|IES)")
# Leading comment punctuation: `//`, `///`, `//!`, `#`, and any `*** ` banner.
LEAD = re.compile(r"^\s*(?://[/!]?|#)\s*|^\s*\*+\s*")


def headline(lines, i):
    """The record's first sentence, folded from however many lines it spans."""
    parts = []
    for raw in lines[i : i + 6]:
        text = LEAD.sub("", raw).strip().strip("*").strip()
        if not text:
            break
        parts.append(text)
        if text.endswith(".") and len(" ".join(parts)) > 40:
            break
    one = " ".join(parts)
    one = re.sub(r"\s+", " ", one)
    return one[:150]


def main():
    query = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    found = []
    for dirpath, _, names in os.walk(SRC):
        for name in sorted(names):
            if not name.endswith(".rs"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
            for i, line in enumerate(lines):
                if not STEM.search(line):
                    continue
                # Skip the checker's own vocabulary and cross-references, which
                # would otherwise triple the index: only a line that OPENS a
                # record (the stem within its first 40 chars) counts.
                if STEM.search(line[:40]) is None:
                    continue
                text = headline(lines, i)
                if query and query not in text.lower() and query not in rel.lower():
                    continue
                found.append((rel, i + 1, text))

    if not found:
        print("no FALSIFY records matched %r" % query if query else "no FALSIFY records")
        return 1

    width = max(len("%s:%d" % (f, n)) for f, n, _ in found)
    last = None
    for rel, num, text in found:
        if rel != last:
            print()
            last = rel
        print("  %-*s  %s" % (width, "%s:%d" % (rel, num), text))
    print("\n%d records. A FALSIFY note is BINDING (hard stop #2): touching one needs a" % len(found))
    print("REOPEN: line naming a NEW mechanism. Prose that PRESCRIBES one needs the same check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
