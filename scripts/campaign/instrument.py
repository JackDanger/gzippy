#!/usr/bin/env python3
"""Refuse to measure with an unidentified instrument. Reads `fulcrum version --json` on stdin.

CLAUDE.md hard stop #7: "A measurement from an unidentified binary is not a measurement."
Hard stop #6: "If the tool is missing on a box, FIX THE BOX — a stale instrument set is
what produced the substitute."

Receipt for this file: the session that wrote it opened with fulcrum refusing every
measurement command because the deployed binary was DIRTY and one commit behind
origin/main. That refusal is correct, but on its own it only stops the good path — the
hand-rolled three-file table is still available, and that is exactly what two binding
FALSIFY records were built from. So the fix must also say what to run.

Exit 0 = safe to measure. Exit 4 = refused, with the repair command printed.
"""
import json
import sys


def main() -> int:
    try:
        info = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"CAMPAIGN REFUSES: could not parse `fulcrum version --json`: {exc}", file=sys.stderr)
        return 4

    commit = info.get("commit") or ""
    dirty = bool(info.get("dirty"))
    src = info.get("src_dir") or "~/www/fulcrum"

    if dirty:
        print(
            "CAMPAIGN REFUSES: the fulcrum binary is a DIRTY build.\n"
            f"  commit: {commit[:12]}-dirty\n"
            "Artifacts from a dirty instrument cannot be reproduced, and fulcrum stamps\n"
            "them '-dirty' precisely so they cannot be banked.\n"
            f"  repair:  (cd {src} && git status --short && cargo build --release)",
            file=sys.stderr,
        )
        return 4

    if not commit:
        print("CAMPAIGN REFUSES: fulcrum reported no commit — unidentified binary.", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
