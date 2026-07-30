#!/usr/bin/env python3
"""Resolve a declared corpus set to filenames, or REFUSE and name what is missing.

Argv: <corpus_split.json> <tune|gate|all> <corpus_root>

Prints one filename per line on success. On any missing declared member it prints the
names to stderr and exits 3 — it never returns a smaller set.

WHY IT NEVER SHRINKS. The campaign plan's headline "165 failing cells" was measured on
"17 files staged of the 20 canonical members". The three absent members were never named,
so nobody could tell whether the missing rows were the interesting ones. A census that
quietly covers less than it claims is worse than one that refuses: the number it prints
still gets quoted.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    split_path, set_name, root = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(split_path) as fh:
        split = json.load(fh)

    if set_name == "all":
        declared = list(split["gate"]["files"]) + list(split["tune"]["files"])
    else:
        declared = list(split[set_name]["files"])

    seen, ordered = set(), []
    for name in declared:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    missing = [n for n in ordered if not os.path.isfile(os.path.join(root, n))]
    if missing:
        print(
            f"CAMPAIGN REFUSES: corpus set '{set_name}' declares {len(ordered)} members; "
            f"{len(missing)} are absent from {root}:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  MISSING  {name}", file=sys.stderr)
        print(
            "\nA census must not silently cover less than it claims. Either stage the\n"
            "missing members or amend corpus_split.json in a commit that says why.\n"
            "Note: goal::MIN_CORPORA members are a subset of GATE by construction, so a\n"
            "missing GATE member can remove the mandated minimum surface.",
            file=sys.stderr,
        )
        return 3

    for name in ordered:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
