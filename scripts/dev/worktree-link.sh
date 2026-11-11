#!/bin/bash
# Fresh `git worktree`s do not materialize submodules, and without these two
# vendored paths the build fails (`isal-sys` needs its patch path, the ISA-L
# FFI needs vendor/isa-l). Link them from the MAIN checkout into a worktree:
#
#   scripts/dev/worktree-link.sh ../w-my-change    # relative or absolute
#   scripts/dev/worktree-link.sh                    # default: $PWD
#
# Idempotent; skips paths that already exist as real (populated) checkouts.
# A fresh worktree's submodule paths can materialize as EMPTY directories —
# git creates the path but does not populate it. An empty directory is treated
# as un-materialized (it is replaced by the link), not skipped: `isal-sys` is
# a path-patch crate, so leaving an empty `vendor/isa-l` behind breaks cargo.
set -euo pipefail

WT="${1:-$PWD}"
if ! GIT_COMMON=$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null); then
    echo "error: $WT is not a git worktree or repository" >&2
    exit 1
fi
ROOT=$(cd "$GIT_COMMON/.." && pwd)

mkdir -p "$WT/vendor/isal-rs"
for tgt in "$WT/vendor/isal-rs/isal-sys" "$WT/vendor/isa-l"; do
    if [ -e "$tgt" ] && [ ! -L "$tgt" ] && [ -n "$(ls -A "$tgt" 2>/dev/null)" ]; then
        echo "skip: $tgt already materialized (submodule checkout?) — not replacing" >&2
    else
        if [ "$tgt" = "$WT/vendor/isal-rs/isal-sys" ]; then
            src="$ROOT/vendor/isal-rs/isal-sys"
        else
            src="$ROOT/vendor/isa-l"
        fi
        # `ln -sfn` into a REAL directory would nest the link INSIDE it, so an
        # empty leftover dir must be rmdir'd first (rmdir refuses a populated
        # dir, which never reaches this branch).
        if [ -d "$tgt" ] && [ ! -L "$tgt" ]; then
            rmdir "$tgt" 2>/dev/null || true
        fi
        ln -sfn "$src" "$tgt"
        echo "linked $tgt -> $src"
    fi
done
