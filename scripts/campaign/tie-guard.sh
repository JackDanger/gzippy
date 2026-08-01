#!/usr/bin/env bash
# tie-guard.sh — THE CHEAPEST FALSIFIER FOR ANY CHANGE THAT ALTERS T1 OUTPUT.
#
# WHY THIS EXISTS. We are BYTE-IDENTICAL to libdeflate on most T1 cells (154 of 198 on the
# frozen census). A tie PASSES the board (not bigger) but has ZERO tolerance: one byte in
# either direction on a tied file flips a passing cell to failing, and promotion clause 3
# forbids that absolutely.
#
# Two levers died on exactly this in one session, each after a full ~20 minute `fulcrum try`:
#   * hash3 chaining      6 closed / 12 tied cells flipped
#   * zlib good_match    31 closed / 17 tied cells flipped (data.csv L2 T1 1.0000 -> 1.0431)
# Both were NET T1 IMPROVEMENTS. Net-positive is not the bar — non-worse on EVERY tie is.
#
# A hand-picked 9-file sample of tied cells found ZERO flips for good_match and missed the two
# worst files entirely. Sampling does not work here; enumerate the ties.
#
# THIS IS A PRE-CHECK, NOT A VERDICT. It runs the tie subset only, in ~2 minutes instead of
# ~20, so a doomed change dies before the graded run. A PASS here does NOT authorise anything:
# `fulcrum try` remains the only adjudicator.
#
#   scripts/campaign/tie-guard.sh <after-ref> [--levels 2,6,9]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
AFTER="${1:?usage: tie-guard.sh <after-ref> [--levels L,L,L]}"; shift || true
LEVELS="2,6,9"; [ "${1:-}" = "--levels" ] && { LEVELS="$2"; }
CORPUS="${CAMPAIGN_CORPUS_ROOT:-$HOME/www/gzippy-bench/corpus}"
LD="$(command -v libdeflate-gzip || echo /opt/homebrew/bin/libdeflate-gzip)"
[ -x "$LD" ] || { echo "tie-guard: no libdeflate-gzip — the tie rival is unavailable" >&2; exit 2; }
git rev-parse --verify "$AFTER" >/dev/null 2>&1 || { echo "tie-guard: '$AFTER' is not a ref" >&2; exit 2; }

WT="$(mktemp -d)/after"
git worktree add -q --detach "$WT" "$AFTER" || exit 2
rm -rf "$WT/vendor" && ln -s "$PWD/vendor" "$WT/vendor"
( cd "$WT" && cargo build --release >/dev/null 2>&1 ) || { echo "tie-guard: after-arm build FAILED" >&2; exit 2; }
AFTER_BIN="$WT/target/release/gzippy"
# BASE MUST COME FROM A REF, NOT THE WORKING TREE. The first version of this guard used
# $PWD/target/release/gzippy, which is whatever the last `cargo build` produced — on a feature
# branch that is the AFTER arm, so base==after, every cell looks tied-or-equal and NOTHING can
# flip. It reported PASS for a change `fulcrum try` says flips 17 ties. A guard that cannot fail
# is worse than no guard. (CLAUDE.md: verify the tree and binary on EVERY measurement.)
BASE_REF="${CAMPAIGN_BASE:-origin/main}"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || { echo "tie-guard: base '$BASE_REF' is not a ref" >&2; exit 2; }
BWT="$(mktemp -d)/base"
git worktree add -q --detach "$BWT" "$BASE_REF" || exit 2
rm -rf "$BWT/vendor" && ln -s "$PWD/vendor" "$BWT/vendor"
( cd "$BWT" && cargo build --release >/dev/null 2>&1 ) || { echo "tie-guard: base-arm build FAILED" >&2; exit 2; }
BASE_BIN="$BWT/target/release/gzippy"
[ "$(shasum -a 256 "$BASE_BIN" | cut -d' ' -f1)" != "$(shasum -a 256 "$AFTER_BIN" | cut -d' ' -f1)" ] \
  || { echo "tie-guard: REFUSED — base and after binaries are identical (NO-OP or wrong refs)" >&2; exit 2; }

echo "TIE-GUARD  base=$BASE_REF  after=$AFTER  levels=$LEVELS  (cheapest falsifier; NOT a verdict)"
ties=0; flips=0; checked=0
for f in "$CORPUS"/*; do
  [ -f "$f" ] || continue
  for L in ${LEVELS//,/ }; do
    r=$("$LD" -"$L" -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')
    b=$("$BASE_BIN"  -"$L" -p1 -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')
    checked=$((checked+1))
    [ "$b" = "$r" ] || continue          # only TIED cells have zero tolerance
    ties=$((ties+1))
    a=$("$AFTER_BIN" -"$L" -p1 -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')
    if [ "$a" -gt "$r" ]; then
      flips=$((flips+1))
      printf "  FLIP  %-22s L%-2s  tied at %s -> %s  (+%s)\n" "$(basename "$f")" "$L" "$r" "$a" "$((a-r))"
    fi
  done
done
git worktree remove --force "$WT" >/dev/null 2>&1; git worktree remove --force "$BWT" >/dev/null 2>&1; git worktree prune
echo "  cells probed=$checked  TIED=$ties  FLIPPED=$flips"
[ "$flips" -eq 0 ] && { echo "  PASS — no tie flipped. NEXT: fulcrum try (this is a pre-check, not a verdict)."; exit 0; }
echo "  FAIL — clause 3 would refuse this. Fix before spending a graded run."; exit 1
