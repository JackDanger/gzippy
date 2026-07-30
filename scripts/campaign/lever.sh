#!/usr/bin/env bash
# lever.sh — judge a change. ONE command, the promotion rule applied clause by clause.
#
#   scripts/campaign/lever.sh <after-ref> [--base origin/main] [extra fulcrum try args]
#   scripts/campaign/lever.sh perf/blockend-recalibrate --size-only
#
# WHY THIS WRAPPER EXISTS. `fulcrum try` already implements docs/promotion-rule.md in code:
# it builds both arms from git refs in throwaway worktrees (so a stale control is
# impossible), refuses NO-OPs by binary hash (clause 2), verifies roundtrip correctness
# (clause 1), runs size + paired wall censuses on both arms, REFUSES single-level verdicts
# (hard stop #3: never generalise across levels), and emits SHIP / NO-SHIP(clause+numbers)
# / UNDECIDED(what to re-run).
#
# The campaign nevertheless kept hand-writing its falsifiers. The campaign plan's §5
# "Falsifier" paragraph is a prose re-description of this command, and the two block-budget
# FALSIFY records were built from a hand-typed three-file table on files that are not
# declared corpus members — which is how a binding falsification came to rest on evidence
# nobody can re-run. `CLAUDE.md` hard stop #6 already says "Never hand-roll a measurement.
# Check Fulcrum's command list first"; this wrapper removes the excuse by making the
# graded path shorter to type than the hand-rolled one.
#
# CHEAPEST FALSIFIER FIRST (CLAUDE.md): pass --size-only. Size is an exact integer,
# deterministic, arch-invariant and roundtrip-VOIDed, so it costs one build and no rig. If
# a lever dies on size it never needed a wall run.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

[ $# -ge 1 ] || die "usage: $0 <after-ref> [--base <ref>] [fulcrum try args...]" \
  "cheapest falsifier first:  $0 <after-ref> --size-only"
AFTER="$1"; shift

git -C "$CAMPAIGN_REPO" rev-parse --verify "$AFTER" >/dev/null 2>&1 \
  || die "'$AFTER' is not a ref in $CAMPAIGN_REPO" "levers are judged from git refs, never from a dirty tree"

# Exploration runs on TUNE. A SHIP verdict is judged on GATE, and corpus_split.json is
# explicit that fitting on GATE voids the promotion regardless of the numbers — so the set
# is chosen by CAMPAIGN_PROMOTE, never silently.
SET_NAME="${CAMPAIGN_SET:-tune}"
[ "${CAMPAIGN_PROMOTE:-0}" != 1 ] || SET_NAME="${CAMPAIGN_SET:-all}"
campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"
campaign_rival_args

# Shallow AND deep, always. Hard stop #3: "Never generalise a measurement across levels" —
# measuring L2 alone and generalising shipped a 6.2% L6 and a 9.9% L9 regression. fulcrum
# try refuses a level set that does not span shallow<=4 and deep>=6; this default satisfies
# it rather than discovering the refusal.
LEVELS="${CAMPAIGN_LEVELS:-2,6,9}"
OUT="${CAMPAIGN_OUT:-$(campaign_outdir "lever-$(echo "$AFTER" | tr '/' '-')")}"

note "lever" "after=$AFTER set=$SET_NAME levels=$LEVELS"
note "out" "$OUT"
note "verdict" "SHIP / NO-SHIP(clause) / UNDECIDED — from docs/promotion-rule.md, applied by fulcrum"

set -x
exec "$CAMPAIGN_FULCRUM" try "$AFTER" \
  --repo "$CAMPAIGN_REPO" \
  "${CAMPAIGN_RIVAL_ARGS[@]}" \
  "${CAMPAIGN_CORPUS_ARGS[@]}" \
  --levels "$LEVELS" \
  --out "$OUT" \
  "$@"
