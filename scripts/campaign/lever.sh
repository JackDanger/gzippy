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

# G2, before ANYTHING else spends box time: the full local test suite, seconds, no box.
# See lib.sh campaign_test_gate's doc comment for the receipt (c8bbde67).
campaign_test_gate "$AFTER"

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
#
# ⚠ THIS SAMPLES 3 OF THE 9 LEVELS THAT CARRY FAILURES, AND CLAUSE 3 IS BLIND OUTSIDE THEM.
# That is not merely a gap in coverage — it fails the promotion rule's OWN Scope clause,
# `docs/promotion-rule.md:62-66`: "The board is per-label ... over the corpus. A change
# evaluated on a NARROWER SLICE HAS NOT BEEN EVALUATED; say so rather than generalising
# from it." Running the full-level SIZE leg is therefore the rule AS WRITTEN, not a
# stricter standard.
# Receipt, 2026-08-01: PR #227 scales max_search_depth in `params_parallel`, which applies
# at EVERY level, and was gated at 2,6,9. It flipped libdeflate:engine.wasm:L8:{T2,T4}:size
# from PASS to FAIL (libdeflate 396,254 B; ours 396,096 -> 396,302) and the gate could not
# see the cell. The same blind spot UNDERCOUNTS the benefit: 21 of that change's 30 T4
# closures are at L5/L7/L8. An 11-hour frozen-box run would have returned SHIP on a
# clause-3 violation.
#
# The SIZE census is deterministic and cheap and `board-size.sh` already defaults to 1-9.
# The wall census is what makes a full-level sweep expensive. So: keep the WALL at a
# sampled set, and run the SIZE leg across ALL levels before trusting any verdict.
LEVELS="${CAMPAIGN_LEVELS:-2,6,9}"

case ",$LEVELS," in
  *,1-9,*|*"1-9"*) ;;
  *)
    note "levels" "SCOPE VIOLATION: promotion-rule.md:62-66 says a change evaluated on"
    note "levels" "  a NARROWER SLICE HAS NOT BEEN EVALUATED. This grades $LEVELS —"
    note "levels" "  3 of the 9 levels that carry failures. Clause 3 is BLIND at the rest."
    note "levels" "params_parallel-style changes act at EVERY level. Before trusting this"
    note "levels" "verdict, run the FULL-level size leg (it is deterministic and cheap):"
    note "levels" "    CAMPAIGN_LEVELS=1-9 make board-size          # this ref"
    note "levels" "and diff its census against main's. Receipt: #227, engine.wasm L8."
    ;;
esac
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
