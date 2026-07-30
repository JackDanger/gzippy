#!/usr/bin/env bash
# board-size.sh — the SIZE axis of the per-label board, in one command.
#
# Size is an exact integer byte count: deterministic, arch-invariant (verified across
# aarch64/Zen2/Intel), immune to box load, and roundtrip-VOIDed by the census itself so a
# corrupt-but-smaller output can never score as a win. That makes this the CHEAPEST
# FALSIFIER for almost every encoder lever, and CLAUDE.md says to run the cheapest
# falsifier first. Run this before any wall work.
#
#   scripts/campaign/board-size.sh                    # tune set, L1-9, T1+T4
#   CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all   # the promotion board
#
# Args: [tune|gate|all]   env: CAMPAIGN_LEVELS, CAMPAIGN_THREADS, CAMPAIGN_OUT
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

SET_NAME="${1:-tune}"
LEVELS="${CAMPAIGN_LEVELS:-1-9}"
THREADS="${CAMPAIGN_THREADS:-1,4}"

campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"
campaign_rival_args

GZ="$CAMPAIGN_REPO/target/release/gzippy"
[ -x "$GZ" ] || die "no gzippy at $GZ" "build it: (cd $CAMPAIGN_REPO && cargo build --release)"
GZ_SHA="$(shasum -a 256 "$GZ" | cut -d' ' -f1)"

OUT="${CAMPAIGN_OUT:-$(campaign_outdir "size-${SET_NAME}-$(git -C "$CAMPAIGN_REPO" rev-parse --short HEAD)")}"
note "binary" "gzippy sha256=${GZ_SHA:0:16} levels=$LEVELS threads=$THREADS"
note "out" "$OUT"

# -p {threads} is passed EXPLICITLY at every cell. Receipt: without an explicit -pN,
# gzippy silently uses T=num_cpus, and an agent once measured T10 believing it was T1.
set -x
exec "$CAMPAIGN_FULCRUM" board size \
  --ours "$GZ -{level} -p {threads} -c {input}" \
  "${CAMPAIGN_RIVAL_ARGS[@]}" \
  --levels "$LEVELS" \
  --threads "$THREADS" \
  "${CAMPAIGN_CORPUS_ARGS[@]}" \
  --roundtrip-cmd 'gzip -dc' \
  --ours-commit "$CAMPAIGN_GZIPPY_SHA" \
  --out "$OUT"
