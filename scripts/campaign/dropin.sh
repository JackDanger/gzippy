#!/usr/bin/env bash
# dropin.sh — the THIRD goal axis, as a gate.
#
# Size and wall have censuses. CLI BEHAVIOUR did not, and the charter says so:
# "`dropin` is the THIRD goal axis (CLI behaviour) and went unmeasured for the
# whole campaign." A board can be 100% green on level x rival x corpus x threads
# while `gzip --rsyncable file` exits 2 where gzip exits 0.
#
# First run, 2026-08-22, main @ 15bdcc85: 208 cells, 182 MATCH, 26 DIVERGENT.
# Of those 26, EIGHTEEN were pigz-habit differences where we correctly follow
# gzip's contract (non-negotiable 4: cite a contract, never a vendor's habit) —
# each hand-verified against gzip directly and declared in
# `dropin-declared.json` with its reason. The remaining EIGHT are one defect:
# `--rsyncable`, which gzip AND pigz accept (exit 0) and we reject (exit 2).
#
# So the axis reduces to a single live defect. Keep it that way: every new
# DIVERGENT cell is a bug to fix or an exception to declare WITH A REASON.
# `fulcrum dropin` exits nonzero iff any cell is DIVERGENT or ERROR.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
OURS="${1:-$PWD/target/release/gzippy}"
OUT="${2:-${TMPDIR:-/tmp}/dropin-$(git rev-parse --short HEAD)}"
CORPUS="${CAMPAIGN_CORPUS_ROOT:-$HOME/www/gzippy-bench/corpus}"
[ -x "$OURS" ] || { echo "dropin: no binary at $OURS — cargo build --release first" >&2; exit 2; }
command -v fulcrum >/dev/null || { echo "dropin: fulcrum not on PATH" >&2; exit 2; }
exec fulcrum dropin \
  --ours "$OURS" \
  --rival gzip=gzip --rival pigz=pigz \
  --fixture "$CORPUS/dickens" \
  --fixture "$CORPUS/engine.wasm" \
  --declared "$PWD/scripts/campaign/dropin-declared.json" \
  --out "$OUT"
