#!/usr/bin/env bash
# residual.sh — host-side driver for the structural-residual sizing oracle (P-owner
# owner/structural-residual-sizing). Brackets _residual_guest.sh with the bench-lock
# host freeze (pause noisy LXCs, no_turbo=1, quiet-gate) and the same double-hop as
# parity.sh/oracle.sh. Reuses guest.env pins so it can never disagree about WHERE.
#
# It does NOT build — run `scripts/bench/parity.sh --build --feature gzippy-isal`
# FIRST (that rsyncs the worktree incl. this tooling AND stamps the stale-binary
# fingerprint the guest runner verifies). Then:
#   scripts/bench/residual.sh --mode output-floor   -T 1 -N 15
#   scripts/bench/residual.sh --mode output-floor   -T 4 -N 15
#   scripts/bench/residual.sh --mode marker-perturb -T 4 -N 11 --slow-pct 50
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"

MODE=""; T=4; N=15; SLOW_PCT=50; DO_LOCK=1; HOST_FROZEN="${HOST_FROZEN:-0}"; ALLOW_LOAD="${ALLOW_LOAD:-0}"
while [ "$#" -gt 0 ]; do case "$1" in
  --mode) MODE="$2"; shift;; --mode=*) MODE="${1#*=}";;
  -T) T="$2"; shift;; -T*) T="${1#-T}";;
  -N) N="$2"; shift;; -N*) N="${1#-N}";;
  --slow-pct) SLOW_PCT="$2"; shift;; --slow-pct=*) SLOW_PCT="${1#*=}";;
  --no-lock) DO_LOCK=0;; --host-frozen) HOST_FROZEN=1;; --allow-load) ALLOW_LOAD=1;;
  *) echo "residual.sh: unknown arg '$1'" >&2; exit 2;;
esac; shift; done
[ -n "$MODE" ] || { echo "residual.sh: --mode {output-floor|marker-perturb} required" >&2; exit 2; }

pin_mask() { case "$1" in 1) echo 0;; 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;;
  16) echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;; *) echo "";; esac; }
MASK="$(pin_mask "$T")"; [ -n "$MASK" ] || { echo "residual.sh: unsupported -T $T" >&2; exit 2; }
ARTDIR="${ARTDIR_BASE}/residual-$MODE-T$T"

if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then echo "## host is QUIET — proceeding.";
  else echo "## WARN: host did not reach quiet — trust RATIO only, not ABSOLUTE."; ALLOW_LOAD=1; fi
  HOST_FROZEN=1
fi

REMOTE_ENV="GUEST_SRC='$GUEST_SRC' GZIPPY_BIN='$GZIPPY_BIN' CORPUS='$CORPUS' \
CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256' RG='$RG' RG_TRACE='$RG_TRACE' \
T='$T' N='$N' MASK='$MASK' GOV='$GOV' NO_TURBO='$NO_TURBO' MODE='$MODE' SLOW_PCT='$SLOW_PCT' \
HOST_FROZEN='$HOST_FROZEN' ALLOW_LOAD='$ALLOW_LOAD' ARTDIR='$ARTDIR'"

echo "=== residual run mode=$MODE T=$T N=$N slow_pct=$SLOW_PCT mask=$MASK ==="
# shellcheck disable=SC2029
$SSH_GUEST "mkdir -p '$ARTDIR'; env $REMOTE_ENV bash '$GUEST_SRC/scripts/bench/_residual_guest.sh'"
rc=$?
[ "$DO_LOCK" = 1 ] && hostlock_release
exit $rc
