#!/usr/bin/env bash
# run_marker_probe.sh — mac-side driver: freeze neurotic, rsync tree to guest,
# build gzippy-native, run the MARKER-mode causal-perturbation probe, release.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"

GMSRC=/mnt/internal/gz-marker
DO_BUILD=${DO_BUILD:-1}

# Freeze the box (release ALWAYS via the lib_hostlock EXIT trap).
if hostlock_acquire; then echo "## host QUIET"; else echo "## WARN host not fully quiet — RATIO/slope still valid, ABSOLUTE not bankable"; fi

echo "=== rsync tree -> $GUEST:$GMSRC (excl target/.git/vendor) ==="
timeout 30 $SSH_GUEST "mkdir -p '$GMSRC'"
timeout 600 rsync -az --delete --exclude 'target/' --exclude '.git/' --exclude 'vendor/' \
  -e "ssh -o ConnectTimeout=15 -J $JUMP" \
  "$ROOT/src" "$ROOT/build.rs" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" \
  "$ROOT/scripts" \
  "$GUEST_USER@$GUEST:$GMSRC/"

echo "=== run marker probe on guest ==="
timeout 1500 $SSH_GUEST \
  "cd '$GMSRC'; chmod +x scripts/bench/_marker_probe_guest.sh; \
   DO_BUILD='$DO_BUILD' SRC='$GMSRC' N='${N:-8}' T='${T:-4}' \
   RG='$RG_BIN' bash scripts/bench/_marker_probe_guest.sh"
RC=$?
hostlock_release
exit "$RC"
