#!/usr/bin/env bash
# perf_attr.sh — LOCAL wrapper for the +40%-instruction ATTRIBUTION (STEP 1).
# rsync the worktree -> guest, bench-lock freeze, run _perf_attr_guest.sh (build
# symboled gzippy-isal + perf record/stat/annotate), release. Numbers valid only
# bench-locked; release guaranteed by the lib_hostlock EXIT trap.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"

T="${T:-4}"; REPS="${REPS:-6}"; PERIOD="${PERIOD:-300000}"
ART="${ARTDIR_BASE}/perfattr"
REFSHA="$CORPUS_RAW_SHA256"
pin_mask() { case "$1" in 1) echo 0;; 4) echo "0,2,4,6";; 8) echo "0,2,4,6,8,10,12,14";; *) echo "0,2,4,6";; esac; }
MASK="$(pin_mask "$T")"
DO_SYNC="${DO_SYNC:-1}"

RSYNC_PATHS=(src crates examples build.rs Cargo.toml Cargo.lock benches scripts vendor)

if [ "$DO_SYNC" = 1 ]; then
  echo "=== rsync worktree -> $GUEST_USER@$GUEST:$GUEST_SRC (via -J $JUMP) ==="
  timeout 30 $SSH_GUEST "mkdir -p '$GUEST_SRC'"
  # shellcheck disable=SC2086
  timeout 600 rsync -az --delete --exclude 'target/' --exclude '.git/' \
    -e "ssh -o ConnectTimeout=15 -J $JUMP" \
    "${RSYNC_PATHS[@]/#/$ROOT/}" "$GUEST_USER@$GUEST:$GUEST_SRC/"
else
  echo "=== ship guest perf-attr script only (no full sync) ==="
  timeout 60 scp -o ConnectTimeout=15 -J "$JUMP" "$HERE/_perf_attr_guest.sh" \
    "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/" >/dev/null
fi

if hostlock_acquire; then echo "## QUIET — proceeding"; else echo "## WARN: not fully quiet"; fi

echo ""
echo "########################## PERF-ATTR T=$T ##########################"
timeout 1200 $SSH_GUEST "cd '$GUEST_SRC' && chmod +x scripts/bench/_perf_attr_guest.sh scripts/cargo-lock.sh 2>/dev/null; \
  GUEST_SRC='$GUEST_SRC' CORPUS='$CORPUS' RG='$RG' MASK='$MASK' T='$T' REPS='$REPS' \
  PERIOD='$PERIOD' ART='$ART' REFSHA='$REFSHA' CARGO_LOCK='scripts/cargo-lock.sh' \
  sh scripts/bench/_perf_attr_guest.sh" 2>&1

hostlock_release
echo "PERF_ATTR_DONE"
