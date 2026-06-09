#!/usr/bin/env bash
# incr_growth.sh — LOCAL wrapper for the incremental-output-growth footprint A/B
# (DIS-17 owed footprint falsifier). Bench-lock freeze, run the guest A/B at each
# T (default 4 8), release ALWAYS (lib_hostlock trap). Numbers valid ONLY
# bench-locked. Measures peak RSS / dTLB MPKI / page-faults / wall for
# gzippy-isal OFF (8x reserve) vs ON (incremental grow) vs rapidgzip, asserting
# 0 fallbacks on the ON arms.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"

REPS="${REPS:-6}"; N="${N:-7}"; TS="${TS:-4 8}"; FACTORS="${FACTORS:-4 2 1}"
ART="${ARTDIR_BASE}/incr"
REFSHA="$CORPUS_RAW_SHA256"
pin_mask() { case "$1" in 1) echo 0;; 4) echo "0,2,4,6";; 8) echo "0,2,4,6,8,10,12,14";; 16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";; esac; }

echo "=== ship guest A/B script ==="
timeout 60 scp -o ConnectTimeout=15 -J "$JUMP" "$HERE/_incr_growth_guest.sh" \
  "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/" >/dev/null

if hostlock_acquire; then echo "## QUIET — proceeding"; else echo "## WARN: not fully quiet — RELATIVE numbers still valid"; fi
HOST_FROZEN=1

for T in $TS; do
  MASK="$(pin_mask "$T")"
  echo ""
  echo "############### INCR-GROWTH T=$T ###############"
  timeout 900 $SSH_GUEST "cd '$GUEST_SRC' && chmod +x scripts/bench/_incr_growth_guest.sh && \
    GZ='$GZIPPY_BIN' RG='$RG' CORPUS='$CORPUS' REFSHA='$REFSHA' MASK='$MASK' T='$T' \
    REPS='$REPS' N='$N' ART='$ART' FACTORS='$FACTORS' GOV='$GOV' NO_TURBO='$NO_TURBO' \
    HOST_FROZEN='$HOST_FROZEN' sh scripts/bench/_incr_growth_guest.sh" 2>&1
done

hostlock_release
echo "INCR_GROWTH_DONE"
