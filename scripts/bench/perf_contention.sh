#!/usr/bin/env bash
# perf_contention.sh — STEP 1 local wrapper: bench-lock freeze, then run the
# guest perf-contention characterization at T4 and T8, then release. Numbers are
# only valid bench-locked; release is guaranteed by the lib_hostlock trap.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"

REPS="${REPS:-6}"
TS="${TS:-4 8}"
ART="${ARTDIR_BASE}/perfc"

pin_mask() { case "$1" in 1) echo 0;; 4) echo "0,2,4,6";; 8) echo "0,2,4,6,8,10,12,14";; 16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";; esac; }

# decompressed-corpus sha pin (correctness oracle)
REFSHA="$CORPUS_RAW_SHA256"

echo "=== ship guest perf script ==="
timeout 60 scp -o ConnectTimeout=15 -J "$JUMP" "$HERE/_perf_contention_guest.sh" \
  "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/" >/dev/null

if hostlock_acquire; then echo "## QUIET — proceeding"; else echo "## WARN: not fully quiet — ratios still valid"; fi

for T in $TS; do
  MASK="$(pin_mask "$T")"
  echo ""
  echo "########################## RUN T=$T ##########################"
  timeout 600 $SSH_GUEST "cd '$GUEST_SRC' && chmod +x scripts/bench/_perf_contention_guest.sh && \
    GZ='$GZIPPY_BIN' RG='$RG' CORPUS='$CORPUS' MASK='$MASK' T='$T' REPS='$REPS' \
    ART='$ART' REFSHA='$REFSHA' sh scripts/bench/_perf_contention_guest.sh" 2>&1
done

hostlock_release
echo "PERF_CONTENTION_DONE"
