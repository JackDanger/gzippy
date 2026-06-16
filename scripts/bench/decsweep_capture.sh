#!/usr/bin/env bash
# decsweep_capture.sh — mac-side bracket for the STEP-13 decoded-size adaptive
# chunk-policy wall+P sweep. Sibling of chunksweep_capture.sh; sweeps the DECODED
# target (GZIPPY_TARGET_DECODED_KIB) on a pre-built ADAPTIVE binary instead of
# the fixed compressed spacing.
#
#   ADAPTIVE_BIN=/dev/shm/gz-adaptive-target/release/gzippy \
#   scripts/bench/decsweep_capture.sh --t-list "4 7" -N 9 \
#       --corpus silesia [--dkibs "4096 5120 6144 8192"] [--no-lock]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"
# shellcheck source=/dev/null
. "$HERE/lib_hostlock.sh"

T_LIST="4"; N=9; CORPUS_NAME="silesia"; DO_LOCK=1
DKIBS="4096 5120 6144 8192"
ADAPTIVE_BIN="${ADAPTIVE_BIN:-$GZIPPY_BIN}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --t) T_LIST="$2"; shift;;
    --t-list) T_LIST="$2"; shift;;
    -N) N="$2"; shift;;
    --corpus) CORPUS_NAME="$2"; shift;;
    --dkibs) DKIBS="$2"; shift;;
    --bin) ADAPTIVE_BIN="$2"; shift;;
    --no-lock) DO_LOCK=0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
  shift
done

if [ "$CORPUS_NAME" = "silesia" ]; then CORPUS_PATH="$CORPUS"; OVSHA="$CORPUS_RAW_SHA256";
else CORPUS_PATH="/root/$CORPUS_NAME.gz"; OVSHA=""; fi

HOST_FROZEN=0
if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then echo "## host QUIET"; else echo "## WARN: freeze not quiet — ratios survive."; fi
  HOST_FROZEN=1
fi

echo "=== ship guest scripts ==="
timeout 60 $SSH_GUEST "mkdir -p '$GUEST_SRC/scripts/bench'"
timeout 60 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -J "$JUMP" \
  "$HERE/_decsweep_guest.sh" "$HERE/lib_decide_guest.sh" \
  "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/"

for T in $T_LIST; do
  RUNID="decsweep_${CORPUS_NAME}_t${T}_$(date -u '+%Y%m%dT%H%M%SZ')"
  LOCAL_ART="$ROOT/artifacts/decsweep/$RUNID"
  echo "=== run sweep corpus=$CORPUS_NAME T=$T N=$N bin=$ADAPTIVE_BIN ==="
  set +e
  timeout 1800 $SSH_GUEST \
    "cd '$GUEST_SRC' && chmod +x scripts/bench/_decsweep_guest.sh && \
     GZIPPY_BIN='$ADAPTIVE_BIN' RG='$RG' CORPUS='$CORPUS_PATH' T='$T' N='$N' \
     ARTDIR='${ARTDIR_BASE}/decsweep' RUNID='$RUNID' DKIBS='$DKIBS' \
     GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' \
     CORPUS_RAW_SHA256='$OVSHA' \
     bash scripts/bench/_decsweep_guest.sh" \
    | tee "/tmp/decsweep_$RUNID.log"
  GRC=${PIPESTATUS[0]}
  set -e
  if ! grep -q 'DECSWEEP_GUEST_DONE' "/tmp/decsweep_$RUNID.log"; then
    hostlock_release
    echo "decsweep: guest did not complete (rc=$GRC) — see /tmp/decsweep_$RUNID.log" >&2
    exit "${GRC:-1}"
  fi
  if grep -q 'DECIDE_FAIL=' "/tmp/decsweep_$RUNID.log"; then
    hostlock_release
    grep 'DECIDE_FAIL=' "/tmp/decsweep_$RUNID.log" >&2
    exit 1
  fi
  mkdir -p "$LOCAL_ART"
  timeout 300 rsync -az -e "ssh -o ConnectTimeout=15 -J $JUMP" \
    "$GUEST_USER@$GUEST:${ARTDIR_BASE}/decsweep/$RUNID/" "$LOCAL_ART/"
  echo "=== chunk-count attestation T=$T ==="
  cat "$LOCAL_ART/chunkcounts.txt" 2>/dev/null || true
  echo "=== analyze T=$T ==="
  python3 "$HERE/chunksweep_analyze.py" "$LOCAL_ART"
  echo "ARTIFACTS: $LOCAL_ART"
done

hostlock_release
