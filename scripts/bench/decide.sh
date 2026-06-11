#!/usr/bin/env bash
# decide.sh — host-side driver for `fulcrum decide` (plans/fulcrum2-charter.md).
#
# ONE command: freeze the box, run every requested cell's wall interleave +
# trace/prof captures + in-tree kill-switch knob A/Bs on the guest, pull the
# artifacts back, and render the ONE ranked decision table (fulcrum_decide.py)
# ending in a DO-THIS-NEXT line.
#
# Usage:
#   scripts/bench/decide.sh [--bin /root/bin-p35-native] [--feature gzippy-native]
#       [--cells silesia:1,silesia:4,silesia:8,silesia:16,model:8]
#       [--knob-cells silesia:1,silesia:16] [--no-knobs]
#       [-N 9] [--knob-n 7] [--allow-thaw] [--no-lock] [--dry-run]
#       [--analyze-only <local-artifact-dir>]
#
# Discipline: host freeze via lib_hostlock.sh (bench-lock on neurotic), guest-side
# freeze readback + procs_running quiet gate, canonical pin masks, regular-file
# sinks, sha-verify EVERY measured run (any mismatch voids the cell). Counter/
# trace/prof captures are labeled unfrozen-counters (they are never wall numbers).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"
# shellcheck source=/dev/null
. "$HERE/lib_hostlock.sh"

BIN="${BIN:-/root/bin-p35-native}"
FEATURE="${FEATURE:-gzippy-native}"
CELLS="silesia:1,silesia:4,silesia:8,silesia:16,model:8"
KNOB_CELLS="silesia:1,silesia:16"
KNOB_FILTER=""
N=9; KNOB_N=7; DO_KNOBS=1; DO_LOCK=1; DRY=0; ALLOW_THAW=0
ANALYZE_ONLY=""
HOST_FROZEN="${HOST_FROZEN:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bin) BIN="$2"; shift;;
    --bin=*) BIN="${1#*=}";;
    --feature) FEATURE="$2"; shift;;
    --feature=*) FEATURE="${1#*=}";;
    --cells) CELLS="$2"; shift;;
    --cells=*) CELLS="${1#*=}";;
    --knob-cells) KNOB_CELLS="$2"; shift;;
    --knob-cells=*) KNOB_CELLS="${1#*=}";;
    --no-knobs) DO_KNOBS=0;;
    --knobs) KNOB_FILTER="$2"; shift;;
    --knobs=*) KNOB_FILTER="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --knob-n) KNOB_N="$2"; shift;;
    --knob-n=*) KNOB_N="${1#*=}";;
    --allow-thaw) ALLOW_THAW=1;;
    --no-lock) DO_LOCK=0;;
    --host-frozen) HOST_FROZEN=1;;
    --dry-run) DRY=1;;
    --analyze-only) ANALYZE_ONLY="$2"; shift;;
    --analyze-only=*) ANALYZE_ONLY="${1#*=}";;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "decide.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

RUNID="decide_$(date -u '+%Y%m%dT%H%M%SZ')"
LOCAL_ART="$ROOT/artifacts/decide/$RUNID"
ANALYZER=(python3 "$ROOT/scripts/fulcrum_decide.py")
[ "$ALLOW_THAW" = 1 ] && ANALYZER+=(--allow-thaw)

if [ -n "$ANALYZE_ONLY" ]; then
  exec "${ANALYZER[@]}" "$ANALYZE_ONLY"
fi

remote_env() {
  cat <<EOF
GUEST_SRC='$GUEST_SRC' BIN='$BIN' FEATURE='$FEATURE' CELLS='$CELLS' \
N='$N' KNOB_N='$KNOB_N' DO_KNOBS='$DO_KNOBS' KNOB_CELLS='$KNOB_CELLS' \
KNOB_FILTER='$KNOB_FILTER' \
ARTDIR='${ARTDIR_BASE}/decide' RUNID='$RUNID' RG='$RG' RG_TRACE='$RG_TRACE' \
GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' \
ALLOW_LOAD='${ALLOW_LOAD:-0}' CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256'
EOF
}

if [ "$DRY" = 1 ]; then
  echo "## DRY-RUN — plan only, nothing executed"
  echo "bin=$BIN feature=$FEATURE cells=$CELLS knobs=$DO_KNOBS knob_cells=$KNOB_CELLS N=$N knob_n=$KNOB_N"
  echo "guest artifacts: ${ARTDIR_BASE}/decide/$RUNID  ->  local $LOCAL_ART"
  echo "remote env: $(remote_env | tr '\n' ' ')"
  echo "runtime estimate: walls ~N*(gz+rg) per cell; knobs ~KNOB_N*2*wall per knob per knob-cell;"
  echo "  defaults on silesia+model fit well inside the 1800s freeze TTL."
  exit 0
fi

# Freeze BEFORE measuring; release ALWAYS (trap in lib_hostlock).
if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then
    echo "## host is QUIET — proceeding."
  else
    echo "## WARN: host freeze did not reach the quiet threshold."
    echo "##       RELATIVE ratios are jitter-immune; do NOT bank ABSOLUTE numbers."
  fi
  HOST_FROZEN=1
fi

echo "=== ship decide guest scripts -> $GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/ ==="
timeout 60 $SSH_GUEST "mkdir -p '$GUEST_SRC/scripts/bench'"
timeout 60 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -J "$JUMP" \
  "$HERE/_decide_guest.sh" "$HERE/lib_decide_guest.sh" \
  "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/"

echo "=== run decide on guest (cells=$CELLS knobs=$DO_KNOBS) ==="
set +e
timeout 1500 $SSH_GUEST \
  "cd '$GUEST_SRC' && chmod +x scripts/bench/_decide_guest.sh && $(remote_env) bash scripts/bench/_decide_guest.sh" \
  | tee /tmp/decide_guest_$RUNID.log
GRC=${PIPESTATUS[0]}
set -e
if ! grep -q 'DECIDE_GUEST_DONE' /tmp/decide_guest_$RUNID.log; then
  hostlock_release
  echo "decide.sh: guest runner did not complete (rc=$GRC) — see /tmp/decide_guest_$RUNID.log" >&2
  exit "${GRC:-1}"
fi

echo "=== pull artifacts -> $LOCAL_ART ==="
mkdir -p "$LOCAL_ART"
timeout 300 rsync -az -e "ssh -o ConnectTimeout=15 -J $JUMP" \
  "$GUEST_USER@$GUEST:${ARTDIR_BASE}/decide/$RUNID/" "$LOCAL_ART/"

hostlock_release

if grep -q 'DECIDE_FAIL=' /tmp/decide_guest_$RUNID.log; then
  echo "decide.sh: guest reported a hard failure:" >&2
  grep 'DECIDE_FAIL=' /tmp/decide_guest_$RUNID.log >&2
  exit 1
fi

echo "=== analyze ==="
"${ANALYZER[@]}" "$LOCAL_ART"
