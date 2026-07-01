#!/usr/bin/env bash
# standing.sh — THE ONE-COMMAND GROUND-TRUTH RIG.
#
#   "Where does gzippy-native stand vs the state of the art?"  ->  one invocation.
#
# Builds gzippy-native (pure-Rust, C-FFI OFF the decode graph) at a pinned sha ON
# THE GUEST, GATE-0 self-validates the binary+box, then runs an interleaved
# best-of-N matrix vs rapidgzip (T>1 SOTA) and igzip (T1 SOTA) and prints ONE
# gated table with a FORK verdict. Replaces the ad-hoc per-night bench scripts as
# the canonical standing-status tool.
#
# Usage:
#   scripts/bench/standing/standing.sh                       # HEAD of kernel-converge-A
#   scripts/bench/standing/standing.sh --sha <sha>
#   scripts/bench/standing/standing.sh --corpora "silesia monorepo" --threads "1 2 4 8" -N 13
#   scripts/bench/standing/standing.sh --no-build            # reuse the last-built binary
#   scripts/bench/standing/standing.sh --analyze-only <local-art-dir>
#
# Self-tests it enforces (a run that fails any prints FAIL and emits no number):
#   build-flavor == parallel-sm+pure (FFI-off native build proven)
#   built sha == requested sha
#   path=ParallelSM asserted per corpus (production routing)
#   every arm sha-verified == zcat (non-inert, correct)
#   rapidgzip & igzip present; same /dev/null sink for all arms
#   A/A self-tests (rg-vs-rg, gz-vs-gz) ~= 1.0 license trusting ratios on a loaded box
#   GATE-1: interleaved best-of-N>=13; Delta<spread => TIE
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/scripts/bench/guest.env"

BOX="${BOX:-neurotic}"
SHA="${SHA:-}"
CORPORA="silesia monorepo nasa"
THREADS="1 2 4 8"
N=13
BUILD=1
ANALYZE_ONLY=""
GUEST_GZ_SRC="${STANDING_GZ_SRC:-/mnt/internal/gz-head}"
GUEST_STAGE=/root/standing            # stable (survives the in-tree git reset)
RUNID="standing_$(date -u '+%Y%m%dT%H%M%SZ')"
LOCAL_ART="$ROOT/artifacts/standing/$RUNID"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --box) BOX="$2"; shift;;
    --box=*) BOX="${1#*=}";;
    --sha) SHA="$2"; shift;;
    --sha=*) SHA="${1#*=}";;
    --corpora) CORPORA="$2"; shift;;
    --corpora=*) CORPORA="${1#*=}";;
    --threads) THREADS="$2"; shift;;
    --threads=*) THREADS="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --no-build) BUILD=0;;
    --analyze-only) ANALYZE_ONLY="$2"; shift;;
    --analyze-only=*) ANALYZE_ONLY="${1#*=}";;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "standing.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

if [ -n "$ANALYZE_ONLY" ]; then
  exec python3 "$HERE/standing_report.py" "$ANALYZE_ONLY"
fi

# --- per-box abstraction (default neurotic == identical to the prior behaviour) --
# shellcheck source=/dev/null
BOX="$BOX" . "$ROOT/scripts/bench/boxes.sh"
SSH_GUEST="$BOX_SSH"; JUMP="$BOX_JUMP"; GUEST="${BOX_GUEST#*@}"; GUEST_USER="${BOX_GUEST%@*}"
RG="$BOX_RG"; COMP_IGZIP="$BOX_IGZIP"; GUEST_GZ_SRC="$BOX_SRC"
SCP_J="$BOX_SCP_JFLAG"
if [ "$BOX_ARCH" != "intel" ]; then
  echo "!! standing.sh on NON-INTEL box '$BOX_NAME' — verify boxes.sh BOX_* paths are live first."
fi

# Default subject = HEAD of kernel-converge-A (origin is truth).
if [ -z "$SHA" ]; then
  SHA="$(git ls-remote origin kernel-converge-A | cut -f1)"
fi
echo "== standing.sh — subject sha=$SHA  corpora='$CORPORA'  threads='$THREADS'  N=$N =="

echo "=== ship rig -> $GUEST_USER@$GUEST:$GUEST_STAGE/ ==="
timeout 60 $SSH_GUEST "mkdir -p '$GUEST_STAGE'"
timeout 120 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new $SCP_J \
  "$HERE/_standing_guest.sh" "$HERE/standing_report.py" \
  "$ROOT/scripts/parallel_sm_tail_metric.py" \
  "$GUEST_USER@$GUEST:$GUEST_STAGE/"

REMOTE_ENV="SHA='$SHA' CORPORA='$CORPORA' THREADS='$THREADS' N='$N' BUILD='$BUILD' \
GZ_SRC='$GUEST_GZ_SRC' GZ_TARGET='/dev/shm/standing-target' EFFCORES='${EFFCORES:-0}' \
RG='$RG' IGZIP='$COMP_IGZIP' PINBASE='0' CORPUS_DIR='/root'"

echo "=== run on guest (detached; build+gate0+measure+analyze) ==="
# Detach with setsid so a dropped ssh cannot orphan the run; poll our own DONE file.
timeout 60 $SSH_GUEST \
  "chmod +x '$GUEST_STAGE/_standing_guest.sh'; \
   setsid bash -c \"$REMOTE_ENV bash '$GUEST_STAGE/_standing_guest.sh'\" >/dev/null 2>&1 < /dev/null & echo launched pid=\$!"

echo "=== poll for completion (DONE marker) ==="
DEADLINE=$(( $(date +%s) + 1500 ))
while :; do
  if timeout 30 $SSH_GUEST "test -f /dev/shm/standing.DONE && cat /dev/shm/standing.DONE" 2>/dev/null | grep -qE 'PASS|FAIL'; then
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "standing.sh: timed out waiting for /dev/shm/standing.DONE" >&2
    timeout 30 $SSH_GUEST "tail -30 /dev/shm/standing.log" 2>&1 || true
    exit 1
  fi
  sleep 15
done

echo "=== pull artifacts -> $LOCAL_ART ==="
mkdir -p "$LOCAL_ART"
timeout 120 rsync -az -e "ssh -o ConnectTimeout=15 $SCP_J" \
  "$GUEST_USER@$GUEST:/dev/shm/standing-art/" "$LOCAL_ART/" 2>/dev/null || true
timeout 60 scp -o ConnectTimeout=15 $SCP_J \
  "$GUEST_USER@$GUEST:/dev/shm/standing.log" "$LOCAL_ART/standing.log" 2>/dev/null || true

if timeout 30 $SSH_GUEST "cat /dev/shm/standing.DONE" 2>/dev/null | grep -q FAIL; then
  echo "### STANDING GATE FAILED ###"
  grep -E 'STANDING_FAIL|GATE0|build-flavor|sha mismatch|missing' "$LOCAL_ART/standing.log" 2>/dev/null || tail -40 "$LOCAL_ART/standing.log"
  exit 1
fi

echo
echo "############################## GROUND-TRUTH REPORT ##############################"
cat "$LOCAL_ART/REPORT.txt" 2>/dev/null || cat "$LOCAL_ART/standing.log"
echo
echo "(full log + raw CSVs: $LOCAL_ART)"
