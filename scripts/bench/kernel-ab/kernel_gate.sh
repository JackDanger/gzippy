#!/usr/bin/env bash
# kernel_gate.sh — THE ONE-COMMAND KERNEL-CHANGE VERDICT.
#
#   "Did this kernel change help, hurt, or wash?"  ->  one invocation, one verdict.
#
# Given a CANDIDATE sha and a BASELINE sha, it builds BOTH gzippy-native binaries
# FRESH on the chosen box (C-FFI off, target-cpu=native, symbols kept), GATE-0
# self-validates (FFI-off flavor + sha pin + NON-INERT run_contig .o-diff +
# sha==zcat both arms + path=ParallelSM + same /dev/null sink), runs a PAIRED
# INTERLEAVED best-of-N>=15 production-wall A/B (A1,A2,B per rep), and prints a
# per-corpus KEEP / TIE / REVERT verdict with cyc/B Δ + paired-bootstrap CI +
# inter-run spread (Gate-1: Δ<spread => TIE) + an A/A self-test that licenses
# trusting the ratios on a loaded shared box.
#
# This is THE tool every future inner-kernel attempt uses to KEEP/REVERT.
#
# Usage:
#   scripts/bench/kernel-ab/kernel_gate.sh                       # HEAD vs HEAD~1 (origin)
#   scripts/bench/kernel-ab/kernel_gate.sh --sha <cand> --base <baseline>
#   scripts/bench/kernel-ab/kernel_gate.sh --corpora "silesia nasa" --threads 1 -N 21
#   scripts/bench/kernel-ab/kernel_gate.sh --box solvency        # AMD (stage; offline now)
#   scripts/bench/kernel-ab/kernel_gate.sh --self-test           # reproduce N40 KEEP (46f74d69 vs parent)
#
# Self-tests it ENFORCES (a run that fails any prints FAIL and emits no number):
#   build-flavor == parallel-sm+pure on BOTH binaries (FFI-off native proof)
#   built sha == requested sha (both)
#   run_contig disasm sha A != B  (non-inert; if ==, "no kernel change, TIE by construction")
#   every arm sha-verified == zcat (correct + non-inert); path=ParallelSM
#   same /dev/null sink both arms; A/A self-test ~= 0 (analyzer); GHz spread reported
#   GATE-1: paired Wilcoxon + bootstrap CI; |Δ| <= inter-run spread => TIE
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/scripts/bench/guest.env"

BOX="${BOX:-neurotic}"
SHA=""            # candidate
BASE=""           # baseline
CORPORA="silesia monorepo nasa"
THREADS=1
N=15
SELFTEST=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --box) BOX="$2"; shift;;
    --box=*) BOX="${1#*=}";;
    --sha) SHA="$2"; shift;;
    --sha=*) SHA="${1#*=}";;
    --base) BASE="$2"; shift;;
    --base=*) BASE="${1#*=}";;
    --corpora) CORPORA="$2"; shift;;
    --corpora=*) CORPORA="${1#*=}";;
    --threads) THREADS="$2"; shift;;
    --threads=*) THREADS="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --self-test) SELFTEST=1;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "kernel_gate.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

# shellcheck source=/dev/null
BOX="$BOX" . "$ROOT/scripts/bench/boxes.sh"

# --self-test: reproduce N40's KEEP — commit 46f74d69 (d0 anchor hoist) vs parent.
if [ "$SELFTEST" = 1 ]; then
  SHA=46f74d69
  BASE=46f74d69^
  echo "== kernel_gate SELF-TEST: candidate=$SHA (N40 d0-anchor-hoist) vs baseline=$BASE (parent) — expect KEEP =="
fi

# Resolve shas against the branch (origin is truth). Default cand=HEAD, base=HEAD~1.
resolve() {  # $1 = ref-or-empty -> full sha
  local ref="$1"
  if [ -z "$ref" ]; then return 0; fi
  git rev-parse "$ref" 2>/dev/null || git rev-parse "origin/$BOX_BRANCH" 2>/dev/null
}
if [ -z "$SHA" ]; then SHA="$(git ls-remote origin "$BOX_BRANCH" | cut -f1)"; fi
SHA="$(git rev-parse "$SHA" 2>/dev/null || echo "$SHA")"
if [ -z "$BASE" ]; then BASE="$(git rev-parse "${SHA}^" 2>/dev/null || echo "${SHA}^")"; else BASE="$(git rev-parse "$BASE" 2>/dev/null || echo "$BASE")"; fi

echo "== kernel_gate.sh  box=$BOX_NAME($BOX_ARCH)  base=$BASE  cand=$SHA =="
echo "   corpora='$CORPORA'  threads=$THREADS  N=$N"
echo "   freeze: $BOX_FREEZE_NOTE"

if [ "$BOX_ARCH" != "intel" ]; then
  echo "!! NON-INTEL box '$BOX_NAME' — verify BOX_* paths in boxes.sh are live before trusting numbers."
fi

RUNID="kgate_$(date -u '+%Y%m%dT%H%M%SZ')"
LOCAL_ART="$ROOT/artifacts/kernel-gate/$RUNID"

echo "=== ship rig -> $BOX_GUEST:$BOX_STAGE/ ==="
$BOX_SSH "mkdir -p '$BOX_STAGE'"
# shellcheck disable=SC2086
scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new $BOX_SCP_JFLAG \
  "$HERE/_kernel_gate_guest.sh" "$HERE/_kernel_gate_analyze.py" \
  "$BOX_GUEST:$BOX_STAGE/"

REMOTE_ENV="BASE='$BASE' CAND='$SHA' CORPORA='$CORPORA' THREADS='$THREADS' N='$N' \
SRC='$BOX_SRC' TARGET='$BOX_TARGET' RG='$BOX_RG' IGZIP='$BOX_IGZIP' PINBASE='$BOX_PINBASE' \
FEATURES='$BOX_FEATURES' FLAVOR='$BOX_FLAVOR' CORPUS_DIR='$BOX_CORPUS_DIR' BRANCH='$BOX_BRANCH'"

echo "=== run on $BOX_NAME (detached; build x2 + gate0 + measure + analyze) ==="
$BOX_SSH \
  "chmod +x '$BOX_STAGE/_kernel_gate_guest.sh'; \
   setsid bash -c \"$REMOTE_ENV bash '$BOX_STAGE/_kernel_gate_guest.sh'\" >/dev/null 2>&1 < /dev/null & echo launched pid=\$!"

echo "=== poll for /dev/shm/kgate.DONE ==="
DEADLINE=$(( $(date +%s) + 2400 ))   # two builds + N*3*corpora reps
while :; do
  if $BOX_SSH "test -f /dev/shm/kgate.DONE && cat /dev/shm/kgate.DONE" 2>/dev/null | grep -qE 'PASS|FAIL'; then
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "kernel_gate.sh: timed out waiting for /dev/shm/kgate.DONE" >&2
    $BOX_SSH "tail -40 /dev/shm/kgate.log" 2>&1 || true
    exit 1
  fi
  sleep 20
done

echo "=== pull artifacts -> $LOCAL_ART ==="
mkdir -p "$LOCAL_ART"
# shellcheck disable=SC2086
scp -o ConnectTimeout=15 $BOX_SCP_JFLAG \
  "$BOX_GUEST:/dev/shm/kgate.log" "$LOCAL_ART/kgate.log" 2>/dev/null || true
# shellcheck disable=SC2086
scp -o ConnectTimeout=15 $BOX_SCP_JFLAG \
  "$BOX_GUEST:/dev/shm/kgate-art/REPORT.txt" "$LOCAL_ART/REPORT.txt" 2>/dev/null || true

if $BOX_SSH "cat /dev/shm/kgate.DONE" 2>/dev/null | grep -q FAIL; then
  echo "### KERNEL-GATE GATE-0 FAILED ###"
  grep -E 'KGATE_FAIL|build|sha|mismatch|missing|run_contig' "$LOCAL_ART/kgate.log" 2>/dev/null | tail -20 \
    || tail -40 "$LOCAL_ART/kgate.log"
  exit 1
fi

echo
echo "############################## KERNEL-GATE VERDICT ##############################"
cat "$LOCAL_ART/REPORT.txt" 2>/dev/null || cat "$LOCAL_ART/kgate.log"
echo
echo "(box=$BOX_NAME $BOX_ARCH; NOT-YET-LAW — replicate on the other arch for LAW. full log: $LOCAL_ART)"
