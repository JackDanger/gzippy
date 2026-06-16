#!/usr/bin/env bash
# optgate_capture.sh — ONE command to A/B a candidate change through the
# self-validated fulcrum optgate cyc/byte gate.
#
#   scripts/bench/optgate_capture.sh --base <BASE_BIN> --after <AFTER_BIN> \
#       [--corpus silesia] [--t 1] [--clean-t 1] [-N 13] [--arch <label>]
#       [--cross-arch] [--allow-load] [--no-lock] [--analyze-only <art-dir>]
#
# It (1) freezes the host (neighbor-pause + governor/no_turbo via bench-lock on
# the jump host — a container can't freeze its own host), (2) ships the guest
# capture, (3) runs the interleaved base/after/rg + clean-path-T1 perf-stat
# capture (N>=12), (4) pulls the artifacts, (5) releases the freeze, (6) builds
# optgate.json and renders BOTH the optgate cyc/byte verdict AND the ENV-INVARIANT
# auto-VOID gate (`fulcrum provenance`).
#
# This REPLACES the ad-hoc per-agent cyc/byte scripts (_memcpy_cyc_*.sh etc.).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"
# shellcheck source=/dev/null
. "$HERE/lib_hostlock.sh"

# Allow overriding the corpus raw-sha pin for non-silesia corpus controls
# (guest.env pins silesia's sha; the per-run REFSHA byte-exact check still runs).
CORPUS_RAW_SHA256="${OVERRIDE_CORPUS_SHA:-$CORPUS_RAW_SHA256}"

BASE_BIN="${BASE_BIN:-$GZIPPY_BIN}"
AFTER_BIN="${AFTER_BIN:-$GZIPPY_BIN}"
CORPUS_NAME="silesia"
T=1; CLEAN_T=1; N=13
ARCH="${ARCH:-intel-i7-13700T}"
CROSS_ARCH=0
ALLOW_LOAD=0; DO_LOCK=1; ANALYZE_ONLY=""
# deliberate-corruption smoke knobs (validation b/c): override the pinned AFTER
# sha to force a BINARY-DRIFT VOID, without touching /dev/null.
FORCE_BAD_PIN="${FORCE_BAD_PIN:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base) BASE_BIN="$2"; shift;;
    --base=*) BASE_BIN="${1#*=}";;
    --after) AFTER_BIN="$2"; shift;;
    --after=*) AFTER_BIN="${1#*=}";;
    --corpus) CORPUS_NAME="$2"; shift;;
    --corpus=*) CORPUS_NAME="${1#*=}";;
    --t) T="$2"; shift;;
    --t=*) T="${1#*=}";;
    --clean-t) CLEAN_T="$2"; shift;;
    --clean-t=*) CLEAN_T="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --arch) ARCH="$2"; shift;;
    --arch=*) ARCH="${1#*=}";;
    --cross-arch) CROSS_ARCH=1;;
    --allow-load) ALLOW_LOAD=1;;
    --no-lock) DO_LOCK=0;;
    --force-bad-pin) FORCE_BAD_PIN="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";;
    --analyze-only) ANALYZE_ONLY="$2"; shift;;
    --analyze-only=*) ANALYZE_ONLY="${1#*=}";;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "optgate_capture.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

# corpus path on the guest (guest.env CORPUS is silesia; others under /root/<n>.gz)
if [ "$CORPUS_NAME" = "silesia" ]; then CORPUS_PATH="$CORPUS"; else CORPUS_PATH="/root/$CORPUS_NAME.gz"; fi

FULCRUM_HOME="${FULCRUM_HOME:-$HOME/www/fulcrum}"
FULCRUM_BIN="${FULCRUM_BIN:-$FULCRUM_HOME/target/release/fulcrum}"
if [ ! -x "$FULCRUM_BIN" ]; then
  echo "optgate_capture.sh: fulcrum binary not found at $FULCRUM_BIN" >&2
  echo "  Build it: cd '$FULCRUM_HOME' && cargo build --release  (or set FULCRUM_BIN)" >&2
  exit 1
fi

render() { # <local-art-dir>
  local art="$1"
  echo "=== build optgate.json ==="
  python3 "$HERE/optgate_build_artifact.py" "$art"
  echo
  echo "=== fulcrum optgate (cyc/byte WALL-WIN-OR-NO-WIN) ==="
  "$FULCRUM_BIN" optgate "$art/optgate.json" || true
  echo
  echo "=== fulcrum provenance (ENV-INVARIANT auto-VOID gate) ==="
  "$FULCRUM_BIN" provenance "$art" || true
}

if [ -n "$ANALYZE_ONLY" ]; then
  render "$ANALYZE_ONLY"
  exit 0
fi

RUNID="optgate_$(date -u '+%Y%m%dT%H%M%SZ')"
LOCAL_ART="$ROOT/artifacts/optgate/$RUNID"

# Pin the AFTER binary's sha by READING it on the guest first, so the ENV-INVARIANT
# re-read-vs-pinned check is meaningful (matches on a clean run; the
# --force-bad-pin smoke overrides it to a value that cannot match -> VOID).
echo "=== read AFTER bin sha on guest (pin source) ==="
GUEST_AFTER_SHA="$(timeout 40 $SSH_GUEST "sha256sum '$AFTER_BIN' 2>/dev/null | cut -d' ' -f1" || true)"
GUEST_RG_SHA="$(timeout 40 $SSH_GUEST "sha256sum '$RG' 2>/dev/null | cut -d' ' -f1" || true)"
AFTER_SHA_PINNED="${FORCE_BAD_PIN:-$GUEST_AFTER_SHA}"
echo "   after_sha(guest)=$GUEST_AFTER_SHA  pinned=$AFTER_SHA_PINNED"

# Freeze BEFORE measuring; release ALWAYS (trap in lib_hostlock).
HOST_FROZEN=0
if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then echo "## host is QUIET — proceeding.";
  else echo "## WARN: host freeze did not reach quiet; ratios survive, ABSOLUTE cyc/byte will be flagged."; fi
  HOST_FROZEN=1
fi

remote_env() {
  cat <<EOF
BASE_BIN='$BASE_BIN' AFTER_BIN='$AFTER_BIN' RG='$RG' CORPUS='$CORPUS_PATH' \
T='$T' CLEAN_T='$CLEAN_T' N='$N' \
ARTDIR='${ARTDIR_BASE}/optgate' RUNID='$RUNID' \
GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' ALLOW_LOAD='$ALLOW_LOAD' \
CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256' \
BASE_SHA_PINNED='$GUEST_AFTER_SHA' AFTER_SHA_PINNED='$AFTER_SHA_PINNED' \
RG_SHA_PINNED='$GUEST_RG_SHA' ARCH='$ARCH' CROSS_ARCH='$CROSS_ARCH' \
BASE_COMMIT='${BASE_COMMIT:-}' AFTER_COMMIT='${AFTER_COMMIT:-}'
EOF
}

echo "=== ship optgate guest scripts -> $GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/ ==="
timeout 60 $SSH_GUEST "mkdir -p '$GUEST_SRC/scripts/bench'"
timeout 60 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -J "$JUMP" \
  "$HERE/_optgate_guest.sh" "$HERE/lib_decide_guest.sh" \
  "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/"

echo "=== run optgate capture on guest (corpus=$CORPUS_NAME T=$T clean_t=$CLEAN_T N=$N) ==="
set +e
timeout 1200 $SSH_GUEST \
  "cd '$GUEST_SRC' && chmod +x scripts/bench/_optgate_guest.sh && $(remote_env) bash scripts/bench/_optgate_guest.sh" \
  | tee "/tmp/optgate_guest_$RUNID.log"
GRC=${PIPESTATUS[0]}
set -e
if ! grep -q 'OPTGATE_GUEST_DONE' "/tmp/optgate_guest_$RUNID.log"; then
  hostlock_release
  echo "optgate_capture.sh: guest runner did not complete (rc=$GRC) — see /tmp/optgate_guest_$RUNID.log" >&2
  exit "${GRC:-1}"
fi

echo "=== pull artifacts -> $LOCAL_ART ==="
mkdir -p "$LOCAL_ART"
timeout 300 rsync -az -e "ssh -o ConnectTimeout=15 -J $JUMP" \
  "$GUEST_USER@$GUEST:${ARTDIR_BASE}/optgate/$RUNID/" "$LOCAL_ART/"

hostlock_release

if grep -q 'DECIDE_FAIL=' "/tmp/optgate_guest_$RUNID.log"; then
  echo "optgate_capture.sh: guest reported a hard failure:" >&2
  grep 'DECIDE_FAIL=' "/tmp/optgate_guest_$RUNID.log" >&2
  exit 1
fi

render "$LOCAL_ART"
echo
echo "ARTIFACTS: $LOCAL_ART"
