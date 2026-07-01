#!/usr/bin/env bash
# decide.sh — host-side driver for `fulcrum decide` (plans/fulcrum2-charter.md).
#
# ONE command: freeze the box, run every requested cell's wall interleave +
# trace/prof captures + in-tree kill-switch knob A/Bs on the guest, pull the
# artifacts back, and render the ONE ranked decision table (`fulcrum decide`)
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
# Comparator champions beside rapidgzip (gzippy = SUBJECT, ranked vs EACH). Paths
# come from guest.env (overridable per run). DO_COMP=0 reverts to the pure gz<->rg
# matrix. AA_N = per-tool A/A self-stability iterations.
DO_COMP="${DO_COMP:-1}"; DO_COMP_AA="${DO_COMP_AA:-1}"; AA_N="${AA_N:-5}"
COMP_LIBDEFLATE="${COMP_LIBDEFLATE:-/usr/bin/libdeflate-gunzip}"
COMP_IGZIP="${COMP_IGZIP:-/usr/bin/igzip}"
COMP_ZLIBNG="${COMP_ZLIBNG:-}"
COMP_PIGZ="${COMP_PIGZ:-/usr/bin/pigz}"

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
    --no-comp) DO_COMP=0;;
    --no-comp-aa) DO_COMP_AA=0;;
    --aa-n) AA_N="$2"; shift;;
    --aa-n=*) AA_N="${1#*=}";;
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
# Analyzer = the Rust `fulcrum decide` (the Python decide/ engine was removed
# 2026-06-15 after a clean whole-pipeline cross-check). It reads FULCRUM_LEDGER.
FULCRUM_HOME="${FULCRUM_HOME:-$HOME/www/fulcrum}"
FULCRUM_BIN="${FULCRUM_BIN:-$FULCRUM_HOME/target/release/fulcrum}"
if [ ! -x "$FULCRUM_BIN" ]; then
  echo "decide.sh: Rust fulcrum binary not found at $FULCRUM_BIN" >&2
  echo "  Build it: cd '$FULCRUM_HOME' && cargo build --release  (or set FULCRUM_BIN)" >&2
  exit 1
fi
export FULCRUM_LEDGER="${FULCRUM_LEDGER:-$ROOT/artifacts/fulcrum/ledger.jsonl}"
ANALYZER=("$FULCRUM_BIN" decide)
[ "$ALLOW_THAW" = 1 ] && ANALYZER+=(--allow-thaw)

if [ -n "$ANALYZE_ONLY" ]; then
  exec "${ANALYZER[@]}" "$ANALYZE_ONLY"
fi

remote_env() {
  cat <<EOF
GUEST_SRC='$GUEST_SRC' BIN='$BIN' FEATURE='$FEATURE' CELLS='$CELLS' \
N='$N' KNOB_N='$KNOB_N' DO_KNOBS='$DO_KNOBS' KNOB_CELLS='$KNOB_CELLS' \
KNOB_FILTER='$KNOB_FILTER' \
DO_COMP='$DO_COMP' DO_COMP_AA='$DO_COMP_AA' AA_N='$AA_N' \
COMP_LIBDEFLATE='$COMP_LIBDEFLATE' COMP_IGZIP='$COMP_IGZIP' \
COMP_ZLIBNG='$COMP_ZLIBNG' COMP_PIGZ='$COMP_PIGZ' \
ARTDIR='${ARTDIR_BASE}/decide' RUNID='$RUNID' RG='$RG' RG_TRACE='$RG_TRACE' \
GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' \
ALLOW_LOAD='${ALLOW_LOAD:-0}' CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256'
EOF
}

if [ "$DRY" = 1 ]; then
  echo "## DRY-RUN — plan only, nothing executed"
  echo "bin=$BIN feature=$FEATURE cells=$CELLS knobs=$DO_KNOBS knob_cells=$KNOB_CELLS N=$N knob_n=$KNOB_N"
  echo "guest artifacts: ${ARTDIR_BASE}/decide/$RUNID  ->  local $LOCAL_ART"
  if [ "$DO_COMP" = 1 ]; then
    echo "comparator arms (gzippy=SUBJECT, ranked vs EACH at bar 0.99x; same SINK_C regular-file sink as gz/rg):"
    echo "  - rapidgzip  : $RG  (-d -c -f -P <T>)                 [primary; existing arm]"
    echo "  - libdeflate : $COMP_LIBDEFLATE  (-c <f>)             [single-shot champion]"
    echo "  - igzip      : $COMP_IGZIP  (-d -c <f>)               [ISA-L single-shot]"
    echo "  - zlibng     : ${COMP_ZLIBNG:-<unset — minigzip absent on guest; build & set COMP_ZLIBNG>}  (-d < <f>)"
    echo "  - pigz       : $COMP_PIGZ  (-d -c -p <T> <f>)         [parallel champion]"
    echo "  each present tool: interleaved wall (sha-verified) + A/A self-stability (AA_N=$AA_N); absent => comp_absent, skipped"
  else
    echo "comparator arms: DISABLED (--no-comp) — pure gzippy<->rapidgzip matrix"
  fi
  echo "remote env: $(remote_env | tr '\n' ' ')"
  echo "runtime estimate: walls ~N*(gz+rg+#comp) per cell; A/A ~AA_N*2*#comp per cell;"
  echo "  knobs ~KNOB_N*2*wall per knob per knob-cell; defaults on silesia+model fit inside the 1800s freeze TTL."
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
