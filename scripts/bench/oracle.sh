#!/usr/bin/env bash
# oracle.sh — ONE parametrized oracle/perturbation driver on the parity spine (P5).
#
# Retires the per-turn one-off driver explosion (guest_ceiling / guest_clean_only /
# guest_engine_isolation / guest_same_sink_floor / guest_step0 + their run_*.sh
# pairs). A new oracle is a `--kind` case, NOT a new 100-line file pair. Every kind
# rides the SAME sync + host-lock + regular-file sink + interleaved best-of-N +
# sha-policy as parity.sh, so oracles are consistent with the parity number.
#
# Paths/values come from scripts/bench/guest.env (same pin as parity.sh).
#
# Usage:
#   scripts/bench/oracle.sh --kind <clean-only|engine-isolation|same-sink|ceiling|perturb>
#       [--build] [--feature F] [-T 8] [-N 9] [--slow GZIPPY_SLOW_DECODE=50]
#       [--no-sync] [--dry-run]
#
# Kinds (CLASS in plans/KNOBS.md):
#   same-sink         production knobs, byte-exact control (a floor; PRODUCTION).
#   ceiling           decode-removed FLOOR via bypass replay (byte-exact, NOT-prod).
#   engine-isolation  ISA-L engine oracle (byte-exact, off the prod decode graph).
#   clean-only        SEEDED clean-engine ceiling (masks-binder; SHA-NOT-CHECKED).
#   perturb           +PCT% slow-injection at a region (--slow KNOB=PCT); byte-exact.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}"; }

KIND=""; DO_BUILD=0; DO_SYNC=1; DRY=0
FEATURE="${DEFAULT_FEATURE:-gzippy-native}"
T=8; N=9; SLOW_KNOB=""; SLOW_PCT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --kind) KIND="$2"; shift;;
    --kind=*) KIND="${1#*=}";;
    --build) DO_BUILD=1;;
    --feature) FEATURE="$2"; shift;;
    --feature=*) FEATURE="${1#*=}";;
    -T) T="$2"; shift;; -T*) T="${1#-T}";;
    -N) N="$2"; shift;; -N*) N="${1#-N}";;
    --slow) SLOW_KNOB="${2%%=*}"; SLOW_PCT="${2#*=}"; shift;;
    --slow=*) v="${1#*=}"; SLOW_KNOB="${v%%=*}"; SLOW_PCT="${v#*=}";;
    --no-sync) DO_SYNC=0;;
    --dry-run) DRY=1;;
    *) echo "oracle.sh: unknown arg '$1'" >&2; usage; exit 2;;
  esac
  shift
done

case "$KIND" in
  clean-only|engine-isolation|same-sink|ceiling|perturb) ;;
  "") echo "oracle.sh: --kind is required" >&2; usage; exit 2;;
  *) echo "oracle.sh: unknown --kind '$KIND'" >&2; usage; exit 2;;
esac
if [ "$KIND" = perturb ]; then
  [ -n "$SLOW_KNOB" ] && [ -n "$SLOW_PCT" ] || { echo "oracle.sh: --kind perturb needs --slow KNOB=PCT" >&2; exit 2; }
  case "$SLOW_KNOB" in GZIPPY_SLOW_*) ;; *) echo "oracle.sh: --slow knob must be a GZIPPY_SLOW_* perturbation (see plans/KNOBS.md)" >&2; exit 2;; esac
fi

pin_mask() { case "$1" in 1) echo 0;; 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;;
  16) echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;; *) echo "";; esac; }
MASK="$(pin_mask "$T")"; [ -n "$MASK" ] || { echo "oracle.sh: unsupported -T $T" >&2; exit 2; }

ARTDIR="${ARTDIR_BASE}/oracle-$KIND"

# Build the remote env line. ceiling needs a capture file produced on the guest
# first; oracle.sh asks the guest to do the capture inline (CAP on /dev/shm).
remote_setup_cap=""
CAP=""
if [ "$KIND" = ceiling ]; then
  CAP="/dev/shm/gz_oracle_bypass.bin"
  remote_setup_cap="GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_CAPTURE='$CAP' \
    taskset -c '$MASK' '$GZIPPY_BIN' -d -c -p '$T' '$CORPUS' >/dev/null 2>'$ARTDIR/cap.stderr' || true;"
fi

remote_env() {
  cat <<EOF
GUEST_SRC='$GUEST_SRC' GZIPPY_BIN='$GZIPPY_BIN' CORPUS='$CORPUS' \
CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256' RG='$RG' RG_TRACE='$RG_TRACE' \
T='$T' N='$N' MASK='$MASK' GOV='$GOV' NO_TURBO='$NO_TURBO' \
KIND='$KIND' SLOW_KNOB='$SLOW_KNOB' SLOW_PCT='$SLOW_PCT' CAP='$CAP' \
ARTDIR='$ARTDIR'
EOF
}

if [ "$DRY" = 1 ]; then
  echo "## DRY-RUN — plan only"
  echo "kind=$KIND feature=$FEATURE T=$T N=$N build=$DO_BUILD sync=$DO_SYNC"
  [ "$KIND" = perturb ] && echo "slow: $SLOW_KNOB=+$SLOW_PCT%"
  [ "$KIND" = ceiling ] && echo "ceiling capture file: $CAP"
  echo "ssh: $SSH_GUEST"
  echo "remote env: $(remote_env | tr '\n' ' ')"
  echo "remote runner: $GUEST_SRC/scripts/bench/_oracle_guest.sh"
  exit 0
fi

# If a build is requested, reuse parity.sh's sync+build path (DRY would not build);
# we run parity.sh --build --no-... only to sync+build, then run the oracle. To
# keep this self-contained and avoid a double measure, we replicate the sync+build
# minimally here.
if [ "$DO_SYNC" = 1 ]; then
  echo "=== rsync working tree -> $GUEST_USER@$GUEST:$GUEST_SRC ==="
  timeout 30 $SSH_GUEST "mkdir -p '$GUEST_SRC'"
  # shellcheck disable=SC2086
  timeout 600 rsync -az --exclude 'target/' --exclude '.git/' \
    -e "ssh -o ConnectTimeout=15 -J $JUMP" \
    "$ROOT/src" "$ROOT/build.rs" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" \
    "$ROOT/benches" "$ROOT/scripts" "$ROOT/vendor" \
    "$GUEST_USER@$GUEST:$GUEST_SRC/"
fi

if [ "$DO_BUILD" = 1 ]; then
  echo "=== build on guest (feature=$FEATURE) ==="
  timeout 1200 $SSH_GUEST "cd '$GUEST_SRC' && RUSTFLAGS='$RUSTFLAGS_PIN' \
    sh scripts/cargo-lock.sh cargo build --release --no-default-features --features '$FEATURE' \
    2>&1 | grep -E 'Compiling gzippy|Finished|error' | tail -5"
fi

echo "=== ensure corpus on guest ==="
timeout 600 $SSH_GUEST "cd '$GUEST_SRC' && bash scripts/bench/ensure-corpus.sh"

echo "=== oracle run (kind=$KIND T=$T N=$N) ==="
timeout 1200 $SSH_GUEST \
  "mkdir -p '$ARTDIR'; cd '$GUEST_SRC'; chmod +x scripts/bench/_oracle_guest.sh; \
   $remote_setup_cap $(remote_env) bash scripts/bench/_oracle_guest.sh"
