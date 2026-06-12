#!/usr/bin/env bash
# parity.sh — THE canonical guest sync+build+measure-vs-rapidgzip spine (P0).
#
# Collapses the multi-minute, every-turn ritual (double-hop sync + guest build +
# interleaved best-of-N measure + sha-verify) into ONE command, and enforces the
# correctness rules MECHANICALLY so a contaminated number is structurally
# impossible:
#   Rule 4 (sha-verify): EVERY measured run is verified against the decompressed-
#     corpus pin; ANY mismatch ABORTS — a fast wrong-bytes win is a loss.
#   Rule 6 (frozen host, interleaved best-of-N, production-path):
#     host-lock readback warn; gzippy and rapidgzip interleaved per trial; the
#     GZIPPY_DEBUG path=ParallelSM assertion runs before measuring.
#   REGULAR-FILE sink, NEVER a pipe (a pipe backpressure-inflated writev — phantom).
#   WINDOW-ABSENT-PRESERVING: NO seeding, NO engine oracle (guest side aborts if
#     any leaked in).
#
# All paths/values come from scripts/bench/guest.env — parity.sh and the owner can
# never disagree about WHERE (the stale-binary trap).
#
# Usage:
#   scripts/bench/parity.sh [--build] [--feature gzippy-native|gzippy-isal|pure-rust-inflate]
#                           [-T 8] [-N 11] [--fulcrum] [--no-sync] [--dry-run]
#
#   --build      rsync the working tree then build on the guest (omit to reuse the
#                already-built $GZIPPY_BIN — but then YOU own its freshness).
#   --feature F  build feature (default: $DEFAULT_FEATURE = gzippy-native = production).
#   -T N         thread count (default 8 — the campaign's primary parity cell).
#   -N M         best-of-N trials (default 11; warmup iter0 is dropped).
#   --fulcrum    additionally capture a window-absent trace and print the analyze cmd.
#   --no-sync    skip the rsync (build/measure the tree already on the guest).
#   --lock/--no-lock  bracket the run with the host freeze (default --lock): pause
#                Plex + ALL noisy LXCs, verify QUIET (procs_running), arm a watchdog,
#                and always thaw on exit. --no-lock = an external manager owns it.
#   --host-frozen  acknowledge the host is frozen when sysfs governor/no_turbo can't
#                  be read back (e.g. an LXC guest); without it a thawed/NA readback
#                  is a HARD FAIL (a thawed-host number is contaminated).
#   --dry-run    print the plan + the exact remote commands, run nothing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"
# shellcheck source=/dev/null
. "$HERE/lib_hostlock.sh"

usage() { sed -n '2,34p' "${BASH_SOURCE[0]}"; }

DO_BUILD=0; DO_FULCRUM=0; DO_SYNC=1; DRY=0
# --lock (default ON): bracket the measure with the host freeze (bench-lock) so
# Plex + the noisy neighbors are paused and the box is verified-quiet. With the
# freeze held, the guest-side host-frozen readback (LXC sysfs hidden) is
# acknowledged automatically (HOST_FROZEN=1). --no-lock leaves freezing to an
# external manager (then YOU must pass --host-frozen and ensure quiet).
DO_LOCK=1
HOST_FROZEN="${HOST_FROZEN:-0}"
FEATURE="${DEFAULT_FEATURE:-gzippy-native}"
# Optional second binary for three-way interleave (rg vs gz1 vs gz2).
GZIPPY_BIN2="${GZIPPY_BIN2:-}"
FEATURE2="${FEATURE2:-}"
# EXPECT_PATH — the production DecodePath this cell must take (Rule 6 routing
# assertion). Default ParallelSM. The gzippy-isal T1 cell routes to single-shot
# ISA-L, so measure it with `--expect-path IsalSingleShot` (DIS-15).
EXPECT_PATH="${EXPECT_PATH:-ParallelSM}"
T=8; N=11
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --build) DO_BUILD=1;;
    --feature) FEATURE="$2"; shift;;
    --feature=*) FEATURE="${1#*=}";;
    -T) T="$2"; shift;;
    -T*) T="${1#-T}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --expect-path) EXPECT_PATH="$2"; shift;;
    --expect-path=*) EXPECT_PATH="${1#*=}";;
    --fulcrum|--decompose) DO_FULCRUM=1;;
    --no-sync) DO_SYNC=0;;
    --lock) DO_LOCK=1;;
    --no-lock) DO_LOCK=0;;
    --host-frozen) HOST_FROZEN=1;;
    --dry-run) DRY=1;;
    --bin2) GZIPPY_BIN2="$2"; shift;;
    --bin2=*) GZIPPY_BIN2="${1#*=}";;
    --feature2) FEATURE2="$2"; shift;;
    --feature2=*) FEATURE2="${1#*=}";;
    *) echo "parity.sh: unknown arg '$1'" >&2; usage; exit 2;;
  esac
  shift
done

# thread-count -> pin mask (matches the campaign convention)
pin_mask() {
  case "$1" in
    1) echo "0";; 4) echo "0,2,4,6";;
    8) echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *) echo "";;
  esac
}
MASK="$(pin_mask "$T")"
[ -n "$MASK" ] || { echo "parity.sh: unsupported -T $T (use 1,4,8,16)" >&2; exit 2; }

# feature sanity (refuse a typo'd feature that would silently build the wrong path)
case "$FEATURE" in
  gzippy-native|gzippy-isal|pure-rust-inflate) ;;
  *) echo "parity.sh: unknown --feature '$FEATURE'" >&2; exit 2;;
esac

GUEST_SCRIPT="$HERE/_parity_guest.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "parity.sh: missing $GUEST_SCRIPT" >&2; exit 1; }

# The env the guest runner needs (from guest.env + flags). Exported into the
# remote shell so the guest script reads exactly the pinned values.
remote_env() {
  cat <<EOF
GUEST_SRC='$GUEST_SRC' GZIPPY_BIN='$GZIPPY_BIN' CORPUS='$CORPUS' \
CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256' CORPUS_GZ_SHA256='${CORPUS_GZ_SHA256:-}' \
RG='$RG' RG_BIN='${RG_BIN:-}' RG_BIN_SHA='${RG_BIN_SHA:-}' RG_WHEEL_BIN='${RG_WHEEL_BIN:-}' \
RG_TRACE='$RG_TRACE' FEATURE='$FEATURE' T='$T' N='$N' MASK='$MASK' \
GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' RUSTFLAGS_PIN='$RUSTFLAGS_PIN' \
ALLOW_LOAD='${ALLOW_LOAD:-0}' MAX_LOADAVG='${MAX_LOADAVG:-2.0}' \
GZIPPY_BIN2='$GZIPPY_BIN2' FEATURE2='$FEATURE2' \
EXPECT_PATH='$EXPECT_PATH' \
DO_BUILD='$DO_BUILD' DO_FULCRUM='$DO_FULCRUM' ARTDIR='${ARTDIR_BASE}/parity'
EOF
}

# rsync the working tree (tracked+modified) to the ONE canonical $GUEST_SRC over
# the -J jump in a single command. Only the files the build needs; exclude target
# and .git so a stale-object/huge-artifact transfer can't happen.
RSYNC_PATHS=(src crates examples build.rs Cargo.toml Cargo.lock benches scripts vendor)
do_sync() {
  echo "=== rsync working tree -> $GUEST_USER@$GUEST:$GUEST_SRC (via -J $JUMP) ==="
  # ensure the dest root exists
  timeout 30 $SSH_GUEST "mkdir -p '$GUEST_SRC'"
  # NOTE: --delete (NOT --delete-excluded). openrsync (macOS /usr/bin/rsync) aborts
  # with a recv_rules buffer overflow on --delete-excluded + multiple --exclude; and
  # --delete-excluded would purge the dest's target/ build cache (forcing a cold
  # rebuild every turn). Plain --delete keeps the excluded target/ intact (warm
  # incremental builds) while still removing tracked files that vanished from the
  # working tree. Excluded dirs are never transferred so they are never deleted.
  # shellcheck disable=SC2086
  timeout 600 rsync -az --delete \
    --exclude 'target/' --exclude '.git/' \
    -e "ssh -o ConnectTimeout=15 -J $JUMP" \
    "${RSYNC_PATHS[@]/#/$ROOT/}" \
    "$GUEST_USER@$GUEST:$GUEST_SRC/"
}

# Ship the corpus-ensure + guest runner, then execute the runner remotely.
run_remote() {
  echo "=== ship guest scripts to $GUEST_SRC/scripts/bench/ ==="
  # rsync already carried scripts/ when --build synced; if --no-sync, push just
  # the two scripts so the guest definitely has the current versions.
  if [ "$DO_SYNC" = 0 ]; then
    timeout 60 $SSH_GUEST "mkdir -p '$GUEST_SRC/scripts/bench'"
    timeout 60 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      -J "$JUMP" \
      "$HERE/_parity_guest.sh" "$HERE/ensure-corpus.sh" "$HERE/guest.env" \
      "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/"
  fi

  echo "=== ensure corpus on guest ==="
  timeout 600 $SSH_GUEST \
    "cd '$GUEST_SRC' && chmod +x scripts/bench/ensure-corpus.sh && bash scripts/bench/ensure-corpus.sh"

  echo "=== build+measure on guest (feature=$FEATURE T=$T N=$N) ==="
  # The remote runner owns its own lifecycle; we timeout-bound the whole hop so a
  # dead connection cannot wedge the local tool channel.
  timeout 1200 $SSH_GUEST \
    "cd '$GUEST_SRC' && chmod +x scripts/bench/_parity_guest.sh && $(remote_env) bash scripts/bench/_parity_guest.sh"
}

if [ "$DRY" = 1 ]; then
  echo "## DRY-RUN — plan only, nothing executed"
  echo "feature=$FEATURE T=$T N=$N mask=$MASK build=$DO_BUILD fulcrum=$DO_FULCRUM sync=$DO_SYNC"
  echo "guest_src=$GUEST_SRC binary=$GZIPPY_BIN corpus=$CORPUS"
  echo "ssh: $SSH_GUEST"
  if [ "$DO_SYNC" = 1 ]; then
    echo "rsync: ${RSYNC_PATHS[*]/#/$ROOT/}  ->  $GUEST_USER@$GUEST:$GUEST_SRC/  (exclude target .git)"
  fi
  echo "remote env: $(remote_env | tr '\n' ' ')"
  echo "remote runner: $GUEST_SRC/scripts/bench/_parity_guest.sh"
  exit 0
fi

# Acquire the host freeze BEFORE syncing/measuring so the build+measure both run
# on a quiet box; release ALWAYS (trap in lib_hostlock). With the freeze held the
# guest's sysfs readback is hidden (LXC), so we acknowledge frozen automatically.
if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then
    echo "## host is QUIET — proceeding."
  else
    echo "## WARN: host freeze did not reach the quiet threshold (a neighbor escaped, or a VM is busy)."
    echo "##       The RELATIVE ratio is jitter-immune (both contenders share the pin), but do NOT bank the ABSOLUTE number."
  fi
  HOST_FROZEN=1   # the freeze is held out-of-band from the guest's view
fi

[ "$DO_SYNC" = 1 ] && do_sync
run_remote
RC=$?
hostlock_release
exit "$RC"
