#!/usr/bin/env bash
# rss_vs_t.sh — the cache-residency-mandate measurement spine
# (plans/gzippy-native-design-mandate.md). Sibling of parity.sh: same double-hop
# sync + guest build + host-freeze bracket, but the deliverable is MEMORY, not
# wall:
#   (a) PEAK RSS of gzippy-native vs rapidgzip at T=1/8/16 (the mandate wants RSS
#       roughly FLAT as T rises — a shared/small working set, NOT N large
#       per-thread buffers growing ~linearly);
#   (b) the in-process per-thread WORKING-SET byte accounting (GZIPPY_MEM_STATS=1,
#       native Block engine) — KiB/thread + aggregate + shared-table bytes;
#   (c) the POSITIVE CONTROL (GZIPPY_MEM_BALLAST_MIB=N) — the RSS-vs-T mechanism
#       must recover a known per-thread slope (validate-the-instrument, Rule 4);
#   (d) optional perf-stat MPKI / mem-stall for gzippy-native vs rapidgzip,
#       validated first with the ballast control.
#
# Correctness is enforced MECHANICALLY (the guest runner sha-verifies EVERY decode
# against the corpus pin and asserts path=ParallelSM). All paths come from
# scripts/bench/guest.env — the build cwd and measured binary can never drift.
#
# Usage:
#   scripts/bench/rss_vs_t.sh [--build] [--feature gzippy-native|gzippy-isal]
#                             [-T "1 8 16"] [-N 5] [--ballast "8 16 32"]
#                             [--no-perf] [--no-sync] [--no-lock] [--host-frozen]
#                             [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
. "$HERE/guest.env"
# shellcheck source=/dev/null
. "$HERE/lib_hostlock.sh"

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}"; }

DO_BUILD=0; DO_SYNC=1; DO_LOCK=1; DO_PERF=1; DRY=0
HOST_FROZEN="${HOST_FROZEN:-0}"
FEATURE="${DEFAULT_FEATURE:-gzippy-native}"
TLIST="1 8 16"; N=5; BALLAST="8 16 32"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --build) DO_BUILD=1;;
    --feature) FEATURE="$2"; shift;;
    --feature=*) FEATURE="${1#*=}";;
    -T) TLIST="$2"; shift;;
    -N) N="$2"; shift;;
    --ballast) BALLAST="$2"; shift;;
    --no-perf) DO_PERF=0;;
    --no-sync) DO_SYNC=0;;
    --no-lock) DO_LOCK=0;;
    --host-frozen) HOST_FROZEN=1;;
    --dry-run) DRY=1;;
    *) echo "rss_vs_t.sh: unknown arg '$1'" >&2; usage; exit 2;;
  esac
  shift
done

case "$FEATURE" in
  gzippy-native|gzippy-isal|pure-rust-inflate) ;;
  *) echo "rss_vs_t.sh: unknown --feature '$FEATURE'" >&2; exit 2;;
esac

GUEST_SCRIPT="$HERE/_rss_vs_t_guest.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "rss_vs_t.sh: missing $GUEST_SCRIPT" >&2; exit 1; }

remote_env() {
  cat <<EOF
GUEST_SRC='$GUEST_SRC' GZIPPY_BIN='$GZIPPY_BIN' CORPUS='$CORPUS' \
CORPUS_RAW_SHA256='$CORPUS_RAW_SHA256' RG='$RG' RG_TRACE='$RG_TRACE' \
FEATURE='$FEATURE' TLIST='$TLIST' N='$N' BALLAST_MIB='$BALLAST' DO_PERF='$DO_PERF' \
GOV='$GOV' NO_TURBO='$NO_TURBO' HOST_FROZEN='$HOST_FROZEN' RUSTFLAGS_PIN='$RUSTFLAGS_PIN' \
ALLOW_LOAD='${ALLOW_LOAD:-0}' DO_BUILD='$DO_BUILD' ARTDIR='${ARTDIR_BASE}/rss'
EOF
}

RSYNC_PATHS=(src crates examples build.rs Cargo.toml Cargo.lock benches scripts vendor)
do_sync() {
  echo "=== rsync working tree -> $GUEST_USER@$GUEST:$GUEST_SRC (via -J $JUMP) ==="
  timeout 30 $SSH_GUEST "mkdir -p '$GUEST_SRC'"
  # shellcheck disable=SC2086
  timeout 600 rsync -az --delete \
    --exclude 'target/' --exclude '.git/' \
    -e "ssh -o ConnectTimeout=15 -J $JUMP" \
    "${RSYNC_PATHS[@]/#/$ROOT/}" \
    "$GUEST_USER@$GUEST:$GUEST_SRC/"
}

run_remote() {
  if [ "$DO_SYNC" = 0 ]; then
    timeout 60 $SSH_GUEST "mkdir -p '$GUEST_SRC/scripts/bench'"
    timeout 60 scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      -J "$JUMP" \
      "$HERE/_rss_vs_t_guest.sh" "$HERE/ensure-corpus.sh" "$HERE/guest.env" \
      "$GUEST_USER@$GUEST:$GUEST_SRC/scripts/bench/"
  fi
  echo "=== ensure corpus on guest ==="
  timeout 600 $SSH_GUEST \
    "cd '$GUEST_SRC' && chmod +x scripts/bench/ensure-corpus.sh && bash scripts/bench/ensure-corpus.sh"
  echo "=== build+measure RSS on guest (feature=$FEATURE T='$TLIST' N=$N) ==="
  timeout 1800 $SSH_GUEST \
    "cd '$GUEST_SRC' && chmod +x scripts/bench/_rss_vs_t_guest.sh && $(remote_env) bash scripts/bench/_rss_vs_t_guest.sh"
}

if [ "$DRY" = 1 ]; then
  echo "## DRY-RUN — plan only"
  echo "feature=$FEATURE T='$TLIST' N=$N ballast='$BALLAST' perf=$DO_PERF build=$DO_BUILD sync=$DO_SYNC lock=$DO_LOCK"
  echo "guest_src=$GUEST_SRC binary=$GZIPPY_BIN corpus=$CORPUS"
  echo "remote env: $(remote_env | tr '\n' ' ')"
  exit 0
fi

if [ "$DO_LOCK" = 1 ]; then
  if hostlock_acquire; then echo "## host is QUIET — proceeding."
  else echo "## WARN: host freeze did not reach quiet threshold; do NOT bank ABSOLUTE RSS."; fi
  HOST_FROZEN=1
fi

[ "$DO_SYNC" = 1 ] && do_sync
run_remote
RC=$?
hostlock_release
exit "$RC"
