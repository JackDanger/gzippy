#!/usr/bin/env bash
# clean_bench.sh — LAPTOP entry point for the clean gzippy-vs-rapidgzip bench.
#
# This is the ONE trustworthy way to compare gzippy vs rapidgzip wall time.
# It ships the harness to the host, then hands the entire lock/gate/restore
# lifecycle to host_lock_and_bench.sh (which owns the trap + watchdog). The
# laptop side is a DUMB streaming viewer — if the laptop's ssh dies, the host
# watchdog still restores the box (the restore guarantee does NOT live here).
#
# Usage:
#   scripts/bench/clean_bench.sh [THREADS] [extra guest_bench args...]
#   THREADS default "8"; e.g.  scripts/bench/clean_bench.sh "4 8 16"
#   Pass-through flags: --allow-dirty  --lever
#
# Env:
#   HOST=neurotic                ssh alias for the bare-metal Proxmox host
#   HOST_DIR=/root/gzippy-bench  where the scripts land on the host
#   WATCHDOG_TTL=2700            restore-watchdog TTL (seconds) on the host
set -u

HOST="${HOST:-neurotic}"
HOST_DIR="${HOST_DIR:-/root/gzippy-bench}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

THREADS="${1:-8}"; shift 2>/dev/null || true
EXTRA="$*"
# Collapse the thread list to a comma token so it survives word-splitting
# across the laptop->host->guest ssh layers as a SINGLE argument.
THREADS_TOK="$(echo "$THREADS" | tr ' ' ',')"

echo "## clean_bench: host=$HOST host_dir=$HOST_DIR threads='$THREADS' extra='$EXTRA'"

# ---- 1. ship the harness to the host ----------------------------------------
echo "## shipping scripts/bench/* -> $HOST:$HOST_DIR ..."
if ! ssh -o ConnectTimeout=15 "$HOST" "mkdir -p $HOST_DIR" 2>&1; then
  echo "## FATAL: cannot reach host $HOST" >&2; exit 2
fi
if ! scp -q -o ConnectTimeout=15 \
     "$HERE/host_lock_and_bench.sh" \
     "$HERE/gzippy_bench_restore.sh" \
     "$HERE/guest_bench.sh" \
     "$HERE/lib_state.sh" \
     "$HERE/lib_gate.sh" \
     "$HERE/cstate_hold.c" \
     "$HOST:$HOST_DIR/" 2>&1; then
  echo "## FATAL: scp of harness failed" >&2; exit 2
fi

# ---- 2. ship guest_bench.sh to guest 199 via the host (one hop) -------------
# host_lock_and_bench.sh runs guest_bench.sh from GUEST_SCRIPT_DIR on the guest;
# push it there now so the host script just invokes it.
echo "## shipping guest_bench.sh -> guest 199:/root/gzippy-bench/ ..."
ssh -o ConnectTimeout=15 "$HOST" "
  set -e
  ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes root@10.30.0.199 'mkdir -p /root/gzippy-bench'
  scp -q -o StrictHostKeyChecking=accept-new $HOST_DIR/guest_bench.sh root@10.30.0.199:/root/gzippy-bench/
" 2>&1 || { echo "## FATAL: could not stage guest_bench.sh on guest 199" >&2; exit 2; }

# ---- 3. hand off to the host lifecycle owner; stream its log ----------------
echo "## launching host lifecycle (it owns the trap + watchdog) ..."
echo "## ---- live host+guest output ----"
ssh -o ConnectTimeout=15 "$HOST" \
  "WATCHDOG_TTL=${WATCHDOG_TTL:-2700} bash $HOST_DIR/host_lock_and_bench.sh THREADS=$THREADS_TOK $EXTRA"
rc=$?
echo "## ---- host lifecycle exited rc=$rc ----"

# ---- 4. paranoia: independently confirm the box is at baseline --------------
echo "## post-run baseline check (independent of the host script):"
ssh -o ConnectTimeout=15 "$HOST" '
  nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo);
  un=0x$(rdmsr -p 0 0x620);
  frozen=$(for f in /sys/fs/cgroup/lxc/*/cgroup.freeze; do [ "$(cat "$f" 2>/dev/null)" = 1 ] && echo "${f}"; done);
  echo "   no_turbo=$nt (want 0)";
  echo "   uncore_0x620=$un (want 0x82b)";
  echo "   still-frozen=${frozen:-<none>} (want <none>)";
  [ "$nt" = 0 ] && [ "$un" = 0x82b ] && [ -z "$frozen" ] && echo "   BASELINE_OK=true" || echo "   BASELINE_OK=false";
' 2>&1

exit $rc
