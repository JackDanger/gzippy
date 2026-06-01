#!/usr/bin/env bash
# _selftest_watchdog.sh — RUNS ON HOST. Proves the watchdog restores the box
# even when the lock owner is kill -9'd (the laptop-ssh-died scenario).
#
# It reuses the REAL apply-lock code path by invoking host_lock_and_bench.sh
# with a SHORT watchdog TTL and a guest-bench command replaced by a long sleep,
# then kill -9's the host script. The trap CANNOT run on SIGKILL, so ONLY the
# watchdog can restore. We then verify baseline.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/lib_state.sh"

TTL="${TTL:-90}"
echo "=== watchdog kill -9 test (TTL=${TTL}s) ==="

echo "--- pre-state ---"
echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
echo "uncore=0x$(rdmsr -p 0 0x620)"

# Launch the real host script but stub the guest hop with a long sleep so the
# box stays LOCKED while we kill the owner. We do this by setting GUEST_IP to a
# command that just sleeps — simplest is to run host_lock_and_bench.sh in a mode
# where the guest step is a no-op sleep. We inject via env: override the ssh by
# pointing GUEST_SCRIPT_DIR at a sleeper. Cleanest: run a trimmed inline lock.
#
# To exercise the ACTUAL apply+watchdog code with zero divergence we instead run
# host_lock_and_bench.sh with TESTSLEEP set; the script honors it (see below).
TESTSLEEP=120 WATCHDOG_TTL="$TTL" bash "$HERE/host_lock_and_bench.sh" THREADS=8 >/tmp/wd_host.log 2>&1 &
OWNER=$!
echo "owner pid=$OWNER; waiting for lock to apply..."

# wait until no_turbo flips to 1 (lock applied) or timeout
for i in $(seq 1 30); do
  [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 1 ] && break
  sleep 1
done

echo "--- locked-state (mid-run) ---"
echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo) (want 1)"
echo "uncore=0x$(rdmsr -p 0 0x620) (want 0x1e1e)"
echo "frozen=$(for f in /sys/fs/cgroup/lxc/*/cgroup.freeze; do [ "$(cat "$f" 2>/dev/null)" = 1 ] && basename "$(dirname "$f")"; done | tr '\n' ' ')"
echo "watchdog timer:"; systemctl list-timers gzippy-bench-restore.timer --no-pager 2>/dev/null | head -3

echo "--- KILL -9 the owner (trap cannot fire) ---"
# kill the whole process group to also stop the stubbed sleeper, but NOT the
# watchdog (it's a detached systemd transient unit, survives).
kill -9 "$OWNER" 2>/dev/null
pkill -9 -P "$OWNER" 2>/dev/null
# also kill any lingering TESTSLEEP sleeper
pkill -9 -f "sleep 120" 2>/dev/null
sleep 2
echo "owner alive? $(kill -0 "$OWNER" 2>/dev/null && echo YES || echo NO)"
echo "box still locked (expected, watchdog hasn't fired yet): no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"

echo "--- waiting for watchdog (TTL ${TTL}s) to restore ---"
end=$(( $(date +%s) + TTL + 30 ))
while [ "$(date +%s)" -lt "$end" ]; do
  nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
  if [ "$nt" = 0 ]; then echo "watchdog fired at $(date -u +%H:%M:%S)"; break; fi
  sleep 3
done

echo "--- post-watchdog state ---"
nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
un=0x$(rdmsr -p 0 0x620)
frozen=$(for f in /sys/fs/cgroup/lxc/*/cgroup.freeze; do [ "$(cat "$f" 2>/dev/null)" = 1 ] && basename "$(dirname "$f")"; done | tr '\n' ' ')
echo "no_turbo=$nt (want 0)"
echo "uncore=$un (want 0x82b)"
echo "frozen=${frozen:-<none>} (want <none>)"
[ "$nt" = 0 ] && [ "$un" = 0x82b ] && [ -z "$frozen" ] && echo "WATCHDOG_RESTORE_OK=true" || echo "WATCHDOG_RESTORE_OK=false"

# cleanup any holder leftovers + disarm any leftover timer
systemctl stop gzippy-bench-restore.timer 2>/dev/null
systemctl reset-failed gzippy-bench-restore.service 2>/dev/null
echo "=== watchdog test done ==="
