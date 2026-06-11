#!/bin/bash
# Frozen bracket for the asm increment-(a) A/B (p35_run.sh pattern).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"
export HOSTLOCK_TTL=${HOSTLOCK_TTL:-1800}
hostlock_acquire
ACQ=$?
if [ "$ACQ" = 2 ]; then echo "ABORT: freeze failed to acquire"; exit 2; fi
if [ "$ACQ" = 1 ]; then echo "## WARN: freeze acquired but box not quiet"; fi
echo "== guest freeze readback =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo); echo governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); echo loadavg=$(cut -d" " -f1 /proc/loadavg)'
echo "== T1 silesia n=9 =="
$SSH_GUEST 'bash /root/asma_ab.sh silesia 1 0 9'
echo "== T8 model n=9 (regression check) =="
$SSH_GUEST 'bash /root/asma_ab.sh model 8 0,2,4,6,8,10,12,14 9'
hostlock_release
echo "== post-release no_turbo readback (want 0) =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'
