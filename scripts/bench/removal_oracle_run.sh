#!/bin/bash
# removal_oracle_run.sh — laptop-side FROZEN bracket for the removal-oracle arms.
# Acquire host freeze (TTL 2400) -> silesia T1 arms -> model T8 arms -> release
# (ALWAYS, via lib_hostlock trap) -> verify no_turbo back to 0.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"
export HOSTLOCK_TTL=2400

N="${N:-9}"
hostlock_acquire
ACQ=$?
if [ "$ACQ" = 2 ]; then
  echo "ABORT: freeze failed to acquire"
  exit 2
fi
if [ "$ACQ" = 1 ]; then
  echo "## WARN: freeze acquired but box not reported quiet — numbers carry that caveat"
fi
echo "== guest freeze readback =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo); echo governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); echo loadavg=$(cut -d" " -f1 /proc/loadavg)'
echo "== silesia T1 (taskset 0) N=$N =="
$SSH_GUEST "bash /root/oracle_arms.sh silesia 1 0 $N /dev/shm/sym-sil-t1.bin"
echo "== model T8 (taskset 0,2,..,14) N=$N =="
$SSH_GUEST "bash /root/oracle_arms.sh model 8 0,2,4,6,8,10,12,14 $N /dev/shm/sym-model-t8.bin"
hostlock_release
echo "== post-release no_turbo readback (want 0) =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'
