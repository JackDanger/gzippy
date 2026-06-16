#!/bin/bash
# _q1q2_frozen_run.sh — laptop-side FROZEN bracket for Q1 ceiling + Q2 split.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"
export HOSTLOCK_TTL=2400
N="${N:-9}"
MASK=0,2,4,6

hostlock_acquire; ACQ=$?
[ "$ACQ" = 2 ] && { echo "ABORT: freeze failed to acquire"; exit 2; }
[ "$ACQ" = 1 ] && echo "## WARN: freeze acquired but box not reported quiet — caveat"

echo "== guest freeze readback =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load=$(cut -d" " -f1 /proc/loadavg)'

for C in silesia squishy; do
  echo "########## Q1 CEILING $C T4 N=$N ##########"
  $SSH_GUEST "bash /root/_q1_ceiling_guest.sh $C 4 $MASK $N"
done
for C in silesia squishy; do
  echo "########## Q2 SPLIT $C T4 N=$N ##########"
  $SSH_GUEST "bash /root/_q2_split_guest.sh $C 4 $MASK $N"
done

hostlock_release
echo "== post-release no_turbo (want 0) =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'
