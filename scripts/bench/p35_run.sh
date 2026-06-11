#!/bin/bash
# p35_run.sh — laptop-side FROZEN bracket for the P3.5 decode-chain arms.
# Acquire host freeze -> run the requested cells -> release (ALWAYS, via
# lib_hostlock trap) -> verify no_turbo back to 0.
#
# Usage: p35_run.sh "CELLSPEC;CELLSPEC;..."
#   CELLSPEC = CORPUS T MASK N BIN BIN...   (whitespace-separated)
# Example:
#   p35_run.sh "silesia 1 0 9 bin-c0 bin-c1 bin-c14 bin-c142"
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"
export HOSTLOCK_TTL=${HOSTLOCK_TTL:-2400}

SPEC="${1:?cellspec}"
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
IFS=';' read -ra CELLS <<<"$SPEC"
for cell in "${CELLS[@]}"; do
  echo "== cell: $cell =="
  $SSH_GUEST "bash /root/p35_arms.sh $cell"
done
hostlock_release
echo "== post-release no_turbo readback (want 0) =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'
