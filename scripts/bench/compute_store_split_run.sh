#!/bin/bash
# compute_store_split_run.sh — FROZEN bracket for the COMPUTE-vs-STORE split of
# the silesia T4 decode+emit region (pinned b22e1b14 binary). Acquire host
# freeze -> regenerate the NODECODE record out-of-wall -> interleaved N reps of
# base/nostore/nodecode at T4 -> release (always, via trap) -> readback.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/guest.env"
. "$HERE/lib_hostlock.sh"
export HOSTLOCK_TTL=2400
N="${N:-14}"
MASK=0,2,4,6
REC=/dev/shm/sym-sil-t4.bin

hostlock_acquire; ACQ=$?
[ "$ACQ" = 2 ] && { echo "ABORT: freeze failed to acquire"; exit 2; }
[ "$ACQ" = 1 ] && echo "## WARN: freeze acquired but box not reported quiet — caveat"

echo "== guest freeze readback =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo); echo gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); echo loadavg=$(cut -d" " -f1 /proc/loadavg)'

echo "== regenerate NODECODE record (OUT-OF-WALL) =="
$SSH_GUEST "test -s $REC && echo 'record present:' \$(ls -la $REC) || { echo 'regen record'; env GZIPPY_ORACLE_RECORD=$REC GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $MASK /dev/shm/gz-b22-target/release/gzippy -d -c -p4 /root/silesia.gz >/dev/null 2>&1; }"

echo "== silesia T4 arms N=$N (CSV: ARM,corpus,T,arm,rep,wall_ms,rc,warm_ms,hits/miss,cyc_core) =="
$SSH_GUEST "bash /root/_oracle_arms_b22.sh silesia 4 $MASK $N $REC"

hostlock_release
echo "== post-release no_turbo (want 0) =="
$SSH_GUEST 'echo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'
