#!/usr/bin/env bash
# step0_frozen_driver.sh — run the intel_gz_rg_cycbyte AXIS instrument across the
# silesia loss cell + nasa/monorepo negative controls, on a FROZEN/quiet box.
# Detached (setsid) + writes /dev/shm/step0/DONE for poll-from-mac. NOT a decode-path change.
set -uo pipefail
GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
N="${N:-13}"
PIN_T1="${PIN_T1:-3}"
PIN_T4="${PIN_T4:-2,4,6,8}"
INSTR=/root/gzippy/scripts/bench/kernel-ab/intel_gz_rg_cycbyte_guest.sh
BASE=/dev/shm/step0
rm -rf "$BASE"; mkdir -p "$BASE"
LOG="$BASE/step0.log"
{
  echo "==== STEP-0 FROZEN AXIS DRIVER  $(date -u '+%FT%TZ') ===="
  echo "gz=$GZ sha=$(sha256sum "$GZ"|cut -c1-16)  rg=$RG  N=$N  pin_t1=$PIN_T1 pin_t4=$PIN_T4"
  echo "loadavg@start: $(cat /proc/loadavg)  procs_running: $(awk '/procs_running/{print $2}' /proc/stat)"
  for c in silesia nasa monorepo; do
    CORP="/root/$c.gz"
    echo; echo "################################################################"
    echo "#### CORPUS=$c  ($CORP)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"
    echo "################################################################"
    OUT="$BASE/$c" GZ="$GZ" RG="$RG" CORP="$CORP" PIN_T1="$PIN_T1" PIN_T4="$PIN_T4" N="$N" \
      bash "$INSTR" 2>&1
  done
  echo; echo "loadavg@end: $(cat /proc/loadavg)"
  echo "ALL_DONE_STEP0"
} >"$LOG" 2>&1
echo "PASS" >"$BASE/DONE"
