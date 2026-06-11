#!/bin/bash
# _p35_arms_guest.sh — guest-side interleaved arm runner for the P3.5
# decode-chain ladder (plans/orchestrator-status.md "decode chain" lever).
#
# Usage: _p35_arms_guest.sh CORPUS T MASK N BIN...
#   CORPUS  silesia|model|... (reads /root/$CORPUS.gz)
#   T       threads (-pT)
#   MASK    taskset CPU list
#   N       reps (interleaved best-of-N)
#   BIN...  binary names under /root/ (e.g. bin-c0 bin-c1 bin-c14 bin-c142)
#
# All arms sink to /dev/null (uniform sink; byte-correctness verified in
# separate UNTIMED sha runs by the driver). Emits CSV:
#   ARM,corpus,T,bin,rep,wall_ms,rc
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}; shift 4
BINS=("$@")
[ ${#BINS[@]} -ge 2 ] || { echo "need >=2 bins"; exit 2; }
for i in $(seq 1 "$N"); do
  for b in "${BINS[@]}"; do
    t0=$(date +%s%N)
    GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
      "/root/$b" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>/tmp/p35-arm.err
    rc=$?
    t1=$(date +%s%N)
    echo "ARM,$C,$T,$b,$i,$(( (t1 - t0) / 1000000 )),$rc"
  done
done
