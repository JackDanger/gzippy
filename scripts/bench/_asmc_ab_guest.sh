#!/bin/bash
# Interleaved same-binary env A/B + cross-binary control for rung (c).
# Arms per rep: on (bin-asmc-native), off (bin-asmc-native + kill), base (bin-asmc-base).
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}
for i in $(seq 1 "$N"); do
  for arm in on off base; do
    case $arm in
      on)   BIN=/root/bin-asmc-native; K="" ;;
      off)  BIN=/root/bin-asmc-native; K="GZIPPY_ASM_KERNEL=0" ;;
      base) BIN=/root/bin-asmc-base;   K="" ;;
    esac
    t0=$(date +%s%N)
    env $K GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
      "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>/tmp/asmc-arm.err
    rc=$?
    t1=$(date +%s%N)
    echo "ARM,$C,$T,$arm,$i,$(( (t1 - t0) / 1000000 )),$rc"
  done
done
