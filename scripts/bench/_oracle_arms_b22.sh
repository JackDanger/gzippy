#!/bin/bash
# _oracle_arms_b22.sh — corrected removal-oracle arm runner using the PINNED
# b22e1b14 binary (/dev/shm/gz-b22-target/release/gzippy), not the stale
# /root/bin-oracle-native the old oracle_arms.sh hardcoded.
#
# Usage: _oracle_arms_b22.sh CORPUS T MASK N RECFILE
# Arms interleaved per rep: base / nostore / nodecode. ALL sink /dev/null.
# CSV: ARM,corpus,T,arm,rep,wall_ms,rc,warm_ms,hits/misses,cyc_core
# adjusted nodecode wall = wall_ms - warm_ms (map load is out-of-wall).
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}; REC=${5:-/dev/null}
BIN=/dev/shm/gz-b22-target/release/gzippy
ERR=/tmp/removal-arm.err
test -c /dev/null || { echo "FATAL /dev/null not char-special"; exit 9; }
for i in $(seq 1 "$N"); do
  for arm in base nostore nodecode; do
    case $arm in
      base)     EK=""; EV="" ;;
      nostore)  EK="GZIPPY_ORACLE_NOSTORE"; EV="1" ;;
      nodecode) EK="GZIPPY_ORACLE_NODECODE"; EV="$REC" ;;
    esac
    t0=$(date +%s%N)
    if [ -n "$EK" ]; then
      env "$EK=$EV" GZIPPY_FORCE_PARALLEL_SM=1 \
        perf stat -e cpu_core/cycles/ -x, -o /tmp/arm.perf \
        taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>"$ERR"
    else
      env GZIPPY_FORCE_PARALLEL_SM=1 \
        perf stat -e cpu_core/cycles/ -x, -o /tmp/arm.perf \
        taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>"$ERR"
    fi
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    warm=$(sed -n 's/.*warm_replay: loaded .* in \([0-9.]*\)ms.*/\1/p' "$ERR" | head -1)
    [ -z "$warm" ] && warm=0
    hits=$(sed -n 's/.*replay: hits=\([0-9]*\) misses=\([0-9]*\).*/\1\/\2/p' "$ERR" | head -1)
    [ -z "$hits" ] && hits=-
    cyc=$(sed -n 's/^\([0-9]*\),.*cpu_core\/cycles\/.*/\1/p' /tmp/arm.perf | head -1)
    [ -z "$cyc" ] && cyc=0
    echo "ARM,$C,$T,$arm,$i,$wall_ms,$rc,$warm,$hits,$cyc"
  done
done
test -c /dev/null || { echo "FATAL /dev/null clobbered post-run"; exit 9; }
