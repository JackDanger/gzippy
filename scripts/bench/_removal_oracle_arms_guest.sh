#!/bin/bash
# _removal_oracle_arms_guest.sh — guest-side interleaved arm runner for the
# contig-loop REMOVAL ORACLES (plans/removal-oracle-ceilings.md).
#
# Usage: _removal_oracle_arms_guest.sh CORPUS T MASK N RECFILE
#   CORPUS   silesia|model (reads /root/$CORPUS.gz)
#   T        threads (-pT)
#   MASK     taskset CPU list
#   N        reps (interleaved best-of-N)
#   RECFILE  symbol-stream capture for the NODECODE arm
#
# Arms per rep, interleaved: base / nostore / nodecode. ALL arms sink to
# /dev/null (uniform sink: NOSTORE refuses regular files, and a file-vs-null
# sink difference would confound the arm deltas; byte-correctness of base and
# nodecode is verified in separate UNTIMED runs by the driver). Emits CSV:
#   ARM,corpus,T,arm,rep,wall_ms,rc,warm_ms
# warm_ms = the NODECODE replay-map load+parse reported out-of-wall by the
# binary (0 for other arms); adjusted nodecode wall = wall_ms - warm_ms.
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}; REC=${5:-/dev/null}
BIN=/root/bin-oracle-native
ERR=/tmp/removal-arm.err
for i in $(seq 1 "$N"); do
  for arm in base nostore nodecode; do
    case $arm in
      base)     EK=""; EV="" ;;
      nostore)  EK="GZIPPY_ORACLE_NOSTORE"; EV="1" ;;
      nodecode) EK="GZIPPY_ORACLE_NODECODE"; EV="$REC" ;;
    esac
    t0=$(date +%s%N)
    if [ -n "$EK" ]; then
      env "$EK=$EV" GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
        "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>"$ERR"
    else
      env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
        "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>"$ERR"
    fi
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    warm=$(sed -n 's/.*warm_replay: loaded .* in \([0-9.]*\)ms.*/\1/p' "$ERR" | head -1)
    [ -z "$warm" ] && warm=0
    # replay hit/miss honesty line (nodecode only)
    hits=$(sed -n 's/.*replay: hits=\([0-9]*\) misses=\([0-9]*\).*/\1\/\2/p' "$ERR" | head -1)
    [ -z "$hits" ] && hits=-
    echo "ARM,$C,$T,$arm,$i,$wall_ms,$rc,$warm,$hits"
  done
done
