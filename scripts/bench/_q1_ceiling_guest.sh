#!/bin/bash
# _q1_ceiling_guest.sh — Q1 clean-decode REMOVAL-ORACLE ceiling, guest-side.
# Usage: _q1_ceiling_guest.sh CORPUS T MASK N
# Arms interleaved per rep (ALL sink /dev/null, same sink as rg):
#   base     normal gzippy decode (sha-gated separately)
#   nostore  GZIPPY_ORACLE_NOSTORE=1            (store/copy removed, decode+coord kept)
#   nodecode GZIPPY_ORACLE_NODECODE=$REC        (Huffman decode removed, store+coord kept)
#   sleep0   GZIPPY_BYPASS_DECODE=$CAP GZIPPY_SLEEP_DECODE_NS=0  (whole kernel removed -> coord+alloc floor)
#   rg       rapidgzip-native                   (comparator)
# CSV: ARM,corpus,T,arm,rep,wall_ms,adj_ms,rc,warm_ms,hits,misses
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}
BIN=/dev/shm/gz-head-native
RG=/root/oracle_c/rapidgzip-native
REC=/dev/shm/q1-rec-$C.bin
CAP=/dev/shm/q1-cap-$C.bin
ERR=/tmp/q1-arm.err
GZ=/root/$C.gz
test -c /dev/null || { echo "FATAL /dev/null not char-special"; exit 9; }

# ---- out-of-wall capture generation (fresh, with the gz-head-native binary) ----
echo "## gen NODECODE record $REC"
env GZIPPY_ORACLE_RECORD=$REC taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>/tmp/gen-rec.err
grep -a "ORACLE_RECORD wrote" /tmp/gen-rec.err | tail -1
echo "## gen META-ONLY bypass capture $CAP"
env GZIPPY_BYPASS_CAPTURE=$CAP GZIPPY_BYPASS_META_ONLY=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>/tmp/gen-cap.err
grep -a "BYPASS_CAPTURE wrote" /tmp/gen-cap.err | tail -1
ls -la "$REC" "$CAP" 2>/dev/null

# ---- correctness gate: base output sha == rg output sha (untimed) ----
SB=$(taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" 2>/dev/null | sha256sum | cut -c1-16)
SR=$("$RG" -d -c -P "$T" "$GZ" 2>/dev/null | sha256sum | cut -c1-16)
echo "## SHA base=$SB rg=$SR  $( [ "$SB" = "$SR" ] && echo MATCH || echo MISMATCH )"

echo "ARM,corpus,T,arm,rep,wall_ms,adj_ms,rc,warm_ms,hits,misses"
for i in $(seq 1 "$N"); do
  for arm in base nostore nodecode sleep0 rg; do
    t0=$(date +%s%N)
    case $arm in
      base)     taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      nostore)  env GZIPPY_ORACLE_NOSTORE=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      nodecode) env GZIPPY_ORACLE_NODECODE=$REC taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      sleep0)   env GZIPPY_BYPASS_DECODE=$CAP GZIPPY_SLEEP_DECODE_NS=0 taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      rg)       taskset -c "$MASK" "$RG" -d -c -P "$T" "$GZ" >/dev/null 2>"$ERR" ;;
    esac
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    warm=$(sed -n -e 's/.*warm_replay: loaded .* in \([0-9.]*\)ms.*/\1/p' -e 's/.*warm_prebuilt: built .* in \([0-9.]*\)ms.*/\1/p' "$ERR" | head -1)
    [ -z "$warm" ] && warm=0
    hm=$(sed -n 's/.*replay: hits=\([0-9]*\) misses=\([0-9]*\).*/\1 \2/p' "$ERR" | head -1)
    hits=$(echo "$hm" | awk '{print ($1==""?"-":$1)}')
    miss=$(echo "$hm" | awk '{print ($2==""?"-":$2)}')
    adj=$(awk -v w="$wall_ms" -v m="$warm" 'BEGIN{printf "%d", w - m}')
    echo "ARM,$C,$T,$arm,$i,$wall_ms,$adj,$rc,$warm,$hits,$miss"
  done
done
test -c /dev/null || { echo "FATAL /dev/null clobbered post-run"; exit 9; }
