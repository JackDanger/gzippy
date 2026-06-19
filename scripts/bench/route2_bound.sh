#!/usr/bin/env bash
# route2_bound.sh — bound ROUTE 2 (output-streaming / residency) in cyc/byte on the
# CURRENT (post-chunk-win) baseline. A=baseline(no env), B=$ENVB. SAME binary.
# Reuses the committed paired analyzer. medΔ=(B-A1): NEGATIVE => residency faster.
set -u
GZIPPY=${GZIPPY:-/root/bin/gzippy-chunkt1}
ENVB=${ENVB:-GZIPPY_RESIDENT_OUTPUT_POOL=1}
PIN=${PIN:-4}
REPS=${REPS:-11}
CORPORA=${CORPORA:-"nasa silesia"}
ANALYZE=/root/distpreload-harness/_distpreload_paired_analyze.py
OUT=/tmp/route2_bound
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock,page-faults"
declare -A BYTES=( [nasa]=205242368 [silesia]=211968000 )
echo "==== ROUTE2 BOUND A=baseline B=$ENVB  pin=cpu$PIN reps=$REPS  $(date) ===="
echo "GZIPPY=$GZIPPY sha=$(sha256sum "$GZIPPY"|cut -c1-12) load:$(cat /proc/loadavg)"
# Gate-0
GF=0
for c in $CORPORA; do F=/root/$c.gz; REF=$(zcat "$F"|sha256sum|cut -c1-16)
  ENTR=$(GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>&1 >/dev/null|grep "asm-kernel:c"|tail -1|sed -n 's/.*entries=\([0-9]*\).*/\1/p')
  AS=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  BS=$(env $ENVB GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  [ "${ENTR:-0}" -gt 0 ] || GF=1; [ "$AS" = "$REF" ] || GF=1; [ "$BS" = "$REF" ] || GF=1
  echo "  $c KERN=${ENTR:-0} A=$([ "$AS" = "$REF" ]&&echo OK||echo BAD) B=$([ "$BS" = "$REF" ]&&echo OK||echo BAD)"
done
[ "$GF" != 0 ] && { echo "GATE0 FAIL"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/bytes.txt"
for c in $CORPORA; do F=/root/$c.gz; echo "$c ${BYTES[$c]}" >> "$OUT/bytes.txt"
  for r in $(seq 1 $REPS); do for arm in A1 A2 B; do
    [ "$arm" = B ] && PRE="$ENVB" || PRE=""
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- env $PRE GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" >/dev/null 2>"$OUT/$c.$arm.$r.csv"
  done; done
done
python3 "$ANALYZE" "$OUT" --tag "route2 residency bound" $CORPORA
echo "--- faults mean ---"
for c in $CORPORA; do for arm in A1 B; do m=$(for f in "$OUT"/$c.$arm.*.csv; do awk -F, '/page-faults/{print $1}' "$f"; done|awk '{s+=$1;n++}END{if(n)printf "%.0f",s/n}'); echo "  $c $arm faults=$m"; done; done
echo "DONE_ROUTE2_BOUND"
