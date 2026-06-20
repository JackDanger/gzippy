#!/usr/bin/env bash
# NIGHT35 — TABLE-BUILD MULT slope at HEAD (post NIGHT29/33/34). One binary
# (gz-n35b), arms = GZIPPY_TBUILD_MULT {1,1b,2,4} interleaved per rep on the
# SAME pinned P-core, SAME /dev/null sink. IG/IG2 self-test. cyc/B freq-inv.
set -u
B=/root/gz-n35b
IG=/usr/bin/igzip
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia monorepo"}
OUT=/dev/shm/n35_tbuild
LOG=${OUT}.log
DONE=${OUT}.DONE
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock,cpu_core/LLC-load-misses/"
echo "== NIGHT35 tbuild slope == binary sha=$(sha256sum $B|cut -c1-12) load=$(cat /proc/loadavg)"

GATE_FAIL=0
for corp in $CORPORA; do
  F=/root/$corp.gz; [ -f "$F" ] || { echo "NO FILE $F"; GATE_FAIL=1; continue; }
  REF=$(zcat "$F"|sha256sum|cut -c1-16)
  for m in 1 2 4; do
    S=$(GZIPPY_TBUILD_MULT=$m GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $B -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
    echo "GATE0 $corp mult=$m sha=$([ "$S" = "$REF" ] && echo OK || { echo BAD; GATE_FAIL=1; })"
  done
done
P=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $B -d -c -p1 /root/silesia.gz 2>&1 >/dev/null|grep -o "path=[A-Za-z]*"|head -1)
echo "GATE4 $P"; [ "$P" = "path=ParallelSM" ] || GATE_FAIL=1
[ "$GATE_FAIL" = 0 ] || { echo "GATE FAIL"; echo FAIL>"$DONE"; exit 2; }
echo "GATE PASS"

run_one(){ local CORP=$1 ARM=$2 R=$3 TM=$4 F=/root/$1.gz CSV="$OUT/$1.$2.$3.csv"
  if [ "$ARM" = IG ] || [ "$ARM" = IG2 ]; then
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN $IG -d -c "$F" >/dev/null 2>"$CSV"
  else
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- env GZIPPY_TBUILD_MULT=$TM GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$B" -d -c -p1 "$F" >/dev/null 2>"$CSV"
  fi
}
for corp in $CORPORA; do
  for r in $(seq 1 $REPS); do
    run_one $corp IG  $r 1; run_one $corp IG2 $r 1
    run_one $corp T1  $r 1; run_one $corp T1b $r 1
    run_one $corp T2  $r 2; run_one $corp T4  $r 4
  done
done
echo "load_end=$(cat /proc/loadavg)"; echo PASS>"$DONE"
