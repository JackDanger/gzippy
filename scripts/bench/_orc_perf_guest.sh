#!/usr/bin/env bash
# _orc_perf_guest.sh — TASK2 stateless-removal-oracle attribution perf.
# Arms (per-rep RANDOMIZED order, same pinned P-core, same /dev/null sink):
#   IG  = igzip (bar) ; IG2 = igzip (self-test, CI-include 0) ;
#   A   = /root/gz-A-native (current A kernel) ; ORC = /dev/shm/gz-oracle (glue-stripped)
# Measures cyc/B, instr/B, IPC. Attribution: of A's +instr/B gap to igzip, how much
# does removing the resumable/marker glue (A-ORC) close?
set -u
LOG=/dev/shm/orc_perf.log; DONE=/dev/shm/orc_perf.DONE; OUT=/dev/shm/orc_perf
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"; exec > "$LOG" 2>&1
IG=/usr/bin/igzip; A=/root/gz-A-native; ORC=/dev/shm/gz-oracle
PIN=${PIN:-4}; REPS=${REPS:-21}; SEED=${SEED:-20260619}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock"
echo "== TASK2 ORACLE perf (seed=$SEED) =="; echo "load_start: $(cat /proc/loadavg)"
echo "ORC sha=$(sha256sum $ORC|cut -c1-12)  A sha=$(sha256sum $A|cut -c1-12)"
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA'"
# Gate-0: sha==zcat for A + ORC
declare -A BYTES; GF=0
for c in $CORPORA; do F=/root/$c.gz; REF=$(zcat $F|sha256sum|cut -c1-16); BYTES[$c]=$(zcat $F|wc -c)
  for b in A ORC; do eval "p=\$$b"; S=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $p -d -c -p1 $F 2>/dev/null|sha256sum|cut -c1-16)
    [ "$S" = "$REF" ] && echo "  GATE0 $c $b OK" || { echo "  GATE0 $c $b BAD"; GF=1; }; done; done
[ "$GF" = 0 ] || { echo GATE0_FAIL; echo FAIL>"$DONE"; exit 2; }
run_one(){ local ARM=$1 C=$2 R=$3 F=/root/$2.gz CSV="$OUT/$2.$1.$3.csv" BIN
  case $ARM in IG|IG2) BIN=$IG;; A) BIN=$A;; ORC) BIN=$ORC;; esac
  if [ "$ARM" = IG ]||[ "$ARM" = IG2 ]; then
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN $BIN -d -c $F >/dev/null 2>"$CSV"
  else
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $BIN -d -c -p1 $F >/dev/null 2>"$CSV"; fi; }
gen_orders(){ python3 -c "
import sys,random
r=random.Random($SEED); a=['IG','IG2','A','ORC']
for _ in range($1):
  x=a[:]; r.shuffle(x); print(' '.join(x))"; }
echo "--- MEASURE (randomized order) ---"
for c in $CORPORA; do rep=0
  while IFS= read -r order; do rep=$((rep+1)); echo "  $c rep=$rep: $order"
    for arm in $order; do run_one $arm $c $rep; done
  done < <(gen_orders $REPS); done
echo "load_end: $(cat /proc/loadavg)"
: > "$OUT/bytes.txt"; for c in $CORPORA; do echo "$c ${BYTES[$c]}" >> "$OUT/bytes.txt"; done
echo "=== ANALYZE ==="; python3 /root/_orc_analyze.py "$OUT" $CORPORA
echo PASS > "$DONE"; echo "=== DONE ($(date +%T)) ==="
