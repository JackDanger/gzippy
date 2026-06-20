#!/usr/bin/env bash
# NIGHT35 — PRODUCTION-WALL KERNEL INSTRUCTION INJECTOR slope.
# One binary (gz-n35), arms = GZIPPY_KERNEL_INJECT mult {0,0b,1,2,4} for a
# given MODE (0=dependent latency chain, 1=independent throughput), interleaved
# per rep on the SAME pinned P-core, SAME /dev/null sink. IG/IG2 = igzip bar +
# self-test. cyc/B is frequency-invariant; GHz + LLC reported per arm.
set -u
MODE=${MODE:-0}
B=/root/gz-n35
IG=/usr/bin/igzip
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia monorepo"}
OUT=/dev/shm/n35_inject_m${MODE}
LOG=${OUT}.log
DONE=${OUT}.DONE
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock,cpu_core/LLC-load-misses/"

echo "== NIGHT35 injector slope MODE=$MODE =="
echo "binary sha=$(sha256sum $B|cut -c1-12)  load_start=$(cat /proc/loadavg)"
echo "igzip: $($IG --version 2>&1|head -1)  pin=cpu$PIN reps=$REPS corpora='$CORPORA'"

# ---- GATE-0: every mult sha==zcat (byte-transparent) + routing ----
GATE_FAIL=0
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "NO FILE $F"; GATE_FAIL=1; continue; }
  REF=$(zcat "$F"|sha256sum|cut -c1-16)
  BY=$(zcat "$F"|wc -c)
  echo "GATE0 $corp ref=$REF bytes=$BY"
  IGS=$(taskset -c $PIN $IG -d -c "$F" 2>/dev/null|sha256sum|cut -c1-16)
  [ "$IGS" = "$REF" ] || { echo "  igzip BAD"; GATE_FAIL=1; }
  for m in 0 1 2 4; do
    S=$(GZIPPY_KERNEL_INJECT=$m GZIPPY_KERNEL_INJECT_MODE=$MODE GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $B -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
    echo "  mult=$m sha=$([ "$S" = "$REF" ] && echo OK || { echo BAD; GATE_FAIL=1; })"
  done
done
P=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $B -d -c -p1 /root/silesia.gz 2>&1 >/dev/null|grep -o "path=[A-Za-z]*"|head -1)
echo "GATE4 routing $P"
[ "$P" = "path=ParallelSM" ] || GATE_FAIL=1
if [ "$GATE_FAIL" != 0 ]; then echo "GATE FAIL"; echo FAIL>"$DONE"; exit 2; fi
echo "GATE0/4 PASS"

run_one(){ # bin corp arm rep injmult
  local BIN=$1 CORP=$2 ARM=$3 R=$4 IM=$5 F=/root/$2.gz CSV="$OUT/$2.$3.$4.csv"
  if [ "$ARM" = IG ] || [ "$ARM" = IG2 ]; then
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN "$BIN" -d -c "$F" >/dev/null 2>"$CSV"
  else
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- env GZIPPY_KERNEL_INJECT=$IM GZIPPY_KERNEL_INJECT_MODE=$MODE GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN" -d -c -p1 "$F" >/dev/null 2>"$CSV"
  fi
}

echo "--- MEASURE interleaved (IG,IG2,M0,M0b,M1,M2,M4) ---"
for corp in $CORPORA; do
  for r in $(seq 1 $REPS); do
    run_one $IG  $corp IG  $r 0
    run_one $IG  $corp IG2 $r 0
    run_one $B   $corp M0  $r 0
    run_one $B   $corp M0b $r 0
    run_one $B   $corp M1  $r 1
    run_one $B   $corp M2  $r 2
    run_one $B   $corp M4  $r 4
  done
done
echo "load_end=$(cat /proc/loadavg)"
echo PASS > "$DONE"
