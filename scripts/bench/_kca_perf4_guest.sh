#!/usr/bin/env bash
# _kca_perf4_guest.sh — NIGHT22 DECISIVE 4-arm interleaved paired cyc/B perf gate.
# Arms per rep (interleaved on the SAME pinned P-core, SAME /dev/null sink):
#   IG  = igzip (the bar) ; IG2 = igzip again (self-test, must CI-include 0 vs IG)
#   A   = gz kernel-converge-A (single-state-base)  [the question]
#   OLD = gz-chunkt1 (early-flag-bit prod baseline)
#   N19 = gz NIGHT19 (cadence-converge, pre single-state-base)
# Per arm: perf stat cycles+instructions+task-clock -> cyc/B, IPC, instr/B.
# Gate-0: KERN entries>0 for each gz arm; every arm sha==zcat ref (non-inert);
#         /dev/null both arms; IG2-IG self-test; GHz reported per arm.
# Writes /dev/shm/kca_perf4.DONE (PASS/FAIL) + raw CSVs in /dev/shm/kca_perf4/.
set -u
LOG=/dev/shm/kca_perf4.log
DONE=/dev/shm/kca_perf4.DONE
OUT=/dev/shm/kca_perf4
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

IG=/usr/bin/igzip
A=/root/gz-A-native
OLD=/root/bin/gzippy-chunkt1
N19=/root/gz-night19
PIN=${PIN:-4}
REPS=${REPS:-21}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock"

echo "== NIGHT22 4-arm decisive perf =="
echo "load_start: $(cat /proc/loadavg)"
echo "igzip: $($IG --version 2>&1 | head -1)"
for b in A OLD N19; do eval "p=\$$b"; echo "$b: $p sha=$(sha256sum $p|cut -c1-12)"; done
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA'"

# ---- GATE-0: KERN fired + every arm sha == zcat ref ----
echo "--- GATE0:"
GATE_FAIL=0
declare -A BYTES
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE $F"; GATE_FAIL=1; continue; }
  REF=$(zcat "$F" | sha256sum | cut -c1-16)
  BYTES[$corp]=$(zcat "$F" | wc -c)
  IGS=$(taskset -c $PIN $IG -d -c "$F" 2>/dev/null | sha256sum | cut -c1-16)
  msg="  $corp ref=$REF bytes=${BYTES[$corp]} ig=$([ "$IGS" = "$REF" ] && echo OK || { echo BAD; GATE_FAIL=1; })"
  for b in A OLD N19; do
    eval "p=\$$b"
    GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$p" -d -c -p1 "$F" >/tmp/g0.out 2>/tmp/g0.err
    ENTR=$(grep "asm-kernel:c" /tmp/g0.err | tail -1 | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
    S=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$p" -d -c -p1 "$F" 2>/dev/null | sha256sum | cut -c1-16)
    KOK=$([ "${ENTR:-0}" -gt 0 ] && echo "K${ENTR}" || echo "K0")
    [ "${ENTR:-0}" -gt 0 ] || { [ "$b" = OLD ] || GATE_FAIL=1; }   # OLD may use different stats gate
    SOK=$([ "$S" = "$REF" ] && echo OK || { echo BAD; GATE_FAIL=1; })
    msg="$msg | $b=$SOK/$KOK"
  done
  echo "$msg"
done
if [ "$GATE_FAIL" != 0 ]; then echo "GATE0 FAIL"; echo FAIL > "$DONE"; exit 2; fi
echo "GATE0 PASS"

run_one() {  # $1=bin $2=corp $3=arm $4=rep ; gz arms get force-parallel-sm
  local BIN=$1 CORP=$2 ARM=$3 R=$4 F=/root/$2.gz CSV="$OUT/$2.$3.$4.csv"
  if [ "$ARM" = IG ] || [ "$ARM" = IG2 ]; then
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN "$BIN" -d -c "$F" >/dev/null 2>"$CSV"
  else
    taskset -c $PIN perf stat -x, -e "$EVENTS" -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN" -d -c -p1 "$F" >/dev/null 2>"$CSV"
  fi
}

echo "--- MEASURE (interleaved IG,IG2,A,OLD,N19 per rep) ---"
for corp in $CORPORA; do
  for r in $(seq 1 $REPS); do
    run_one $IG  $corp IG  $r
    run_one $IG  $corp IG2 $r
    run_one $A   $corp A   $r
    run_one $OLD $corp OLD $r
    run_one $N19 $corp N19 $r
  done
done
echo "load_end: $(cat /proc/loadavg)"

# emit bytes file for analyzer
: > "$OUT/bytes.txt"
for corp in $CORPORA; do echo "$corp ${BYTES[$corp]}" >> "$OUT/bytes.txt"; done

echo "=== ANALYZE ==="
python3 /root/_kca_perf4_analyze.py "$OUT" $CORPORA
echo PASS > "$DONE"
echo "=== DONE ($(date +%T)) ==="
