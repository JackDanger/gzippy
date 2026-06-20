#!/usr/bin/env bash
# _distpreload_paired_guest.sh — PAIRED-STAT + BANDWIDTH-CONTROL cyc/byte A/B
# for the asm_kernel dist-entry preload.
#
#   arm A = 8383a2eb (baseline)            BIN_A=/root/gz-baseline/.../gzippy
#   arm B = 2b10aa48 (speculative preload) BIN_B=/root/gz-fullrewrite/.../gzippy
#
# This supersedes _distpreload_cycbyte_guest.sh's min-of-N + Δ-vs-spread estimator
# (advisor-downgraded). It (1) takes N>=21 INTERLEAVED reps so B_r,A1_r,A2_r see
# the same instantaneous box state -> PAIRED samples; (2) hands them to a paired
# analyzer (median paired Δ + bootstrap 95% CI + Wilcoxon signed-rank p); and
# (3) runs the WHOLE A/B TWICE — once unstressed, once with a co-resident
# memory-BANDWIDTH STRESSOR pinned to sibling cores — to test whether the Δ is a
# bandwidth artifact (grows/flips under contention) or a real IPC win (stable).
#
# Measurement core (PIN, default 4) and its SMT sibling are EXCLUDED from the
# stressor so only LLC/MC bandwidth — not SMT execution ports — is contended.
#
# Makes NO host mutations (no governor/no_turbo/online writes); the stressor is a
# user process killed at the end. Nothing to restore.
#
# Usage: PIN=4 REPS=21 CORPORA="silesia monorepo" \
#        STRESS_CORES="0,2,3,8,10,12,13,14,15,16,17,19,20,21" STRESS_THREADS=14 \
#        _distpreload_paired_guest.sh
set -u

BIN_A=${BIN_A:-/root/gz-baseline/target/release/gzippy}
BIN_B=${BIN_B:-/root/gz-fullrewrite/target/release/gzippy}
PIN=${PIN:-4}
REPS=${REPS:-21}
CORPORA=${CORPORA:-"silesia monorepo nasa"}
STRESS_CORES=${STRESS_CORES:-"0,2,3,8,10,12,13,14,15,16,17,19,20,21"}
STRESS_THREADS=${STRESS_THREADS:-14}
STRESS_BUF_MB=${STRESS_BUF_MB:-96}
SKIP_STRESS=${SKIP_STRESS:-0}
HERE="$(cdv=$(dirname "$0"); cd "$cdv" && pwd)"
ANALYZE="$HERE/_distpreload_paired_analyze.py"
MEMSTRESS_SRC="$HERE/memstress.c"
MEMSTRESS_BIN=/tmp/memstress
OUT_BASE=/tmp/distpreload_paired
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock"

echo "================ DIST-PRELOAD PAIRED cyc/byte A/B + BANDWIDTH CONTROL ================"
echo "BIN_A(8383a2eb)=$BIN_A  sha=$(sha256sum $BIN_A 2>/dev/null|cut -c1-16)"
echo "BIN_B(2b10aa48)=$BIN_B  sha=$(sha256sum $BIN_B 2>/dev/null|cut -c1-16)"
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA' stress_cores='$STRESS_CORES' threads=$STRESS_THREADS"
echo "load:$(cat /proc/loadavg)"

SHA_A=$(sha256sum "$BIN_A" | cut -c1-16); SHA_B=$(sha256sum "$BIN_B" | cut -c1-16)
echo "--- GATE0(c) NON-INERT: sha_A=$SHA_A sha_B=$SHA_B  $([ "$SHA_A" != "$SHA_B" ] && echo PASS || echo FAIL-SAME-BINARY)"

# build the stressor
if [ "$SKIP_STRESS" != "1" ]; then
  gcc -O2 -pthread -o "$MEMSTRESS_BIN" "$MEMSTRESS_SRC" && echo "--- memstress built: $MEMSTRESS_BIN" || { echo "memstress build FAILED — running UNSTRESSED only"; SKIP_STRESS=1; }
fi

# ---------- GATE-0(a) kernel engaged + GATE0 sha-correctness ----------
FIRST=$(echo $CORPORA | awk '{print $1}'); FF=/root/$FIRST.gz
echo "--- GATE0(a) CONSUMER (run_contig engaged) + SHA:"
REF=$(zcat "$FF" | sha256sum | cut -c1-16)
for tag in A B; do
  B_=$([ $tag = A ] && echo "$BIN_A" || echo "$BIN_B")
  GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
    taskset -c $PIN "$B_" -d -c -p1 "$FF" >/tmp/g0a.out 2>/tmp/g0a.err
  L=$(grep "asm-kernel:c" /tmp/g0a.err | tail -1)
  ENTR=$(echo "$L" | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
  OO=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$B_" -d -c -p1 "$FF" 2>/dev/null | sha256sum | cut -c1-16)
  echo "    arm$tag entries=${ENTR:-0} $([ "${ENTR:-0}" -gt 0 ] && echo KERN_OK || echo KERN_ZERO)  sha=$OO $([ "$OO" = "$REF" ] && echo SHA_OK || echo SHA_MISMATCH)"
done

# ---------- measurement function ----------
measure_phase() {  # $1=tag(unstressed|stressed) $2=outdir
  local TAG=$1 OUT=$2
  rm -rf "$OUT"; mkdir -p "$OUT"
  : > "$OUT/bytes.txt"
  for corp in $CORPORA; do
    local F=/root/$corp.gz
    [ -f "$F" ] || { echo "  SKIP $corp (no $F)"; continue; }
    local BYTES=$(zcat "$F" | wc -c)
    echo "$corp $BYTES" >> "$OUT/bytes.txt"
    for r in $(seq 1 $REPS); do
      for arm in A1 A2 B; do
        case $arm in A1|A2) BIN=$BIN_A;; B) BIN=$BIN_B;; esac
        taskset -c $PIN perf stat -x, -e "$EVENTS" \
          -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN" -d -c -p1 "$F" \
          >/dev/null 2>"$OUT/$corp.$arm.$r.csv"
      done
    done
  done
}

# ---------- PHASE 1: UNSTRESSED ----------
echo "=== PHASE 1: UNSTRESSED (interleaved A1,A2,B per rep) ==="
measure_phase unstressed "$OUT_BASE/unstressed"
python3 "$ANALYZE" "$OUT_BASE/unstressed" --tag unstressed $CORPORA

# ---------- PHASE 2: STRESSED ----------
if [ "$SKIP_STRESS" != "1" ]; then
  echo "=== PHASE 2: STRESSED (memstress on cores $STRESS_CORES) ==="
  taskset -c "$STRESS_CORES" "$MEMSTRESS_BIN" "$STRESS_THREADS" "$STRESS_BUF_MB" 0 &
  STRESS_PID=$!
  sleep 2  # let bandwidth pressure ramp
  echo "    stressor pid=$STRESS_PID load:$(cat /proc/loadavg)"
  measure_phase stressed "$OUT_BASE/stressed"
  kill "$STRESS_PID" 2>/dev/null; wait "$STRESS_PID" 2>/dev/null
  echo "    stressor stopped. load:$(cat /proc/loadavg)"
  python3 "$ANALYZE" "$OUT_BASE/stressed" --tag stressed $CORPORA
else
  echo "=== PHASE 2 SKIPPED (no stressor) ==="
fi

echo "DONE_DISTPRELOAD_PAIRED"
