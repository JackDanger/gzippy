#!/usr/bin/env bash
# _gzippy_vs_igzip_paired_guest.sh — THE MISSION INSTRUMENT.
# Paired-stat + bandwidth-control single-core T1 cyc/byte A/B of
#   arm A (A1,A2) = igzip (ISA-L)         — the bar to beat
#   arm B         = gzippy-native (pure-Rust, FFI-off, ParallelSM @ T1)
# medΔ = (B - A1) cyc/byte:  NEGATIVE => gzippy FASTER than igzip (mission goal).
#
# Reuses the committed paired analyzer (_distpreload_paired_analyze.py): A1,A2,B
# interleaved per rep -> paired diffs; median Δ + bootstrap 95% CI + Wilcoxon
# signed-rank p; A2-A1 self-test (igzip-vs-igzip, must include 0); + a co-resident
# memory-bandwidth STRESSOR phase to prove the Δ is IPC not bandwidth luck.
#
# GATE-0 self-validation enforced before any number:
#   (a) gzippy run_contig CONSUMER fired: KERN entries>0 (VERBOSE+ASM_STATS gated);
#   (b) igzip arm REAL + NON-INERT: prints `igzip --version`; decoded sha == zcat ref;
#   (c) gzippy arm NON-INERT: decoded sha == zcat ref == igzip sha;
#   (d) SAME sink (/dev/null via perf >/dev/null) + SAME pin both arms;
#   (e) GHz-spread + LLC-miss reported by the analyzer.
#
# Makes NO host mutations; stressor is a user process killed at the end.
#
# Usage: PIN=4 REPS=21 CORPORA="silesia monorepo nasa" \
#        IGZIP=/usr/bin/igzip GZIPPY=/root/gz-fullrewrite/target/release/gzippy \
#        STRESS_CORES="0,2,3,8,10,12,13,14,15,16,17,19,20,21" STRESS_THREADS=14 \
#        _gzippy_vs_igzip_paired_guest.sh
set -u

IGZIP=${IGZIP:-/usr/bin/igzip}                                  # arm A (the bar)
GZIPPY=${GZIPPY:-/root/gz-fullrewrite/target/release/gzippy}    # arm B (us)
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
OUT_BASE=/tmp/gzippy_vs_igzip_paired
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock"

echo "================ GZIPPY-NATIVE vs IGZIP — T1 PAIRED cyc/byte + BANDWIDTH CONTROL ================"
echo "arm A (bar) = igzip : $IGZIP"
igzip --version 2>&1 | head -1
echo "arm B (us)  = gzippy: $GZIPPY  sha=$(sha256sum "$GZIPPY" 2>/dev/null|cut -c1-16)"
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA' stress_cores='$STRESS_CORES' threads=$STRESS_THREADS"
echo "load:$(cat /proc/loadavg)"

# the stressor
if [ "$SKIP_STRESS" != "1" ]; then
  gcc -O2 -pthread -o "$MEMSTRESS_BIN" "$MEMSTRESS_SRC" && echo "--- memstress built: $MEMSTRESS_BIN" || { echo "memstress build FAILED — UNSTRESSED only"; SKIP_STRESS=1; }
fi

# ---------- GATE-0 (a)(b)(c): consumer fired + both arms non-inert + correct ----------
echo "--- GATE0 self-validation per corpus:"
GATE_FAIL=0
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "    $corp: NO FILE $F — skip"; continue; }
  REF=$(zcat "$F" | sha256sum | cut -c1-16)
  # gzippy: KERN entries + sha
  GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
    taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" >/tmp/g0.gz.out 2>/tmp/g0.gz.err
  ENTR=$(grep "asm-kernel:c" /tmp/g0.gz.err | tail -1 | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
  GZSHA=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>/dev/null | sha256sum | cut -c1-16)
  IGSHA=$(taskset -c $PIN "$IGZIP" -d -c "$F" 2>/dev/null | sha256sum | cut -c1-16)
  K=$([ "${ENTR:-0}" -gt 0 ] && echo "KERN_OK(${ENTR})" || { echo "KERN_ZERO"; GATE_FAIL=1; })
  GS=$([ "$GZSHA" = "$REF" ] && echo "GZ_SHA_OK" || { echo "GZ_SHA_MISMATCH"; GATE_FAIL=1; })
  IS=$([ "$IGSHA" = "$REF" ] && echo "IGZIP_SHA_OK" || { echo "IGZIP_SHA_MISMATCH"; GATE_FAIL=1; })
  echo "    $corp: ref=$REF  $K  $GS(gz=$GZSHA)  $IS(ig=$IGSHA)"
done
if [ "$GATE_FAIL" != "0" ]; then echo "!!! GATE-0 FAILED — numbers below are INVALID. Aborting."; exit 2; fi
echo "--- GATE0 PASS (consumer fired, both arms non-inert + byte-correct)"

# ---------- measurement: A1,A2 = igzip ; B = gzippy ; interleaved per rep ----------
measure_phase() {  # $1=tag $2=outdir
  local TAG=$1 OUT=$2
  rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/bytes.txt"
  for corp in $CORPORA; do
    local F=/root/$corp.gz
    [ -f "$F" ] || { echo "  SKIP $corp"; continue; }
    local BYTES=$(zcat "$F" | wc -c)
    echo "$corp $BYTES" >> "$OUT/bytes.txt"
    for r in $(seq 1 $REPS); do
      for arm in A1 A2 B; do
        case $arm in
          A1|A2) taskset -c $PIN perf stat -x, -e "$EVENTS" \
                   -- taskset -c $PIN "$IGZIP" -d -c "$F" \
                   >/dev/null 2>"$OUT/$corp.$arm.$r.csv" ;;
          B)     taskset -c $PIN perf stat -x, -e "$EVENTS" \
                   -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" \
                   >/dev/null 2>"$OUT/$corp.$arm.$r.csv" ;;
        esac
      done
    done
  done
}

echo "=== PHASE 1: UNSTRESSED (interleaved igzip,igzip,gzippy per rep) ==="
measure_phase unstressed "$OUT_BASE/unstressed"
python3 "$ANALYZE" "$OUT_BASE/unstressed" --tag "igzip-vs-gzippy UNSTRESSED" $CORPORA

if [ "$SKIP_STRESS" != "1" ]; then
  echo "=== PHASE 2: STRESSED (memstress on cores $STRESS_CORES) ==="
  taskset -c "$STRESS_CORES" "$MEMSTRESS_BIN" "$STRESS_THREADS" "$STRESS_BUF_MB" 0 &
  STRESS_PID=$!
  sleep 2
  echo "    stressor pid=$STRESS_PID load:$(cat /proc/loadavg)"
  measure_phase stressed "$OUT_BASE/stressed"
  kill "$STRESS_PID" 2>/dev/null; wait "$STRESS_PID" 2>/dev/null
  echo "    stressor stopped. load:$(cat /proc/loadavg)"
  python3 "$ANALYZE" "$OUT_BASE/stressed" --tag "igzip-vs-gzippy STRESSED" $CORPORA
else
  echo "=== PHASE 2 SKIPPED ==="
fi

echo "REMINDER: medΔ=(B-A1) NEGATIVE => gzippy FASTER than igzip."
echo "DONE_GZIPPY_VS_IGZIP_PAIRED"
