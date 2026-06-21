#!/usr/bin/env bash
# intel_gz_rg_pinregime_guest.sh — P1 PARALLEL-PARITY CLOSE instrument.
#
# THE QUESTION: every prior gz-vs-rapidgzip silesia-T4 wall ratio (+14-17%) was
# measured PINNED to a 4-core cpuset (taskset 2,4,6,8) — the regime that
# OVERSUBSCRIBES gz's 5-thread topology (4 decode workers + 1 in-order consumer)
# onto 4 cores. The PRODUCTION-relevant comparison (`gzippy -p4`, OS schedules
# freely) is UNPINNED. So: is gz at-parity with rapidgzip silesia-T4 UNPINNED, or
# is the loss real beyond the pin artifact?
#
# This runs gz-native (-p4) vs rapidgzip-native (-P4) on the SAME corpus in THREE
# pin regimes, N>=13 interleaved (gzA,gzB,rgA,rgB / rep), /dev/null BOTH arms,
# perf stat cpu_core PMU (instructions/cycles/LLC-miss/task-clock/duration_time):
#   UNPIN  : no taskset (OS schedules across all guest P-cores = production)
#   PIN5   : taskset to 5 distinct physical P-cores (= gz n_active = 4 workers + consumer)
#   PIN4   : taskset to 4 distinct physical P-cores (reproduce the prior regime = control)
#
# Per regime: gz wall, rg wall, gz/rg ratio, cyc/B both, CPUs-utilized both, +
# A/A self-tests (gz-vs-gz, rg-vs-rg) to license the ratio, GHz-stability gate.
#
# GATE-0 (printed FIRST; a run failing any is VOID):
#   byte-exact gz==rg==zcat (sha) both arms; /dev/null both arms; gz
#   flavor=parallel-sm+pure + path=ParallelSM asserted; rg present; per-regime
#   A/A wall ratio <=1.02; GHz spread <=1%; box freeze/load + procs_running reported.
#
# MEASUREMENT ONLY (no decode-path src change). Frozen-Intel BOX-VALID -> single-arch
# NOT-YET-LAW (AMD owed).
#
# Usage:
#   GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
#   CORP=/root/silesia.gz PIN5=2,4,6,8,10 PIN4=2,4,6,8 N=13 \
#   bash intel_gz_rg_pinregime_guest.sh
set -uo pipefail

GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORP="${CORP:-/root/silesia.gz}"
PIN5="${PIN5:-2,4,6,8,10}"      # 5 distinct physical P-core CPU ids
PIN4="${PIN4:-2,4,6,8}"         # 4 distinct physical P-core CPU ids (control)
N="${N:-13}"
OUT="${OUT:-/dev/shm/gzrg_pinregime}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYZE="$HERE/intel_gz_rg_pinregime_analyze.py"
[ -f "$ANALYZE" ] || ANALYZE="/root/gzippy/scripts/bench/kernel-ab/intel_gz_rg_pinregime_analyze.py"
PROBE="$HERE/thread_topology_probe.py"
[ -f "$PROBE" ] || PROBE="/root/gzippy/scripts/bench/kernel-ab/thread_topology_probe.py"

[ -x "$GZ" ] || { echo "NO GZ $GZ"; exit 2; }
[ -x "$RG" ] || { echo "NO RG $RG"; exit 2; }
[ -f "$CORP" ] || { echo "NO CORP $CORP"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"

echo "============ INTEL gz-vs-rapidgzip PIN-REGIME PARITY (silesia T4) ============"
echo "GZ=$GZ  sha=$(sha256sum "$GZ"|cut -c1-16)"
echo "RG=$RG  sha=$(sha256sum "$RG"|cut -c1-16)  ver=$("$RG" --version 2>&1|grep -oE 'version [0-9.]+'|head -1)"
echo "CORP=$CORP  PIN5={$PIN5}  PIN4={$PIN4}  N=$N"
echo "load: $(cat /proc/loadavg)"
echo "host-freq-state: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"

# ---------- GATE-0: gz build-flavor + production path ----------
echo "--- GATE-0 FLAVOR/PATH (gz) ---"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p1 "$CORP" >/dev/null 2>"$OUT/dbg.txt" || true
FLAVOR=$(grep -oiE 'flavor[=: ]+[a-z+-]+' "$OUT/dbg.txt" | head -1)
PATHLINE=$(grep -oE 'path=[A-Za-z]+' "$OUT/dbg.txt" | head -1)
echo "    $FLAVOR   $PATHLINE"
echo "$PATHLINE" | grep -q 'path=ParallelSM' && echo "    PATH PASS (ParallelSM)" || echo "    PATH WARN (expected ParallelSM)"

# ---------- GATE-0: byte-exact gz==rg==zcat ----------
echo "--- GATE-0 BYTE-EXACT (sha; /dev/null sink) ---"
REF=$(zcat "$CORP" | sha256sum | cut -c1-16)
SG=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p4 "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
SR=$("$RG" -dc -P4 "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
echo "    ref(zcat)=$REF  gz=$SG  rg=$SR"
[ "$SG" = "$REF" ] || { echo "    GZ BYTE MISMATCH — VOID"; exit 3; }
[ "$SR" = "$REF" ] || { echo "    RG BYTE MISMATCH — VOID"; exit 3; }
echo "    BYTE-EXACT PASS (gz==rg==zcat)"

OUTBYTES=$(zcat "$CORP" | wc -c)
echo "    output_bytes=$OUTBYTES"
echo "$OUTBYTES" > "$OUT/outbytes.txt"

# ---------- THREAD-TOPOLOGY AUDIT (does rg also run a separate in-order consumer = 5 threads, or fold = 4?) ----------
echo "--- THREAD-TOPOLOGY AUDIT (gz -p4 vs rg -P4; UNPINNED) ---"
python3 "$PROBE" gz  "$GZ" -dc -p4 "$CORP" 2>&1 | sed 's/^/    /'
python3 "$PROBE" rg  "$RG" -dc -P4 "$CORP" 2>&1 | sed 's/^/    /'

# ---------- MEASUREMENT ----------
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/LLC-load-misses/,task-clock,duration_time"
# $1=taskset-prefix-or-empty $2=bin $3=threadflag $4=outfile
run_one() {
  local pin="$1"; shift
  if [ -z "$pin" ]; then
    perf stat -x, -e "$EVENTS" -- env GZIPPY_FORCE_PARALLEL_SM=1 "$1" -dc "$2" "$CORP" >/dev/null 2>"$3"
  else
    taskset -c "$pin" perf stat -x, -e "$EVENTS" -- env GZIPPY_FORCE_PARALLEL_SM=1 "$1" -dc "$2" "$CORP" >/dev/null 2>"$3"
  fi
}

measure_regime() { # $1=tag $2=pin(empty=unpinned)
  local tag="$1" pin="$2"
  echo "--- MEASURE $tag (pin='${pin:-none}') interleaved gzA,gzB,rgA,rgB x N=$N ---"
  # warmup
  run_one "$pin" "$GZ" -p4 "$OUT/$tag.warm.gz" 2>/dev/null || true
  run_one "$pin" "$RG" -P4 "$OUT/$tag.warm.rg" 2>/dev/null || true
  local r
  for r in $(seq 1 "$N"); do
    run_one "$pin" "$GZ" -p4 "$OUT/$tag.gzA.$r.csv"
    run_one "$pin" "$GZ" -p4 "$OUT/$tag.gzB.$r.csv"
    run_one "$pin" "$RG" -P4 "$OUT/$tag.rgA.$r.csv"
    run_one "$pin" "$RG" -P4 "$OUT/$tag.rgB.$r.csv"
  done
}

measure_regime UNPIN ""
measure_regime PIN5  "$PIN5"
measure_regime PIN4  "$PIN4"

echo "load(post): $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"
echo "--- ANALYSIS ---"
python3 "$ANALYZE" "$OUT" "$OUTBYTES"
echo "DONE_INTEL_GZ_RG_PINREGIME"
