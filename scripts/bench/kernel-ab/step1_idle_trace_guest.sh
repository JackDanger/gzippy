#!/usr/bin/env bash
# STEP-1 idle-localization trace driver (FROZEN box, guest 199).
# Captures, in the SAME run, the GZIPPY_TIMELINE wall-span trace AND a perf-stat
# (task-clock/wall = CPUs-utilized) of the IDENTICAL traced command — the only
# honest trace<->perf reconciliation. Runs PINNED (STEP-0 4-P-core conditions)
# and UNPINNED (to test whether the 4-core pin manufactures the util deficit).
# Also a clean UNTRACED perf run per mode to confirm STEP-0's 2.458 reproduces.
#
# Usage: GZ=/dev/shm/gzrg-target/release/gzippy CORP=/root/silesia.gz \
#        OUT=/dev/shm/step1 PINSET=2,4,6,8 N=3 bash step1_idle_trace_guest.sh
set -u
GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
CORP="${CORP:-/root/silesia.gz}"
OUT="${OUT:-/dev/shm/step1}"
PINSET="${PINSET:-2,4,6,8}"
N="${N:-3}"
mkdir -p "$OUT"
echo "GZ=$GZ sha=$(sha256sum "$GZ"|cut -c1-16)  CORP=$CORP  PINSET=$PINSET  N=$N"
echo "rg-style ref out size = $($GZ -d -p4 -c "$CORP" | wc -c)"

perf_cpus() {  # $1=logfile -> echo "task_ms wall_ms cpus"
  awk '
    /task-clock/      {gsub(/,/,"",$1); tc=$1}
    /seconds time elapsed/ {wt=$1*1000}
    /CPUs utilized/   {for(i=1;i<=NF;i++) if($i=="CPUs") cu=$(i-1)}
    END{printf "%.1f %.1f %s", tc, wt, cu}
  ' "$1"
}

run_mode() {  # $1=label  $2=taskset-prefix
  local label="$1" ts="$2"
  echo "===== MODE $label ====="
  # warm
  $ts $GZ -d -p4 -c "$CORP" >/dev/null 2>&1
  # (a) clean UNTRACED perf — confirm STEP-0 2.458 reproduces under freeze
  for r in $(seq 1 "$N"); do
    perf stat -e task-clock $ts $GZ -d -p4 -c "$CORP" >/dev/null 2> "$OUT/$label.clean.$r.perf"
    echo "  clean.$r  CPUs-utilized: $(perf_cpus "$OUT/$label.clean.$r.perf")"
  done
  # (b) TRACED run + perf on the SAME command (same-run reconciliation)
  for r in $(seq 1 "$N"); do
    perf stat -e task-clock env GZIPPY_TIMELINE="$OUT/$label.trace.$r.json" \
      $ts $GZ -d -p4 -c "$CORP" >/dev/null 2> "$OUT/$label.traced.$r.perf"
    echo "  traced.$r CPUs-utilized: $(perf_cpus "$OUT/$label.traced.$r.perf")  trace=$(wc -l < "$OUT/$label.trace.$r.json") lines"
  done
}

run_mode "pinned"   "taskset -c $PINSET"
run_mode "unpinned" ""
echo "DONE. traces+perf in $OUT"
