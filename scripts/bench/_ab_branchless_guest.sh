#!/bin/bash
# Interleaved best-of-N A/B: baseline (62fd59ef) vs branchless (f7810238)
# Primary metric = per-process cpu_core/cycles (frequency-independent, load-robust).
# Wall (duration_time) reported as secondary (noisy: host is shared, freq uncontrolled).
set -u
N=${N:-11}
BASE=/root/bin-ab-base
BRANCH=/root/bin-ab-branchless
declare -A RAW=( [silesia]=211968000 [squishy]=400391411 )
RAWF=/tmp/ab_raw.csv
: > "$RAWF"

pinmask() { case "$1" in 1) echo 2;; 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;; esac; }

runone() { # $1=bin $2=pin $3=T -> "wall_ns cycles instr"
  local bin="$1" pin="$2" T="$3" c="$4"
  taskset -c "$pin" perf stat -x, -e duration_time,cycles,instructions \
    -- taskset -c "$pin" "$bin" -d -c -p"$T" /root/"$c".gz >/dev/null 2>/tmp/ps
  local W CY IN
  W=$(awk -F, '/duration_time/{print $1}' /tmp/ps)
  CY=$(awk -F, '/cpu_core\/cycles\//{print $1}' /tmp/ps)
  IN=$(awk -F, '/cpu_core\/instructions\//{print $1}' /tmp/ps)
  echo "$W $CY $IN"
}

echo "AB START load:$(cat /proc/loadavg) N=$N"
echo "base=$(sha256sum $BASE|cut -c1-12) branch=$(sha256sum $BRANCH|cut -c1-12)"
for c in silesia squishy; do
  for T in 1 4 8; do
    pin=$(pinmask "$T")
    # warmup (cold-cache rep, discarded)
    runone "$BASE" "$pin" "$T" "$c" >/dev/null
    runone "$BRANCH" "$pin" "$T" "$c" >/dev/null
    for r in $(seq 1 "$N"); do
      read W CY IN < <(runone "$BASE" "$pin" "$T" "$c")
      echo "$c,$T,base,$r,$W,$CY,$IN" >> "$RAWF"
      read W CY IN < <(runone "$BRANCH" "$pin" "$T" "$c")
      echo "$c,$T,branch,$r,$W,$CY,$IN" >> "$RAWF"
    done
    echo "  done $c T$T  load:$(cat /proc/loadavg | cut -d' ' -f1)"
  done
done
echo "AB RAW DONE -> $RAWF"
