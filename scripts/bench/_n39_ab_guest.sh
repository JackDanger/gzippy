#!/bin/bash
# NIGHT39 production-wall A/B (cyc/B, interleaved, paired) for the clean-path
# speculative-guard gating. Arms:
#   new   = fef1f8c7 (clean path build elides the per-symbol short-loop guard)
#   base  = de05fc80 (NIGHT38; guard present on every path)
#   new2  = byte-copy of new  -> Gate-0 A-vs-A self-test (must bracket 0)
#   igzip = /usr/bin/igzip     -> remaining-gap context
# cyc/B is frequency-independent; GHz reported for Gate-0. Both gzippy arms
# decode -p1 to /dev/null with GZIPPY_FORCE_PARALLEL_SM=1 (clean T1 path).
set -u
NEW=/dev/shm/gz-new
BASE=/dev/shm/gz-base
NEW2=/dev/shm/gz-new2
IGZIP=/usr/bin/igzip
cp "$NEW" "$NEW2"
PIN="${PIN:-4}"; N="${N:-15}"
OUT="${OUT:-/dev/shm/n39_ab}"; rm -rf "$OUT"; mkdir -p "$OUT"
CSV="$OUT/raw.csv"; echo "corpus,tool,rep,instructions,cycles,task_clock_ms,llc_misses" > "$CSV"
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock,cpu_core/LLC-misses/"
stat_one() { local corpus=$1 tool=$2 f=$3 rep=$4 bin line ins cyc tc llc
  case "$tool" in new) bin=$NEW;; base) bin=$BASE;; new2) bin=$NEW2;; igzip) bin=$IGZIP;; esac
  if [ "$tool" = igzip ]; then
    line=$(taskset -c $PIN perf stat -x, -e $EVENTS -- taskset -c $PIN "$bin" -d -c "$f" 2>&1 >/dev/null)
  else
    line=$(taskset -c $PIN perf stat -x, -e $EVENTS -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$bin" -d -c -p1 "$f" 2>&1 >/dev/null)
  fi
  ins=$(echo "$line"|awk -F, '/instructions/{print $1}'|head -1)
  cyc=$(echo "$line"|awk -F, '/cycles/{print $1}'|head -1)
  tc=$(echo "$line"|awk -F, '/task-clock/{print $1}'|head -1)
  llc=$(echo "$line"|awk -F, '/LLC-misses/{print $1}'|head -1)
  echo "$corpus,$tool,$rep,$ins,$cyc,$tc,$llc" >> "$CSV"
}
SIL_BYTES=$(GZIPPY_FORCE_PARALLEL_SM=1 "$NEW" -d -c -p1 /root/silesia.gz 2>/dev/null|wc -c)
MONO_BYTES=$(GZIPPY_FORCE_PARALLEL_SM=1 "$NEW" -d -c -p1 /root/monorepo.gz 2>/dev/null|wc -c)
echo "SIL_BYTES=$SIL_BYTES MONO_BYTES=$MONO_BYTES" > "$OUT/sizes.txt"
echo "### A/B N=$N pin=$PIN arms {new,base,new2,igzip} x {silesia,monorepo}"
for rep in $(seq 1 $N); do
  for arm in $(printf "new\nbase\nnew2\nigzip\n"|shuf); do
    stat_one silesia "$arm" /root/silesia.gz $rep
    stat_one monorepo "$arm" /root/monorepo.gz $rep
  done
done
echo "### DONE"; cat "$OUT/sizes.txt"
