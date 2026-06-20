#!/bin/bash
# NIGHT39 TBUILD slope before/after: per-(extra litlen rebuild) cyc/B for
# new vs base. The clean-path build now elides the per-symbol short-loop guard,
# so if that cut real on-wall build work, new's per-build slope ((m4-m1)/3) is
# BELOW base's. silesia (largest build share). GZIPPY_TBUILD_MULT repeats the
# litlen LUT build into idempotent state (default 1).
set -u
NEW=/dev/shm/gz-new
BASE=/dev/shm/gz-base
PIN="${PIN:-4}"; N="${N:-13}"; F=/root/silesia.gz
OUT=/dev/shm/n39_slope; rm -rf "$OUT"; mkdir -p "$OUT"
CSV="$OUT/raw.csv"; echo "tool,mult,rep,cycles" > "$CSV"
EVENTS="cpu_core/cycles/"
one(){ local tool=$1 mult=$2 rep=$3 bin line cyc
  case "$tool" in new) bin=$NEW;; base) bin=$BASE;; esac
  line=$(taskset -c $PIN perf stat -x, -e $EVENTS -- env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TBUILD_MULT=$mult taskset -c $PIN "$bin" -d -c -p1 "$F" 2>&1 >/dev/null)
  cyc=$(echo "$line"|awk -F, '/cycles/{print $1}'|head -1)
  echo "$tool,$mult,$rep,$cyc" >> "$CSV"
}
B=$(GZIPPY_FORCE_PARALLEL_SM=1 "$NEW" -d -c -p1 "$F" 2>/dev/null|wc -c)
echo "BYTES=$B" > "$OUT/sizes.txt"
for rep in $(seq 1 $N); do
  for cell in $(printf "new:1\nnew:4\nbase:1\nbase:4\n"|shuf); do
    one "${cell%%:*}" "${cell##*:}" $rep
  done
done
echo DONE
