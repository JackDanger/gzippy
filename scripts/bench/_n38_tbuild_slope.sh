#!/bin/bash
# NIGHT38 TBUILD slope before/after: per-(extra litlen rebuild) cyc/B for
# conv vs base. If the redundant-clear removal cut real build work, conv's
# per-build slope ((m4-m1)/3) is BELOW base's. silesia (largest build share).
set -u
CONV=/dev/shm/n38-conv/release/gzippy
BASE=/dev/shm/n38-base/release/gzippy
PIN="${PIN:-4}"; N="${N:-11}"; F=/root/silesia.gz
OUT=/dev/shm/n38_slope; rm -rf "$OUT"; mkdir -p "$OUT"
CSV="$OUT/raw.csv"; echo "tool,mult,rep,cycles" > "$CSV"
EVENTS="cpu_core/cycles/"
one(){ local tool=$1 mult=$2 rep=$3 bin line cyc
  case "$tool" in conv) bin=$CONV;; base) bin=$BASE;; esac
  line=$(taskset -c $PIN perf stat -x, -e $EVENTS -- env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TBUILD_MULT=$mult taskset -c $PIN "$bin" -d -c -p1 "$F" 2>&1 >/dev/null)
  cyc=$(echo "$line"|awk -F, '/cycles/{print $1}'|head -1)
  echo "$tool,$mult,$rep,$cyc" >> "$CSV"
}
B=$(GZIPPY_FORCE_PARALLEL_SM=1 "$CONV" -d -c -p1 "$F" 2>/dev/null|wc -c)
echo "BYTES=$B" > "$OUT/sizes.txt"
for rep in $(seq 1 $N); do
  for cell in $(printf "conv:1\nconv:4\nbase:1\nbase:4\n"|shuf); do
    one "${cell%%:*}" "${cell##*:}" $rep
  done
done
echo DONE
