#!/usr/bin/env bash
# A/B: pre-inline (old) vs inline-fast-match-copy (new), inside lxc/199.
set -u
cd /root/gzippy
F=benchmark_data/silesia-large.gz; RAW=503627776
OLD=/tmp/bench-bin/gzippy-purerust-old
NEW=/tmp/bench-bin/gzippy-purerust-sp
REF=e114dd2baa2e7c4aa1ef72de54eda2ec698a8689c6e5ec12c9a9a5b2976bb092

# correctness
for b in "$OLD" "$NEW"; do
  h=$("$b" -d -c -p 16 "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "$(basename "$b") sha=$([ "$h" = "$REF" ] && echo OK || echo BAD)"
done

med() { echo "$1" | awk -v raw="$RAW" '{for(i=1;i<=NF;i++)a[i]=$i;n=NF;for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){x=a[i];a[i]=a[j];a[j]=x}m=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;printf "%.0f",raw/m/1e6}'; }

for spec in "T4 0,2,4,6 4" "T16 0-15 16"; do
  set -- $spec; lbl=$1; cpus=$2; t=$3
  o=""; n=""
  for i in $(seq 1 12); do
    s=$(date +%s.%N); taskset -c $cpus $OLD -d -c -p $t $F >/dev/null 2>&1; e=$(date +%s.%N); o="$o $(awk "BEGIN{printf \"%.4f\",$e-$s}")"
    s=$(date +%s.%N); taskset -c $cpus $NEW -d -c -p $t $F >/dev/null 2>&1; e=$(date +%s.%N); n="$n $(awk "BEGIN{printf \"%.4f\",$e-$s}")"
  done
  echo "$lbl: old=$(med "$o") MB/s  new=$(med "$n") MB/s"
done
