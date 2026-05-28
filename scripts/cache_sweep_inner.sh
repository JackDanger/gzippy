#!/usr/bin/env bash
# Sweep GZIPPY_CACHE_CAP at T=16: does bounding in-flight depth recycle
# buffers and cut page-faults? Runs inside lxc/199; neighbors frozen.
set -u
cd /root/gzippy
F=benchmark_data/silesia-large.gz
RAW=503627776; MB=$(awk "BEGIN{print $RAW/1048576}")
B=/tmp/bench-bin/gzippy-purerust-sp
REF=e114dd2baa2e7c4aa1ef72de54eda2ec698a8689c6e5ec12c9a9a5b2976bb092
TS="taskset -c 0-15"

printf "%-6s %8s %8s %12s %10s %8s %8s %6s\n" cap maxlive fetched min-faults flt/MB u8miss wall_s sha
for cap in 6 8 12 16 24 32; do
  V=$(GZIPPY_CACHE_CAP=$cap GZIPPY_VERBOSE=1 $TS $B -d -c -p 16 $F 2>&1 >/dev/null)
  ml=$(echo "$V" | awk '/concurrently-live/{print $6}')
  ft=$(echo "$V" | awk '/Total Fetched/{print $4}')
  u8m=$(echo "$V" | awk '/Buffer pool u8/{for(i=1;i<=NF;i++)if($i~/misses=/){gsub(/misses=/,"",$i);print $i}}')
  M=$(GZIPPY_CACHE_CAP=$cap $TS perf stat -r 5 -e minor-faults $B -d -c -p 16 $F 2>&1 >/dev/null)
  mf=$(echo "$M" | awk '/minor-faults/{gsub(/,/,"",$1);print $1;exit}')
  wall=$(echo "$M" | awk '/elapsed/{print $1;exit}')
  fpm=$(awk -v m="$mf" -v mb="$MB" 'BEGIN{printf "%.0f", m/mb}')
  h=$(GZIPPY_CACHE_CAP=$cap $TS $B -d -c -p 16 $F 2>/dev/null | sha256sum | cut -d' ' -f1)
  ok=$([ "$h" = "$REF" ] && echo OK || echo BAD)
  printf "%-6s %8s %8s %12s %10s %8s %8s %6s\n" "$cap" "$ml" "$ft" "$mf" "$fpm" "$u8m" "$wall" "$ok"
done
