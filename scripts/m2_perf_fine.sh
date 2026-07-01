#!/bin/bash
# Finer-grained interleaved timing (bash TIMEFORMAT %R, 3 decimals), N=9.
set -u
MASK="0,2,4,6,8,10,12,14"
cell() { # arc threads
  local arc=$1 t=$2 i old new
  for i in 1 2 3 4 5 6 7 8 9; do
    TIMEFORMAT=%R
    old=$( { time taskset -c "$MASK" /root/bin-bar-native -d -c -p "$t" "/root/$arc.gz" >/dev/null 2>/dev/null; } 2>&1 )
    new=$( { time taskset -c "$MASK" /root/bin-m2-native -d -c -p "$t" "/root/$arc.gz" >/dev/null 2>/dev/null; } 2>&1 )
    echo "$arc T$t rep$i old=$old new=$new"
  done
}
cell silesia 8
cell silesia 4
cell model 8
