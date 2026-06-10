#!/bin/bash
# M2 masked perf sanity: interleaved x3, old (/root/bin-bar-native) vs new
# (/root/bin-m2-native), silesia T4+T8, model T8. taskset P-cores.
set -u
MASK="0,2,4,6,8,10,12,14"
run() { # bin file threads -> seconds
  /usr/bin/time -f "%e" taskset -c "$MASK" "$1" -d -c -p "$3" "$2" 2>&1 >/dev/null | tail -1
}
for round in 1 2 3; do
  for cell in "silesia 4" "silesia 8" "model 8"; do
    set -- $cell
    arc=$1; t=$2
    old=$(run /root/bin-bar-native "/root/$arc.gz" "$t")
    new=$(run /root/bin-m2-native "/root/$arc.gz" "$t")
    echo "round$round $arc T$t old=$old new=$new"
  done
done
