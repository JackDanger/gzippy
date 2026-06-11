#!/bin/bash
# P3.2 A/B: bin-p31-native (old) vs bin-p32-native (new), interleaved best-of-N.
# Masked taskset; sha-verified per arm before timing.
set -u
N=${N:-7}
F=${F:-/root/silesia.gz}
T=${T:-1}
CPUS=${CPUS:-0}
declare -a NAMES CMDS
NAMES=(p31 p32)
CMDS=(
  "/root/bin-p31-native"
  "/root/bin-p32-native"
)
REF=$(zcat "$F" | sha256sum | cut -d' ' -f1)
for i in "${!NAMES[@]}"; do
  S=$(${CMDS[$i]} -d -c -p "$T" "$F" | sha256sum | cut -d' ' -f1)
  [ "$S" = "$REF" ] && ok=OK || ok="SHA-MISMATCH($S)"
  echo "sha ${NAMES[$i]}: $ok"
done
# warmup
for i in "${!NAMES[@]}"; do ${CMDS[$i]} -d -c -p "$T" "$F" >/dev/null; done
declare -A BEST
for n in "${NAMES[@]}"; do BEST[$n]=999999; done
for r in $(seq 1 "$N"); do
  for i in "${!NAMES[@]}"; do
    t0=$(date +%s%N)
    taskset -c "$CPUS" ${CMDS[$i]} -d -c -p "$T" "$F" >/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    nm=${NAMES[$i]}
    echo "run $r $nm ${ms}ms"
    [ "$ms" -lt "${BEST[$nm]}" ] && BEST[$nm]=$ms
  done
done
echo "=== BEST OF $N (T$T, cpus $CPUS, $F) ==="
for n in "${NAMES[@]}"; do echo "$n: ${BEST[$n]}ms"; done
