#!/bin/bash
# P3.1 bisect: which M-step owns the native-T1 -4%?
# Interleaved best-of-N, masked taskset -c 0, T1 silesia.
set -u
N=${N:-7}
F=${F:-/root/silesia.gz}
T=${T:-1}
CPU=${CPU:-0}
declare -a NAMES CMDS
NAMES=(bar m6 m6-seed0 m6-exact0 m6-nostored)
CMDS=(
  "/root/bin-bar-native"
  "/root/bin-m6-native"
  "env GZIPPY_SEEDED_BLOCK=0 /root/bin-m6-native"
  "env GZIPPY_EXACT_BLOCK=0 /root/bin-m6-native"
  "env GZIPPY_NO_STORED_FLIP=1 /root/bin-m6-native"
)
# sha sanity once per arm
REF=$( /root/bin-bar-native -d -c -p "$T" "$F" | sha256sum | cut -d' ' -f1 )
echo "ref sha: $REF"
for i in "${!NAMES[@]}"; do
  S=$( ${CMDS[$i]} -d -c -p "$T" "$F" | sha256sum | cut -d' ' -f1 )
  [ "$S" = "$REF" ] && ok=OK || ok="SHA-MISMATCH($S)"
  echo "sha ${NAMES[$i]}: $ok"
done
# warmup
for i in "${!NAMES[@]}"; do ${CMDS[$i]} -d -c -p "$T" "$F" >/dev/null; done
# interleaved timing
declare -A BEST
for n in "${!NAMES[@]}"; do BEST[${NAMES[$n]}]=999; done
for r in $(seq 1 "$N"); do
  for i in "${!NAMES[@]}"; do
    t0=$(date +%s%N)
    taskset -c "$CPU" ${CMDS[$i]} -d -c -p "$T" "$F" >/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1-t0)/1000000 ))
    nm=${NAMES[$i]}
    echo "run $r $nm ${ms}ms"
    if [ "$ms" -lt "${BEST[$nm]%ms}" ] 2>/dev/null || [ "${BEST[$nm]}" = "999" ]; then BEST[$nm]=$ms; fi
  done
done
echo "=== BEST OF $N (T$T, cpu $CPU) ==="
for n in "${NAMES[@]}"; do echo "$n: ${BEST[$n]}ms"; done
