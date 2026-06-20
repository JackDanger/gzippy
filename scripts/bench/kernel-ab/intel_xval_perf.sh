#!/usr/bin/env bash
# D2/D3 interleaved instruction-count A/B on the Intel guest.
#   D2 (engine A vs engine B, SAME asm-off binary via kill-switch):
#       A  = gz-asmoff (engine A, flat)         B = gz-asmoff GZIPPY_FLAT_CLEAN=0 (engine B)
#   D3 (engine A asm-off vs run_contig asm-on):
#       A  = gz-asmoff (engine A)               B = gz-asmon (x86 BMI2 run_contig)
#   self-test A2 = re-run of arm A (same config) -> A2/A must be ~1.0 (licenses ratios)
# Deterministic primitive: cpu_core/instructions/. Pinned to P-core cpu4, -p1,
# /dev/null sink BOTH arms, interleaved A,B,A2 per rep, N reps.
set -uo pipefail
O=/dev/shm/ixv
PIN="taskset -c 4"
N=${N:-15}
CORPORA="silesia monorepo nasa"
OUT=/dev/shm/ixv_perf.csv
: > "$OUT"
echo "test,corpus,arm,rep,instructions,cycles" >> "$OUT"

instr() {  # $@ = command; echoes "instr cycles" (cpu_core). SINK LAW: cmd stdout->/dev/null
  perf stat -x, -e cpu_core/instructions/,cpu_core/cycles/ -o /tmp/ps.txt $PIN "$@" >/dev/null 2>/dev/null
  local i c
  i=$(awk -F, '/cpu_core\/instructions\// {print $1; exit}' /tmp/ps.txt)
  c=$(awk -F, '/cpu_core\/cycles\// {print $1; exit}' /tmp/ps.txt)
  echo "$i $c"
}

# warm page cache
for c in $CORPORA; do cat /root/$c.gz >/dev/null; done

for c in $CORPORA; do
  for ((r=1;r<=N;r++)); do
    # ---- D2 ----
    a=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    b=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    a2=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    echo "D2,$c,A_engineA,$r,${a/ /,}" >> "$OUT"
    echo "D2,$c,B_engineB,$r,${b/ /,}" >> "$OUT"
    echo "D2,$c,A2_self,$r,${a2/ /,}" >> "$OUT"
    # ---- D3 ----
    d=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    e=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmon"  -d -c -p1 /root/$c.gz)
    d2=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    echo "D3,$c,A_engineA_asmoff,$r,${d/ /,}" >> "$OUT"
    echo "D3,$c,B_runcontig_asmon,$r,${e/ /,}" >> "$OUT"
    echo "D3,$c,A2_self,$r,${d2/ /,}" >> "$OUT"
  done
done
echo "IXV_PERF_DONE rows=$(wc -l < "$OUT")"
