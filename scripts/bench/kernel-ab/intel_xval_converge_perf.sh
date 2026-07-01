#!/usr/bin/env bash
# Interleaved instruction-count A/B for the SOLE-PATH convergence (@cffa61ee), x86.
#   DB (the -2% DOUBLE-BUILD drop): BOTH asm-off, BOTH GZIPPY_FLAT_CLEAN=1 (engine A bulk active)
#       A = gz-asmoff (HEAD cffa61ee: flat resumable careful tail, lazy lut, NO double-build)
#       B = gz-hybrid (c4c3cc97  : engine-B careful tail + per-block engine-B lut_litlen build)
#       B/A>1 => the converged binary uses FEWER instructions (the double-build drop).
#   D2 (engine A vs engine B, SAME asm-off binary kill-switch):
#       A = gz-asmoff (engine A)   B = gz-asmoff GZIPPY_FLAT_CLEAN=0 (engine B)
#   D3 (engine A asm-off vs run_contig asm-on):
#       A = gz-asmoff (engine A)   B = gz-asmon (x86 BMI2 run_contig)
#   self-test A2 = re-run of arm A -> A2/A ~1.0 licenses the ratios.
# Deterministic primitive: cpu_core/instructions/. Pinned P-core cpu4, -p1,
# /dev/null sink BOTH arms, interleaved A,B,A2 per rep, N reps.
set -uo pipefail
O=/dev/shm/ixv
PIN="taskset -c 4"
N=${N:-15}
CORPORA="silesia monorepo nasa"
OUT=/dev/shm/ixv_conv_perf.csv
: > "$OUT"
echo "test,corpus,arm,rep,instructions,cycles" >> "$OUT"

instr() { # $@=cmd; echoes "instr cycles"; SINK LAW: stdout->/dev/null
  perf stat -x, -e cpu_core/instructions/,cpu_core/cycles/ -o /tmp/ps.txt $PIN "$@" >/dev/null 2>/dev/null
  local i c
  i=$(awk -F, '/cpu_core\/instructions\// {print $1; exit}' /tmp/ps.txt)
  c=$(awk -F, '/cpu_core\/cycles\// {print $1; exit}' /tmp/ps.txt)
  echo "$i $c"
}

for c in $CORPORA; do cat /root/$c.gz >/dev/null; done

for c in $CORPORA; do
  for ((r=1;r<=N;r++)); do
    # ---- DB: double-build drop (converged vs hybrid, both asm-off, engine A) ----
    a=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    b=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=1 "$O/gz-hybrid" -d -c -p1 /root/$c.gz)
    a2=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    echo "DB,$c,A_converged,$r,${a/ /,}" >> "$OUT"
    echo "DB,$c,B_hybrid,$r,${b/ /,}" >> "$OUT"
    echo "DB,$c,A2_self,$r,${a2/ /,}" >> "$OUT"
    # ---- D2: engine A vs engine B (kill-switch) ----
    d=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    e=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    d2=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    echo "D2,$c,A_engineA,$r,${d/ /,}" >> "$OUT"
    echo "D2,$c,B_engineB,$r,${e/ /,}" >> "$OUT"
    echo "D2,$c,A2_self,$r,${d2/ /,}" >> "$OUT"
    # ---- D3: engine A asm-off vs run_contig asm-on ----
    f=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    g=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmon"  -d -c -p1 /root/$c.gz)
    f2=$(instr env GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/$c.gz)
    echo "D3,$c,A_engineA_asmoff,$r,${f/ /,}" >> "$OUT"
    echo "D3,$c,B_runcontig_asmon,$r,${g/ /,}" >> "$OUT"
    echo "D3,$c,A2_self,$r,${f2/ /,}" >> "$OUT"
  done
done
echo "IXV_CONV_PERF_DONE rows=$(wc -l < "$OUT")"
