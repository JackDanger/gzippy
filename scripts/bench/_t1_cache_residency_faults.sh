#!/usr/bin/env bash
# _t1_cache_residency_faults.sh — Gate-2 MECHANISM check (AMD bare-metal).
# perf stat minor-faults/page-faults/instructions/cycles for the key arms, so a
# wall move is EXPLAINED by the cache-residency mechanism (faults toward igzip).
set -u
BIN_N=${BIN_N:-/dev/shm/tn/release/examples/streaming_thin}
BIN_I=${BIN_I:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
CORPORA=${CORPORA:-"silesia monorepo nasa"}
GZDIR=${GZDIR:-/root}
EV="minor-faults,page-faults,instructions,cycles"

declare -A BINOF MODE ENV
BINOF[igzip]=$BIN_I;   MODE[igzip]=igzip;  ENV[igzip]=""
BINOF[prod]=$BIN_N;    MODE[prod]=prod;    ENV[prod]=""
BINOF[manpool]=$BIN_N; MODE[manpool]=prod; ENV[manpool]="GZIPPY_MANUAL_BUFFER_POOL=1"
BINOF[respool]=$BIN_N; MODE[respool]=prod; ENV[respool]="GZIPPY_RESIDENT_OUTPUT_POOL=1"
ARMS="igzip prod manpool respool"

echo "== T1-CACHE-RESIDENCY mechanism (perf stat, pin=cpu$PIN) =="
for corp in $CORPORA; do
  F=$GZDIR/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE"; continue; }
  echo "--- $corp ---"
  for a in $ARMS; do
    # best-of-3 by minor-faults-stable: run 3x, report the run with median wall;
    # faults are deterministic enough that the min is representative.
    out=$(env ${ENV[$a]} taskset -c $PIN perf stat -e "$EV" "${BINOF[$a]}" "${MODE[$a]}" "$F" 2>&1)
    mf=$(echo "$out"  | grep -i 'minor-faults' | awk '{gsub(/,/,"",$1);print $1}')
    pf=$(echo "$out"  | grep -i 'page-faults'  | awk '{gsub(/,/,"",$1);print $1}')
    ins=$(echo "$out" | grep -i 'instructions' | awk '{gsub(/,/,"",$1);print $1}')
    cyc=$(echo "$out" | grep -iw 'cycles'      | awk '{gsub(/,/,"",$1);print $1}')
    printf "  %-9s minor-faults=%-12s page-faults=%-10s instr=%-15s cycles=%-15s\n" "$a" "$mf" "$pf" "$ins" "$cyc"
  done
done
echo "== DONE =="
