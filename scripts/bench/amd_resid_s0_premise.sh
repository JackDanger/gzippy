#!/usr/bin/env bash
# S0: re-confirm the AMD-T4 (+T2) WORK-bound premise at HEAD. gz_total vs rg_total
# perf `cycles` (same metric both arms = valid gz-vs-rg comparison). taskset-isolate,
# interleaved gz,rg,gzAA, best-of-N. NO freeze, NO llama pause (load-robust via taskset
# + GHz-stability gate). perf cycles + duration_time -> achieved GHz spread gate.
set -u
GZ=/dev/shm/ar-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/amd-resid-s0
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
echo "tool,corpus,threads,rep,cycles,instructions,dur_ns,ghz" > "$CSV"
CORPORA=(/root/silesia.gz /root/monorepo.gz /root/squishy.gz)
THREADLIST=(${THREADS:-2 4})

run() { # tool corpus threads -> appends row; echoes "cycles dur"
  local tool="$1" corpus="$2" th="$3" name perfout
  name=$(basename "$corpus" .gz)
  perfout="$OUT/perf.$$"
  if [ "$tool" = "rg" ]; then
    taskset -c "$CORES" perf stat -e cycles,instructions,duration_time -- \
      env RAPIDGZIP_WA_PROF= "$RG" -d -P "$th" -o /dev/null "$corpus" >/dev/null 2>"$perfout"
  else
    taskset -c "$CORES" perf stat -e cycles,instructions,duration_time -- \
      env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -p "$th" -c "$corpus" >/dev/null 2>"$perfout"
  fi
  local cyc insn dur ghz
  cyc=$(grep -E '[0-9].*cycles' "$perfout" | head -1 | grep -oE '[0-9,]+' | head -1 | tr -d ,)
  insn=$(grep -E '[0-9].*instructions' "$perfout" | head -1 | grep -oE '[0-9,]+' | head -1 | tr -d ,)
  dur=$(grep -E 'duration_time' "$perfout" | head -1 | grep -oE '[0-9,]+' | head -1 | tr -d ,)
  ghz=$(awk -v c="$cyc" -v d="$dur" 'BEGIN{ if(d>0) printf "%.4f", c/d; else print "0"}')
  echo "$tool,$name,$th,$REP,$cyc,$insn,$dur,$ghz" >> "$CSV"
  rm -f "$perfout"
}

for th in "${THREADLIST[@]}"; do
  for corpus in "${CORPORA[@]}"; do
    for REP in $(seq 1 "$N"); do
      run gz   "$corpus" "$th"
      run rg   "$corpus" "$th"
      run gzAA "$corpus" "$th"
    done
    echo "  done $(basename "$corpus") T$th" >&2
  done
done
echo "CSV=$CSV"
