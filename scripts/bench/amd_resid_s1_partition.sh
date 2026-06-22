#!/usr/bin/env bash
# S1: gz exclusive-TSC region partition + perf totals (gz & rg), FROZEN so TSC ~=
# core cycles (gov=performance, boost=0). NO llama pause. trap+watchdog GUARANTEED
# restore. taskset 0-3, interleaved gz,rg per rep, best-of-N. Captures gz perf_total
# + R_WORKER/R_MARKERPP/R_OUTPUT (TSC) -> R_OTHER = perf_total - sum. rg perf_total.
set -u
GZ=/dev/shm/ar-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/amd-resid-s1
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
TH="${TH:-4}"
CORPORA=(/root/silesia.gz /root/squishy.gz)
echo "corpus,rep,gz_perf,gz_worker,gz_markerpp,gz_output,rg_perf" > "$CSV"

# --- box save + trap/watchdog (NO llama pause) ---
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GF GV
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do GF+=("$f"); GV+=("$(cat "$f")"); done
restore(){ for i in "${!GF[@]}"; do echo "${GV[$i]}" > "${GF[$i]}" 2>/dev/null; done
           echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null; }
trap restore EXIT INT TERM
setsid bash -c "sleep 600; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand>\$f 2>/dev/null; done; echo 1>/sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WD=$!
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
sleep 1
echo "FROZEN gov=performance boost=0 (no llama touch); watchdog=$WD cores=$CORES N=$N TH=$TH" >&2

perfcyc(){ grep -E 'cycles' "$1" | grep -vE 'region|TSC' | grep -oE '[0-9,]+' | head -1 | tr -d ,; }
reg(){ grep "$1" "$2" | grep -oE 'cyc= *[0-9]+' | head -1 | grep -oE '[0-9]+'; }

for corpus in "${CORPORA[@]}"; do
  name=$(basename "$corpus" .gz)
  for r in $(seq 1 "$N"); do
    taskset -c "$CORES" perf stat -o "$OUT/gz.perf" -e cycles -- env GZIPPY_FORCE_PARALLEL_SM=1 \
      GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$TH" -c "$corpus" \
      >/dev/null 2>"$OUT/gz.err"
    gp=$(perfcyc "$OUT/gz.perf")
    gw=$(reg R_WORKER "$OUT/gz.err"); gm=$(reg R_MARKERPP "$OUT/gz.err"); go=$(reg R_OUTPUT "$OUT/gz.err")
    taskset -c "$CORES" perf stat -o "$OUT/rg.perf" -e cycles -- env RAPIDGZIP_WA_PROF= "$RG" -d -P "$TH" -o /dev/null "$corpus" \
      >/dev/null 2>"$OUT/rg.err"
    rp=$(perfcyc "$OUT/rg.perf")
    echo "$name,$r,$gp,$gw,$gm,$go,$rp" >> "$CSV"
  done
  echo "  done $name" >&2
done
restore; kill "$WD" 2>/dev/null; trap - EXIT INT TERM
echo "=== RESTORED gov0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost) ===" >&2
echo "CSV=$CSV"
