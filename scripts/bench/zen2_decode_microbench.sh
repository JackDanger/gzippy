#!/usr/bin/env bash
# ZEN2-DECODE-MICROBENCH driver (runs ON solvency). Symmetric window-absent
# (marker) decode cyc/B: gz (GZIPPY_MFAST_PROF) vs rg (RAPIDGZIP_WA_PROF).
# FROZEN box + llama SIGSTOP'd for the timed window; GUARANTEED resume
# (trap + <=600s watchdog). Interleaved best-of-N. cyc/B is intensive.
set -u

GZ=/dev/shm/zen-mb-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/zen2-mb-out
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
THREADS="${THREADS:-4}"
CORPORA=(/root/silesia.gz /root/squishy.gz /root/monorepo.gz)

echo "tool,corpus,rep,cyc,bytes,cyc_per_byte" > "$CSV"

# ---- box state save ----
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GOV_FILES GOV_VALS
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  GOV_FILES+=("$f"); GOV_VALS+=("$(cat "$f")")
done
LLAMA_PIDS=$(pgrep -x llama-server | tr '\n' ' ')

restore() {
  for p in $LLAMA_PIDS; do kill -CONT "$p" 2>/dev/null; done
  for i in "${!GOV_FILES[@]}"; do echo "${GOV_VALS[$i]}" > "${GOV_FILES[$i]}" 2>/dev/null; done
  echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
}
trap 'restore' EXIT INT TERM
# detached guaranteed-resume backstop (<=600s)
setsid bash -c "sleep 600; for p in $LLAMA_PIDS; do kill -CONT \$p 2>/dev/null; done; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand > \$f 2>/dev/null; done; echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WATCHDOG=$!

# ---- freeze + pause ----
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
for p in $LLAMA_PIDS; do kill -STOP "$p" 2>/dev/null; done
sleep 1
echo "FROZEN gov=performance boost=0; llama STOPPED: $LLAMA_PIDS; watchdog=$WATCHDOG cores=$CORES N=$N" >&2

gz_run() {  # corpus -> "cyc bytes cpb"
  taskset -c "$CORES" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MFAST_PROF=1 GZIPPY_VERBOSE=1 \
    "$GZ" -d -p "$THREADS" -c "$1" 2>"$OUT/gz.err" >/dev/null
  # cyc = MFAST_CYC+CAREFUL_CYC ; bytes = total_bytes ; cpb from the WA-CYCB line
  awk '/\[WA-CYCB\] gz_marker_decode_cyc_per_byte/{cpb=$2} END{}' "$OUT/gz.err" >/dev/null
  local cpb b
  cpb=$(grep -oE 'gz_marker_decode_cyc_per_byte=[0-9.]+' "$OUT/gz.err" | head -1 | cut -d= -f2)
  b=$(grep -oE 'total_bytes=[0-9]+' "$OUT/gz.err" | head -1 | cut -d= -f2)
  local c
  c=$(awk -v cpb="$cpb" -v b="$b" 'BEGIN{printf "%.0f", cpb*b}')
  echo "$c $b $cpb"
}
rg_run() {
  taskset -c "$CORES" env RAPIDGZIP_WA_PROF=1 "$RG" -d -P "$THREADS" -o /dev/null "$1" 2>"$OUT/rg.err" >/dev/null
  local line c b cpb
  line=$(grep 'RG-WA-CYCB' "$OUT/rg.err" | head -1)
  c=$(echo "$line"  | grep -oE 'cyc=[0-9]+' | cut -d= -f2)
  b=$(echo "$line"  | grep -oE 'bytes=[0-9]+' | cut -d= -f2)
  cpb=$(echo "$line"| grep -oE 'rg_marker_decode_cyc_per_byte=[0-9.]+' | cut -d= -f2)
  echo "$c $b $cpb"
}

for corpus in "${CORPORA[@]}"; do
  name=$(basename "$corpus" .gz)
  for r in $(seq 1 "$N"); do
    read gc gb gcpb <<<"$(gz_run "$corpus")"
    echo "gz,$name,$r,$gc,$gb,$gcpb" >> "$CSV"
    read rc rb rcpb <<<"$(rg_run "$corpus")"
    echo "rg,$name,$r,$rc,$rb,$rcpb" >> "$CSV"
    # A/A: a second gz run interleaved
    read gc2 gb2 gcpb2 <<<"$(gz_run "$corpus")"
    echo "gzAA,$name,$r,$gc2,$gb2,$gcpb2" >> "$CSV"
  done
  echo "  done $name" >&2
done

restore
kill "$WATCHDOG" 2>/dev/null
trap - EXIT INT TERM
echo "=== BOX RESTORED ===" >&2
for p in $LLAMA_PIDS; do echo -n "llama $p: "; ps -o stat= -p "$p" 2>/dev/null || echo gone; done >&2
echo "gov0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost)" >&2
echo "CSV=$CSV"
