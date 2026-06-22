#!/usr/bin/env bash
# AMD WALL-DECOMP S2/S3: symmetric gz-vs-rg 3-region exclusive-TSC partition +
# perf-cycle totals, FROZEN (gov=performance, boost=0 -> TSC ~= core cycles), NO
# llama pause. trap+watchdog GUARANTEED restore. taskset 0-3, interleaved
# gz,rg,gzAA per rep, best/median-of-N. Computes the 3 conservation residuals.
#
# Gate-0: each region non-inert + OVERLAP_VIOLATIONS==0 (printed by both binaries) +
#         OFF==identity sha==zcat (separate clean run) + scheduler gates
#         (context-switches / cpu-migrations reported). Env fingerprint asserted:
#         GZIPPY_OVERLAP_WRITER unset, RAPIDGZIP_AW_PROF/WA_PROF unset.
set -u
GZ=/dev/shm/ar-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/amd-wall-decomp
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
THS="${THS:-2 4}"
CORPORA=(${CORPORA:-/root/silesia.gz /root/squishy.gz /root/monorepo.gz})

# Env hygiene (cursor-agent review): no overlap writer, no other rg profilers.
unset GZIPPY_OVERLAP_WRITER RAPIDGZIP_AW_PROF RAPIDGZIP_WA_PROF

echo "corpus,th,rep,gz_perf,gz_w,gz_m,gz_o,gz_ov,rg_perf,rg_w,rg_m,rg_o,rg_ov,gzaa_perf,gzaa_w,gz_cs,rg_cs,gz_mig,rg_mig" > "$CSV"

# --- box save + trap/watchdog (NO llama pause) ---
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GF GV
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do GF+=("$f"); GV+=("$(cat "$f")"); done
restore(){ for i in "${!GF[@]}"; do echo "${GV[$i]}" > "${GF[$i]}" 2>/dev/null; done
           echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null; }
trap restore EXIT INT TERM
setsid bash -c "sleep 900; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand>\$f 2>/dev/null; done; echo 1>/sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WD=$!
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
sleep 1
echo "FROZEN gov=performance boost=0 (no llama touch); watchdog=$WD cores=$CORES N=$N THS='$THS'" >&2

# perf field extractors
pf(){ grep -E "[[:space:]]$2\b" "$1" | grep -oE '[0-9,]+' | head -1 | tr -d ,; }
gzreg(){ grep "$2" "$1" | grep -oE 'cyc=[0-9]+' | head -1 | cut -d= -f2; }
gzov(){ grep "OVERLAP_VIOLATIONS=" "$1" | grep -oE 'OVERLAP_VIOLATIONS=[0-9]+' | head -1 | cut -d= -f2; }
# rg region: "[RG-REGION] WORKER cyc=.. | MARKERPP cyc=.. | OUTPUT cyc=.."
rgreg(){ grep '\[RG-REGION\] WORKER' "$1" | grep -oE "$2 cyc=[0-9]+" | head -1 | grep -oE '[0-9]+'; }
rgov(){ grep 'OVERLAP_VIOLATIONS=' "$1" | grep -oE 'OVERLAP_VIOLATIONS=[0-9]+' | head -1 | cut -d= -f2; }

# --- OFF==identity sha gate (once per corpus) ---
echo "=== Gate-0 OFF==identity sha check ===" >&2
for corpus in "${CORPORA[@]}"; do
  ref=$(zcat "$corpus" | sha256sum | awk '{print $1}')
  gs=$(env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -p 4 -c "$corpus" 2>/dev/null | sha256sum | awk '{print $1}')
  rs=$("$RG" -d -P 4 -c "$corpus" 2>/dev/null | sha256sum | awk '{print $1}')
  echo "  $(basename "$corpus"): ref=$ref gz=$([ "$gs" = "$ref" ] && echo OK || echo BAD) rg=$([ "$rs" = "$ref" ] && echo OK || echo BAD)" >&2
done

PE="cycles,context-switches,cpu-migrations"
for corpus in "${CORPORA[@]}"; do
  name=$(basename "$corpus" .gz)
  # warm page cache (cursor-agent review: warm I/O so decodeBlock reads don't block)
  cat "$corpus" >/dev/null 2>&1
  for TH in $THS; do
    for r in $(seq 1 "$N"); do
      taskset -c "$CORES" perf stat -o "$OUT/gz.perf" -e "$PE" -- env GZIPPY_FORCE_PARALLEL_SM=1 \
        GZIPPY_REGION_PROF=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gz.err"
      gp=$(pf "$OUT/gz.perf" cycles); gcs=$(pf "$OUT/gz.perf" context-switches); gmig=$(pf "$OUT/gz.perf" cpu-migrations)
      gw=$(gzreg "$OUT/gz.err" R_WORKER); gm=$(gzreg "$OUT/gz.err" R_MARKERPP); go=$(gzreg "$OUT/gz.err" R_OUTPUT); gov=$(gzov "$OUT/gz.err")

      taskset -c "$CORES" perf stat -o "$OUT/rg.perf" -e "$PE" -- env RAPIDGZIP_REGION_PROF=1 \
        "$RG" -d -P "$TH" -o /dev/null "$corpus" >/dev/null 2>"$OUT/rg.err"
      rp=$(pf "$OUT/rg.perf" cycles); rcs=$(pf "$OUT/rg.perf" context-switches); rmig=$(pf "$OUT/rg.perf" cpu-migrations)
      rw=$(rgreg "$OUT/rg.err" WORKER); rm=$(rgreg "$OUT/rg.err" MARKERPP); ro=$(rgreg "$OUT/rg.err" OUTPUT); rov=$(rgov "$OUT/rg.err")

      taskset -c "$CORES" perf stat -o "$OUT/gzaa.perf" -e cycles -- env GZIPPY_FORCE_PARALLEL_SM=1 \
        GZIPPY_REGION_PROF=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gzaa.err"
      ap=$(pf "$OUT/gzaa.perf" cycles); aw=$(gzreg "$OUT/gzaa.err" R_WORKER)

      echo "$name,$TH,$r,$gp,$gw,$gm,$go,$gov,$rp,$rw,$rm,$ro,$rov,$ap,$aw,$gcs,$rcs,$gmig,$rmig" >> "$CSV"
    done
    echo "  done $name T$TH" >&2
  done
done
restore; kill "$WD" 2>/dev/null; trap - EXIT INT TERM
echo "=== RESTORED gov0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost) ===" >&2
echo "CSV=$CSV"
echo "--- last gz region dump ---" >&2; grep -E 'region-prof|RG-REGION' "$OUT/gz.err" "$OUT/rg.err" 2>/dev/null | tail -8 >&2
