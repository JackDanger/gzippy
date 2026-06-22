#!/usr/bin/env bash
# ZEN2 STEP-1 CEILING ORACLE timed sweep (5-arm interleave, Gate-0 done separately).
# Arms per r: GZ(base) RG GZC(u8) GZCU(u16-consumer-serial) GZCW(u16-worker-parallel)
#             GZ2 RG2 (A/A spread). All decode to /dev/null (SINK LAW).
# Ceiling outputs are wrong-on-purpose (perturbation) — wall captured by perf
# duration_time; terminal CRC fail is AFTER the full decode+resolve+write wall.
set -u
: "${GZ:?}"; : "${RG:?}"; : "${CORPUS_DIR:=/root}"; : "${OUT:=/dev/shm/zen-ceil-out}"
: "${N:=9}"; : "${CELLS:?}"
mkdir -p "$OUT"; LOG="$OUT/measure.log"; exec > "$LOG" 2>&1
rm -f "$OUT/DONE"
echo "== ZEN ceiling measure $(date -u +%FT%TZ) =="
echo "host: $(uname -srm) cores=$(nproc)  load_start: $(cat /proc/loadavg)"
echo "gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
echo "CELLS=$CELLS N=$N"

run(){ perf stat -x, -e duration_time -- "$@" >/dev/null 2>/tmp/pf.$$; grep ',ns,duration_time' /tmp/pf.$$|cut -d, -f1; }
gz(){    run env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p"$1" "$2"; }
gzc8(){  run env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING=1      "$GZ" -d -c -p"$1" "$2"; }
gzcu(){  run env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING_U16=1  "$GZ" -d -c -p"$1" "$2"; }
gzcw(){  run env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING_U16W=1 "$GZ" -d -c -p"$1" "$2"; }
rg(){    run "$RG" -d -c -P"$1" "$2"; }

RAW="$OUT/raw.csv"; : > "$RAW"
echo "--- interleaved (GZ,RG,GZC,GZCU,GZCW,GZ2,RG2) N=$N ---"
for cell in $CELLS; do
  corp="${cell%%:*}"; T="${cell##*:}"; F="$CORPUS_DIR/$corp.gz"
  [ -f "$F" ] || { echo "MISSING $F"; continue; }
  echo "  cell $corp T$T ..."
  for r in $(seq 1 "$N"); do
    echo "$corp,$T,GZ,$r,$(gz    "$T" "$F")"  >> "$RAW"
    echo "$corp,$T,RG,$r,$(rg    "$T" "$F")"  >> "$RAW"
    echo "$corp,$T,GZC,$r,$(gzc8 "$T" "$F")"  >> "$RAW"
    echo "$corp,$T,GZCU,$r,$(gzcu "$T" "$F")" >> "$RAW"
    echo "$corp,$T,GZCW,$r,$(gzcw "$T" "$F")" >> "$RAW"
    echo "$corp,$T,GZ2,$r,$(gz    "$T" "$F")" >> "$RAW"
    echo "$corp,$T,RG2,$r,$(rg    "$T" "$F")" >> "$RAW"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "=== ANALYZE ==="
python3 "$OUT/report.py" "$RAW" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$OUT/DONE"
echo "=== ZEN_DONE $(date -u +%FT%TZ) ==="
