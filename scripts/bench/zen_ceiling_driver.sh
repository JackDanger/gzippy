#!/usr/bin/env bash
# ZEN ceiling AMD driver: pause llama (GUARANTEED resume) + freeze + TIMED SWEEP + restore.
# Gate-0 is run SEPARATELY and UNPAUSED (correctness is load-independent) so the freeze
# window holds only the timed sweep. HARD NO-ORPHAN: short (<=600s) guaranteed-resume
# watchdog + EXIT/INT/TERM trap restore; caller runs this SYNCHRONOUSLY in-turn.
set -u
OUT=/dev/shm/zen-ceil-out
mkdir -p "$OUT"
LOG="$OUT/driver.log"; exec > "$LOG" 2>&1
PIDFILE="$OUT/llama.pids"
WATCHDOG_S="${WATCHDOG_S:-600}"

echo "== ZEN ceiling driver $(date -u +%FT%TZ) =="
PIDS="$( { pgrep -x llama-server; pgrep -f '/qsweep/chain_final.sh'; pgrep -f 'qsweep/niah2.py'; pgrep -f 'qsweep/eval'; } | sort -un | tr '\n' ' ')"
echo "$PIDS" | tr ' ' '\n' | grep -E '^[0-9]+$' > "$PIDFILE"
echo "workload PIDs to pause: $(tr '\n' ' ' < "$PIDFILE")"
ps -o pid,ppid,stat,comm $(tr '\n' ' ' < "$PIDFILE") 2>/dev/null || true

ORIG_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null||echo ondemand)"
ORIG_BOOST="$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null||echo 1)"
echo "orig gov=$ORIG_GOV boost=$ORIG_BOOST"

restore(){
  echo "=== RESTORE $(date -u +%FT%TZ) ==="
  while read -r p; do [ -n "$p" ] && kill -CONT "$p" 2>/dev/null; done < "$PIDFILE"
  echo ondemand | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1
  echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
  echo "post-restore gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
  echo "workload states after CONT:"; while read -r p; do [ -n "$p" ] && ps -o pid,stat,comm -p "$p" 2>/dev/null|tail -1; done < "$PIDFILE"
}
trap restore EXIT INT TERM

setsid bash -c "sleep $WATCHDOG_S; while read -r p; do [ -n \"\$p\" ] && kill -CONT \"\$p\" 2>/dev/null; done < '$PIDFILE'; echo ondemand | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1; echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null; echo watchdog-fired-$(date -u +%FT%TZ) >> '$OUT/watchdog.log'" >/dev/null 2>&1 < /dev/null &
WD=$!
echo "watchdog pid=$WD ttl=${WATCHDOG_S}s"

while read -r p; do [ -n "$p" ] && kill -STOP "$p" 2>/dev/null; done < "$PIDFILE"
sleep 1
echo "paused states:"; while read -r p; do [ -n "$p" ] && ps -o pid,stat,comm -p "$p" 2>/dev/null|tail -1; done < "$PIDFILE"
echo "loadavg right after pause: $(cat /proc/loadavg)"
sleep 3
echo "loadavg +3s: $(cat /proc/loadavg)"

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
echo "frozen gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"

cp /dev/shm/zen_ceiling_report.py "$OUT/report.py"
GZ=/dev/shm/zen-tgt/release/gzippy \
RG=/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip \
CORPUS_DIR=/root OUT="$OUT" N="${N:-9}" \
CELLS="${CELLS:-silesia:4 squishy:4 silesia:2 monorepo:2 nasa:4}" \
bash /dev/shm/zen_ceiling_measure.sh
echo "measure rc=$?"

kill "$WD" 2>/dev/null
echo "=== ZEN driver done, trap will restore $(date -u +%FT%TZ) ==="
