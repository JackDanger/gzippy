#!/usr/bin/env bash
# ZEN marker-loop SLOWDOWN perturbation (Gate-2): does slowing the window-absent
# marker decode move the AMD-T4 wall? spin + sleep(freq-neutral control). Pauses
# llama (guaranteed resume) + freeze + measure + restore. Byte-transparent knob
# (sha verified ==ref pre-run). VERDICT: monotonic+proportional wall rise that
# SURVIVES the sleep control => marker loop on the T4 critical wall path.
set -u
OUT=/dev/shm/zen-slow-out; mkdir -p "$OUT"
LOG="$OUT/driver.log"; exec > "$LOG" 2>&1
PIDFILE="$OUT/llama.pids"; WATCHDOG_S="${WATCHDOG_S:-1800}"
GZ=/dev/shm/zen-tgt/release/gzippy
RG=/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
N="${N:-9}"; MODES="${MODES:-0 50 100 200}"; CELLS="${CELLS:-silesia:4 squishy:4 silesia:2}"

echo "== ZEN slowdown driver $(date -u +%FT%TZ) =="
PIDS="$( { pgrep -x llama-server; pgrep -f '/qsweep/chain_final.sh'; pgrep -f 'qsweep/niah2.py'; pgrep -f 'qsweep/eval'; } | sort -un | tr '\n' ' ')"
echo "$PIDS" | tr ' ' '\n' | grep -E '^[0-9]+$' > "$PIDFILE"
echo "workload PIDs: $(tr '\n' ' ' < "$PIDFILE")"
ORIG_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null||echo ondemand)"
ORIG_BOOST="$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null||echo 1)"
echo "orig gov=$ORIG_GOV boost=$ORIG_BOOST"
restore(){
  echo "=== RESTORE $(date -u +%FT%TZ) ==="
  while read -r p; do [ -n "$p" ] && kill -CONT "$p" 2>/dev/null; done < "$PIDFILE"
  echo "$ORIG_GOV" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1
  echo "$ORIG_BOOST" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
  echo "post-restore gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"
  while read -r p; do [ -n "$p" ] && ps -o pid,stat,comm -p "$p" 2>/dev/null|tail -1; done < "$PIDFILE"
}
trap restore EXIT INT TERM
setsid bash -c "sleep $WATCHDOG_S; while read -r p; do [ -n \"\$p\" ] && kill -CONT \"\$p\" 2>/dev/null; done < '$PIDFILE'; echo '$ORIG_GOV' | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1; echo '$ORIG_BOOST' > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 < /dev/null &
WD=$!; echo "watchdog pid=$WD ttl=${WATCHDOG_S}s"
while read -r p; do [ -n "$p" ] && kill -STOP "$p" 2>/dev/null; done < "$PIDFILE"
sleep 1; echo "loadavg after pause: $(cat /proc/loadavg)"; sleep 3; echo "loadavg +3s: $(cat /proc/loadavg)"
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
echo "frozen gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"

# Gate-0 byte-transparent proof (sha==ref at mode 0 and 200, both kinds)
SIL=/root/silesia.gz; REF=$(zcat $SIL|sha256sum|cut -c1-16)
for M in 0 200; do for K in spin sleep; do
  S=$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SLOW_MFAST_MODE=$M GZIPPY_SLOW_KIND=$K $GZ -d -c -p4 $SIL 2>/dev/null|sha256sum|cut -c1-16)
  echo "gate0 mode=$M kind=$K $([ "$S" = "$REF" ]&&echo BYTE-EXACT||echo DIFF-FAIL)"
done; done

RAW="$OUT/raw.csv"; : > "$RAW"
runw(){ perf stat -x, -e duration_time -- env "$@" >/dev/null 2>/tmp/pf.$$; grep ',ns,duration_time' /tmp/pf.$$|cut -d, -f1; }
echo "--- SWEEP interleaved N=$N modes='$MODES' (spin+sleep) + rg anchor ---"
for cell in $CELLS; do
  corp="${cell%%:*}"; T="${cell##*:}"; F="/root/$corp.gz"
  echo "cell $corp T$T"
  for r in $(seq 1 "$N"); do
    # rg anchor
    w=$(runw "$RG" -d -c -P"$T" "$F"); echo "$corp,$T,rg,-,$r,$w" >> "$RAW"
    for K in spin sleep; do
      for M in $MODES; do
        w=$(runw GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SLOW_MFAST_MODE=$M GZIPPY_SLOW_KIND=$K "$GZ" -d -c -p"$T" "$F")
        echo "$corp,$T,$K,$M,$r,$w" >> "$RAW"
      done
    done
  done
done
echo "load_end: $(cat /proc/loadavg)"
cp /dev/shm/zen_slowdown_report.py "$OUT/report.py"
python3 "$OUT/report.py" "$RAW" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$OUT/DONE"
kill "$WD" 2>/dev/null
echo "=== ZEN slowdown done, trap restores $(date -u +%FT%TZ) ==="
