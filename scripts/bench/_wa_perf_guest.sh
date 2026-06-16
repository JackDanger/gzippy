#!/usr/bin/env sh
# _wa_perf_guest.sh — STAGE b: name the window-absent inflate mechanism.
# perf stat (cyc/instr/IPC) + perf record/report (symbol share) + perf annotate
# (hot source lines) for BOTH gz (native symboled) and rg, silesia T4.
set -u
GZ="${GZ:-/dev/shm/wa/gz-native-sym}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
C="${C:-/root/silesia.gz}"
T="${T:-4}"; MASK="${MASK:-0-7}"; REPS="${REPS:-7}"; PERIOD="${PERIOD:-200000}"
ART="${ART:-/dev/shm/wa}"; mkdir -p "$ART"
SINK="$ART/sink.bin"; rm -f "$SINK"
REFSHA="028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f"

EV="instructions:u"
if perf stat -e cpu_core/instructions/ -- true >/dev/null 2>&1; then
  IEV="cpu_core/instructions/u"; CEV="cpu_core/cycles/u"
else IEV="instructions:u"; CEV="cycles:u"; fi
echo "=== events: $IEV $CEV ; MASK=$MASK T=$T REPS=$REPS ==="

# decoded byte count (window-absent ~98% of this)
NBYTES="$($GZ -d -c -p "$T" "$C" 2>/dev/null | wc -c)"
echo "=== decoded bytes = $NBYTES ==="

echo "############ perf stat gz ############"
perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/gz.stat" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$C" >"$SINK" 2>>"$ART/run.err"
s1="$(sha256sum "$SINK" | cut -d' ' -f1)"; [ "$s1" = "$REFSHA" ] && echo "  gz SHA OK" || echo "  !! gz SHA MISMATCH $s1"
grep -E 'instructions|cycles|elapsed' "$ART/gz.stat" | sed 's/^/  /'

echo "############ perf stat rg ############"
perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/rg.stat" -- \
  taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$C" >"$SINK" 2>>"$ART/run.err"
s2="$(sha256sum "$SINK" | cut -d' ' -f1)"; [ "$s2" = "$REFSHA" ] && echo "  rg SHA OK" || echo "  !! rg SHA MISMATCH $s2"
grep -E 'instructions|cycles|elapsed' "$ART/rg.stat" | sed 's/^/  /'

echo "############ MFAST_PROF gz (rdtsc cyc/event mfast vs careful) ############"
GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MFAST_PROF=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$C" >"$SINK" 2>"$ART/mfastprof.txt" || true
grep -iE 'mfast|careful|cyc|event' "$ART/mfastprof.txt" | head -20 | sed 's/^/  /'

echo "############ perf record gz (cycles -c $PERIOD) ############"
perf record -e "$CEV" -c "$PERIOD" -g -o "$ART/gz.data" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$C" >"$SINK" 2>>"$ART/run.err"
echo "--- gz top symbols (self) ---"
perf report -i "$ART/gz.data" --stdio --no-children 2>/dev/null | grep -vE '^#' | grep -E '[0-9]+\.[0-9]+%' | head -25 | sed 's/^/  /'

echo "############ perf record rg (cycles -c $PERIOD) ############"
perf record -e "$CEV" -c "$PERIOD" -g -o "$ART/rg.data" -- \
  taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$C" >"$SINK" 2>>"$ART/run.err"
echo "--- rg top symbols (self) ---"
perf report -i "$ART/rg.data" --stdio --no-children 2>/dev/null | grep -vE '^#' | grep -E '[0-9]+\.[0-9]+%' | head -25 | sed 's/^/  /'

echo "WA_PERF_DONE"
