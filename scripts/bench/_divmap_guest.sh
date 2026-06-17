#!/usr/bin/env sh
# _divmap_guest.sh — whole-program gz-vs-rg DIVERGENCE MAP (measurement only).
# perf stat (instr/cyc/IPC headline ratio) + perf record/report (per-symbol self%
# for BOTH instructions and cycles) for gz (fresh symboled native @HEAD) and rg,
# on two corpora, T4, /dev/null sink, taskset P-cores. NO code change.
set -u
GZ="${GZ:-/dev/shm/gzmap-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
T="${T:-4}"; MASK="${MASK:-0-7}"; REPS="${REPS:-7}"; PERIOD="${PERIOD:-300000}"
ART="${ART:-/dev/shm/divmap}"; mkdir -p "$ART"

if perf stat -e cpu_core/instructions/ -- true >/dev/null 2>&1; then
  IEV="cpu_core/instructions/u"; CEV="cpu_core/cycles/u"
else IEV="instructions:u"; CEV="cycles:u"; fi
echo "=== events: $IEV $CEV ; MASK=$MASK T=$T REPS=$REPS PERIOD=$PERIOD ==="

run_corpus() {
  CORPUS="$1"; TAG="$2"; REFSHA="$3"
  echo
  echo "################################################################"
  echo "## CORPUS=$TAG  ($CORPUS)"
  echo "################################################################"
  NB="$($GZ -d -c -p "$T" "$CORPUS" 2>/dev/null | wc -c)"
  echo "=== decoded bytes = $NB ==="

  echo "---- perf stat gz ----"
  perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/$TAG.gz.stat" -- \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ART/run.err"
  s="$(env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  [ "$s" = "$REFSHA" ] && echo "  gz SHA OK" || echo "  !! gz SHA MISMATCH $s"
  grep -E 'instructions|cycles|elapsed' "$ART/$TAG.gz.stat" | sed 's/^/  /'

  echo "---- perf stat rg ----"
  perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/$TAG.rg.stat" -- \
    taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS" >/dev/null 2>>"$ART/run.err"
  s="$(taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  [ "$s" = "$REFSHA" ] && echo "  rg SHA OK" || echo "  !! rg SHA MISMATCH $s"
  grep -E 'instructions|cycles|elapsed' "$ART/$TAG.rg.stat" | sed 's/^/  /'

  for EVK in instructions cycles; do
    if [ "$EVK" = instructions ]; then EV="$IEV"; else EV="$CEV"; fi
    echo "---- perf record gz  event=$EVK ----"
    perf record -e "$EV" -c "$PERIOD" -o "$ART/$TAG.gz.$EVK.data" -- \
      env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ART/run.err"
    echo "--- gz top symbols (self %, $EVK) ---"
    perf report -i "$ART/$TAG.gz.$EVK.data" --stdio --no-children 2>/dev/null \
      | grep -vE '^#' | grep -E '[0-9]+\.[0-9]+%' | head -35 | sed 's/^/  /'

    echo "---- perf record rg  event=$EVK ----"
    perf record -e "$EV" -c "$PERIOD" -o "$ART/$TAG.rg.$EVK.data" -- \
      taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS" >/dev/null 2>>"$ART/run.err"
    echo "--- rg top symbols (self %, $EVK) ---"
    perf report -i "$ART/$TAG.rg.$EVK.data" --stdio --no-children 2>/dev/null \
      | grep -vE '^#' | grep -E '[0-9]+\.[0-9]+%' | head -35 | sed 's/^/  /'
  done
}

run_corpus /root/silesia.gz silesia 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f
run_corpus /root/squishy.gz squishy 4ee30c1571688ca4fa34b2c78865fe8c9faef1f67ed82a2ee0e8abf470a3d498
echo DIVMAP_DONE
