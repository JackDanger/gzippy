#!/usr/bin/env sh
# _perf_contention_guest.sh — STEP 1 contention/cache-residency characterization.
# Runs perf stat (cpu_core PMU, P-cores) on gzippy-isal vs rapidgzip at one T,
# capturing cache/TLB/LLC + context-switch/migration/page-fault counters with
# per-run variance (perf -r REPS gives mean +- stddev%). Read-only; sink to
# /dev/shm regular file; sha-verified.
#
# Env (exported by the local wrapper): GZ RG CORPUS MASK T REPS ART REFSHA
set -u
: "${GZ:?}"; : "${RG:?}"; : "${CORPUS:?}"; : "${MASK:?}"; : "${T:?}"
REPS="${REPS:-6}"; ART="${ART:-/dev/shm/perfc}"; REFSHA="${REFSHA:-}"
mkdir -p "$ART"
SINK="$ART/sink.bin"; rm -f "$SINK"

# cpu_core (P-core) PMU events. Two groups to limit multiplexing.
G_MEM="cpu_core/cycles/,cpu_core/instructions/,cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/,cpu_core/dTLB-load-misses/,cpu_core/L1-dcache-load-misses/"
G_SW="cycles,instructions,context-switches,cpu-migrations,page-faults,task-clock"

run_one() { # <label> <group> <env-prefix> <cmd...>
  label="$1"; grp="$2"; shift 2
  out="$ART/${label}.perf"
  # perf stat -r REPS: repeats the workload, reports mean +- stddev (the variance).
  perf stat -r "$REPS" -e "$grp" -o "$out" -- "$@" >"$SINK" 2>>"$ART/run.stderr"
  sha="$(sha256sum "$SINK" | cut -d' ' -f1)"
  if [ -n "$REFSHA" ] && [ "$sha" != "$REFSHA" ]; then echo "!! SHA MISMATCH $label sha=$sha"; fi
  echo "---- $label ----"
  # Print the counter lines + the elapsed-time variance line.
  grep -E 'cache-misses|cache-references|LLC-load|dTLB-load|L1-dcache-load-misses|context-switches|cpu-migrations|page-faults|instructions|cycles|seconds time elapsed|task-clock' "$out" \
    | sed 's/^ */   /'
}

echo "################ T=$T MASK=$MASK REPS=$REPS ################"
echo "==== MEMORY/CACHE GROUP ===="
run_one "gz_mem_T$T"  "$G_MEM" env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS"
run_one "rg_mem_T$T"  "$G_MEM" taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS"
echo "==== SW/CONTENTION GROUP ===="
run_one "gz_sw_T$T"   "$G_SW"  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS"
run_one "rg_sw_T$T"   "$G_SW"  taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS"
rm -f "$SINK"
echo "PERFC_T${T}_DONE"
