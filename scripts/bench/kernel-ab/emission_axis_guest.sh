#!/usr/bin/env bash
# STEP-0 EMISSION AXIS — gated single-core isolated A/B of gz clean-path emission
# kernel (ARM A = Block::decode_clean_into_contig = decode+copy/store+loop) vs
# igzip decode_huffman_code_block_stateless_04 (ARM B), on the SAME real silesia
# clean dynamic block, pinned to ONE P-core, interleaved, best/median-of-N.
#
# Counts are isolated to the kernel LOOP via the DIFFERENCE METHOD: each arm is
# run at reps=R and reps=2R under perf; counts(2R)-counts(R) == exactly R reps of
# pure loop (the one-time setup/compress/warm cancels). Reports per arm:
#   instr/B (load-immune, deterministic), core-cyc/B, IPC, LLC-load-miss/B, GHz.
# Also captures the example's own rdtsc loop-only cyc/B as an independent cross-check.
#
# Gate-0: byte-exact gz==_04==flate2 asserted inside the example (it dies otherwise);
# A/A self-test (split-half ratio per arm ~1.0); GHz-stability self-test;
# non-inert proven by construction (ARM B calls _04 directly; ARM A asserts asm path).
#
# Usage: emission_axis_guest.sh <BIN> <CORE> <R> <N> <OUTCSV>
set -euo pipefail
BIN="${1:?bin}"
CORE="${2:-3}"
R="${3:-2000}"
N="${4:-15}"
OUT="${5:-/dev/shm/emission_axis.csv}"
R2=$((R*2))

EVENTS="cycles,instructions,LLC-load-misses,task-clock"

echo "round,arm,reps,bytes,cycles,instructions,llc_load_misses,task_clock_ms,rdtsc_cyc,rdtsc_cpb" > "$OUT"

run_one() {  # arm reps -> prints "bytes,cycles,instructions,llc,taskclock,rdtsc,rdtsccpb"
  local arm="$1" reps="$2"
  local perffile stdoutfile
  perffile=$(mktemp); stdoutfile=$(mktemp)
  perf stat -x, -e "$EVENTS" taskset -c "$CORE" "$BIN" --arm "$arm" --reps "$reps" \
      >"$stdoutfile" 2>"$perffile" || { echo "RUN FAILED arm=$arm reps=$reps" >&2; cat "$perffile" >&2; exit 3; }
  # stdout: ARM=A bytes=NNN rdtsc_cyc=NNN rdtsc_cyc_per_byte=NNN
  local bytes rdtsc rdtsccpb
  bytes=$(grep -oP '(?<=bytes=)[0-9]+' "$stdoutfile" | tail -1)
  rdtsc=$(grep -oP '(?<=rdtsc_cyc=)[0-9]+' "$stdoutfile" | tail -1)
  rdtsccpb=$(grep -oP '(?<=rdtsc_cyc_per_byte=)[0-9.]+' "$stdoutfile" | tail -1)
  # perf: parse cpu_core lines only (pinned to a P-core; cpu_atom is <not counted>)
  local cyc ins llc tclk
  cyc=$(awk -F, '/cpu_core\/cycles\//{print $1}' "$perffile" | head -1)
  ins=$(awk -F, '/cpu_core\/instructions\//{print $1}' "$perffile" | head -1)
  llc=$(awk -F, '/cpu_core\/LLC-load-misses\//{print $1}' "$perffile" | head -1)
  tclk=$(awk -F, '$3=="task-clock"{print $1}' "$perffile" | head -1)
  rm -f "$perffile" "$stdoutfile"
  if [[ -z "$cyc" || -z "$ins" || -z "$bytes" ]]; then
    echo "PARSE FAILED arm=$arm reps=$reps cyc=$cyc ins=$ins bytes=$bytes" >&2; exit 4
  fi
  echo "$bytes,$cyc,$ins,${llc:-0},$tclk,$rdtsc,$rdtsccpb"
}

echo "# emission_axis: BIN=$BIN CORE=$CORE R=$R R2=$R2 N=$N" >&2
uptime >&2
for ((i=1;i<=N;i++)); do
  # interleave the 4 cells each round to keep box-state symmetric
  for cell in "a $R" "b $R" "a $R2" "b $R2"; do
    set -- $cell; arm="$1"; reps="$2"
    line=$(run_one "$arm" "$reps")
    echo "$i,$arm,$reps,$line" >> "$OUT"
  done
  echo "round $i done" >&2
done
echo "WROTE $OUT" >&2
