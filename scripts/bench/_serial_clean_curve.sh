#!/usr/bin/env bash
# Serial-clean selector all-T scaling curve (Intel trainer).
# For each (corpus,T): fulcrum abmeasure with the SAME binary, base arm =
# selector OFF (GZIPPY_PARALLEL_CROSSOVER_MARGIN=0, the old parallel-always-
# below-cap behaviour that regressed), after arm = selector ON (default).
# Parses ABMEASURE-WALL: base(old) / after(selector) / comparator wall(ms).
set -u
FUL=/dev/shm/fulcrum-target/release/fulcrum
BIN=/dev/shm/sel-target/release/gzippy
N=${N:-11}
CORES=${CORES:-0-15}
OUT=${OUT:-/dev/shm/serial_clean_curve.tsv}
: > "$OUT"
echo -e "corpus\tT\tcompetitor\tbase_old_ms\tafter_sel_ms\tcomp_ms\tafter_over_base\tafter_over_comp" | tee -a "$OUT"

run_cell() {
  local corpus_path="$1" cname="$2" T="$3" rgcmd="$4" rglabel="$5"
  local line
  line=$(taskset -c "$CORES" "$FUL" abmeasure \
      --base-bin "$BIN" --after-bin "$BIN" \
      --base-env "GZIPPY_PARALLEL_CROSSOVER_MARGIN=0" \
      --after-env "" \
      --common-env "GZIPPY_FORCE_PARALLEL_SM=1" \
      --gz-args "-d -c -p$T" \
      --rg-cmd "$rgcmd" --rg-label "$rglabel" \
      --corpus "$corpus_path" --core "$CORES" --n "$N" --no-gate 2>&1 \
      | grep "ABMEASURE-WALL")
  # ABMEASURE-WALL <corpus>: base 707ms  after 697ms  <rglabel> 687ms ...
  local b a c
  b=$(echo "$line" | grep -oE "base [0-9]+ms"  | grep -oE "[0-9]+")
  a=$(echo "$line" | grep -oE "after [0-9]+ms" | grep -oE "[0-9]+")
  c=$(echo "$line" | grep -oE "$rglabel [0-9]+ms" | grep -oE "[0-9]+")
  local rab rac
  rab=$(python3 -c "print(round($a/$b,3))" 2>/dev/null || echo NA)
  rac=$(python3 -c "print(round($a/$c,3))" 2>/dev/null || echo NA)
  echo -e "$cname\t$T\t$rglabel\t${b:-NA}\t${a:-NA}\t${c:-NA}\t$rab\t$rac" | tee -a "$OUT"
}

# Corpora to sweep (exercise the selector: ratio between 1 and the hard cap 8).
# Format: path:name:Tlist
SPEC=(
  "/dev/shm/silesia.gz:silesia:1 2 3 4 6 8 12 16"
  "/dev/shm/monorepo.gz:monorepo:1 2 3 4 6 8 12 16"
  "/dev/shm/squishy_realdata.gz:squishy:1 2 3 4 8 16"
  "/dev/shm/nasa.gz:nasa:1 2 4 8"
)
COMPETITOR=${COMPETITOR:-igzip}
for entry in "${SPEC[@]}"; do
  IFS=':' read -r path name tlist <<< "$entry"
  [ -f "$path" ] || { echo "MISSING $path"; continue; }
  for T in $tlist; do
    if [ "$COMPETITOR" = "rapidgzip" ]; then
      run_cell "$path" "$name" "$T" "rapidgzip -P $T -d -c" "rapidgzip"
    else
      run_cell "$path" "$name" "$T" "igzip -d -c" "igzip"
    fi
  done
done
echo "DONE -> $OUT"
