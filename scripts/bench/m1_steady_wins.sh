#!/bin/bash
# =============================================================================
# m1_steady_wins.sh — COMMITTED, REPRODUCIBLE steady-wall env-toggle oracles for
# the three accumulated Apple-Silicon (M1) arm64 decode wins on
# reimplement-isa-l@b2d1a2db. Each win is regenerated as a SINGLE-BINARY
# env-toggle removal-oracle: the SAME production gzippy is run under two tiny
# native Mach-O wrapper binaries; one wrapper sets the toggle env (the change
# REMOVED), the other sets nothing (the change PRESENT / default). The env flips
# exactly one code path and does NOT change output bytes (asserted by sha).
#
# The three oracles (default aarch64 behaviour = the WIN being defended):
#   1. PMULL      gz=default(PMULL fold)        ref=GZIPPY_CRC_PMULL=0 (crc32x fold3)
#   2. tablebuild gz=default(SINGLE litlen)     ref=GZIPPY_LITLEN_MULTISYM=triple
#   3. crossover  gz=default(selector OFF,0.0)  ref=GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0
#
# GATE-0 NON-INERT PROOF (done out-of-band, recorded here, NOT a production edit):
# a throwaway instrumented build (GZIPPY_GATE0_PROBE eprintln at each toggle
# site) confirmed the env actually flips the executed path:
#   PMULL:      default→"crc_path=PMULL"   ref→"crc_path=FOLD3"            NON-INERT
#   tablebuild: default→"multisym=SINGLE"  ref→"multisym=TRIPLE"           NON-INERT
#   crossover:  T1 default→"early-return; crossover NOT consulted"
#               T1 ref    →"early-return; crossover NOT consulted"         *** INERT at T1 ***
#               T2 default→"out=2 (parallel)"  T2 ref→"out=1 (SERIAL_CLEAN_FLOOR)"  NON-INERT
# => crossover is STRUCTURALLY INERT at T1 (effective_parallel_threads early-
#    returns when num_threads<=1, before reading the env). It is therefore
#    measured at T2 (the lowest T where the selector is consulted). The T1
#    crossover cells are run anyway, to EMPIRICALLY exhibit the inert ~1.0 A/A.
#
# The instrument is `fulcrum wall --steady` (~/www/fulcrum-mac), a freq-pinned,
# warmed-up, throttle-filtered, multi-cohort paired A/B whose gz-vs-gz A/A
# cross-cohort spread is the reproducibility FLOOR; a verdict is REPRODUCIBLE
# only if the effect clears that floor in every cohort with a consistent sign.
#
# Reproduce on this M1 Pro:
#   sudo true   # fulcrum --steady needs root for kpc
#   bash scripts/bench/m1_steady_wins.sh
#
# Override points (env): GZ_BIN, FULCRUM, SILESIA, LOGSBIG, COHORTS, PAIRS,
# WARMUP, COOLDOWN, OUTDIR, BUILD=1 (force rebuild the clean gzippy).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
DATE="$(date +%Y-%m-%d)"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo b2d1a2db)"

GZ_BIN="${GZ_BIN:-$REPO/target/release/gzippy}"
FULCRUM="${FULCRUM:-$HOME/www/fulcrum-mac/target/release/fulcrum}"
WORK="${WORK:-${TMPDIR:-/tmp}/m1-steady-work}"
# silesia: prefer an existing single-member .gz; else regenerate from the
# tracked benchmark_data/silesia.tar.xz (unxz -> single-member gzip).
SILESIA="${SILESIA:-$REPO/benchmark_data/silesia-gzip.tar.gz}"
LOGSBIG="${LOGSBIG:-$WORK/logsbig.gz}"
OUTDIR="${OUTDIR:-$REPO/artifacts/m1-steady/$DATE}"
COHORTS="${COHORTS:-5}"; PAIRS="${PAIRS:-9}"
WARMUP="${WARMUP:-20}"; COOLDOWN="${COOLDOWN:-5}"
PARSER="$HERE/parse_steady.py"

mkdir -p "$WORK" "$OUTDIR"

echo "== m1_steady_wins =="
echo "repo=$REPO commit=$COMMIT date=$DATE"
echo "gz=$GZ_BIN fulcrum=$FULCRUM outdir=$OUTDIR"

# ---- 0. build the clean gzippy (b2d1a2db, byte-exact) if missing -------------
if [ ! -x "$GZ_BIN" ] || [ "${BUILD:-0}" = "1" ]; then
  echo "-- building clean gzippy (--no-default-features --features pure-rust-inflate) --"
  ( cd "$REPO" && RUSTFLAGS="-C target-cpu=native" \
      cargo build --release --no-default-features --features pure-rust-inflate ) || exit 1
fi
# prepare silesia .gz if absent
if [ ! -f "$SILESIA" ]; then
  XZ="$REPO/benchmark_data/silesia.tar.xz"
  if [ -f "$XZ" ]; then
    echo "-- regenerating silesia single-member .gz from $XZ --"
    SILESIA="$WORK/silesia.tar.gz"
    if [ ! -f "$SILESIA" ]; then xz -dc "$XZ" | gzip -c > "$SILESIA"; fi
  else
    echo "FATAL: no silesia corpus ($SILESIA) and no $XZ to regenerate from"; exit 1
  fi
fi

GZIPPY_DEBUG=1 "$GZ_BIN" -d -c -p1 "$SILESIA" >/dev/null 2> "$WORK/route.txt"
grep -q "path=ParallelSM" "$WORK/route.txt" || { echo "FATAL: gz not on ParallelSM path"; cat "$WORK/route.txt"; exit 1; }
echo "Gate-4: path=ParallelSM confirmed"

# ---- 1. logsbig corpus: VARIED ~150MB jsonl logs (NOT one file repeated) -----
# Built by concatenating MANY DISTINCT specstories conversation jsonl files so
# every block has different Huffman statistics, then single-member gzip.
if [ ! -f "$LOGSBIG" ]; then
  echo "-- generating logsbig corpus --"
  RAW="$WORK/logsbig.raw"; : > "$RAW"; n=0
  while IFS= read -r f; do
    gzip -dc "$f" >> "$RAW" 2>/dev/null || true; n=$((n+1))
    [ "$(stat -f%z "$RAW")" -ge $((150*1024*1024)) ] && break
  done < <(find "$HOME/www/specstories/conversations" -name '*.jsonl.gz' 2>/dev/null | sort)
  gzip -c "$RAW" > "$LOGSBIG"
  echo "logsbig: $n distinct files, raw=$(stat -f%z "$RAW") gz=$(stat -f%z "$LOGSBIG")"
fi
# corpus ISIZE/deflate ratios (for the crossover record)
ratio_of(){ python3 - "$1" <<'PY'
import sys,struct
p=sys.argv[1]; d=open(p,'rb').read()
isize=struct.unpack('<I', d[-4:])[0]; dl=len(d)
print(f"{isize/dl:.3f}")
PY
}
SIL_RATIO="$(ratio_of "$SILESIA")"; LB_RATIO="$(ratio_of "$LOGSBIG")"
echo "silesia ratio=$SIL_RATIO  logsbig ratio=$LB_RATIO"

# ---- 2. compile the native Mach-O env-toggle wrappers ------------------------
WC="$HERE/wrapper.c"
cc -O2 -DGZ="\"$GZ_BIN\"" -o "$WORK/w_default" "$WC"
cc -O2 -DGZ="\"$GZ_BIN\"" -DENV_NAME="\"GZIPPY_CRC_PMULL\"" -DENV_VAL="\"0\"" -o "$WORK/w_pmull_off" "$WC"
cc -O2 -DGZ="\"$GZ_BIN\"" -DENV_NAME="\"GZIPPY_LITLEN_MULTISYM\"" -DENV_VAL="\"triple\"" -o "$WORK/w_multisym_triple" "$WC"
cc -O2 -DGZ="\"$GZ_BIN\"" -DENV_NAME="\"GZIPPY_PARALLEL_CROSSOVER_MARGIN\"" -DENV_VAL="\"1.0\"" -o "$WORK/w_xover_on" "$WC"

# ---- 3. run one steady A/B cell + emit JSON ----------------------------------
# args: oracle gz_arm ref_bin ref_label corpus threads ratio non_inert
run_cell(){
  local oracle="$1" gz_label="$2" ref_bin="$3" ref_label="$4" corpus="$5" thr="$6" ratio="$7" inert="$8"
  local tag="${oracle}_$(basename "$corpus" | tr '.' '_')_T${thr}"
  local log="$OUTDIR/$tag.log" json="$OUTDIR/$tag.json"
  echo; echo "### CELL $tag  (gz=$gz_label  ref=$ref_label  non_inert=$inert) ###"
  sudo -n -E "$FULCRUM" wall --steady \
      --gz "$WORK/w_default" --ref "$ref_bin" --ref-args "-d,-c,-p${thr}" \
      --corpus "$corpus" --threads "$thr" \
      --cohorts "$COHORTS" --pairs "$PAIRS" --warmup-secs "$WARMUP" --cooldown-secs "$COOLDOWN" \
      2>&1 | tee "$log"
  python3 "$PARSER" "$log" --oracle "$oracle" --corpus "$(basename "$corpus")" \
      --threads "$thr" --gz-arm "$gz_label" --ref-arm "$ref_label" \
      --non-inert "$inert" --ratio "$ratio" --commit "$COMMIT" > "$json"
  echo "-> $json"
}

# PMULL — critical: real wall win or microbench-only?
run_cell pmull "PMULL fold (default)" "$WORK/w_pmull_off" "GZIPPY_CRC_PMULL=0 (crc32x fold3)" "$SILESIA" 1 "$SIL_RATIO" yes
run_cell pmull "PMULL fold (default)" "$WORK/w_pmull_off" "GZIPPY_CRC_PMULL=0 (crc32x fold3)" "$LOGSBIG" 1 "$LB_RATIO" yes
# tablebuild — SINGLE (aarch64 default) vs TRIPLE (x86 default)
run_cell tablebuild "SINGLE litlen (default)" "$WORK/w_multisym_triple" "GZIPPY_LITLEN_MULTISYM=triple" "$SILESIA" 1 "$SIL_RATIO" yes
run_cell tablebuild "SINGLE litlen (default)" "$WORK/w_multisym_triple" "GZIPPY_LITLEN_MULTISYM=triple" "$LOGSBIG" 1 "$LB_RATIO" yes
# crossover — INERT at T1 (run to exhibit), NON-INERT at T2 (the real test)
run_cell crossover "selector OFF (default 0.0)" "$WORK/w_xover_on" "GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0" "$SILESIA" 1 "$SIL_RATIO" "no-INERT-at-T1"
run_cell crossover "selector OFF (default 0.0)" "$WORK/w_xover_on" "GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0" "$LOGSBIG" 1 "$LB_RATIO" "no-INERT-at-T1"
run_cell crossover "selector OFF (default 0.0)" "$WORK/w_xover_on" "GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0" "$SILESIA" 2 "$SIL_RATIO" yes
run_cell crossover "selector OFF (default 0.0)" "$WORK/w_xover_on" "GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0" "$LOGSBIG" 2 "$LB_RATIO" yes

echo; echo "-- rolling up index.json --"
python3 "$HERE/make_index.py" "$OUTDIR" > "$OUTDIR/index.json"

echo; echo "== all cells done; JSON in $OUTDIR =="
ls -1 "$OUTDIR"/*.json
