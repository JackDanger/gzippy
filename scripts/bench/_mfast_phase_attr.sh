#!/usr/bin/env bash
# _mfast_phase_attr.sh — per-PHASE instruction attribution INSIDE the 'mfast
# marker inner loop (read_internal_compressed_specialized<true>).
#
# Symboled production-codegen build (fat LTO + debuginfo2 — instruction COUNT
# unchanged vs shipped) of gzippy-native; perf record -e instructions on a T4
# silesia decode; perf annotate the marker decode symbol by SOURCE LINE; the
# wrapper aggregates instruction-sample % by marker_inflate.rs line so the lines
# can be bucketed into phases (litlen-decode / literal-extract / dist-decode /
# backref-emit / refill / guard).
set -u
SRC="${SRC:-/root/gz-lever}"
CORPUS="${CORPUS:-/root/silesia.gz}"
T="${T:-4}"
MASK="${MASK:-0,2,4,6}"
PERIOD="${PERIOD:-200000}"
ART="${ART:-/dev/shm/mfast-phase}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/dev/shm/gzl-tgt}"
RUSTFLAGS_SYM="-C target-cpu=native -C strip=none -C debuginfo=2"
mkdir -p "$ART"
SINK="$ART/sink.bin"
cd "$SRC" || { echo "FAIL no-src:$SRC"; exit 5; }

echo "### MFAST-PHASE-ATTR T=$T MASK=$MASK PERIOD=$PERIOD SRC=$SRC ###"
echo "=== df ==="; df -h /dev/shm | tail -1

echo "=== BUILD gzippy-native symboled ==="
RUSTFLAGS="$RUSTFLAGS_SYM" cargo build --release --no-default-features \
  --features gzippy-native --bin gzippy > "$ART/build.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "FAIL build rc=$rc"; grep -E 'error' "$ART/build.log" | head -25; exit 8; fi
grep -E 'Finished' "$ART/build.log" | tail -1 | sed 's/^/   /'
GZ="$CARGO_TARGET_DIR/release/gzippy"
echo "   symbols=$(nm "$GZ" 2>/dev/null | wc -l)"

echo "=== ASSERT path=ParallelSM ==="
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null)"
echo "$DBG" | grep -E 'path=' | head -1 | sed 's/^/   DBG /'
echo "$DBG" | grep -q 'ParallelSM' || { echo "FAIL routing-not-ParallelSM"; exit 9; }

EV="instructions:u"
if perf stat -e cpu_core/instructions/ -- true >/dev/null 2>&1; then EV="cpu_core/instructions/u"; fi
echo "=== perf event = $EV ==="

echo "=== perf stat instructions (REPS=5) ==="
perf stat -r 5 -e "$EV" -o "$ART/gz.stat" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
grep -E 'instructions|elapsed' "$ART/gz.stat" | sed 's/^/   /'
echo "   sink-sha=$(sha256sum "$SINK" | cut -d' ' -f1)"

echo "=== perf record $EV -c $PERIOD ==="
perf record -e "$EV" -c "$PERIOD" -o "$ART/gz.data" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
echo "--- total samples ---"; perf report -i "$ART/gz.data" --stdio 2>/dev/null | grep -iE 'event count|samples' | head -2 | sed 's/^/   /'

echo "=== TOP symbols (self) ==="
perf report -i "$ART/gz.data" --stdio -n --no-children --sort symbol 2>/dev/null | grep -vE '^#|^$' | head -20

# The marker inner loop monomorph: read_internal_compressed_specialized (markers=true).
SYM="$(perf report -i "$ART/gz.data" --stdio -n --no-children --sort symbol 2>/dev/null \
  | grep -vE '^#|^$' | grep -i 'read_internal_compressed_specialized' \
  | awk '{ for(i=1;i<=NF;i++) if($i=="[.]"){ s=""; for(j=i+1;j<=NF;j++) s=s (j>i+1?" ":"") $j; print s; break } }' \
  | head -1)"
echo "=== marker symbol = [$SYM] ==="

echo "=== annotate $SYM : per-source-line instruction sample %% (marker_inflate.rs lines) ==="
perf annotate -i "$ART/gz.data" --stdio -l --no-source --symbol="$SYM" 2>"$ART/annot.err" > "$ART/annot.txt"
echo "   annot.err head:"; head -3 "$ART/annot.err" | sed 's/^/     /'
echo "   annot.txt lines=$(wc -l < "$ART/annot.txt")"

rm -f "$SINK"
echo "MFAST_PHASE_DONE"
