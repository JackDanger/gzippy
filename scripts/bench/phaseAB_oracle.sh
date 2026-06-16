#!/usr/bin/env bash
# phaseAB_oracle.sh — CONTROLLED seed-oracle A/B with capture-then-seed (the fix).
#
# The existing _oracle_guest.sh `clean-only` kind set GZIPPY_SEED_WINDOWS=1, which
# is NOT a file path: load_store() does File::open("1") -> ENOENT -> empty_seed(),
# so the oracle NEVER fired (every chunk took the normal path, labeled as oracle,
# CHECK_SHA=0 masking it). This script:
#   1. CAPTURE pass (p=1) -> writes a real seed file (start_bit -> window).
#   2. asserts the seed file is non-empty.
#   3. interleaved A(ON)/B(OFF)/A/B best-of-N, sha-verified EVERY run.
#   4. asserts the oracle FIRED: SEED hits>0 on every ON run.
#   5. min +/- spread per arm.
#
# Usage: phaseAB_oracle.sh <gzippy-bin> <corpus.gz> <T> <N> <mask> <workdir>
set -u
BIN="$1"; CORPUS="$2"; T="$3"; N="$4"; MASK="$5"; WD="$6"
mkdir -p "$WD"
SEED="$WD/seed.bin"
REF="$WD/ref.bin"
SINK="$WD/sink.bin"
export GZIPPY_FORCE_PARALLEL_SM=1

echo "=== phaseAB_oracle: bin=$(sha256sum "$BIN"|cut -c1-16) corpus=$(basename "$CORPUS") T=$T N=$N mask=$MASK ==="
"$BIN" --version

# 0. reference output (gzip ground truth)
gzip -dc "$CORPUS" > "$REF" || { echo "FATAL: gzip ref failed"; exit 2; }
REF_SHA=$(sha256sum "$REF" | cut -d' ' -f1)
echo "ref_sha=$REF_SHA raw_bytes=$(wc -c < "$REF")"

# 1. CAPTURE pass at p=1 (natural clean path records every aligned window)
rm -f "$SEED"
env GZIPPY_SEED_WINDOWS_CAPTURE="$SEED" taskset -c "$MASK" "$BIN" -d -c -p 1 "$CORPUS" > "$SINK" 2>"$WD/cap.stderr"
CAP_SHA=$(sha256sum "$SINK" | cut -d' ' -f1)
[ "$CAP_SHA" = "$REF_SHA" ] || { echo "FATAL: capture-pass sha mismatch ($CAP_SHA != $REF_SHA)"; exit 3; }
[ -s "$SEED" ] || { echo "FATAL: seed file empty after capture (oracle would never fire)"; cat "$WD/cap.stderr"; exit 4; }
echo "seed file bytes=$(wc -c < "$SEED")"

# 1b. count window-absent chunks in a NORMAL p=T run (reconciliation target for A2)
env GZIPPY_VERBOSE=1 taskset -c "$MASK" "$BIN" -d -c -p "$T" "$CORPUS" > "$SINK" 2>"$WD/norm.stderr"
[ "$(sha256sum "$SINK"|cut -d' ' -f1)" = "$REF_SHA" ] || { echo "FATAL: normal p=T sha mismatch"; exit 5; }
echo "normal-run verbose tail:"; grep -iE "flip_to_clean|window_seeded|isal_chunks" "$WD/norm.stderr" | head

run_one() {  # $1=ENVSTRING-or-empty ; echoes "secs sha hits misses"
  local extra="$1" s e secs sha hits misses
  s=$(date +%s.%N)
  # shellcheck disable=SC2086
  env $extra taskset -c "$MASK" "$BIN" -d -c -p "$T" "$CORPUS" > "$SINK" 2>"$WD/run.stderr"
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}')
  sha=$(sha256sum "$SINK" | cut -d' ' -f1)
  hits=$(sed -n 's/.*SEED_WINDOWS replay: hits=\([0-9]*\).*/\1/p' "$WD/run.stderr"); hits=${hits:-NA}
  misses=$(sed -n 's/.*misses=\([0-9]*\).*/\1/p' "$WD/run.stderr"); misses=${misses:-NA}
  echo "$secs $sha $hits $misses"
}

ON_T=""; OFF_T=""; FAIL=0; ONHITS=""
# warmup (discarded)
run_one "GZIPPY_SEED_WINDOWS=$SEED" >/dev/null
run_one "" >/dev/null
for ((i=1;i<=N;i++)); do
  read on_s on_sha on_h on_m < <(run_one "GZIPPY_SEED_WINDOWS=$SEED")
  read off_s off_sha off_h off_m < <(run_one "")
  [ "$on_sha"  = "$REF_SHA" ] || { echo "!! ON  i=$i SHA DIVERGENCE $on_sha";  FAIL=1; }
  [ "$off_sha" = "$REF_SHA" ] || { echo "!! OFF i=$i SHA DIVERGENCE $off_sha"; FAIL=1; }
  [ "$on_h" != "NA" ] && [ "$on_h" -gt 0 ] 2>/dev/null || { echo "!! ON i=$i ORACLE DID NOT FIRE (hits=$on_h)"; FAIL=1; }
  printf "i=%d ON=%s (hits=%s miss=%s)  OFF=%s\n" "$i" "$on_s" "$on_h" "$on_m" "$off_s"
  ON_T="$ON_T $on_s"; OFF_T="$OFF_T $off_s"; ONHITS="$ONHITS $on_h"
done

stats() { awk '{for(i=1;i<=NF;i++){v[i]=$i}}END{n=NF;asort(v);min=v[1];max=v[n];printf "min=%.4f max=%.4f spread=%.4f",min,max,max-min}' <<< "$1"; }
echo "---- RESULTS T=$T corpus=$(basename "$CORPUS") ----"
echo "ON  ($(echo $ONHITS | tr ' ' ',')): $(echo $ON_T | tr ' ' '\n' | sort -n | head -1 | xargs -I{} echo min={}); all:$ON_T"
echo "OFF: $(echo $OFF_T | tr ' ' '\n' | sort -n | head -1 | xargs -I{} echo min={}); all:$OFF_T"
on_min=$(echo $ON_T | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
off_min=$(echo $OFF_T | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
on_max=$(echo $ON_T | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
off_max=$(echo $OFF_T | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
echo "ON  min=$on_min max=$on_max spread=$(awk -v a=$on_min -v b=$on_max 'BEGIN{printf "%.4f",b-a}')"
echo "OFF min=$off_min max=$off_max spread=$(awk -v a=$off_min -v b=$off_max 'BEGIN{printf "%.4f",b-a}')"
echo "DELTA(OFF-ON) min-to-min = $(awk -v a=$off_min -v b=$on_min 'BEGIN{printf "%+.4f s (%.1f%% of OFF)",a-b,(a-b)/a*100}')"
echo "FAIL=$FAIL"
[ "$FAIL" = 0 ] || echo "## RUN INVALID (sha divergence or oracle-did-not-fire)"
