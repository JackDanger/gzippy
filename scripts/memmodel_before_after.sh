#!/usr/bin/env bash
# memmodel_before_after.sh — BEFORE/AFTER for the segmented-marker memory-model port.
#
# Runs ON neurotic (/root/gzippy). Builds the monolithic-model binary (BEFORE =
# reimplement-isa-l tip) and the segmented-model binary (AFTER = the port
# branch), then measures both INTERLEAVED (jitter-immune relative delta) on
# silesia at T4/T8/T16, sha-verified, plus the mechanism counters
# (dtlb_store_misses.walk_completed + minor-faults) the residency hypothesis
# predicts should move.
#
# No nested heredocs, no inline python. Wrapped operations are bounded by the
# caller's `timeout`.
set -u

REPO=/root/gzippy
BEFORE_REF=origin/reimplement-isa-l
AFTER_REF=origin/feat/rapidgzip-memory-model
BD="$REPO/benchmark_data"
RAW_TAR="$BD/silesia.tar"
GZ="$BD/silesia-gzip.tar.gz"
N="${N:-9}"
THREADS="${THREADS:-4 8 16}"
WORK=/dev/shm/memmodel-ba
mkdir -p "$WORK"

cd "$REPO" || { echo "no $REPO"; exit 1; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true

echo "## host: $(uname -m) nproc=$(nproc) loadavg=$(cat /proc/loadavg | awk '{print $1}')"
echo "## no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

# ---- prepare the silesia gzip fixture (single-member, what gzip(1) makes) ----
git fetch origin --quiet 2>&1 | tail -1
if [ ! -f "$RAW_TAR" ]; then
  [ -f "$BD/silesia.tar.xz" ] && { echo "extracting silesia.tar"; xz -dk "$BD/silesia.tar.xz" -c > "$RAW_TAR"; }
fi
[ -f "$RAW_TAR" ] || { echo "ERROR: no $RAW_TAR"; exit 1; }
if [ ! -s "$GZ" ]; then
  echo "creating $GZ (gzip -6 single-member)"
  gzip -6 -c "$RAW_TAR" > "$GZ"
fi
RAW_BYTES=$(stat -c %s "$RAW_TAR")
REF_SHA=$(gzip -dc "$GZ" | sha256sum | awk '{print $1}')
echo "## fixture: $GZ ($(stat -c %s "$GZ") B compressed, $RAW_BYTES B raw) ref_sha=$REF_SHA"

build_one () {
  local ref="$1" out="$2"
  echo "## building $ref -> $out"
  git checkout -f -B _bench_tmp "$ref" >/dev/null 2>&1
  git submodule update --init --recursive >/dev/null 2>&1
  # x86 production build: isal-compression enables parallel_sm + the ISA-L path.
  cargo build --release --no-default-features --features isal-compression \
    >/dev/null 2>"$WORK/build_$out.log" || { echo "BUILD FAILED $ref"; tail -20 "$WORK/build_$out.log"; return 1; }
  cp target/release/gzippy "$WORK/gzippy_$out"
  echo "   ok: $WORK/gzippy_$out"
}

build_one "$BEFORE_REF" before || exit 1
build_one "$AFTER_REF"  after  || exit 1
git checkout -f "$AFTER_REF" >/dev/null 2>&1

BEFORE="$WORK/gzippy_before"
AFTER="$WORK/gzippy_after"

# ---- correctness gate: both binaries sha-match gzip(1) at every T ----
echo ""; echo "## correctness (sha vs gzip ref):"
for T in $THREADS; do
  for tag in before after; do
    bin="$WORK/gzippy_$tag"
    s=$(GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p "$T" "$GZ" 2>/dev/null | sha256sum | awk '{print $1}')
    if [ "$s" = "$REF_SHA" ]; then echo "   $tag T$T: MATCH"; else echo "   $tag T$T: MISMATCH ($s)"; fi
  done
done

# ---- interleaved wall (best-of-N), per T, relative delta ----
# We time wall via /usr/bin/time -v (elapsed). Interleave before/after each
# trial so both see the same per-trial contention; report min/median per tag.
median () { sort -n | awk '{a[NR]=$1} END{n=NR; if(n%2){print a[(n+1)/2]} else {print (a[n/2]+a[n/2+1])/2}}'; }
minof  () { sort -n | head -1; }
sdof   () { awk '{x[NR]=$1; s+=$1} END{m=s/NR; for(i=1;i<=NR;i++){d=x[i]-m; v+=d*d} printf "%.4f", sqrt(v/NR)}'; }

echo ""; echo "## interleaved wall (N=$N), seconds elapsed:"
for T in $THREADS; do
  : > "$WORK/w_before_$T"; : > "$WORK/w_after_$T"
  for i in $(seq 1 "$N"); do
    for tag in before after; do
      bin="$WORK/gzippy_$tag"
      # elapsed seconds via bash time builtin formatting
      t=$( { /usr/bin/env bash -c "TIMEFORMAT='%R'; time GZIPPY_FORCE_PARALLEL_SM=1 '$bin' -d -c -p $T '$GZ' > /dev/null" ; } 2>&1 )
      echo "$t" >> "$WORK/w_${tag}_$T"
    done
  done
  bmin=$(minof < "$WORK/w_before_$T"); bmed=$(median < "$WORK/w_before_$T"); bsd=$(sdof < "$WORK/w_before_$T")
  amin=$(minof < "$WORK/w_after_$T");  amed=$(median < "$WORK/w_after_$T");  asd=$(sdof < "$WORK/w_after_$T")
  ratio=$(awk -v a="$amed" -v b="$bmed" 'BEGIN{printf "%.4f", a/b}')
  pct=$(awk -v a="$amed" -v b="$bmed" 'BEGIN{printf "%+.2f", (a/b-1)*100}')
  echo "  T$T  BEFORE min=$bmin med=$bmed sd=$bsd | AFTER min=$amin med=$amed sd=$asd | after/before=$ratio (${pct}% wall)"
done

# ---- mechanism counters: dtlb_store_misses.walk_completed + minor-faults ----
# Single perf-stat run per (tag,T). paranoid=4 + root: software event
# minor-faults always available; dtlb walk is a hw event (may need the PMU).
echo ""; echo "## mechanism counters (perf stat, T=8, 1 run each):"
for tag in before after; do
  bin="$WORK/gzippy_$tag"
  perf stat -e dtlb_store_misses.walk_completed,minor-faults,cycles,instructions \
    -- env GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p 8 "$GZ" > /dev/null 2> "$WORK/perf_$tag.txt"
  echo "  --- $tag ---"
  grep -E "dtlb_store_misses.walk_completed|minor-faults|cycles|instructions" "$WORK/perf_$tag.txt" \
    | sed 's/^/    /'
done

echo ""; echo "## DONE"
