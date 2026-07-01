#!/usr/bin/env bash
# guest_same_sink_floor.sh — RIDER 2: same-sink production-output floor.
# Decodes silesia-large.gz T8 with BOTH gzippy and rapidgzip to a REAL FILE sink
# (not /dev/null), interleaved best-of-N, sha-verified, so the §3 tie verdict's
# "floor ≤0.54s" can be re-quoted on a same-sink basis. Runs AFTER host freq lock.
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
THREADS="${THREADS:-8}"; T="$(echo "$THREADS" | awk '{print $1}')"; [ -n "$T" ] || T=8
N="${N:-9}"; [ "$N" -ge 9 ] || N=9
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-same-sink}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
RG="${RG:-$REPO/vendor/rapidgzip/librapidarchive/build-trace/src/tools/rapidgzip}"
# Real file sinks on /dev/shm (RAM-backed but a real writev target; / is RO/tiny).
GZOUT=/dev/shm/ss_gz.out
RGOUT=/dev/shm/ss_rg.out

mkdir -p "$ARTDIR"; say() { echo "$@"; }
pin_mask() { case "$1" in 8) echo "0,2,4,6,8,10,12,14";; 4) echo "0,2,4,6";; 1) echo "0";; *) echo "0,2,4,6,8,10,12,14";; esac; }
mask="$(pin_mask "$T")"

cd "$REPO" || { echo "SS_FAILURE=no-repo"; echo "SS_GUEST_DONE"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
say "## build gzippy (pure-rust-inflate) ..."
if ! cargo build --release --no-default-features --features pure-rust-inflate >"$ARTDIR/build.log" 2>&1; then
  echo "SS_FAILURE=gzippy-build"; tail -30 "$ARTDIR/build.log"; echo "SS_GUEST_DONE"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"
[ -x "$RG" ] || { echo "SS_FAILURE=no-rg ($RG)"; echo "SS_GUEST_DONE"; exit 8; }
[ -f "$CORPUS" ] || { echo "SS_FAILURE=corpus"; echo "SS_GUEST_DONE"; exit 7; }

REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"
say "## freq: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
say "## corpus=$CORPUS raw_bytes=$RAW_BYTES ref_sha=$REF_SHA T=$T N=$N (REAL FILE SINK: $GZOUT / $RGOUT)"
say "## gzippy=$(git rev-parse HEAD) rapidgzip=$("$RG" --version 2>&1 | head -1)"

# routing assert
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
case "$DBG" in *ParallelSM*) ;; *) echo "SS_FAILURE=routing $DBG"; echo "SS_GUEST_DONE"; exit 9;; esac

timed() { # <mask> <outfile> <cmd...> -> "secs sha"
  local mask="$1" outf="$2"; shift 2
  local s e secs sha
  s=$(date +%s.%N); taskset -c "$mask" "$@" >"$outf" 2>>"$ARTDIR/run.err"; e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$outf" | cut -d' ' -f1); echo "$secs $sha"
}
stats() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{v[NR]=$1;sum+=$1} END{n=NR;if(n==0){print "0 0 0";exit}min=v[1];mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;mean=sum/n;ss=0;for(i=1;i<=n;i++){d=v[i]-mean;ss+=d*d}sd=(n>1)?sqrt(ss/(n-1)):0;printf "%.4f %.4f %.1f",min,mid,(mean>0)?sd/mean*100:0}'; }

GZ=""; RGT=""; DIV=0
for ((i=0;i<=N;i++)); do
  read gs gh < <(timed "$mask" "$GZOUT" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS")
  read rs rh < <(timed "$mask" "$RGOUT" "$RG" -d -P "$T" -c "$CORPUS")
  [ "$gh" = "$REF_SHA" ] || { say "DIVERGE gzippy i=$i sha=$gh"; DIV=1; }
  [ "$rh" = "$REF_SHA" ] || { say "DIVERGE rapidgzip i=$i sha=$rh"; DIV=1; }
  if [ "$i" -gt 0 ]; then GZ="$GZ $gs"; RGT="$RGT $rs"; fi   # drop iter0
  say "iter $i: gzippy=$gs rapidgzip=$rs"
done
rm -f "$GZOUT" "$RGOUT"
say ""
say "## SAME-SINK FLOOR (real file sink, drop iter0, N=$N, T=$T):"
say "gzippy_same_sink:    min/med/sd% = $(stats "$GZ")"
say "rapidgzip_same_sink: min/med/sd% = $(stats "$RGT")"
say "DIVERGED=$DIV (0=all sha OK)"
echo "SS_GUEST_DONE"
