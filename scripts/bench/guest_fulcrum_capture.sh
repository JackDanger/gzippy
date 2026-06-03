#!/usr/bin/env bash
# guest_fulcrum_capture.sh — frozen-host wall + Chrome-trace artifacts for fulcrum.
#
# RUNS ON GUEST 199 after host_lock_and_bench.sh gate PASS.
# Args: BRANCH=reimplement-isa-l  THREADS=8  N=9
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
THREADS="${THREADS:-8}"
N="${N:-9}"
SD_FAIL_PCT="${SD_FAIL_PCT:-5}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-fulcrum}"
RG_TRACE="${RG_TRACE:-$REPO/vendor/rapidgzip/librapidarchive/build-trace/src/tools/rapidgzip}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"

for a in "$@"; do
  case "$a" in
    BRANCH=*) BRANCH="${a#*=}";;
    THREADS=*) THREADS="$(echo "${a#*=}" | tr ',' ' ')";;
    N=*) N="${a#*=}";;
    SLOW_BOOTSTRAP=*) GZIPPY_SLOW_BOOTSTRAP="${a#*=}";;
  esac
done
# Optional experiment knobs (env or host_lock arguments).
GZIPPY_SLOW_BOOTSTRAP="${GZIPPY_SLOW_BOOTSTRAP:-}"
[ "$N" -ge 9 ] || N=9

mkdir -p "$ARTDIR"
say() { echo "$@"; }

pin_mask() {
  case "$1" in
    4) echo "0,2,4,6";;
    8) echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *) echo "";;
  esac
}

# ---- sync branch (provenance requires clean gzippy tree) --------------------
cd "$REPO" || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=no-repo"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
# Drop stray untracked files (copied scripts, orphan target/) but keep vendor + corpus.
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true

DIRTY_COUNT="$(git status --porcelain --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')"
if [ "$DIRTY_COUNT" != "0" ]; then
  echo "RUN_TRUSTWORTHY=false"
  echo "FAILURE=dirty-tree ($DIRTY_COUNT files after reset)"
  exit 6
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
say "## build gzippy (pure-rust-inflate) ..."
if ! cargo build --release --no-default-features --features pure-rust-inflate >"$ARTDIR/build-gzippy.log" 2>&1; then
  echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=gzippy-build"; tail -30 "$ARTDIR/build-gzippy.log"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"

[ -x "$RG_TRACE" ] || {
  say "## build-trace rapidgzip missing; building ..."
  BDIR="$REPO/vendor/rapidgzip/librapidarchive/build-trace"
  mkdir -p "$BDIR"
  (cd "$BDIR" && cmake .. -DCMAKE_BUILD_TYPE=Release >>"$ARTDIR/build-rg-trace.log" 2>&1 \
    && cmake --build . --target rapidgzip -j"$(nproc)" >>"$ARTDIR/build-rg-trace.log" 2>&1) \
    || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=rapidgzip-trace-build"; exit 8; }
}
[ -x "$RG_TRACE" ] || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=no-rg-trace"; exit 8; }

[ -f "$CORPUS" ] || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=corpus"; exit 7; }
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"
cat "$CORPUS" >/dev/null 2>&1

DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p 8 "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
case "$DBG" in *IsalParallelSM*) ;; *) echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=routing $DBG"; exit 9;; esac

GZIPPY_SHA="$(git rev-parse HEAD)"
RG_VER="$("$RG_TRACE" --version 2>&1 | head -1)"

say "================ PROVENANCE ================"
say "branch=$BRANCH head=$GZIPPY_SHA resolve_ahead=always slow_bootstrap=${GZIPPY_SLOW_BOOTSTRAP:-off}"
say "rapidgzip-trace=$RG_VER"
say "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
say "artifacts=$ARTDIR"
say "==========================================="

run_cmd_timed() {
  local mask="$1"; shift
  local out s e rc
  out="$(mktemp)"
  s=$(date +%s.%N)
  # taskset execve's argv[0]; never pass a shell function here.
  set +e
  taskset -c "$mask" "$@" >"$out" 2>>"$ARTDIR/wall.stderr"
  rc=$?
  set -e
  e=$(date +%s.%N)
  local secs sha
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || say "WARN run_cmd_timed exit=$rc: $*"
  rm -f "$out"
  echo "$secs $sha"
}

# Print argv prefix for gzippy wall/trace runs (pure-rust parallel SM). One line for $(…).
gzippy_wall_cmd() {
  local parts="env GZIPPY_FORCE_PARALLEL_SM=1"
  [ -n "$GZIPPY_SLOW_BOOTSTRAP" ] && parts="$parts GZIPPY_SLOW_BOOTSTRAP=$GZIPPY_SLOW_BOOTSTRAP"
  echo "$parts"
}

stats() {
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1; sum+=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      mean=sum/n; ss=0; for(i=1;i<=n;i++){d=v[i]-mean; ss+=d*d}
      sd=(n>1)?sqrt(ss/(n-1)):0; sdpct=(mean>0)?sd/mean*100:0;
      printf "%.4f %.4f %.1f", min, mid, sdpct;
    }'
}

capture_trace() { # capture_trace <label> <cpu-mask> <cmd...>
  local label="$1" mask="$2"; shift 2
  local tl="$ARTDIR/trace_${label}.json"
  local ml="$ARTDIR/memlife_${label}.json"
  say "## TRACE $label -> $tl"
  # shellcheck disable=SC2046
  GZIPPY_TIMELINE="$tl" GZIPPY_MEMLIFE="$ml" GZIPPY_VERBOSE=1 \
    taskset -c "$mask" $(gzippy_wall_cmd) "$@" >/dev/null 2>>"$ARTDIR/trace.log" || true
  [ -s "$tl" ] || { say "## WARN empty timeline $tl"; return 1; }
  ls -la "$tl" "$ml" 2>/dev/null || true
}

DIVERGED=0
ANY_SD_FAIL=0
say ""
say "================ WALL (interleaved, iter0 dropped) ================"

for T in $THREADS; do
  mask="$(pin_mask "$T")"
  [ -n "$mask" ] || continue
  GZ_T=""; RG_T=""
  for ((i=0; i<=N; i++)); do
    # shellcheck disable=SC2046
    read gsec gsha < <(run_cmd_timed "$mask" \
      $(gzippy_wall_cmd) "$GZIPPY" -d -c -p "$T" "$CORPUS")
    read rsec rsha < <(run_cmd_timed "$mask" "$RG_TRACE" -d -c -f -P "$T" "$CORPUS")
    [ "$i" -eq 0 ] && continue
    GZ_T="$GZ_T $gsec"; RG_T="$RG_T $rsec"
    [ "$gsha" = "$REF_SHA" ] || { say "DIVERGE gzippy T=$T"; DIVERGED=1; }
    [ "$rsha" = "$REF_SHA" ] || { say "DIVERGE rapidgzip T=$T"; DIVERGED=1; }
  done
  read gmin gmed gsd < <(stats "$GZ_T")
  read rmin rmed rsd < <(stats "$RG_T")
  ratio=$(awk -v rg="$rmin" -v gz="$gmin" 'BEGIN{printf "%.3f", (gz>0)?rg/gz:0}')
  margin=$(awk -v a="$gsd" -v b="$rsd" 'BEGIN{m=(a>b)?a:b; print m/100}')
  verdict=$(awk -v r="$ratio" -v m="$margin" 'BEGIN{d=r-1; if(d>m)print "WIN"; else if(d<-m)print "LOSS"; else print "TIE"}')
  gmb=$(awk -v r="$RAW_BYTES" -v t="$gmin" 'BEGIN{printf "%.0f", r/t/1e6}')
  rmb=$(awk -v r="$RAW_BYTES" -v t="$rmin" 'BEGIN{printf "%.0f", r/t/1e6}')
  awk -v s="$gsd" -v f="$SD_FAIL_PCT" 'BEGIN{exit !(s>f)}' && ANY_SD_FAIL=1
  awk -v s="$rsd" -v f="$SD_FAIL_PCT" 'BEGIN{exit !(s>f)}' && ANY_SD_FAIL=1
  printf "T%-4s gzippy min=%.4fs (%s MB/s) rapidgzip min=%.4fs (%s MB/s) ratio=%.3f verdict=%s sd%% gz=%.1f rg=%.1f\n" \
    "$T" "$gmin" "$gmb" "$rmin" "$rmb" "$ratio" "$verdict" "$gsd" "$rsd"
done

say ""
say "================ TRACE CAPTURE (PMU/trace perturb wall — separate runs) ================"
T=8
mask="$(pin_mask "$T")"
capture_trace "gzippy_T${T}" "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" || true
capture_trace "rapidgzip_T${T}" "$mask" "$RG_TRACE" -d -c -f -P "$T" "$CORPUS" || true
capture_trace "gzippy_writev_off_T${T}" "$mask" env GZIPPY_DISABLE_WRITEV=1 \
  "$GZIPPY" -d -c -p "$T" "$CORPUS" || true

# manifest for laptop fulcrum
cat >"$ARTDIR/manifest.json" <<MANIFEST
{
  "branch": "$BRANCH",
  "head": "$GZIPPY_SHA",
  "resolve_ahead": "always",
  "slow_bootstrap": "${GZIPPY_SLOW_BOOTSTRAP:-}",
  "ref_sha": "$REF_SHA",
  "raw_bytes": $RAW_BYTES,
  "threads": "$THREADS",
  "traces": {
    "gzippy": "$ARTDIR/trace_gzippy_T8.json",
    "rapidgzip": "$ARTDIR/trace_rapidgzip_T8.json",
    "gzippy_writev_off": "$ARTDIR/trace_gzippy_writev_off_T8.json"
  }
}
MANIFEST

TRUST=true
[ "$DIVERGED" = 0 ] || TRUST=false
[ "$ANY_SD_FAIL" = 0 ] || TRUST=false
say ""
say "RUN_TRUSTWORTHY=$TRUST diverged=$DIVERGED sd_fail=$ANY_SD_FAIL"
say "FULCRUM_ARTIFACTS=$ARTDIR"
[ "$TRUST" = true ]
