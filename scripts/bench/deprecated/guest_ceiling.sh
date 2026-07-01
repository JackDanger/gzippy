#!/usr/bin/env bash
# guest_ceiling.sh — Stage-5b decode-bypass CEILING oracle (FLOOR + projections).
#
# RUNS ON GUEST 199 after host_lock_and_bench.sh gate PASS (frequency pinned:
# no_turbo=1, governor=performance, min=max — neutralizes the bypass-lowers-
# occupancy turbo deflation).
#
# Bounds the inner-Huffman SPEED-UP ceiling by REMOVING decode (the existing
# decode_bypass oracle) and measuring the interleaved wall. Roles (corrected by
# advisor a62eab35):
#   (A)  FULL-replay, decode≈0, BYTE-EXACT  = THE FLOOR (preserves form-B markers
#        ⇒ apply_window/L_resolve still runs ⇒ keeps the binding publish-chain term)
#   (A2) sleep-0 meta = DECOMPOSITION PROBE (clean zeroed ⇒ ELIDES L_resolve) — not a floor
#   (B)  sleep-66ms meta = rapidgzip-rate projection, resolve-ELIDED ⇒ weak lower bound
# Args: BRANCH=reimplement-isa-l THREADS=8 N=9 (THREADS comma-separated; T8 is the ceiling question)
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
THREADS="${THREADS:-8}"
N="${N:-9}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-ceiling}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
RG_TRACE="${RG_TRACE:-$REPO/vendor/rapidgzip/librapidarchive/build-trace/src/tools/rapidgzip}"
# Capture files live on /dev/shm (RAM, 15G free) — the / disk has only ~1.1G free.
CAPFULL="${CAPFULL:-/dev/shm/gz_bypass_full.bin}"
CAPMETA="${CAPMETA:-/dev/shm/gz_bypass_meta.bin}"
SLEEP_NS_RG="${SLEEP_NS_RG:-66420000}"   # rapidgzip per-chunk window-absent d_w ≈ 66.42ms

for a in "$@"; do
  case "$a" in
    BRANCH=*) BRANCH="${a#*=}";;
    THREADS=*) THREADS="$(echo "${a#*=}" | tr ',' ' ')";;
    N=*) N="${a#*=}";;
  esac
done
[ "$N" -ge 9 ] || N=9
# Ceiling is a T8 question; use the first thread count given, default 8.
T="$(echo "$THREADS" | awk '{print $1}')"; [ -n "$T" ] || T=8

mkdir -p "$ARTDIR"
say() { echo "$@"; }

pin_mask() {
  case "$1" in
    8) echo "0,2,4,6,8,10,12,14";;
    4) echo "0,2,4,6";;
    1) echo "0";;
    *) echo "0,2,4,6,8,10,12,14";;
  esac
}
mask="$(pin_mask "$T")"

# ---- sync branch (provenance requires clean gzippy tree) --------------------
cd "$REPO" || { echo "CEIL_FAILURE=no-repo"; echo "CEILING_GUEST_DONE"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true
DIRTY_COUNT="$(git status --porcelain --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')"
if [ "$DIRTY_COUNT" != "0" ]; then
  echo "CEIL_FAILURE=dirty-tree ($DIRTY_COUNT files)"; echo "CEILING_GUEST_DONE"; exit 6
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
GZIPPY_BUILD_FEATURES="${GZIPPY_BUILD_FEATURES:-pure-rust-inflate}"
say "## build gzippy (${GZIPPY_BUILD_FEATURES}) ..."
if ! cargo build --release --no-default-features --features "${GZIPPY_BUILD_FEATURES}" >"$ARTDIR/build-gzippy.log" 2>&1; then
  echo "CEIL_FAILURE=gzippy-build"; tail -30 "$ARTDIR/build-gzippy.log"; echo "CEILING_GUEST_DONE"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"

# rapidgzip: REUSE the existing trace binary (built earlier today); do NOT rebuild
# (saves ~5min, avoids touching vendor). Only error if truly absent.
if [ ! -x "$RG_TRACE" ]; then
  echo "CEIL_FAILURE=no-rg-trace ($RG_TRACE missing — run guest_fulcrum_capture first to build it)"
  echo "CEILING_GUEST_DONE"; exit 8
fi

[ -f "$CORPUS" ] || { echo "CEIL_FAILURE=corpus"; echo "CEILING_GUEST_DONE"; exit 7; }
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"
cat "$CORPUS" >/dev/null 2>&1

# routing assertion (production path)
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
# Phase-1 naming-truth rename: path is now ParallelSM (was IsalParallelSM).
case "$DBG" in *ParallelSM*) ;; *) echo "CEIL_FAILURE=routing $DBG"; echo "CEILING_GUEST_DONE"; exit 9;; esac

GZIPPY_SHA="$(git rev-parse HEAD)"
say "================ CEILING PROVENANCE ================"
say "branch=$BRANCH head=$GZIPPY_SHA T=$T N=$N"
say "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
say "rapidgzip=$("$RG_TRACE" --version 2>&1 | head -1)"
say "freq lock readback: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
say "mem: $(free -m | awk '/Mem:/{print "avail="$7"M"} /Swap:/{print "swapfree="$4"M"}' | tr '\n' ' ')"
say "==================================================="

drop_caches() { sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true; }
swap_in() { awk '/pswpin/{print $2}' /proc/vmstat; }

# timed_run <ignore_sha:0|1> <mask> <cmd...> -> echoes "secs sha"
timed_run() {
  local _ig="$1" mask="$2"; shift 2
  local out s e rc secs sha
  out="$(mktemp)"
  s=$(date +%s.%N)
  set +e; taskset -c "$mask" "$@" >"$out" 2>>"$ARTDIR/wall.stderr"; rc=$?; set -e
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || say "WARN exit=$rc: $*"
  rm -f "$out"
  echo "$secs $sha"
}

stats() { # min med sdpct
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1; sum+=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      mean=sum/n; ss=0; for(i=1;i<=n;i++){d=v[i]-mean;ss+=d*d}
      sd=(n>1)?sqrt(ss/(n-1)):0; printf "%.4f %.4f %.1f", min, mid, (mean>0)?sd/mean*100:0; }'
}
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

GZWALL="env GZIPPY_FORCE_PARALLEL_SM=1"

# ============================================================================
# CAPTURE PASS 1 — FULL payload capture (for A). One normal decode that records.
# ============================================================================
drop_caches
say ""
say "## CAPTURE-FULL -> $CAPFULL"
rm -f "$CAPFULL"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_CAPTURE="$CAPFULL" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$ARTDIR/cap-full.stderr" || true
CAP_LINE="$(grep -m1 'BYPASS_CAPTURE wrote' "$ARTDIR/cap-full.stderr" || true)"
CAP_CHUNKS="$(echo "$CAP_LINE" | grep -o 'wrote [0-9]*' | grep -o '[0-9]*' || echo 0)"
say "CEIL_CAPTURE_FULL: $CAP_LINE  (file=$(ls -la "$CAPFULL" 2>/dev/null | awk '{print $5}') bytes)"
if [ "${CAP_CHUNKS:-0}" -lt 1 ]; then
  echo "CEIL_FAILURE=capture-wrote-0-chunks"; echo "CEILING_GUEST_DONE"; exit 10
fi

# ============================================================================
# CAPTURE PASS 2 — META-ONLY (tiny) for the sleep probes (A2, B).
# ============================================================================
say "## CAPTURE-META -> $CAPMETA"
rm -f "$CAPMETA"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_META_ONLY=1 GZIPPY_BYPASS_CAPTURE="$CAPMETA" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$ARTDIR/cap-meta.stderr" || true
say "CEIL_CAPTURE_META: $(grep -m1 'BYPASS_CAPTURE wrote' "$ARTDIR/cap-meta.stderr" || echo none)  (file=$(ls -la "$CAPMETA" 2>/dev/null | awk '{print $5}') bytes)"

# ---- standalone hit/miss + sha probe for the FLOOR (A) ----
drop_caches
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE="$CAPFULL" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$ARTDIR/floor_probe.out" 2>"$ARTDIR/floor_probe.stderr" || true
HM="$(grep -m1 'BYPASS_DECODE replay:' "$ARTDIR/floor_probe.stderr" || echo 'replay: hits=? misses=?')"
HITS="$(echo "$HM" | grep -o 'hits=[0-9]*' | grep -o '[0-9]*' || echo 0)"
MISS="$(echo "$HM" | grep -o 'misses=[0-9]*' | grep -o '[0-9]*' || echo 0)"
HITPCT="$(awk -v h="$HITS" -v m="$MISS" 'BEGIN{t=h+m; printf "%.1f", (t>0)?100*h/t:0}')"
PROBE_SHA="$(sha256sum "$ARTDIR/floor_probe.out" | cut -d' ' -f1)"
PROBE_SHA_OK=DIVERGE; [ "$PROBE_SHA" = "$REF_SHA" ] && PROBE_SHA_OK=OK
say "CEIL_FLOOR_PROBE: $HM  hit%=$HITPCT  sha=$PROBE_SHA_OK"
rm -f "$ARTDIR/floor_probe.out"

# ============================================================================
# PASS A — FLOOR, sha-EXACT, interleaved. Contenders (gzip-ref FIRST = sha ref):
#   gzip_ref | gzippy_bypass(FULL) | rapidgzip | gzippy_normal
# ============================================================================
drop_caches
SWAP0="$(swap_in)"
say ""
say "## PASS A — FLOOR interleave (N=$N, drop iter0)"
GZB=""; RG=""; GZN=""; DIVERGED=0
for ((i=0;i<=N;i++)); do
  read bsec bsha < <(timed_run 0 "$mask" $GZWALL GZIPPY_BYPASS_DECODE="$CAPFULL" "$GZIPPY" -d -c -p "$T" "$CORPUS")
  read rsec rsha < <(timed_run 1 "$mask" "$RG_TRACE" -d -c -f -P "$T" "$CORPUS")
  read nsec nsha < <(timed_run 0 "$mask" $GZWALL "$GZIPPY" -d -c -p "$T" "$CORPUS")
  [ "$i" -eq 0 ] && continue
  GZB="$GZB $bsec"; RG="$RG $rsec"; GZN="$GZN $nsec"
  [ "$bsha" = "$REF_SHA" ] || { say "DIVERGE gzippy_bypass i=$i sha=$bsha"; DIVERGED=1; }
  [ "$rsha" = "$REF_SHA" ] || { say "DIVERGE rapidgzip i=$i sha=$rsha"; DIVERGED=1; }
  [ "$nsha" = "$REF_SHA" ] || { say "DIVERGE gzippy_normal i=$i sha=$nsha"; DIVERGED=1; }
done
SWAP1="$(swap_in)"
SWAPPED="$(awk -v a="$SWAP0" -v b="$SWAP1" 'BEGIN{print b-a}')"
read bmin bmed bsd < <(stats "$GZB")
read rmin rmed rsd < <(stats "$RG")
read nmin nmed nsd < <(stats "$GZN")

# ============================================================================
# PASS A2 — sleep-0 decomposition probe (clean zeroed, ELIDES resolve). garbage out.
# ============================================================================
drop_caches
say "## PASS A2 — sleep-0 (resolve-elided probe; garbage output)"
A2=""
for ((i=0;i<=N;i++)); do
  read s _ < <(timed_run 1 "$mask" $GZWALL GZIPPY_SLEEP_DECODE_NS=0 GZIPPY_BYPASS_DECODE="$CAPMETA" "$GZIPPY" -d -c -p "$T" "$CORPUS")
  [ "$i" -eq 0 ] && continue
  A2="$A2 $s"
done
read a2min a2med a2sd < <(stats "$A2")

# ============================================================================
# PASS B — sleep-66ms rapidgzip-rate projection (resolve-elided ⇒ weak lower bound).
# ============================================================================
drop_caches
say "## PASS B — sleep-${SLEEP_NS_RG}ns rapidgzip-rate (resolve-elided; garbage output)"
PB=""
for ((i=0;i<=N;i++)); do
  read s _ < <(timed_run 1 "$mask" $GZWALL GZIPPY_SLEEP_DECODE_NS="$SLEEP_NS_RG" GZIPPY_BYPASS_DECODE="$CAPMETA" "$GZIPPY" -d -c -p "$T" "$CORPUS")
  [ "$i" -eq 0 ] && continue
  PB="$PB $s"
done
read bbmin bbmed bbsd < <(stats "$PB")

# ============================================================================
# FLOOR-PASS TRACE — for binding-component identification (decode≈0).
# ============================================================================
drop_caches
say "## FLOOR TRACE -> $ARTDIR/trace_floor_T${T}.json"
GZIPPY_TIMELINE="$ARTDIR/trace_floor_T${T}.json" GZIPPY_MEMLIFE="$ARTDIR/memlife_floor_T${T}.json" GZIPPY_VERBOSE=1 \
  env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE="$CAPFULL" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ARTDIR/trace_floor.log" || true
ls -la "$ARTDIR/trace_floor_T${T}.json" 2>/dev/null || say "WARN no floor trace"

# manifest for laptop fulcrum
cat >"$ARTDIR/manifest.json" <<EOF
{"head":"$GZIPPY_SHA","T":$T,"ref_sha":"$REF_SHA","raw_bytes":$RAW_BYTES,
 "floor_trace":"$ARTDIR/trace_floor_T${T}.json"}
EOF

# ---- decomposition + verdict ----
decomp="$(awk -v a="$bmin" -v b="$a2min" 'BEGIN{printf "%.4f", a-b}')"
rm -f "$CAPFULL" "$CAPMETA"

say ""
say "================ CEILING SUMMARY (T=$T) ================"
printf "CEIL_BASELINE_NORMAL  min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f\n" "$nmin" "$(mbps "$nmin")" "$nmed" "$nsd"
printf "CEIL_FLOOR_A          min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f  sha=%s hit%%=%s swap_in_pages=%s  [THE FLOOR; preserves L_resolve]\n" \
  "$bmin" "$(mbps "$bmin")" "$bmed" "$bsd" "$([ $DIVERGED = 0 ] && echo OK || echo DIVERGE)" "$HITPCT" "$SWAPPED"
printf "CEIL_RAPIDGZIP        min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsd"
printf "CEIL_A2_SLEEP0        min=%.4fs med=%.4f sd%%=%.1f  [PROBE: resolve-ELIDED, garbage out — NOT the floor]\n" "$a2min" "$a2med" "$a2sd"
printf "CEIL_B_SLEEP66        min=%.4fs med=%.4f sd%%=%.1f  [rapidgzip-rate, resolve-ELIDED weak lower bound, garbage]\n" "$bbmin" "$bbmed" "$bbsd"
printf "CEIL_DECOMP_A_minus_A2= %.4fs  [≈ L_resolve+load; model L_resolve≈0.78s]\n" "$decomp"

# verdict (model predicts FLOOR≈0.78–0.84s ⇒ CAPPED)
VERDICT=VOID; REASON=""
# NOTE: replay misses are STRUCTURAL (run-to-run speculative (start_bit,stop_hint)
# key drift between the separate capture and replay processes; validated locally
# ~89% on a 18-chunk corpus). Misses fall back to REAL decode ⇒ they keep bytes
# correct AND only INFLATE the floor (upper bound). So a moderate miss rate does
# NOT void the run — it makes (A) a (looser) upper bound, which still anchors
# CAPPED. VOID only on a SEVERE miss rate (<90% hit) where contamination is large.
if [ "$DIVERGED" != 0 ]; then REASON="(A) sha DIVERGED — oracle void"
elif awk -v h="$HITPCT" 'BEGIN{exit !(h<90)}'; then REASON="(A) hit%=$HITPCT <90 — floor heavily contaminated by real decode"
elif awk -v s="$bsd" 'BEGIN{exit !(s>5)}'; then REASON="(A) sd%=$bsd >5 — bimodal/swap-thrash, invalid"
else
  if awk -v f="$bmin" 'BEGIN{exit !(f<=0.55)}'; then VERDICT=TIE; REASON="FLOOR(A)=$bmin ≤0.55 ⇒ inner-loop is the arc (CONTRADICTS model 0.78–0.84 — reconcile)"
  elif awk -v f="$bmin" 'BEGIN{exit !(f>0.60)}'; then VERDICT=CAPPED; REASON="FLOOR(A)=$bmin >0.60 with decode≈0 ⇒ another component binds (model: publish-chain/L_resolve)"
  else VERDICT=AMBIGUOUS; REASON="FLOOR(A)=$bmin in (0.55,0.60] grey zone — lean on decomposition+model"
  fi
fi
say "CEIL_VERDICT=$VERDICT  $REASON"
say "CEIL_NOTE: floor trace at $ARTDIR/trace_floor_T${T}.json — analyze for binding component under free decode"
say "================ END CEILING SUMMARY ================"
echo "CEILING_GUEST_DONE"
