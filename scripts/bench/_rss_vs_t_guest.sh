#!/usr/bin/env bash
# _rss_vs_t_guest.sh — GUEST-SIDE half of rss_vs_t.sh (the cache-residency
# mandate instrument: plans/gzippy-native-design-mandate.md). Shipped+executed by
# rss_vs_t.sh over the double-hop; not run directly by the owner.
#
# Measures, on the FROZEN/quiet guest, for gzippy-native vs rapidgzip across a
# thread grid:
#   (a) PEAK RSS per T (/usr/bin/time -v "Maximum resident set size", KiB on
#       Linux) — the mandate wants RSS roughly FLAT as T rises (shared/small),
#       not growing ~linearly;
#   (b) the IN-PROCESS per-thread working-set byte accounting (GZIPPY_MEM_STATS=1,
#       native Block engine) — KiB/thread + aggregate + shared-table bytes;
#   (c) the POSITIVE CONTROL: GZIPPY_MEM_BALLAST_MIB=N must recover a per-thread
#       RSS slope ~= N*threads (linearity validates the RSS-vs-T mechanism);
#   (d) optional MPKI/mem-stall via `perf stat` (L2/L3 miss + cycle stalls) for
#       gzippy-native vs rapidgzip — VALIDATE FIRST with the ballast control
#       (MPKI/RSS must move monotonically with ballast).
#
# Correctness (Rule 4): EVERY decode is sha-verified against the decompressed
# corpus pin; ANY mismatch ABORTS (a fast wrong-bytes win is a loss). Production
# path asserted (GZIPPY_DEBUG -> path=ParallelSM). Host-freeze readback + quiet
# gate mirror _parity_guest.sh.
#
# Inputs (env, passed by rss_vs_t.sh from guest.env + flags):
#   GUEST_SRC GZIPPY_BIN CORPUS CORPUS_RAW_SHA256 RG RG_TRACE FEATURE
#   TLIST (space-separated, e.g. "1 8 16") N GOV NO_TURBO HOST_FROZEN
#   RUSTFLAGS_PIN DO_BUILD ARTDIR  BALLAST_MIB (control N list)  DO_PERF
set -u

fail() { echo "RSS_FAIL=$1"; echo "RSS_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${FEATURE:?}"; : "${TLIST:?}"
N="${N:-5}"
RG="${RG:-rapidgzip}"; RG_TRACE="${RG_TRACE:-}"
RUSTFLAGS_PIN="${RUSTFLAGS_PIN:--C target-cpu=native}"
DO_BUILD="${DO_BUILD:-0}"; DO_PERF="${DO_PERF:-1}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-rss-art}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"; HOST_FROZEN="${HOST_FROZEN:-0}"
BALLAST_MIB="${BALLAST_MIB:-8 16 32}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5

# ---- 1. contamination guard (allowlist scrub) --------------------------------
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    GZIPPY_FORCE_PARALLEL_SM|GZIPPY_DEBUG|GZIPPY_BIN|GZIPPY_MEM_STATS|GZIPPY_MEM_BALLAST_MIB) ;;
    *) SCRUBBED="$SCRUBBED $v=${!v}"; unset "$v";;
  esac
done
if [ -n "$SCRUBBED" ]; then
  echo "## SCRUBBED non-production GZIPPY_* env:$SCRUBBED"
  case "$SCRUBBED" in
    *GZIPPY_SEED*|*GZIPPY_*ORACLE*|*GZIPPY_BYPASS*|*GZIPPY_SLEEP_DECODE*|*GZIPPY_SLOW*)
      fail "contaminated-env (seeding/oracle var present:$SCRUBBED)" 2;;
  esac
fi

# ---- 2. build (cargo-lock serialized) ----------------------------------------
CARGO_LOCK="${CARGO_LOCK:-scripts/cargo-lock.sh}"
if [ "$DO_BUILD" = 1 ]; then
  echo "## build feature=$FEATURE RUSTFLAGS='$RUSTFLAGS_PIN' (cargo-lock serialized)"
  if [ -x "$CARGO_LOCK" ]; then BUILDER=(sh "$CARGO_LOCK"); else BUILDER=(); fi
  RUSTFLAGS="$RUSTFLAGS_PIN" "${BUILDER[@]}" \
    cargo build --release --no-default-features --features "$FEATURE" \
    >"$ARTDIR/build.log" 2>&1
  brc=$?
  if [ "$brc" -ne 0 ]; then grep -E 'error' "$ARTDIR/build.log" | head -30; fail "build rc=$brc" 8; fi
  grep -E 'Finished' "$ARTDIR/build.log" | tail -1 || true
fi
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (run with --build)" 5

# ---- 3. corpus + oracle ------------------------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$REF_SHA" = "$CORPUS_RAW_SHA256" ] || fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256" 7

# ---- 4. production-path assertion --------------------------------------------
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p 8 "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in *ParallelSM*) ;; *) fail "routing not ParallelSM: ${DBG:-<none>}" 9;; esac

# ---- 5. host-freeze readback (ABORT on readable thaw) ------------------------
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
gov_state=MATCH; [ "$ACT_GOV" = "$GOV" ] || gov_state=$([ "$ACT_GOV" = NA ] && echo NA || echo WRONG)
trb_state=MATCH; [ "$ACT_TURBO" = "$NO_TURBO" ] || trb_state=$([ "$ACT_TURBO" = NA ] && echo NA || echo WRONG)
case "$gov_state/$trb_state" in
  MATCH/MATCH) ;;
  *WRONG*) fail "host-not-frozen governor=$ACT_GOV no_turbo=$ACT_TURBO (expected $GOV/$NO_TURBO)" 13;;
  *) [ "$HOST_FROZEN" = 1 ] && echo "## WARN: freeze unreadable (gov=$ACT_GOV turbo=$ACT_TURBO) HOST_FROZEN=1 ack" \
       || fail "host-freeze-unreadable gov=$ACT_GOV turbo=$ACT_TURBO; pass HOST_FROZEN=1" 13;;
esac

# ---- 5b. quiet gate ----------------------------------------------------------
QUIET_MAX_RUNNABLE="${QUIET_MAX_RUNNABLE:-2.0}"; QUIET_SAMPLES="${QUIET_SAMPLES:-4}"
ALLOW_LOAD="${ALLOW_LOAD:-0}"
RUN_SUM=0; RUN_CNT=0
for _qs in $(seq 1 "$QUIET_SAMPLES"); do
  _r="$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA)"
  [ "$_r" = NA ] && break
  RUN_SUM=$((RUN_SUM + _r)); RUN_CNT=$((RUN_CNT + 1))
  [ "$_qs" -lt "$QUIET_SAMPLES" ] && sleep 1
done
if [ "$RUN_CNT" -gt 0 ]; then
  RUN_AVG="$(awk -v s="$RUN_SUM" -v c="$RUN_CNT" 'BEGIN{printf "%.2f", s/c}')"
  if awk -v a="$RUN_AVG" -v m="$QUIET_MAX_RUNNABLE" 'BEGIN{exit !(a+0>m+0)}'; then
    [ "$ALLOW_LOAD" = 1 ] && echo "## WARN runnable_avg=$RUN_AVG>$QUIET_MAX_RUNNABLE ALLOW_LOAD=1" \
      || fail "host-loaded runnable_avg=$RUN_AVG>$QUIET_MAX_RUNNABLE" 13
  fi
  echo "## quiet-gate: runnable_avg=$RUN_AVG (max=$QUIET_MAX_RUNNABLE)"
else RUN_AVG=NA; fi

# rapidgzip presence
RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"
else fail "no-rapidgzip" 12; fi

pin_mask() { case "$1" in 1) echo 0;; 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;;
  16) echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;; *) echo "";; esac; }

BIN_SHA="$(sha256sum "$GZIPPY_BIN" | cut -c1-16)"
echo "================ RSS-VS-T PROVENANCE ================"
echo "feature=$FEATURE TLIST='$TLIST' N=$N binary=$GZIPPY_BIN bin_sha=$BIN_SHA"
echo "corpus=$CORPUS ref_sha=$REF_SHA rapidgzip=$("$RG_CMD" --version 2>&1 | head -1)"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO runnable_avg=${RUN_AVG:-NA} host_frozen=$HOST_FROZEN nproc=$(nproc)"
echo "===================================================="

SINK="$ARTDIR/sink.bin"

# peak_rss <mask> <env-prefix-cmd...> -> echoes "rss_kib sha"
# /usr/bin/time -v prints "Maximum resident set size (kbytes)" on Linux.
peak_rss() {
  local mask="$1"; shift
  rm -f "$SINK"; : > "$SINK"
  local t out rss sha
  out="$ARTDIR/time.out"
  /usr/bin/time -v taskset -c "$mask" "$@" >"$SINK" 2>"$out"
  rss="$(awk -F': ' '/Maximum resident set size/{print $2}' "$out")"
  sha="$(sha256sum "$SINK" | cut -d' ' -f1)"
  echo "${rss:-NA} $sha"
}

# best-of-N MIN peak RSS (RSS is allocation-driven so MIN is the cleanest floor)
min_rss() { # <mask> <env+cmd...>
  local mask="$1"; shift
  local best="" sha=""
  for ((i=0;i<N;i++)); do
    read r s < <(peak_rss "$mask" "$@")
    [ "$s" = "$REF_SHA" ] || { echo "NA SHADIVERGE"; return; }
    if [ -z "$best" ] || awk -v a="$r" -v b="$best" 'BEGIN{exit !(a+0<b+0)}'; then best="$r"; fi
    sha="$s"
  done
  echo "$best OK"
}

echo ""
echo "================ (a) PEAK RSS vs T (KiB; min-of-N) ================"
printf "%-4s %-14s %-14s %-10s\n" "T" "gzippy(KiB)" "rapidgzip(KiB)" "ratio g/rg"
GZ_RSS_FIRST=""; GZ_RSS_LAST=""; FIRST_T=""; LAST_T=""
for T in $TLIST; do
  MASK="$(pin_mask "$T")"; [ -n "$MASK" ] || { echo "T=$T unsupported mask, skip"; continue; }
  read grss gok < <(min_rss "$MASK" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS")
  [ "$gok" = OK ] || fail "gzippy sha-diverge at T=$T" 11
  read rrss rok < <(min_rss "$MASK" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  [ "$rok" = OK ] || fail "rapidgzip sha-diverge at T=$T" 11
  ratio="$(awk -v g="$grss" -v r="$rrss" 'BEGIN{printf "%.2f", (r>0)?g/r:0}')"
  printf "%-4s %-14s %-14s %-10s\n" "$T" "$grss" "$rrss" "$ratio"
  [ -z "$FIRST_T" ] && { FIRST_T="$T"; GZ_RSS_FIRST="$grss"; }
  LAST_T="$T"; GZ_RSS_LAST="$grss"
done
# RSS-flatness: how much did gzippy RSS grow from the smallest to largest T?
if [ -n "$GZ_RSS_FIRST" ] && [ -n "$GZ_RSS_LAST" ]; then
  echo "## gzippy RSS growth T${FIRST_T}->T${LAST_T}: $(awk -v a="$GZ_RSS_FIRST" -v b="$GZ_RSS_LAST" \
    'BEGIN{printf "%+.1f%% (%d->%d KiB)", (a>0)?(b-a)/a*100:0, a, b}')  (mandate wants ~FLAT)"
fi

echo ""
echo "================ (b) per-thread WORKING-SET byte accounting ================"
# Native in-process accounting at the largest T (uniform per-thread size).
BIG_T="$(echo $TLIST | tr ' ' '\n' | sort -n | tail -1)"
BIG_MASK="$(pin_mask "$BIG_T")"
rm -f "$SINK"; : > "$SINK"
GZIPPY_MEM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$BIG_MASK" \
  "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS" >"$SINK" 2>"$ARTDIR/memstats.txt"
ms_sha="$(sha256sum "$SINK" | cut -d' ' -f1)"
[ "$ms_sha" = "$REF_SHA" ] || fail "mem-stats run sha-diverge" 11
echo "## native byte accounting @T=$BIG_T (sha=OK):"
cat "$ARTDIR/memstats.txt"

echo ""
echo "================ (c) POSITIVE CONTROL: ballast slope ================"
# baseline + ballast N MiB/thread @ BIG_T; the incremental RSS slope per +MiB must
# recover ~= thread-count (the instrument can resolve a KNOWN per-thread alloc).
read base_rss bok < <(min_rss "$BIG_MASK" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS")
[ "$bok" = OK ] || fail "ballast-baseline sha-diverge" 11
echo "## baseline RSS @T=$BIG_T: $base_rss KiB"
prev_n=0; prev_rss="$base_rss"
for NB in $BALLAST_MIB; do
  read brss bok < <(min_rss "$BIG_MASK" env GZIPPY_MEM_BALLAST_MIB="$NB" GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS")
  [ "$bok" = OK ] || fail "ballast N=$NB sha-diverge" 11
  # incremental slope vs previous ballast point, in "threads recovered per +1 MiB/thread"
  thr="$(awk -v r1="$prev_rss" -v r0="$base_rss" -v n="$NB" 'BEGIN{ if(n>0) printf "%.2f", (r1*0 + 0); }')"
  inc_thr="$(awk -v cur="$brss" -v prv="$prev_rss" -v dn="$((NB-prev_n))" \
    'BEGIN{ if(dn>0) printf "%.2f", (cur-prv)/1024.0/dn; else print "NA"}')"
  echo "## ballast=${NB}MiB/thread: RSS=$brss KiB  (incremental slope vs ${prev_n}MiB = ${inc_thr} threads-recovered/MiB)"
  prev_n="$NB"; prev_rss="$brss"
done
echo "## CONTROL PASS criterion: incremental slope ~= worker-thread count at T=$BIG_T (validates RSS-vs-T mechanism)"

if [ "$DO_PERF" = 1 ] && command -v perf >/dev/null 2>&1; then
  echo ""
  echo "================ (d) MPKI / mem-stall (perf stat) ================"
  PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo NA)"
  echo "## perf_event_paranoid=$PARANOID (need <=2 or root for counters)"
  EVENTS="instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"
  perf_run() { # <mask> <label> <env+cmd...>
    local mask="$1" label="$2"; shift 2
    rm -f "$SINK"; : > "$SINK"
    perf stat -e "$EVENTS" -- taskset -c "$mask" "$@" >"$SINK" 2>"$ARTDIR/perf_${label}.txt"
    local s; s="$(sha256sum "$SINK" | cut -d' ' -f1)"
    [ "$s" = "$REF_SHA" ] || echo "## WARN perf $label sha-diverge"
    echo "## --- perf $label ---"
    grep -E 'instructions|cache-misses|cache-references|LLC|L1-dcache' "$ARTDIR/perf_${label}.txt" || cat "$ARTDIR/perf_${label}.txt"
    # MPKI = LLC-load-misses / instructions * 1000
    local instr llcm
    instr="$(awk '/instructions/{gsub(/,/,"",$1); print $1; exit}' "$ARTDIR/perf_${label}.txt")"
    llcm="$(awk '/LLC-load-misses/{gsub(/,/,"",$1); print $1; exit}' "$ARTDIR/perf_${label}.txt")"
    if [ -n "$instr" ] && [ -n "$llcm" ]; then
      echo "## LLC-load-miss MPKI ($label) = $(awk -v m="$llcm" -v i="$instr" 'BEGIN{printf "%.3f", (i>0)?m/i*1000:0}')"
    fi
  }
  # Validate perf instrument FIRST with the ballast positive control: ballast must
  # MOVE MPKI/cache-misses monotonically (it touches N MiB/thread of fresh pages).
  echo "## perf ballast control (cache-misses must rise with ballast):"
  perf_run "$BIG_MASK" "gzippy_ballast0" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS"
  perf_run "$BIG_MASK" "gzippy_ballast32" env GZIPPY_MEM_BALLAST_MIB=32 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS"
  echo "## perf gzippy-native vs rapidgzip @T=$BIG_T:"
  perf_run "$BIG_MASK" "gzippy_native" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$BIG_T" "$CORPUS"
  perf_run "$BIG_MASK" "rapidgzip" "$RG_CMD" -d -c -f -P "$BIG_T" "$CORPUS"
else
  echo ""
  echo "## (d) MPKI skipped (DO_PERF=$DO_PERF or perf absent)"
fi

rm -f "$SINK"
echo ""
echo "================ END RSS-VS-T ================"
echo "RSS_GUEST_DONE"
