#!/usr/bin/env bash
# _residual_guest.sh — guest-side MULTI-CONTENDER interleaved measurement for the
# structural-residual sizing campaign (owner/structural-residual-sizing).
#
# Sizes the low-T BAR-1 residual by running, IN ONE interleaved trial loop (so the
# prod-vs-oracle delta is jitter-immune), several gzippy/rapidgzip variants:
#   MODE=output-floor : gz-prod, gz-skipwritev (GZIPPY_SKIP_WRITEV_SYSCALL removal
#                       oracle, output syscall removed), rg-file, rg-null.
#                       => gzippy serial-output share = gz-prod - gz-skipwritev;
#                          rg output exposure         = rg-file - rg-null;
#                          gzippy-SPECIFIC output excess = the two deltas compared.
#   MODE=marker-perturb : gz-off, gz-spinPCT, gz-sleepPCT (GZIPPY_SLOW_MARKER_MODE
#                       slow-injection of the u16 marker bootstrap + the
#                       frequency-neutral sleep CONTROL), rg-file.
#                       => marker bootstrap on the critical path iff gz-off<gz-spin
#                          monotonically AND the rise SURVIVES spin->sleep.
#
# It REUSES the parity/oracle contamination bar verbatim: GZIPPY_* allowlist scrub,
# host-freeze HARD-FAIL on a readable thaw, quiet-gate on procs_running, stale-binary
# content-fingerprint guard, ParallelSM production-path assert, regular-file sink,
# sha-verify on the byte-exact contenders. NOT production code — a measurement
# instrument (worktree tooling).
set -u

fail() { echo "RESIDUAL_FAIL=$1"; echo "RESIDUAL_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${T:?}"; : "${N:?}"; : "${MASK:?}"; : "${MODE:?}"
RG="${RG:-rapidgzip}"; RG_TRACE="${RG_TRACE:-}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-residual-art}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
HOST_FROZEN="${HOST_FROZEN:-0}"; ALLOW_LOAD="${ALLOW_LOAD:-0}"; MAX_LOADAVG="${MAX_LOADAVG:-2.0}"
SLOW_PCT="${SLOW_PCT:-50}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5

# ---- 1. CONTAMINATION GUARD: ALLOWLIST scrub of GZIPPY_* env ------------------
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    GZIPPY_BIN) ;;
    *) SCRUBBED="$SCRUBBED $v=${!v}"; unset "$v";;
  esac
done
if [ -n "$SCRUBBED" ]; then
  echo "## SCRUBBED inherited GZIPPY_* env before measuring (knobs applied per-command, NOT inherited):$SCRUBBED"
  case "$SCRUBBED" in
    *GZIPPY_SEED*|*GZIPPY_*ORACLE*|*GZIPPY_BYPASS*|*GZIPPY_SLOW*|*GZIPPY_SKIP*)
      fail "contaminated-env (a measurement knob was inherited:$SCRUBBED — this runner sets its knobs per-command)" 2;;
  esac
fi

# ---- 2. binary present + CONTENT-FINGERPRINT STALE-BINARY guard --------------
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (parity.sh --build first)" 5
FPRINT="$GZIPPY_BIN.inputs.sha"
input_fingerprint() {
  { find src crates examples build.rs Cargo.toml Cargo.lock vendor benches -type f 2>/dev/null \
      | LC_ALL=C sort | xargs sha256sum 2>/dev/null; } | sha256sum | cut -d' ' -f1
}
[ -f "$FPRINT" ] || fail "no-build-fingerprint:$FPRINT absent — build via parity.sh --build (stamps it)" 6
CUR_FP="$(input_fingerprint)"; STAMP_FP="$(cat "$FPRINT" 2>/dev/null)"
[ "$CUR_FP" = "$STAMP_FP" ] || fail "stale-binary:$GZIPPY_BIN built from different inputs (stamp=$STAMP_FP cur=$CUR_FP)" 6

# ---- 3. corpus + correctness oracle -----------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$REF_SHA" = "$CORPUS_RAW_SHA256" ] || fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256" 7
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# ---- 4. production-path assert + isal_chunks STEP-0 readback -----------------
COVLOG="$ARTDIR/coverage_T$T.txt"
GZIPPY_DEBUG=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  taskset -c "$MASK" "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$COVLOG" || true
DBG="$(grep -m1 'path=' "$COVLOG" || true)"
case "$DBG" in *ParallelSM*) ;; *) fail "routing not ParallelSM: ${DBG:-none}" 9;; esac
COVLINE="$(grep -m1 'isal_chunks=' "$COVLOG" || true)"
ISAL_CHUNKS="$(printf '%s' "$COVLINE" | sed -n 's/.*isal_chunks=\([0-9]*\).*/\1/p')"
ISAL_FB="$(printf '%s' "$COVLINE" | sed -n 's/.*isal_fallbacks=\([0-9]*\).*/\1/p')"

# ---- 5. host-lock READBACK (HARD-FAIL on a readable thaw) --------------------
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
gov_state=MATCH; [ "$ACT_GOV" = "$GOV" ] || gov_state=$([ "$ACT_GOV" = NA ] && echo NA || echo WRONG)
trb_state=MATCH; [ "$ACT_TURBO" = "$NO_TURBO" ] || trb_state=$([ "$ACT_TURBO" = NA ] && echo NA || echo WRONG)
case "$gov_state/$trb_state" in
  MATCH/MATCH) ;;
  *WRONG*) fail "host-not-frozen governor=$ACT_GOV no_turbo=$ACT_TURBO (expected $GOV/$NO_TURBO)" 13;;
  *) if [ "$HOST_FROZEN" = 1 ]; then echo "## WARN: host freeze unreadable (gov=$ACT_GOV turbo=$ACT_TURBO) HOST_FROZEN=1 ack.";
     else fail "host-freeze-unreadable governor=$ACT_GOV no_turbo=$ACT_TURBO — pass HOST_FROZEN=1" 13; fi;;
esac

# ---- 5b. QUIET-BOX readback (instantaneous runnable) ------------------------
QUIET_MAX_RUNNABLE="${QUIET_MAX_RUNNABLE:-2.0}"; QUIET_SAMPLES="${QUIET_SAMPLES:-4}"
LOAD1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA)"
RUN_SUM=0; RUN_CNT=0
for _qs in $(seq 1 "$QUIET_SAMPLES"); do
  _r="$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA)"
  [ "$_r" = NA ] && break
  RUN_SUM=$((RUN_SUM + _r)); RUN_CNT=$((RUN_CNT + 1))
  [ "$_qs" -lt "$QUIET_SAMPLES" ] && sleep 1
done
if [ "$RUN_CNT" -gt 0 ]; then
  RUN_AVG="$(awk -v s="$RUN_SUM" -v c="$RUN_CNT" 'BEGIN{printf "%.2f", s/c}')"
  run_hot="$(awk -v a="$RUN_AVG" -v m="$QUIET_MAX_RUNNABLE" 'BEGIN{print (a+0>m+0)?1:0}')"
  echo "## quiet-gate: runnable_avg=$RUN_AVG (max=$QUIET_MAX_RUNNABLE) loadavg1=$LOAD1(EMA,context)"
  if [ "$run_hot" = 1 ]; then
    if [ "$ALLOW_LOAD" = 1 ]; then echo "## WARN: runnable_avg=$RUN_AVG>$QUIET_MAX_RUNNABLE ALLOW_LOAD=1 — trust RATIO only.";
    else fail "host-loaded runnable_avg=$RUN_AVG>$QUIET_MAX_RUNNABLE — a neighbor escaped the freeze" 13; fi
  fi
else RUN_AVG=NA; fi

RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"; fi
[ -n "$RG_CMD" ] || fail "no-rapidgzip" 8

# ---- 6. build the contender table (parallel arrays) -------------------------
# Each contender: LABELS[i], ENVS[i] (extra env for the cmd), CMDS[i] (full cmd
# WITHOUT taskset; sink appended by timed()), SHACK[i] (1=verify vs REF_SHA),
# SINKTO[i] ("file"=regular-file sink, "null"=/dev/null sink).
declare -a LABELS ENVS CMDS SHACK SINKTO
add() { LABELS+=("$1"); ENVS+=("$2"); CMDS+=("$3"); SHACK+=("$4"); SINKTO+=("$5"); }

GZ="$GZIPPY_BIN -d -c -p $T $CORPUS"
RGC="$RG_CMD -d -c -f -P $T $CORPUS"
case "$MODE" in
  output-floor)
    add "gz-prod"       "GZIPPY_FORCE_PARALLEL_SM=1"                              "$GZ"  1 file
    add "gz-skipwritev" "GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SKIP_WRITEV_SYSCALL=1" "$GZ"  0 file
    add "rg-file"       ""                                                        "$RGC" 1 file
    add "rg-null"       ""                                                        "$RGC" 0 null
    ;;
  marker-perturb)
    add "gz-off"            "GZIPPY_FORCE_PARALLEL_SM=1"                                                       "$GZ" 1 file
    add "gz-spin$SLOW_PCT"  "GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SLOW_MARKER_MODE=$SLOW_PCT GZIPPY_SLOW_KIND=spin"  "$GZ" 1 file
    add "gz-sleep$SLOW_PCT" "GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SLOW_MARKER_MODE=$SLOW_PCT GZIPPY_SLOW_KIND=sleep" "$GZ" 1 file
    add "rg-file"           ""                                                                                 "$RGC" 1 file
    ;;
  *) fail "unknown MODE=$MODE" 2;;
esac

BIN_SHA="$(sha256sum "$GZIPPY_BIN" | cut -c1-16)"
echo "================ RESIDUAL PROVENANCE ================"
echo "mode=$MODE T=$T N=$N mask=$MASK slow_pct=$SLOW_PCT"
echo "guest_src=$GUEST_SRC binary=$GZIPPY_BIN bin_sha=$BIN_SHA inputs_fp_ok=yes"
echo "STEP0 isal_chunks=${ISAL_CHUNKS:-NA} isal_fallbacks=${ISAL_FB:-NA} path=ParallelSM (from GZIPPY_VERBOSE @T=$T)"
echo "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO runnable_avg=${RUN_AVG:-NA} loadavg1=$LOAD1 host_frozen=$HOST_FROZEN"
echo "## NOTE: MEASUREMENT/ORACLE run — NOT a bankable production parity number by itself."
echo "===================================================="

# ---- 7. sinks (defend against planted FIFO/symlink) -------------------------
SINK_FILE="$ARTDIR/sink_file.bin"
rm -f "$SINK_FILE" 2>/dev/null || true; : > "$SINK_FILE" || fail "cannot-create-sink:$SINK_FILE" 14
[ -f "$SINK_FILE" ] && [ ! -L "$SINK_FILE" ] && [ ! -p "$SINK_FILE" ] || fail "sink-not-regular-file:$SINK_FILE" 14

timed() { # $1=sink-target(file|null) ; $2..=env+cmd
  local tgt="$1"; shift; local sink s e secs sha rc
  if [ "$tgt" = null ]; then sink=/dev/null; else sink="$SINK_FILE"; fi
  s=$(date +%s.%N); taskset -c "$MASK" env "$@" >"$sink" 2>>"$ARTDIR/run.stderr"; rc=$?; e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  if [ "$tgt" = null ]; then sha="(null)"; else sha=$(sha256sum "$sink" | cut -d' ' -f1); fi
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  echo "$secs $sha"
}

# ---- 8. interleaved best-of-N (drop warmup iter 0) --------------------------
NC=${#LABELS[@]}
declare -a TIMES; for ((k=0;k<NC;k++)); do TIMES[$k]=""; done
DIVERGED=0
for ((i=0;i<=N;i++)); do
  for ((k=0;k<NC;k++)); do
    # shellcheck disable=SC2086
    read sec sha < <(timed "${SINKTO[$k]}" ${ENVS[$k]} ${CMDS[$k]})
    if [ "$i" -ge 1 ]; then
      TIMES[$k]="${TIMES[$k]} $sec"
      if [ "${SHACK[$k]}" = 1 ] && [ "$sha" != "$REF_SHA" ]; then
        echo "!! SHA DIVERGENCE ${LABELS[$k]} i=$i sha=$sha"; DIVERGED=1
      fi
    fi
  done
done
rm -f "$SINK_FILE"
[ "$DIVERGED" -eq 0 ] || fail "sha-mismatch on a byte-exact contender — run VOID (Rule 4)" 11

stats() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
  { v[NR]=$1 } END { n=NR; if(n==0){print "0 0 0"; exit}
    min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
    printf "%.4f %.4f %.1f", min, mid, (min>0)?(max-min)/min*100:0 }'; }
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

echo ""
echo "================ RESIDUAL SUMMARY (mode=$MODE T=$T N=$N) ================"
declare -a MINS MEDS
for ((k=0;k<NC;k++)); do
  read mn md sp < <(stats "${TIMES[$k]}")
  MINS[$k]="$mn"; MEDS[$k]="$md"
  printf "%-16s min=%.4fs (%4s MB/s) med=%.4f spread=%s%% sha=%s\n" \
    "${LABELS[$k]}" "$mn" "$(mbps "$mn")" "$md" "$sp" \
    "$([ "${SHACK[$k]}" = 1 ] && echo OK || echo NOT-CHK)"
done

# Helper: ms delta + ratio between two labelled contenders.
idx() { local want="$1" k; for ((k=0;k<NC;k++)); do [ "${LABELS[$k]}" = "$want" ] && { echo "$k"; return; }; done; echo -1; }
delta_ms() { awk -v a="${MINS[$1]}" -v b="${MINS[$2]}" 'BEGIN{printf "%.0f", (a-b)*1000}'; }

echo "----------------------------------------------------------------"
if [ "$MODE" = output-floor ]; then
  gp=$(idx gz-prod); gs=$(idx gz-skipwritev); rf=$(idx rg-file); rn=$(idx rg-null)
  echo "gzippy serial-output share = gz-prod - gz-skipwritev = $(delta_ms "$gp" "$gs") ms"
  echo "rapidgzip output exposure  = rg-file - rg-null       = $(delta_ms "$rf" "$rn") ms"
  echo "gz-prod vs rg-file ratio   = $(awk -v g="${MINS[$gp]}" -v r="${MINS[$rf]}" 'BEGIN{printf "%.3f", r/g}')"
  echo "gz-skipwritev vs rg-null   = $(awk -v g="${MINS[$gs]}" -v r="${MINS[$rn]}" 'BEGIN{printf "%.3f", r/g}')  (both output-removed)"
  echo "gz-skipwritev vs rg-file   = $(awk -v g="${MINS[$gs]}" -v r="${MINS[$rf]}" 'BEGIN{printf "%.3f", r/g}')"
elif [ "$MODE" = marker-perturb ]; then
  go=$(idx gz-off); gp=$(idx "gz-spin$SLOW_PCT"); gl=$(idx "gz-sleep$SLOW_PCT"); rf=$(idx rg-file)
  echo "marker slow-inject +$SLOW_PCT% SPIN  : +$(delta_ms "$gp" "$go") ms over gz-off (slope)"
  echo "marker slow-inject +$SLOW_PCT% SLEEP : +$(delta_ms "$gl" "$go") ms over gz-off (freq-neutral control)"
  echo "gz-off vs rg-file ratio = $(awk -v g="${MINS[$go]}" -v r="${MINS[$rf]}" 'BEGIN{printf "%.3f", r/g}')"
fi
echo "================ END RESIDUAL SUMMARY ================"
echo "RESIDUAL_GUEST_DONE"
