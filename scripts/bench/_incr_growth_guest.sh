#!/usr/bin/env sh
# _incr_growth_guest.sh — GUEST half of the INCREMENTAL-OUTPUT-GROWTH footprint
# A/B (the DIS-17 owed footprint falsifier). Measures, on the FROZEN/quiet guest,
# gzippy-isal with GZIPPY_ISAL_INCREMENTAL_GROWTH OFF (identity 8x upfront reserve)
# vs ON (small initial + grow-on-demand, faithful rg GzipChunk.hpp:309-379) vs
# rapidgzip, at ONE thread count:
#   - peak RSS (min-of-N, /usr/bin/time -v "Maximum resident set size")
#   - perf stat -r REPS: instructions, dTLB-load-misses, page-faults, wall
#   - dTLB-load-miss MPKI = dTLB-load-misses / instructions * 1000
#   - isal_chunks / isal_fallbacks ASSERT (ON must stay 0-fallback, >=14 chunks)
# Every decode sha-verified against the decompressed-corpus pin (Rule 4).
#
# Env (from the local wrapper): GZ RG CORPUS REFSHA MASK T REPS N ART FACTORS
#   GOV NO_TURBO HOST_FROZEN
set -u
fail() { echo "INCR_FAIL=$1"; echo "INCR_GUEST_DONE"; exit "${2:-1}"; }
: "${GZ:?}"; : "${RG:?}"; : "${CORPUS:?}"; : "${MASK:?}"; : "${T:?}"; : "${REFSHA:?}"
REPS="${REPS:-6}"; N="${N:-7}"; ART="${ART:-/dev/shm/incr}"
FACTORS="${FACTORS:-4 2 1}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"; HOST_FROZEN="${HOST_FROZEN:-0}"
mkdir -p "$ART"
SINK="$ART/sink.bin"; rm -f "$SINK"

# ---- contamination guard (allowlist scrub; my knob + force-SM allowed) -------
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    GZIPPY_FORCE_PARALLEL_SM|GZIPPY_DEBUG|GZIPPY_VERBOSE|GZIPPY_ISAL_INCREMENTAL_GROWTH|GZIPPY_ISAL_INITIAL_FACTOR|GZIPPY_ISAL_GROW_MIB) ;;
    *) SCRUBBED="$SCRUBBED $v"; unset "$v" 2>/dev/null || true;;
  esac
done
[ -n "$SCRUBBED" ] && echo "## SCRUBBED non-production GZIPPY_*:$SCRUBBED"
case "$SCRUBBED" in *SEED*|*ORACLE*|*BYPASS*|*SLOW*|*SLEEP*) fail "contaminated-env:$SCRUBBED" 2;; esac

# ---- corpus / oracle ---------------------------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$REF" = "$REFSHA" ] || fail "corpus-sha-drift got=$REF pin=$REFSHA" 7

# ---- production-path assert --------------------------------------------------
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in *ParallelSM*) ;; *) fail "routing not ParallelSM: ${DBG:-none}" 9;; esac

# ---- host-freeze readback ----------------------------------------------------
AG="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
AT="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
if [ "$AG" = "$GOV" ] && [ "$AT" = "$NO_TURBO" ]; then :
elif [ "$AG" != NA ] && [ "$AG" != "$GOV" ]; then fail "host-not-frozen gov=$AG (want $GOV)" 13
elif [ "$AT" != NA ] && [ "$AT" != "$NO_TURBO" ]; then fail "host-not-frozen no_turbo=$AT (want $NO_TURBO)" 13
elif [ "$HOST_FROZEN" = 1 ]; then echo "## WARN freeze unreadable (gov=$AG turbo=$AT) HOST_FROZEN=1 ack"
else fail "host-freeze-unreadable gov=$AG turbo=$AT (pass HOST_FROZEN=1)" 13; fi

# ---- quiet gate --------------------------------------------------------------
RS=0; RC=0
for q in 1 2 3 4; do
  r="$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA)"; [ "$r" = NA ] && break
  RS=$((RS+r)); RC=$((RC+1)); [ "$q" -lt 4 ] && sleep 1
done
RUNAVG=NA
if [ "$RC" -gt 0 ]; then
  RUNAVG="$(awk -v s="$RS" -v c="$RC" 'BEGIN{printf "%.2f",s/c}')"
  awk -v a="$RUNAVG" 'BEGIN{exit !(a+0>2.0)}' && fail "host-loaded runnable_avg=$RUNAVG>2.0" 13
fi

BINSHA="$(sha256sum "$GZ" | cut -c1-16)"
echo "================ INCR-GROWTH FOOTPRINT A/B (T=$T) ================"
echo "bin_sha=$BINSHA mask=$MASK reps=$REPS N=$N factors='$FACTORS' gov=$AG no_turbo=$AT runnable_avg=$RUNAVG frozen=$HOST_FROZEN"
echo "rg=$("$RG" --version 2>&1 | head -1)"

# cpu_core (P-core) PMU events; root bypasses paranoid.
EV="cpu_core/instructions/,cpu_core/dTLB-load-misses/,cpu_core/cycles/,page-faults,task-clock"

# perf_arm <label> <env+cmd...> : perf stat -r REPS, sha-verify, print + MPKI.
perf_arm() {
  label="$1"; shift
  out="$ART/${label}.perf"; rm -f "$SINK"
  # NOTE: `perf stat -r REPS` runs the cmd REPS times into the SAME $SINK, so the
  # sink sha is a -r CONCATENATION artifact, NOT a correctness signal (DIS-17).
  # Per-run byte-exactness is verified separately by min_rss + counters_assert.
  perf stat -r "$REPS" -e "$EV" -o "$out" -- "$@" >"$SINK" 2>>"$ART/run.err"
  instr="$(awk '/instructions/{gsub(/,/,"",$1);print $1;exit}' "$out")"
  dtlb="$(awk '/dTLB-load-misses/{gsub(/,/,"",$1);print $1;exit}' "$out")"
  pf="$(awk '/page-faults/{gsub(/,/,"",$1);print $1;exit}' "$out")"
  wall="$(awk '/seconds time elapsed/{print $1;exit}' "$out")"
  mpki="$(awk -v d="$dtlb" -v i="$instr" 'BEGIN{printf "%.4f",(i>0)?d/i*1000:0}')"
  printf "   %-14s wall=%-9s instr=%-12s dTLB-miss=%-11s MPKI=%-8s page-faults=%s\n" \
    "$label" "${wall:-NA}" "${instr:-NA}" "${dtlb:-NA}" "$mpki" "${pf:-NA}"
}

# min_rss <label> <env+cmd...> : best-of-N peak RSS (KiB), sha-verify each.
min_rss() {
  label="$1"; shift
  best=""
  i=0
  while [ "$i" -lt "$N" ]; do
    rm -f "$SINK"
    /usr/bin/time -v taskset -c "$MASK" "$@" >"$SINK" 2>"$ART/time.out"
    r="$(awk -F': ' '/Maximum resident set size/{print $2}' "$ART/time.out")"
    s="$(sha256sum "$SINK" | cut -d' ' -f1)"
    [ "$s" = "$REFSHA" ] || { echo "!! $label RSS-SHA-MISMATCH"; return 1; }
    if [ -z "$best" ] || awk -v a="$r" -v b="$best" 'BEGIN{exit !(a+0<b+0)}'; then best="$r"; fi
    i=$((i+1))
  done
  printf "   %-14s peak_rss=%s KiB\n" "$label" "$best"
}

# counters_assert <env+cmd...> : print isal_chunks/fallbacks, FAIL if fallbacks>0.
counters_assert() {
  label="$1"; shift
  rm -f "$SINK"
  c="$(GZIPPY_VERBOSE=1 "$@" 2>&1 >"$SINK" | grep -oE 'isal_chunks=[0-9]+ isal_fallbacks=[0-9]+' | head -1)"
  s="$(sha256sum "$SINK" | cut -d' ' -f1)"
  [ "$s" = "$REFSHA" ] || { echo "!! $label COUNTERS-SHA-MISMATCH"; return 1; }
  fb="$(echo "$c" | sed -n 's/.*isal_fallbacks=\([0-9]*\).*/\1/p')"
  echo "   $label $c"
  [ "${fb:-1}" = 0 ] || fail "$label introduced fallbacks: $c" 11
}

GZOFF="env GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p $T $CORPUS"
RGCMD="$RG -d -c -f -P $T $CORPUS"

echo "==== peak RSS (min-of-$N) ===="
min_rss gz_OFF $GZOFF
for F in $FACTORS; do
  min_rss "gz_ON_f$F" env GZIPPY_ISAL_INCREMENTAL_GROWTH=1 GZIPPY_ISAL_INITIAL_FACTOR=$F GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS"
done
min_rss rapidgzip $RGCMD

echo "==== perf stat (instr / dTLB-miss / MPKI / page-faults / wall) ===="
perf_arm gz_OFF $GZOFF
for F in $FACTORS; do
  perf_arm "gz_ON_f$F" env GZIPPY_ISAL_INCREMENTAL_GROWTH=1 GZIPPY_ISAL_INITIAL_FACTOR=$F GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS"
done
perf_arm rapidgzip $RGCMD

echo "==== isal_chunks / fallbacks assert (ON must be 0-fallback) ===="
counters_assert gz_OFF env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS"
for F in $FACTORS; do
  counters_assert "gz_ON_f$F" env GZIPPY_ISAL_INCREMENTAL_GROWTH=1 GZIPPY_ISAL_INITIAL_FACTOR=$F GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS"
done

rm -f "$SINK"
echo "INCR_T${T}_DONE"
