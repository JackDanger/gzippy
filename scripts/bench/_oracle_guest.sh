#!/usr/bin/env bash
# _oracle_guest.sh — guest-side runner for oracle.sh (P5). The oracle/perturbation
# sibling of _parity_guest.sh.
#
# Unlike _parity_guest.sh (which ABORTS on any seeding/oracle env because it
# measures PRODUCTION), this runner DELIBERATELY sets oracle/perturbation knobs —
# but it stays honest:
#   - it LABELS every non-production run loudly (NOT-PRODUCTION / output-may-be-garbage);
#   - sha-verify is still ENFORCED for kinds that must stay byte-exact (ceiling/
#     same-sink); kinds that knowingly produce garbage (clean-only seeded, sleep
#     projections) are marked SHA-NOT-CHECKED so a garbage run can never masquerade
#     as a parity number;
#   - it reuses the SAME host-lock readback, regular-file sink, interleaved best-of-N.
#
# Inputs (env, from oracle.sh): the _parity_guest.sh set PLUS:
#   KIND          clean-only|engine-isolation|same-sink|ceiling|perturb
#   SLOW_KNOB     e.g. GZIPPY_SLOW_DECODE   (for KIND=perturb)
#   SLOW_PCT      e.g. 50                    (for KIND=perturb)
set -u

fail() { echo "ORACLE_FAIL=$1"; echo "ORACLE_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${T:?}"; : "${N:?}"; : "${MASK:?}"; : "${KIND:?}"
RG="${RG:-rapidgzip}"; RG_TRACE="${RG_TRACE:-}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-oracle-art}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
SLOW_KNOB="${SLOW_KNOB:-}"; SLOW_PCT="${SLOW_PCT:-}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (oracle.sh --build first)" 5
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7

REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$REF_SHA" = "$CORPUS_RAW_SHA256" ] || fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256" 7
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# Production-path assert (the engine must still be ParallelSM for the oracle to mean anything).
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in *ParallelSM*) ;; *) fail "routing not ParallelSM: ${DBG:-none}" 9;; esac

# Host-lock readback.
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
[ "$ACT_GOV" = "$GOV" ] || echo "## WARN: governor=$ACT_GOV (expected $GOV) — host not frozen."
[ "$ACT_TURBO" = "$NO_TURBO" ] || echo "## WARN: no_turbo=$ACT_TURBO (expected $NO_TURBO)."

RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"; fi

# ---- per-KIND knob set + verification policy --------------------------------
# GZ_ENV  : extra env prepended to the gzippy command.
# CHECK_SHA: 1 = byte-exact required (abort on mismatch); 0 = garbage-by-design.
# WITH_RG : 1 = also run rapidgzip interleaved (for a relative number).
GZ_ENV=""; CHECK_SHA=1; WITH_RG=1; LABEL="$KIND"; PRODUCTION=0
case "$KIND" in
  ceiling)
    # decode-removal FLOOR via bypass replay; preserves markers ⇒ stays byte-exact.
    # The capture pass must have run first (oracle.sh does it); CAP is the file.
    : "${CAP:?ceiling needs CAP=<bypass-capture-file>}"
    GZ_ENV="GZIPPY_BYPASS_DECODE=$CAP"; CHECK_SHA=1
    LABEL="ceiling(decode-removed FLOOR; NOT-PRODUCTION but byte-exact)";;
  clean-only)
    # seeded clean-engine ceiling — MASKS the binder, garbage relative to production.
    GZ_ENV="GZIPPY_SEED_WINDOWS=1"; CHECK_SHA=0
    LABEL="clean-only(SEEDED; masks-binder; SHA-NOT-CHECKED; NOT-PRODUCTION)";;
  engine-isolation)
    # ISA-L engine oracle — C-FFI engine, off the production decode graph.
    GZ_ENV="GZIPPY_ISAL_ENGINE_ORACLE=1"; CHECK_SHA=1
    LABEL="engine-isolation(ISA-L engine; NOT-PRODUCTION; byte-exact)";;
  same-sink)
    # production knobs, same regular-file sink as rapidgzip — a FLOOR/control run.
    GZ_ENV=""; CHECK_SHA=1; PRODUCTION=1
    LABEL="same-sink(production knobs; byte-exact control)";;
  perturb)
    : "${SLOW_KNOB:?perturb needs SLOW_KNOB=GZIPPY_SLOW_*}"; : "${SLOW_PCT:?perturb needs SLOW_PCT=N}"
    GZ_ENV="GZIPPY_SLOW_MODE=$SLOW_PCT $SLOW_KNOB=1 GZIPPY_SLOW_HITS=1"; CHECK_SHA=1
    LABEL="perturb($SLOW_KNOB +$SLOW_PCT%; byte-exact; slope NOT a ceiling)";;
  *) fail "unknown KIND=$KIND" 2;;
esac

echo "================ ORACLE PROVENANCE ================"
echo "kind=$KIND  $LABEL"
echo "guest_src=$GUEST_SRC head=$(git rev-parse --short HEAD 2>/dev/null || echo NA) T=$T N=$N mask=$MASK"
echo "binary=$GZIPPY_BIN corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
echo "gz_env='$GZ_ENV'  check_sha=$CHECK_SHA production=$PRODUCTION"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO"
[ "$PRODUCTION" = 1 ] || echo "## NOTE: this is an ORACLE/PERTURBATION run — NOT a production parity number."
echo "=================================================="

SINK_GZ="$ARTDIR/sink_gz.bin"; SINK_RG="$ARTDIR/sink_rg.bin"
timed() { local sink="$1"; shift; local s e secs sha rc
  s=$(date +%s.%N); set +e; taskset -c "$MASK" "$@" >"$sink" 2>>"$ARTDIR/run.stderr"; rc=$?; set -e 2>/dev/null||true; e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}'); sha=$(sha256sum "$sink" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2; echo "$secs $sha"; }

GZT=""; RGT=""; DIVERGED=0
for ((i=0;i<=N;i++)); do
  # shellcheck disable=SC2086
  read gsec gsha < <(timed "$SINK_GZ" env GZIPPY_FORCE_PARALLEL_SM=1 $GZ_ENV "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS")
  if [ "$WITH_RG" = 1 ] && [ -n "$RG_CMD" ]; then
    read rsec _ < <(timed "$SINK_RG" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  else rsec=""; fi
  [ "$i" -eq 0 ] && continue
  GZT="$GZT $gsec"; [ -n "$rsec" ] && RGT="$RGT $rsec"
  if [ "$CHECK_SHA" = 1 ] && [ "$gsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE gzippy i=$i sha=$gsha"; DIVERGED=1; fi
done
rm -f "$SINK_GZ" "$SINK_RG"

if [ "$CHECK_SHA" = 1 ] && [ "$DIVERGED" -ne 0 ]; then
  fail "sha-mismatch on a byte-exact KIND ($KIND) — oracle VOID (Rule 4)" 11
fi

stats() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
  { v[NR]=$1 } END { n=NR; if(n==0){print "0 0 0"; exit}
    min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
    printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'; }
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

read gmin gmed gsp < <(stats "$GZT")
echo ""
echo "================ ORACLE SUMMARY (kind=$KIND T=$T) ================"
printf "gzippy[%s]  min=%.4fs (%s MB/s) med=%.4f spread=%s%%  sha=%s\n" \
  "$KIND" "$gmin" "$(mbps "$gmin")" "$gmed" "$gsp" \
  "$([ "$CHECK_SHA" = 1 ] && echo OK || echo NOT-CHECKED)"
if [ -n "$RGT" ]; then
  read rmin rmed rsp < <(stats "$RGT")
  RATIO="$(awk -v g="$gmin" -v r="$rmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
  printf "rapidgzip    min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsp"
  printf "gzippy[%s]=%sms  rg=%sms  ratio=%s\n" "$KIND" \
    "$(awk -v t="$gmin" 'BEGIN{printf "%.0f", t*1000}')" \
    "$(awk -v t="$rmin" 'BEGIN{printf "%.0f", t*1000}')" "$RATIO"
fi
[ "$PRODUCTION" = 1 ] || echo "## REMINDER: oracle/perturbation number — do NOT bank as parity. See plans/KNOBS.md + Measurement PROCESS."
echo "================ END ORACLE SUMMARY ================"
echo "ORACLE_GUEST_DONE"
