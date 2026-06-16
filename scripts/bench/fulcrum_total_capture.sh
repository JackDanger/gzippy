#!/usr/bin/env bash
# fulcrum_total_capture.sh — capture a trace + matching counter sidecar for
# `scripts/fulcrum total` (the Rust analyzer), WITHOUT seeding (production
# routing preserved).
#
# This is the companion capture for the trustworthy whole-system instrument. It
# produces TWO files per label so the analyzer can verify window-absent routing:
#   ARTDIR/trace_<label>.json     (GZIPPY_TIMELINE — the Chrome-trace timeline)
#   ARTDIR/verbose_<label>.txt    (GZIPPY_VERBOSE stderr — the counter sidecar)
# `fulcrum total` auto-detects verbose_<label>.txt next to trace_<label>.json.
#
# CRITICAL: this script does NOT set GZIPPY_SEED_WINDOWS and does NOT set the
# ISA-L engine oracle. Seeding routes to the clean engine and MASKS the binder;
# the analyzer's routing guard will REFUSE any run whose sidecar shows
# window_seeded>0. Keep it that way — capture production, then read the verdict.
#
# Usage (on the guest, after the host freq-lock gate PASSES):
#   bash scripts/bench/fulcrum_total_capture.sh LABEL=gzippy_T8 T=8 \
#        CORPUS=/root/gzippy/benchmark_data/silesia-large.gz ARTDIR=/root/ft-art
#   # then, on any host with the trace pulled down:
#   scripts/fulcrum total /root/ft-art/trace_gzippy_T8.json
#
# To capture an ORACLE/SEEDED run ON PURPOSE (for a ceiling, never as production),
# set SEED=1 — the sidecar will record it and the analyzer will correctly REFUSE
# to call it production. This makes the seeded-vs-unseeded distinction auditable.
set -u

REPO="${REPO:-/root/gzippy}"
GZIPPY="${GZIPPY:-$REPO/target/release/gzippy}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-fulcrum-total}"
LABEL="${LABEL:-gzippy_T8}"
T="${T:-8}"
SEED="${SEED:-0}"

for a in "$@"; do
  case "$a" in
    LABEL=*) LABEL="${a#*=}";;
    T=*) T="${a#*=}";;
    CORPUS=*) CORPUS="${a#*=}";;
    ARTDIR=*) ARTDIR="${a#*=}";;
    GZIPPY=*) GZIPPY="${a#*=}";;
    SEED=*) SEED="${a#*=}";;
  esac
done

mkdir -p "$ARTDIR"
[ -x "$GZIPPY" ] || { echo "FAILURE=no-gzippy-binary $GZIPPY"; exit 5; }
[ -f "$CORPUS" ] || { echo "FAILURE=no-corpus $CORPUS"; exit 7; }

pin_mask() {
  case "$1" in
    1) echo "0";; 4) echo "0,2,4,6";;
    8) echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *) echo "";;
  esac
}
MASK="$(pin_mask "$T")"
[ -n "$MASK" ] || { echo "FAILURE=bad-T $T (use 1,4,8,16)"; exit 8; }

TRACE="$ARTDIR/trace_${LABEL}.json"
VERB="$ARTDIR/verbose_${LABEL}.txt"

# Routing assert (must be ParallelSM, else the trace is meaningless).
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
case "$DBG" in *ParallelSM*) ;; *) echo "FAILURE=routing $DBG"; exit 9;; esac

REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"

SEED_ENV=""
if [ "$SEED" = "1" ]; then
  echo "## WARNING: SEED=1 — capturing a SEEDED (clean-engine ceiling) run."
  echo "## The analyzer will (correctly) REFUSE to call this production."
  SEED_ENV="GZIPPY_SEED_WINDOWS=1"
fi

echo "## capture label=$LABEL T=$T mask=$MASK seed=$SEED"
# shellcheck disable=SC2086
env $SEED_ENV GZIPPY_TIMELINE="$TRACE" GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  taskset -c "$MASK" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$ARTDIR/out_${LABEL}.bin" 2>"$VERB"
OUT_SHA="$(sha256sum "$ARTDIR/out_${LABEL}.bin" | cut -d' ' -f1)"
rm -f "$ARTDIR/out_${LABEL}.bin"

# Trust gates: trace non-empty, output byte-exact.
if [ ! -s "$TRACE" ]; then
  echo "FAILURE=empty-trace $TRACE (the empty-output instrument failure class)"; exit 10
fi
if [ "$OUT_SHA" != "$REF_SHA" ]; then
  echo "FAILURE=sha-mismatch out=$OUT_SHA ref=$REF_SHA (a wrong-bytes capture is void)"; exit 11
fi

echo "OK trace=$TRACE verbose=$VERB out_sha=$OUT_SHA (==ref)"
echo "## counter sidecar (routing-relevant lines):"
grep -E 'window_seeded|flip_to_clean|finished_no_flip|isal_oracle' "$VERB" || true
echo ""
echo "## analyze with:  scripts/fulcrum total $TRACE"
