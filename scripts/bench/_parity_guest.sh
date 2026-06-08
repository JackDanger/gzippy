#!/usr/bin/env bash
# _parity_guest.sh — the GUEST-SIDE half of parity.sh (P0). Not run directly by
# the owner; parity.sh ships it to $GUEST_SRC and executes it over the double-hop.
#
# It is self-contained and correctness-critical. It enforces, MECHANICALLY:
#   - build in $GUEST_SRC with the pinned RUSTFLAGS + selected feature (Rule 6);
#   - production-path assertion (GZIPPY_DEBUG -> path=ParallelSM) (Rule 6);
#   - host-lock READBACK warn (governor/no_turbo/taskset) (Rule 6);
#   - REGULAR-FILE sink, NEVER a pipe (a pipe backpressure-inflated writev into a
#     phantom — the contamination this whole wrapper exists to prevent);
#   - WINDOW-ABSENT-PRESERVING: it sets NO GZIPPY_SEED_* and NO engine oracle, and
#     it ABORTS if any seeding env leaked in (a seeded run routes to the clean
#     engine and masks the binder — it is NOT production);
#   - interleaved best-of-N gzippy vs rapidgzip (Rule 6);
#   - MANDATORY sha-verify of EVERY run against the decompressed-corpus pin, with
#     a loud ABORT on any mismatch (Rule 4 — a fast wrong-bytes win is a loss).
#
# Inputs (env, all required, passed by parity.sh from guest.env + flags):
#   GUEST_SRC GZIPPY_BIN CORPUS CORPUS_RAW_SHA256 RG RG_TRACE
#   FEATURE T N MASK GOV NO_TURBO RUSTFLAGS_PIN DO_BUILD DO_FULCRUM ARTDIR
#   CARGO_LOCK (path to scripts/cargo-lock.sh, relative to GUEST_SRC)
set -u

fail() { echo "PARITY_FAIL=$1"; echo "PARITY_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${FEATURE:?}"; : "${T:?}"; : "${N:?}"; : "${MASK:?}"
RG="${RG:-rapidgzip}"
RG_TRACE="${RG_TRACE:-}"
RUSTFLAGS_PIN="${RUSTFLAGS_PIN:--C target-cpu=native}"
DO_BUILD="${DO_BUILD:-0}"
DO_FULCRUM="${DO_FULCRUM:-0}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-parity-art}"
GOV="${GOV:-performance}"
NO_TURBO="${NO_TURBO:-1}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5

# ---- 1. CONTAMINATION GUARD: refuse any leaked seeding / oracle env ----------
# These would route off the production (window-absent marker bootstrap) path.
for bad in GZIPPY_SEED_WINDOWS GZIPPY_SEED_WINDOWS_CAPTURE GZIPPY_SEED_NO_WINDOWS \
           GZIPPY_SEED_NO_BOUNDARIES GZIPPY_ISAL_ENGINE_ORACLE \
           GZIPPY_CLEAN_WINDOW_ORACLE GZIPPY_BYPASS_DECODE GZIPPY_BYPASS_CAPTURE \
           GZIPPY_SLEEP_DECODE_NS GZIPPY_SLOW_MODE; do
  if [ -n "${!bad:-}" ]; then
    fail "contaminated-env:$bad=${!bad} (not production — would mask the binder)" 2
  fi
done

# ---- 2. build (under cargo-lock) --------------------------------------------
CARGO_LOCK="${CARGO_LOCK:-scripts/cargo-lock.sh}"
if [ "$DO_BUILD" = 1 ]; then
  echo "## build feature=$FEATURE RUSTFLAGS='$RUSTFLAGS_PIN' (cargo-lock serialized)"
  if [ -x "$CARGO_LOCK" ]; then BUILDER=(sh "$CARGO_LOCK"); else BUILDER=(); fi
  set +e
  RUSTFLAGS="$RUSTFLAGS_PIN" "${BUILDER[@]}" \
    cargo build --release --no-default-features --features "$FEATURE" \
    >"$ARTDIR/build.log" 2>&1
  brc=$?
  set -e 2>/dev/null || true
  if [ "$brc" -ne 0 ]; then
    grep -E 'error' "$ARTDIR/build.log" | head -30
    fail "build rc=$brc (see $ARTDIR/build.log)" 8
  fi
  grep -E 'Compiling gzippy|Finished' "$ARTDIR/build.log" | tail -3 || true
fi
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (run with --build)" 5

# ---- 3. corpus + correctness oracle -----------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
if [ "$REF_SHA" != "$CORPUS_RAW_SHA256" ]; then
  fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256 (corpus differs from banked)" 7
fi
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# ---- 4. production-path assertion (Rule 6) -----------------------------------
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in
  *ParallelSM*) ;;
  *) fail "routing not ParallelSM: ${DBG:-<no path= line>}" 9;;
esac

# ---- 5. host-lock READBACK (warn, don't change — Rule 6 frozen host) ---------
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
[ "$ACT_GOV" = "$GOV" ] || echo "## WARN: governor=$ACT_GOV (expected $GOV) — host not frozen; absolutes are noise."
[ "$ACT_TURBO" = "$NO_TURBO" ] || echo "## WARN: no_turbo=$ACT_TURBO (expected $NO_TURBO) — turbo not pinned; spin/sleep delta suspect."

# rapidgzip presence (the comparison target).
RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"
else fail "no-rapidgzip (not in PATH and RG_TRACE absent) — cannot measure parity" 12; fi

GZIPPY_SHA="$(git rev-parse HEAD 2>/dev/null || echo NA)"
echo "================ PARITY PROVENANCE ================"
echo "guest_src=$GUEST_SRC head=$GZIPPY_SHA feature=$FEATURE T=$T N=$N mask=$MASK"
echo "binary=$GZIPPY_BIN mtime=$(date -r "$GZIPPY_BIN" '+%F %T' 2>/dev/null || echo NA)"
echo "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
echo "rapidgzip=$("$RG_CMD" --version 2>&1 | head -1)"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO"
echo "=================================================="

# ---- 6. interleaved best-of-N, REGULAR-FILE sink, sha-verify EVERY run -------
# CRITICAL: output goes to a regular file on /dev/shm (NEVER a pipe). A pipe sink
# backpressure-inflated writev into a phantom — the exact contamination class this
# wrapper exists to kill.
SINK_GZ="$ARTDIR/sink_gzippy.bin"
SINK_RG="$ARTDIR/sink_rapidgzip.bin"

timed() { # <sink> <cmd...> -> echoes "secs sha"
  local sink="$1"; shift
  local s e secs sha rc
  s=$(date +%s.%N)
  set +e
  taskset -c "$MASK" "$@" >"$sink" 2>>"$ARTDIR/run.stderr"; rc=$?
  set -e 2>/dev/null || true
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$sink" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  echo "$secs $sha"
}

GZT=""; RGT=""; DIVERGED=0
echo "## interleave (N=$N, drop warmup iter0)"
for ((i=0;i<=N;i++)); do
  read gsec gsha < <(timed "$SINK_GZ" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS")
  read rsec rsha < <(timed "$SINK_RG" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  if [ "$i" -eq 0 ]; then continue; fi
  GZT="$GZT $gsec"; RGT="$RGT $rsec"
  if [ "$gsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE gzippy i=$i sha=$gsha"; DIVERGED=1; fi
  if [ "$rsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE rapidgzip i=$i sha=$rsha"; DIVERGED=1; fi
done
rm -f "$SINK_GZ" "$SINK_RG"

# ABORT on any wrong bytes (Rule 4) — the number is VOID.
if [ "$DIVERGED" -ne 0 ]; then
  fail "sha-mismatch (a wrong-bytes win is a loss — Rule 4); number VOID" 11
fi

stats() { # echoes "min med spreadpct"
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'
}
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

read gmin gmed gsp < <(stats "$GZT")
read rmin rmed rsp < <(stats "$RGT")

# RELATIVE signal (jitter-immune): ratio = rg_time / gzippy_time = gzippy_tput/rg_tput
RATIO="$(awk -v g="$gmin" -v r="$rmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
MARGIN="$(awk -v a="$gsp" -v b="$rsp" 'BEGIN{m=(a>b)?a:b; print m/100.0}')"
VERDICT="$(awk -v x="$RATIO" -v m="$MARGIN" 'BEGIN{d=x-1; if(d>m)print "WIN(gzippy)"; else if(d<-m)print "LOSS"; else print "TIE"}')"

echo ""
echo "================ PARITY SUMMARY (T=$T) ================"
printf "gzippy    min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$gmin" "$(mbps "$gmin")" "$gmed" "$gsp"
printf "rapidgzip min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsp"
echo "RELATIVE: gzippy is ${RATIO}x rapidgzip's throughput => $VERDICT  (TIE margin=${MARGIN})"
# The one canonical summary line the spec asks for:
printf "gzippy=%sms  rg=%sms  ratio=%s  sha=OK  verdict=%s\n" \
  "$(awk -v t="$gmin" 'BEGIN{printf "%.0f", t*1000}')" \
  "$(awk -v t="$rmin" 'BEGIN{printf "%.0f", t*1000}')" \
  "$RATIO" "$VERDICT"

# ---- 7. optional --fulcrum decompose ----------------------------------------
if [ "$DO_FULCRUM" = 1 ]; then
  echo ""
  echo "## --fulcrum: capturing a window-absent trace for fulcrum_total decompose"
  if [ -x scripts/bench/fulcrum_total_capture.sh ]; then
    bash scripts/bench/fulcrum_total_capture.sh \
      LABEL="parity_T${T}" T="$T" CORPUS="$CORPUS" ARTDIR="$ARTDIR" GZIPPY="$GZIPPY_BIN" \
      || echo "## WARN: fulcrum capture failed (non-fatal; parity number above stands)"
    echo "## analyze on any host:  python3 scripts/fulcrum_total.py $ARTDIR/trace_parity_T${T}.json"
  else
    echo "## WARN: scripts/bench/fulcrum_total_capture.sh absent — skipping decompose"
  fi
fi

echo "================ END PARITY SUMMARY ================"
echo "PARITY_GUEST_DONE"
