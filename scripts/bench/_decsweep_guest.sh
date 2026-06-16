#!/usr/bin/env bash
# _decsweep_guest.sh — GUEST-SIDE wall+P capture for the STEP-13 decoded-size
# adaptive chunk policy.
#
# Identical discipline to _chunksweep_guest.sh (freeze readback, quiet gate, env
# scrub, regular-file sink, sha-verify every run, pin_mask, WALL + task-clock ->
# P=avg busy CPUs). The ONLY difference: instead of sweeping the FIXED compressed
# spacing (GZIPPY_CHUNK_KIB), it sweeps the DECODED-size target
# (GZIPPY_TARGET_DECODED_KIB) consumed by the adaptive policy at
# single_member.rs (compressed = target_decoded / ratio, ratio = ISIZE/comp_len).
#
# Arms:
#   k0   : BASE — pure compile-time default (NO chunk env) = fixed 4 MiB.
#   aa   : self-test — GZIPPY_CHUNK_KIB=4096 (explicit 4 MiB). MUST ratio ~1.0 vs
#          k0 (proves the rig sound AND that explicit-4096 == compile default).
#   d<K> : adaptive — GZIPPY_TARGET_DECODED_KIB=<K> for K in $DKIBS.
#   rg   : rapidgzip comparator.
#
# Inputs (env): GZIPPY_BIN RG CORPUS T N ARTDIR RUNID GOV NO_TURBO HOST_FROZEN
#   ALLOW_LOAD CORPUS_RAW_SHA256 DKIBS
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lib_decide_guest.sh"

: "${GZIPPY_BIN:?}"; : "${RG:?}"; : "${CORPUS:?}"; : "${ARTDIR:?}"; : "${RUNID:?}"
T="${T:-4}"; N="${N:-9}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
CORPUS_RAW_SHA256="${CORPUS_RAW_SHA256:-}"
DKIBS="${DKIBS:-4096 5120 6144 8192}"

OUT="$ARTDIR/$RUNID"; mkdir -p "$OUT"; export ARTDIR
: > "$ARTDIR/run.stderr"

scrub_gzippy_env "GZIPPY_CHUNK_KIB GZIPPY_TARGET_DECODED_KIB GZIPPY_FORCE_PARALLEL_SM"
freeze_readback
quiet_gate

[ -x "$GZIPPY_BIN" ] || decide_fail "no-bin:$GZIPPY_BIN" 5
RG_CMD=""
if [ -x "$RG" ]; then RG_CMD="$RG"
elif command -v "$RG" >/dev/null 2>&1; then RG_CMD="$(command -v "$RG")"
else decide_fail "no-rg:$RG" 12; fi
command -v perf >/dev/null 2>&1 || decide_fail "no-perf" 15
[ -f "$CORPUS" ] || decide_fail "no-corpus:$CORPUS" 7
BYTES="$(gzip -dc "$CORPUS" | wc -c)"
REFSHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$BYTES" -gt 0 ] || decide_fail "corpus-zero" 7
if [ -n "$CORPUS_RAW_SHA256" ] && [ "$REFSHA" != "$CORPUS_RAW_SHA256" ]; then
  decide_fail "corpus-sha-drift got=$REFSHA pin=$CORPUS_RAW_SHA256" 7
fi

MASK="$(pin_mask "$T")"; [ -n "$MASK" ] || decide_fail "bad-T:$T" 8
SINK="$OUT/sink.bin"; assert_regular_sink "$SINK"

# perf_timed <mask> <cmd...> -> echoes "wall_s taskclock_ms sha"
perf_timed() {
  local mask="$1"; shift
  local pstat sha rc wall tc
  pstat="$(mktemp /tmp/.ds_perf_XXXXXX)"
  assert_regular_sink "$SINK"
  set +e
  perf stat -e task-clock -o "$pstat" -- taskset -c "$mask" "$@" >"$SINK" 2>>"$ARTDIR/run.stderr"
  rc=$?
  set -e 2>/dev/null || true
  sha="$(sha256sum "$SINK" | cut -d' ' -f1)"
  tc="$(awk '/task-clock/{gsub(/,/,"",$1); print $1; exit}' "$pstat")"
  wall="$(awk '/seconds time elapsed/{print $1; exit}' "$pstat")"
  rm -f "$pstat"
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  [ -n "$tc" ] && [ -n "$wall" ] || { echo "## WARN perf no counts: $*" >&2; tc=0; wall=0; }
  echo "$wall $tc $sha"
}

# arm runners. base/self-test/adaptive all force the SM engine (Gate 4).
run_base() { perf_timed "$MASK" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS"; }
run_aa()   { perf_timed "$MASK" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_CHUNK_KIB=4096 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS"; }
run_d()    { perf_timed "$MASK" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TARGET_DECODED_KIB="$1" "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS"; }

: > "$OUT/samples_k0.txt"; : > "$OUT/samples_aa.txt"; : > "$OUT/samples_rg.txt"
for dk in $DKIBS; do : > "$OUT/samples_d$dk.txt"; done
DIVERGED=0

# one-shot chunk-count attestation per adaptive target (GZIPPY_DEBUG line),
# so the report can VERIFY the ISIZE-ratio read computed the expected spacing.
{
  echo "## chunk-count attestation (GZIPPY_DEBUG, T=$T)"
  GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$OUT/attest.txt" || true
  for dk in $DKIBS; do
    echo "## target_decoded_kib=$dk:"
    GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TARGET_DECODED_KIB="$dk" "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$OUT/attest.txt" || true
  done
} 2>&1
grep -h 'parallel_sm' "$OUT/attest.txt" 2>/dev/null | sort -u > "$OUT/chunkcounts.txt" || true

echo "## decsweep: T=$T mask=$MASK N=$N dkibs='$DKIBS' bytes=$BYTES ref=$REFSHA freeze=$FREEZE_STATE quiet=$QUIET_STATE runnable=${RUN_AVG:-NA}"
for ((i=0;i<=N;i++)); do
  read -r w tc sha < <(run_base)
  [ "$i" -gt 0 ] && { echo "$w $tc $sha" >> "$OUT/samples_k0.txt"; [ "$sha" != "$REFSHA" ] && { echo "!! SHA DIV base"; DIVERGED=1; }; }
  for dk in $DKIBS; do
    read -r w tc sha < <(run_d "$dk")
    [ "$i" -gt 0 ] && { echo "$w $tc $sha" >> "$OUT/samples_d$dk.txt"; [ "$sha" != "$REFSHA" ] && { echo "!! SHA DIV d$dk sha=$sha"; DIVERGED=1; }; }
  done
  read -r w tc sha < <(run_aa)
  [ "$i" -gt 0 ] && { echo "$w $tc $sha" >> "$OUT/samples_aa.txt"; [ "$sha" != "$REFSHA" ] && DIVERGED=1; }
  read -r w tc sha < <(perf_timed "$MASK" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  [ "$i" -gt 0 ] && { echo "$w $tc $sha" >> "$OUT/samples_rg.txt"; [ "$sha" != "$REFSHA" ] && { echo "!! SHA DIV rg sha=$sha"; DIVERGED=1; }; }
done
rm -f "$SINK"
[ "$DIVERGED" -eq 0 ] || decide_fail "sha-mismatch (wrong-bytes — VOID)" 11

{
  echo "BYTES=$BYTES"; echo "REFSHA=$REFSHA"; echo "T=$T"; echo "MASK=$MASK"
  echo "DKIBS=$DKIBS"; echo "N=$N"
  echo "FREEZE_STATE=$FREEZE_STATE"; echo "QUIET_STATE=$QUIET_STATE"
  echo "RUN_AVG=${RUN_AVG:-NA}"; echo "ACT_GOV=${ACT_GOV:-NA}"; echo "ACT_TURBO=${ACT_TURBO:-NA}"
  echo "GZIPPY_SHA=$(sha256sum "$GZIPPY_BIN" | cut -d' ' -f1)"
  echo "RG_SHA=$(sha256sum "$RG_CMD" | cut -d' ' -f1)"
} > "$OUT/meta.env"

echo "=== chunk-count attestation ==="; cat "$OUT/chunkcounts.txt" 2>/dev/null || true
echo "DECSWEEP_ARTIFACTS=$OUT"
echo "DECSWEEP_GUEST_DONE"
