#!/usr/bin/env bash
# _silt4_capture_guest.sh — guest-side S2 discriminator capture.
#
# Gate-0 self-validates the binary+box, then captures GZIPPY_TIMELINE traces of
# gzippy-native on silesia-T4 (the residual) and monorepo-T4 (the tying control),
# plus rapidgzip --verbose chunk counts for the non-inert chunk-count cross-check.
# Reduction to effcores happens MAC-side via scripts/parallel_sm_tail_metric.py.
#
# Inputs (env): GZ, RG, CORPUS_DIR, REPS, OUT
set -u
GZ="${GZ:-/dev/shm/standing-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORPUS_DIR="${CORPUS_DIR:-/root}"
REPS="${REPS:-3}"
OUT="${OUT:-/dev/shm/silt4-art}"
LOG=/dev/shm/silt4.log
DONE=/dev/shm/silt4.DONE
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

fail() { echo "SILT4_FAIL=$*"; echo "FAIL $*" > "$DONE"; exit 2; }

echo "== SILT4 discriminator capture =="
echo "host: $(uname -srm)  load: $(cat /proc/loadavg)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
[ -x "$GZ" ] || fail "gz missing $GZ"
[ -x "$RG" ] || fail "rg missing $RG"

# ---- GATE-0a build-flavor (FFI-off native proof) ----
FERR=/tmp/silt4.flav
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$CORPUS_DIR/monorepo.gz" >/dev/null 2>"$FERR" || true
FLAVOR="$(sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p' "$FERR" | head -1)"
echo "build-flavor: '$FLAVOR' (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "flavor '$FLAVOR' != parallel-sm+pure"
echo "gz sha256: $(sha256sum "$GZ" | cut -c1-12)"
echo "rg version: $("$RG" --version 2>&1 | head -1 | cut -c1-70)"

# pin masks: T4 -> even P-cores 0,2,4,6
MASK4=0,2,4,6

for corp in silesia monorepo; do
  F="$CORPUS_DIR/$corp.gz"
  [ -f "$F" ] || fail "corpus missing $F"
  # ---- GATE-0b routing + sha==zcat ----
  REF="$(zcat "$F" | sha256sum | cut -c1-16)"
  PE=/tmp/silt4.path.$corp
  GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK4" "$GZ" -d -c -p4 "$F" >/dev/null 2>"$PE" || true
  PATHL="$(grep -m1 'path=' "$PE")"
  echo "$corp routing: $PATHL"
  echo "$PATHL" | grep -q 'path=ParallelSM' || fail "$corp not ParallelSM"
  GS="$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK4" "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  [ "$GS" = "$REF" ] || fail "$corp gz sha mismatch ($GS != $REF)"
  echo "$corp sha==zcat OK  bytes=$(zcat "$F" | wc -c)"

  # ---- rg --verbose chunk count (independent non-inert cross-check) ----
  RGV=/tmp/silt4.rgv.$corp
  taskset -c "$MASK4" "$RG" -d -c -P4 --verbose "$F" >/dev/null 2>"$RGV" || true
  echo "$corp rg-verbose chunk-ish lines:"
  grep -iE 'chunk|block.*fetch|Got [0-9]+|partition|decoded [0-9]+ chunks' "$RGV" | head -8 | sed 's/^/    /'
  cp "$RGV" "$OUT/rg_verbose.$corp.txt"

  # ---- capture GZIPPY_TIMELINE traces, /dev/null sink ----
  for r in $(seq 1 "$REPS"); do
    TJ="$OUT/$corp.T4.r$r.json"
    GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TIMELINE="$TJ" taskset -c "$MASK4" \
      "$GZ" -d -c -p4 "$F" >/dev/null 2>>"$OUT/capture.err" || fail "$corp trace r$r failed"
    echo "$corp trace r$r -> $(wc -l < "$TJ") events"
  done
done

echo "load_end: $(cat /proc/loadavg)"
echo PASS > "$DONE"
echo "=== SILT4_CAPTURE_DONE ($(date -u +%FT%TZ)) ==="
