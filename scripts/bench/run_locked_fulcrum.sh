#!/usr/bin/env bash
# run_locked_fulcrum.sh — laptop driver: host lock + guest traces + fulcrum views.
#
#   scripts/bench/run_locked_fulcrum.sh
#   BRANCH=reimplement-isa-l THREADS=8 scripts/bench/run_locked_fulcrum.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"
THREADS="${THREADS:-8}"
N="${N:-9}"
ART_LOCAL="${ART_LOCAL:-/tmp/gzippy-locked-fulcrum-$(date +%Y%m%d-%H%M%S)}"
FULCRUM_BIN="${FULCRUM_BIN:-${HOME}/www/fulcrum/target/release/fulcrum}"
NEUROTIC=(ssh -o ConnectTimeout=15 neurotic)
GUEST=(ssh -o ConnectTimeout=15 -J neurotic root@10.30.0.199)
BENCH_HOST_DIR=/root/gzippy-bench
GUEST_ART=/root/gzippy-bench/artifacts-fulcrum
REPORT="${ART_LOCAL}/fulcrum-report.txt"

if ! git -C "$ROOT" rev-parse "origin/${BRANCH}" >/dev/null 2>&1 \
  || [ -n "$(git -C "$ROOT" log "origin/${BRANCH}"..HEAD 2>/dev/null || true)" ]; then
  echo "Pushing ${BRANCH}..."
  git -C "$ROOT" push -u origin "${BRANCH}"
fi

mkdir -p "$ART_LOCAL"
GUEST_SCRIPT="$ROOT/scripts/bench/guest_fulcrum_capture.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "missing $GUEST_SCRIPT"; exit 1; }

echo "=== Stage guest_fulcrum_capture.sh on host + guest ==="
"${NEUROTIC[@]}" "mkdir -p $BENCH_HOST_DIR && cat > $BENCH_HOST_DIR/guest_fulcrum_capture.sh" <"$GUEST_SCRIPT"
"${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BENCH_HOST_DIR/guest_fulcrum_capture.sh root@10.30.0.199:$BENCH_HOST_DIR/"
"${GUEST[@]}" "chmod +x $BENCH_HOST_DIR/guest_fulcrum_capture.sh"

echo "=== Host lock + gate + guest capture (branch=$BRANCH T=$THREADS N=$N) ==="
RA_ARG=""
if [ -n "${GZIPPY_RESOLVE_AHEAD:-}" ]; then
  RA_ARG="RESOLVE_AHEAD=${GZIPPY_RESOLVE_AHEAD}"
fi
"${NEUROTIC[@]}" "GUEST_SCRIPT=guest_fulcrum_capture.sh bash $BENCH_HOST_DIR/host_lock_and_bench.sh \
  BRANCH=${BRANCH} THREADS=${THREADS} N=${N} ${RA_ARG}" 2>&1 | tee "$ART_LOCAL/host-guest.log"

echo "=== Fetch artifacts from guest 199 ==="
"${GUEST[@]}" "tar czf - -C /root/gzippy-bench artifacts-fulcrum 2>/dev/null" | tar xzf - -C "$ART_LOCAL"
cp "$ART_LOCAL/host-guest.log" "$ART_LOCAL/" 2>/dev/null || true

if [ ! -x "$FULCRUM_BIN" ]; then
  echo "FULCRUM_BIN missing at $FULCRUM_BIN — traces only in $ART_LOCAL"
  exit 0
fi

GZ_TRACE="$ART_LOCAL/artifacts-fulcrum/trace_gzippy_T8.json"
RG_TRACE="$ART_LOCAL/artifacts-fulcrum/trace_rapidgzip_T8.json"
WVO_TRACE="$ART_LOCAL/artifacts-fulcrum/trace_gzippy_writev_off_T8.json"
CFG=gzippy

{
  echo "FULCRUM REPORT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "artifacts=$ART_LOCAL"
  echo ""

  for pair in "gzippy:$GZ_TRACE" "rapidgzip:$RG_TRACE"; do
    label="${pair%%:*}"
    path="${pair#*:}"
    if [ -s "$path" ]; then
      echo "OK $label $(wc -c <"$path") bytes"
    else
      echo "MISSING $label $path"
    fi
  done
  echo ""

  echo "======== fulcrum vs (busy + wall-critical by span) ========"
  "$FULCRUM_BIN" vs "$GZ_TRACE" "$RG_TRACE" --labels gzippy,rapidgzip --config "$CFG" || true
  echo ""

  echo "======== fulcrum flow (stages: wall-crit vs slack) ========"
  "$FULCRUM_BIN" flow "$GZ_TRACE" --config "$CFG" || true
  echo ""

  echo "======== fulcrum critpath ========"
  "$FULCRUM_BIN" critpath "$GZ_TRACE" --config "$CFG" --heavy-ms 20 || true
  echo ""

  echo "======== fulcrum consumer (WAIT/COMPUTE/OUTPUT/IDLE) ========"
  "$FULCRUM_BIN" consumer "$GZ_TRACE" || true
  echo ""

  echo "======== fulcrum schedule (PLACEMENT vs RATE) ========"
  "$FULCRUM_BIN" schedule "$GZ_TRACE" || true
  echo ""

  echo "======== fulcrum causal ========"
  "$FULCRUM_BIN" causal "$GZ_TRACE" --timeline 16 || true
  echo ""

  echo "======== fulcrum decompose (residual) ========"
  "$FULCRUM_BIN" decompose "$GZ_TRACE" --config "$CFG" || true
  echo ""

  echo "======== fulcrum model ========"
  "$FULCRUM_BIN" model "$GZ_TRACE" "$RG_TRACE" --workers 8 --labels gzippy,rapidgzip || true
  echo ""

  if [ -s "$WVO_TRACE" ]; then
    echo "======== fulcrum vs writev-off vs default (gzippy only) ========"
    "$FULCRUM_BIN" vs "$GZ_TRACE" "$WVO_TRACE" --labels writev-on,writev-off --config "$CFG" || true
    echo ""
  fi

  ML_GZ="$ART_LOCAL/artifacts-fulcrum/memlife_gzippy_T8.json"
  ML_RG="$ART_LOCAL/artifacts-fulcrum/memlife_rapidgzip_T8.json"
  if [ -s "$ML_GZ" ] && [ -s "$ML_RG" ]; then
    echo "======== fulcrum memlife vs ========"
    "$FULCRUM_BIN" memlife vs "$ML_GZ" "$ML_RG" 2>/dev/null || true
  elif [ -s "$ML_GZ" ]; then
    echo "======== fulcrum memlife (gzippy only) ========"
    "$FULCRUM_BIN" memlife "$ML_GZ" 2>/dev/null || true
  fi
} | tee "$REPORT"

echo ""
echo "Done. wall+log: $ART_LOCAL/host-guest.log"
echo "fulcrum report: $REPORT"
