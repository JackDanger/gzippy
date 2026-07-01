#!/usr/bin/env bash
# lag_causality_sweep.sh — placement-rescope lag-causality perturbation sweep.
# Runs run_locked_fulcrum.sh at T8 for F in {0,50,100} x kind in {spin,sleep}.
# Each combo: full wall (interleaved best-of-N, sha-verified) + T8 trace WITH the
# stall-residency probe (cb60842d). The STALL COUNT (trace.log) + wall + the
# block_fetcher_get span (trace) are the lag proxies. Falsifier: plans/lag-causality-falsifier.md
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/tmp/lag-causality-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
N="${N:-7}"
echo "lag-causality sweep -> $OUT  (N=$N, T=8)"

run_combo() {
  local F="$1" KIND="$2"
  local tag="F${F}_${KIND}"
  local art="$OUT/$tag"
  echo "=============== COMBO $tag ==============="
  # F=0 needs no kind distinction (knob is hoistable-zero); run it once as spin.
  GZIPPY_SLOW_MODE="$F" GZIPPY_SLOW_KIND="$KIND" \
    ART_LOCAL="$art" BRANCH=reimplement-isa-l THREADS=8 N="$N" \
    bash "$ROOT/scripts/bench/run_locked_fulcrum.sh" >"$OUT/$tag.log" 2>&1
  echo "  exit=$? log=$OUT/$tag.log"
  # extract the headline wall line + stall-residency report
  grep -h "^T8\|STALL_RESIDENCY_PROBE\|RUN_TRUSTWORTHY\|DIVERGE\|GZIPPY_SLOW_HITS\|path=" \
    "$OUT/$tag.log" "$art"/artifacts-fulcrum/trace.log 2>/dev/null | sed "s/^/  [$tag] /"
}

# F=0 once (baseline; knob off == identity, kind irrelevant)
run_combo 0 spin
for KIND in spin sleep; do
  for F in 50 100; do
    run_combo "$F" "$KIND"
  done
done
echo "=============== SWEEP DONE -> $OUT ==============="
