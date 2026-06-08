#!/usr/bin/env bash
# Detached parity-baseline Fulcrum vs capture at HEAD. Writes a sentinel on exit.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ART="${ART_LOCAL:-/tmp/gzippy-parity-baseline}"
SENTINEL="$ART/PARITY_BASELINE_DONE"
mkdir -p "$ART"
rm -f "$SENTINEL"
echo "PARITY_BASELINE_START $(date -u +%FT%TZ) HEAD=$(git -C "$ROOT" rev-parse --short HEAD)" | tee "$ART/run.log"
ART_LOCAL="$ART" THREADS=8 N=9 bash "$ROOT/scripts/bench/run_locked_fulcrum.sh" >>"$ART/run.log" 2>&1
RC=$?
echo "PARITY_BASELINE_DONE rc=$RC $(date -u +%FT%TZ)" | tee -a "$ART/run.log" > "$SENTINEL"
echo "rc=$RC" >> "$SENTINEL"
