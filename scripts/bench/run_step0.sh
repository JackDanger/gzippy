#!/usr/bin/env bash
# run_step0.sh — laptop driver for TIER-3 STEP-0 discriminators.
# Stages guest_step0.sh, runs it under host_lock_and_bench.sh on guest 199 (the
# host script OWNS the ssh lifecycle for the whole run so the guest run is never
# orphaned), fetches artifacts. Host freq state is restored by the host script's
# trap + watchdog regardless of laptop ssh fate.
#
#   scripts/bench/run_step0.sh
#   BRANCH=reimplement-isa-l THREADS=8 scripts/bench/run_step0.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"
THREADS="${THREADS:-8}"; THREADS="${THREADS// /,}"
ART_LOCAL="${ART_LOCAL:-/tmp/gzippy-step0-$(date +%Y%m%d-%H%M%S)}"
NEUROTIC=(ssh -o ConnectTimeout=15 neurotic)
GUEST=(ssh -o ConnectTimeout=15 -J neurotic root@REDACTED_IP)
BENCH_HOST_DIR=/root/gzippy-bench

# ensure origin has the commit the guest will pull
if [ -n "$(git -C "$ROOT" log "origin/${BRANCH}"..HEAD 2>/dev/null || true)" ]; then
  echo "Pushing ${BRANCH}..."; git -C "$ROOT" push --no-verify -u origin "${BRANCH}"
fi

mkdir -p "$ART_LOCAL"
GUEST_SCRIPT="$ROOT/scripts/bench/guest_step0.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "missing $GUEST_SCRIPT"; exit 1; }

echo "=== Stage guest_step0.sh on host + guest ==="
"${NEUROTIC[@]}" "mkdir -p $BENCH_HOST_DIR && cat > $BENCH_HOST_DIR/guest_step0.sh" <"$GUEST_SCRIPT"
"${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BENCH_HOST_DIR/guest_step0.sh root@REDACTED_IP:$BENCH_HOST_DIR/"
"${GUEST[@]}" "chmod +x $BENCH_HOST_DIR/guest_step0.sh"

echo "=== Host lock + guest STEP-0 (branch=$BRANCH T=$THREADS) ==="
"${NEUROTIC[@]}" "GUEST_SCRIPT=guest_step0.sh bash $BENCH_HOST_DIR/host_lock_and_bench.sh \
  BRANCH=${BRANCH} THREADS='${THREADS}'" 2>&1 | tee "$ART_LOCAL/host-guest.log"

echo "=== Fetch artifacts from guest 199 ==="
"${GUEST[@]}" "tar czf - -C /root/gzippy-bench artifacts-step0 2>/dev/null" | tar xzf - -C "$ART_LOCAL" || true

echo ""; echo "Done. log: $ART_LOCAL/host-guest.log ; artifacts: $ART_LOCAL/artifacts-step0"
