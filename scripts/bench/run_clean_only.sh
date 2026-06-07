#!/usr/bin/env bash
# run_clean_only.sh — laptop driver for the STEP-A.2 CLEAN-ONLY ENGINE ORACLE.
# Host lock (freeze/turbo/governor + watchdog + restore via host_lock_and_bench.sh)
# then runs guest_clean_only.sh on guest 199. Holds the ssh for the run duration
# (host_lock_and_bench.sh owns the lifecycle) so the guest run is never orphaned.
#
#   scripts/bench/run_clean_only.sh
#   BRANCH=reimplement-isa-l THREADS=8 N=9 scripts/bench/run_clean_only.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"
THREADS="${THREADS:-8}"
THREADS="${THREADS// /,}"
N="${N:-9}"
ART_LOCAL="${ART_LOCAL:-/tmp/gzippy-clean-only-$(date +%Y%m%d-%H%M%S)}"
NEUROTIC=(ssh -o ConnectTimeout=15 neurotic)
GUEST=(ssh -o ConnectTimeout=15 -J neurotic root@REDACTED_IP)
BENCH_HOST_DIR=/root/gzippy-bench

if ! git -C "$ROOT" rev-parse "origin/${BRANCH}" >/dev/null 2>&1 \
  || [ -n "$(git -C "$ROOT" log "origin/${BRANCH}"..HEAD 2>/dev/null || true)" ]; then
  echo "Pushing ${BRANCH}..."
  git -C "$ROOT" push -u origin "${BRANCH}"
fi

mkdir -p "$ART_LOCAL"
GUEST_SCRIPT="$ROOT/scripts/bench/guest_clean_only.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "missing $GUEST_SCRIPT"; exit 1; }

echo "=== Stage guest_clean_only.sh on host + guest ==="
"${NEUROTIC[@]}" "mkdir -p $BENCH_HOST_DIR && cat > $BENCH_HOST_DIR/guest_clean_only.sh" <"$GUEST_SCRIPT"
"${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BENCH_HOST_DIR/guest_clean_only.sh root@REDACTED_IP:$BENCH_HOST_DIR/"
"${GUEST[@]}" "chmod +x $BENCH_HOST_DIR/guest_clean_only.sh"

echo "=== Host lock + guest clean-only oracle (branch=$BRANCH T=$THREADS N=$N) ==="
"${NEUROTIC[@]}" "GUEST_SCRIPT=guest_clean_only.sh bash $BENCH_HOST_DIR/host_lock_and_bench.sh \
  BRANCH=${BRANCH} THREADS='${THREADS}' N=${N}" 2>&1 | tee "$ART_LOCAL/host-guest.log"

echo "=== Fetch artifacts from guest 199 ==="
"${GUEST[@]}" "tar czf - -C /root/gzippy-bench artifacts-clean-only 2>/dev/null" | tar xzf - -C "$ART_LOCAL" || true

echo ""
echo "Done. wall+log: $ART_LOCAL/host-guest.log"
echo "artifacts: $ART_LOCAL/artifacts-clean-only"
