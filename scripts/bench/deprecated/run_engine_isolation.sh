#!/usr/bin/env bash
# run_engine_isolation.sh — laptop driver for the §2.3 ENGINE ISOLATION BENCH.
# Stages the (uncommitted) bench files + guest script to guest 199, then runs
# host_lock_and_bench.sh (freq pin + watchdog + restore) which executes the
# guest bench. Holds the ssh for the run duration (no orphan).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-reimplement-isa-l}"
ART_LOCAL="${ART_LOCAL:-/tmp/gzippy-engine-isolation-$(date +%Y%m%d-%H%M%S)}"
NEUROTIC=(ssh -o ConnectTimeout=15 neurotic)
GUEST=(ssh -o ConnectTimeout=15 -J neurotic root@10.30.0.199)
BENCH_HOST_DIR=/root/gzippy-bench
STAGE_GUEST=/root/gzippy-bench/engine-bench-stage

mkdir -p "$ART_LOCAL"
GUEST_SCRIPT="$ROOT/scripts/bench/guest_engine_isolation.sh"
[ -f "$GUEST_SCRIPT" ] || { echo "missing $GUEST_SCRIPT"; exit 1; }

echo "=== Stage bench files (uncommitted) + guest script on host + guest ==="
# 1. guest script -> host -> guest
"${NEUROTIC[@]}" "mkdir -p $BENCH_HOST_DIR && cat > $BENCH_HOST_DIR/guest_engine_isolation.sh" <"$GUEST_SCRIPT"
"${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BENCH_HOST_DIR/guest_engine_isolation.sh root@10.30.0.199:$BENCH_HOST_DIR/"
"${GUEST[@]}" "chmod +x $BENCH_HOST_DIR/guest_engine_isolation.sh && mkdir -p $STAGE_GUEST"

# 2. the three uncommitted bench files -> host -> guest stage dir
stage_one() { # <local-path> <staged-name>
  "${NEUROTIC[@]}" "cat > $BENCH_HOST_DIR/_stage_$2" <"$1"
  "${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BENCH_HOST_DIR/_stage_$2 root@10.30.0.199:$STAGE_GUEST/$2"
}
stage_one "$ROOT/benches/engine_isolation.rs" engine_isolation.rs
stage_one "$ROOT/Cargo.toml"                  Cargo.toml
stage_one "$ROOT/src/lib.rs"                  lib.rs
echo "staged engine_isolation.rs + Cargo.toml + lib.rs to guest:$STAGE_GUEST"

echo "=== Host lock + guest engine isolation bench (branch=$BRANCH) ==="
"${NEUROTIC[@]}" "GUEST_SCRIPT=guest_engine_isolation.sh bash $BENCH_HOST_DIR/host_lock_and_bench.sh \
  BRANCH=${BRANCH} THREADS=1 N=11" 2>&1 | tee "$ART_LOCAL/host-guest.log"

echo "=== Fetch artifacts from guest 199 ==="
"${GUEST[@]}" "tar czf - -C /root/gzippy-bench artifacts-engine-isolation 2>/dev/null" | tar xzf - -C "$ART_LOCAL" || true

echo ""
echo "Done. log: $ART_LOCAL/host-guest.log"
echo "artifacts: $ART_LOCAL/artifacts-engine-isolation"
