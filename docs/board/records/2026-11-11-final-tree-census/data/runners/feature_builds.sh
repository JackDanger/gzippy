#!/bin/bash
# FEATURE ARMS — all builds come from wt-a sources (b1502a07 = arm A tree),
# each with its own explicit --target-dir (the shared-target trap).
set -euo pipefail
WT=/Users/jackdanger/www/gzippy/.review-tmp/wC-final/wt-a
TGT=/Users/jackdanger/www/gzippy/.review-tmp/wC-final
cd "$WT"
cargo build --release --features nearopt-fixedpoint-skip --target-dir "$TGT/tgt-skip" 2>&1 | tail -2
cargo build --release --features near-opt-parallel-flush,near-opt-parallel-flush-flipsafe --target-dir "$TGT/tgt-safe" 2>&1 | tail -2
cargo build --release --features anatomy-counters --target-dir "$TGT/tgt-cntr" 2>&1 | tail -2
cargo build --release --features anatomy-counters,nearopt-fixedpoint-skip --target-dir "$TGT/tgt-skipc" 2>&1 | tail -2
cargo build --release --features anatomy-counters,near-opt-parallel-flush,near-opt-parallel-flush-flipsafe --target-dir "$TGT/tgt-flushc" 2>&1 | tail -2
echo FEATURE-BUILDS-DONE
