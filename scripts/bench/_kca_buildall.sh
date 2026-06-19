#!/usr/bin/env bash
# _kca_buildall.sh — build the 4 perf arms for NIGHT22 decisive gate:
#   A-native  (gzippy-native, kernel-converge-A HEAD)  -> /root/gz-A-native
#   A-isal    (gzippy-isal,   kernel-converge-A HEAD)  -> /root/gz-A-isal
#   NIGHT19   (gzippy-native, c4ac5acc)                -> /root/gz-night19
# OLD (chunkt1) + igzip already exist as binaries.
# Writes /dev/shm/kca_build.DONE with PASS/FAIL at end.
set -u
LOG=/dev/shm/kca_build.log
DONE=/dev/shm/kca_build.DONE
rm -f "$DONE"
exec > "$LOG" 2>&1
cd /root/gzippy || { echo "NO_CHECKOUT"; echo FAIL > "$DONE"; exit 1; }
export RUSTFLAGS="-C target-cpu=native"
HEAD=$(git rev-parse --short HEAD)
echo "=== HEAD=$HEAD ($(date +%T)) ==="

echo "=== BUILD A-native gzippy-native ($(date +%T)) ==="
CARGO_TARGET_DIR=/dev/shm/kca cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -3
BN=${PIPESTATUS[0]}
[ "$BN" -eq 0 ] && cp /dev/shm/kca/release/gzippy /root/gz-A-native
echo "BUILD_A_NATIVE_EXIT=$BN sha=$(sha256sum /root/gz-A-native 2>/dev/null | cut -c1-12)"

echo "=== BUILD A-isal gzippy-isal ($(date +%T)) ==="
CARGO_TARGET_DIR=/dev/shm/kci cargo build --release --no-default-features --features gzippy-isal 2>&1 | tail -3
BI=${PIPESTATUS[0]}
[ "$BI" -eq 0 ] && cp /dev/shm/kci/release/gzippy /root/gz-A-isal
echo "BUILD_A_ISAL_EXIT=$BI sha=$(sha256sum /root/gz-A-isal 2>/dev/null | cut -c1-12)"

echo "=== BUILD NIGHT19 (c4ac5acc) gzippy-native ($(date +%T)) ==="
git stash -u >/dev/null 2>&1 || true
CUR=$(git rev-parse HEAD)
git checkout -q c4ac5acc 2>&1 | tail -2
CARGO_TARGET_DIR=/dev/shm/kc19 cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -3
B19=${PIPESTATUS[0]}
[ "$B19" -eq 0 ] && cp /dev/shm/kc19/release/gzippy /root/gz-night19
echo "BUILD_NIGHT19_EXIT=$B19 sha=$(sha256sum /root/gz-night19 2>/dev/null | cut -c1-12)"
git checkout -q "$CUR" 2>&1 | tail -2
echo "restored HEAD=$(git rev-parse --short HEAD)"

echo "=== DONE ($(date +%T)) ==="
if [ "$BN" -eq 0 ] && [ "$BI" -eq 0 ] && [ "$B19" -eq 0 ]; then
  echo PASS > "$DONE"
else
  echo FAIL > "$DONE"
fi
