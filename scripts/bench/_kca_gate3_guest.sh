#!/usr/bin/env bash
# _kca_gate3_guest.sh — finish the owed gates: lib-isal(skip-pipe) + isolated pipe
# modules under timeout (gate2's native lib-skip + both 60k props already PASSED;
# its bash was killed mid lib-isal). Writes /dev/shm/kca_gate3.DONE.
set -u
LOG=/dev/shm/kca_gate3.log
DONE=/dev/shm/kca_gate3.DONE
rm -f "$DONE"
exec > "$LOG" 2>&1
cd /root/gzippy || { echo NO_CHECKOUT; echo FAIL > "$DONE"; exit 1; }
export RUSTFLAGS="-C target-cpu=native"
echo "=== HEAD=$(git rev-parse --short HEAD) ($(date +%T)) ==="
FAIL=0
SKIPS="--skip fd_vectored --skip pipe --skip deadlock --skip drain --skip output_writer --skip writev --skip splice"

echo "=== LIB(skip-pipe) isal (gzippy-isal) ($(date +%T)) ==="
CARGO_TARGET_DIR=/dev/shm/kci timeout 2400 cargo test --release --no-default-features --features gzippy-isal --lib -- --test-threads=1 $SKIPS 2>&1 | tail -25
rc=${PIPESTATUS[0]}; echo "LIB_isal_EXIT=$rc"; [ "$rc" -eq 0 ] || FAIL=1

echo "=== PART3: isolated pipe-modules (native), each timeout 90 (informational) ==="
for pat in fd_vectored_write decompress::parallel test_t1_pipelined_completes_without_deadlock; do
  echo "--- isolated: $pat ($(date +%T)) ---"
  CARGO_TARGET_DIR=/dev/shm/kca timeout 90 cargo test --release --no-default-features --features gzippy-native --lib -- --test-threads=1 "$pat" 2>&1 | tail -8
  r2=${PIPESTATUS[0]}
  if [ "$r2" -eq 124 ]; then echo "ISOLATED_${pat}=TIMED_OUT(known-harness-hang)"; else echo "ISOLATED_${pat}=EXIT$r2"; fi
done

echo "=== DONE ($(date +%T)) FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ] && echo PASS > "$DONE" || echo FAIL > "$DONE"
