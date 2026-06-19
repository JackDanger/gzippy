#!/usr/bin/env bash
# kernel-converge Part B (bit-24 flag removal) byte-exact gate, guest-side.
# Builds both flavors, runs the tri-oracle sha grid + asm-vs-ref differential +
# prop_structured + lut_huffman tests. Writes a .DONE sentinel at the end.
set -u
LOG=/dev/shm/kcb_gate.log
DONE=/dev/shm/kcb_gate.DONE
rm -f "$DONE"
exec > "$LOG" 2>&1
cd /root/gzippy || { echo "NO_CHECKOUT"; echo FAIL > "$DONE"; exit 1; }

export RUSTFLAGS="-C target-cpu=native"
export CARGO_TARGET_DIR=/dev/shm/kcb

echo "=== BUILD native ($(date +%T)) ==="
cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -3
BN=$?
[ $BN -eq 0 ] && cp /dev/shm/kcb/release/gzippy /root/gz-new-native
echo "BUILD_NATIVE_EXIT=$BN"

echo "=== BUILD isal ($(date +%T)) ==="
cargo build --release --no-default-features --features gzippy-isal 2>&1 | tail -3
BI=$?
[ $BI -eq 0 ] && cp /dev/shm/kcb/release/gzippy /root/gz-new-isal
echo "BUILD_ISAL_EXIT=$BI"

if [ $BN -ne 0 ] || [ $BI -ne 0 ]; then
  echo "BUILD_FAILED"; echo FAIL > "$DONE"; exit 1
fi

echo "=== TRIORACLE GATE ($(date +%T)) ==="
NATIVE=/root/gz-new-native ISAL=/root/gz-new-isal bash /root/_trioracle_gate.sh
TRI=$?
echo "TRIORACLE_EXIT=$TRI"

echo "=== ASM-vs-REF DIFFERENTIAL + lut_huffman tests (native, $(date +%T)) ==="
timeout 400 cargo test --release --no-default-features --features gzippy-native --lib \
  -- --test-threads=4 lut_huffman differential_asm_vs_ref 2>&1 | tail -25
echo "DIFF_TESTS_EXIT=${PIPESTATUS[0]}"

echo "=== prop_structured_roundtrip (native, $(date +%T)) ==="
timeout 400 cargo test --release --no-default-features --features gzippy-native --lib \
  -- --test-threads=2 prop_structured 2>&1 | tail -15
echo "PROP_EXIT=${PIPESTATUS[0]}"

echo "=== DONE ($(date +%T)) ==="
if [ $TRI -eq 0 ]; then echo PASS > "$DONE"; else echo FAIL > "$DONE"; fi
