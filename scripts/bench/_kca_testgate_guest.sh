#!/usr/bin/env bash
# _kca_testgate_guest.sh — NIGHT22 owed correctness gates for element A.
#  1. trioracle: quad-oracle(gzip+igzip+libdeflate+pigz) x 5 corpora x T1/T4/T8 x both flavors
#  2. full `cargo test --release --lib` BOTH flavors, --test-threads=1 (dodge parallel pipe-deadlock)
#  3. 60k prop_structured_roundtrip BOTH flavors
# Writes /dev/shm/kca_testgate.DONE (PASS/FAIL).
set -u
LOG=/dev/shm/kca_testgate.log
DONE=/dev/shm/kca_testgate.DONE
rm -f "$DONE"
exec > "$LOG" 2>&1
cd /root/gzippy || { echo NO_CHECKOUT; echo FAIL > "$DONE"; exit 1; }
export RUSTFLAGS="-C target-cpu=native"
echo "=== HEAD=$(git rev-parse --short HEAD) ($(date +%T)) ==="
FAIL=0

echo "=== 1. TRIORACLE (A binaries, 5 corpora, both flavors) ($(date +%T)) ==="
NATIVE=/root/gz-A-native ISAL=/root/gz-A-isal bash /root/_trioracle_gate.sh
TRI=${PIPESTATUS[0]}
echo "TRIORACLE_EXIT=$TRI"; [ "$TRI" -eq 0 ] || FAIL=1

# full lib suite per flavor, serialized. Use separate /dev/shm target dirs (reuse build caches).
run_lib() {  # $1=feature $2=targetdir $3=label
  local FEAT=$1 TD=$2 LBL=$3
  echo "=== 2.$LBL full lib test ($FEAT) --test-threads=1 ($(date +%T)) ==="
  CARGO_TARGET_DIR=$TD timeout 2400 cargo test --release --no-default-features --features "$FEAT" --lib -- --test-threads=1 2>&1 | tail -40
  local rc=${PIPESTATUS[0]}
  echo "LIB_${LBL}_EXIT=$rc"
  [ "$rc" -eq 0 ] || FAIL=1
}
run_lib gzippy-native /dev/shm/kca native
run_lib gzippy-isal   /dev/shm/kci isal

# 60k prop both flavors
run_prop() {  # $1=feature $2=td $3=label
  local FEAT=$1 TD=$2 LBL=$3
  echo "=== 3.$LBL 60k prop_structured_roundtrip ($FEAT) ($(date +%T)) ==="
  CARGO_TARGET_DIR=$TD PROPTEST_CASES=60000 timeout 1800 cargo test --release --no-default-features --features "$FEAT" --lib -- --test-threads=2 prop_structured 2>&1 | tail -15
  local rc=${PIPESTATUS[0]}
  echo "PROP_${LBL}_EXIT=$rc"
  [ "$rc" -eq 0 ] || FAIL=1
}
run_prop gzippy-native /dev/shm/kca native
run_prop gzippy-isal   /dev/shm/kci isal

echo "=== DONE ($(date +%T)) FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ] && echo PASS > "$DONE" || echo FAIL > "$DONE"
