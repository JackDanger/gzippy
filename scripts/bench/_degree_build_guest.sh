#!/usr/bin/env bash
# _degree_build_guest.sh — build multisym-degree variant binaries on the guest.
# Edits ONLY the production lit/len degree flag at lut_huffman.rs:997
# (LutLitLenCode::rebuild_from), builds each variant to a distinct
# CARGO_TARGET_DIR in /dev/shm, prints sha + commit, then reverts the file.
#
# Usage: SRC=/mnt/internal/gz-b22 bash _degree_build_guest.sh SINGLE DOUBLE
set -euo pipefail
SRC="${SRC:-/mnt/internal/gz-b22}"
RF="-C target-cpu=native"
FILE="src/decompress/parallel/lut_huffman.rs"

cd "$SRC"
COMMIT="$(git rev-parse --short HEAD)"
echo "BUILD_BASE_COMMIT=$COMMIT"

# Confirm line 997 is the production TRIPLE flag (12-space indent).
LINE997="$(sed -n '997p' "$FILE")"
echo "LINE997=[$LINE997]"
if ! printf '%s' "$LINE997" | grep -q 'TRIPLE_SYM_FLAG,'; then
  echo "FATAL: line 997 is not the TRIPLE_SYM_FLAG production flag" >&2
  exit 3
fi

for DEG in "$@"; do
  FLAG="${DEG}_SYM_FLAG"
  TGT="/dev/shm/gz-${DEG,,}-target"
  echo "=== building $DEG ($FLAG) -> $TGT ==="
  # revert first to ensure clean start
  git checkout -- "$FILE"
  # swap only line 997
  sed -i "997s/TRIPLE_SYM_FLAG,/${FLAG},/" "$FILE"
  NEW997="$(sed -n '997p' "$FILE")"
  echo "  NEW997=[$NEW997]"
  printf '%s' "$NEW997" | grep -q "${FLAG}," || { echo "FATAL: sed did not apply for $DEG" >&2; git checkout -- "$FILE"; exit 4; }
  CARGO_TARGET_DIR="$TGT" RUSTFLAGS="$RF" \
    cargo build --release --no-default-features --features gzippy-native --bin gzippy \
    > "/tmp/build_${DEG,,}.log" 2>&1 || { echo "BUILD FAILED $DEG (see /tmp/build_${DEG,,}.log)"; tail -20 "/tmp/build_${DEG,,}.log"; git checkout -- "$FILE"; exit 5; }
  git checkout -- "$FILE"
  BIN="$TGT/release/gzippy"
  SHA="$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "DEGREE_BIN $DEG $BIN sha=$SHA"
done
echo "DEGREE_BUILD_DONE"
