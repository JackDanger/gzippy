#!/usr/bin/env bash
# ixv_build.sh — build TWO gzippy-native x86 binaries at sha 834ba516:
#   gz-asmoff : pure-rust-inflate ON, asm-kernel OFF  -> engine A active on x86
#   gz-asmon  : pure-rust-inflate ON, asm-kernel ON   -> x86 BMI2 run_contig (engine B's x86 spec)
# Gate-0: build-flavor=parallel-sm+pure both, sha pin, engine-A wire-in present
# only in asm-off (objdump), run_contig present only in asm-on.
set -euo pipefail
SHA=834ba516
SRC=/mnt/internal/gz-head
TGT=/dev/shm/ixv-target
OUT=/dev/shm/ixv
export RUSTFLAGS="-C target-cpu=native"
export CARGO_TARGET_DIR="$TGT"
mkdir -p "$OUT"
cd "$SRC"
echo "== fetch+reset to $SHA =="
git fetch origin kernel-converge-A >/dev/null 2>&1
git reset --hard "$SHA" >/dev/null 2>&1
git submodule update --init vendor/isal-rs >/dev/null 2>&1 || true
BUILT_SHA=$(git rev-parse HEAD)
echo "BUILT_SHA=$BUILT_SHA"

echo "== BUILD asm-ON (gzippy-native, unmodified tree) =="
cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -3
cp "$TGT/release/gzippy" "$OUT/gz-asmon"

echo "== transient edit: drop asm-kernel from pure-rust-inflate =="
git checkout Cargo.toml
sed -i 's/^pure-rust-inflate = \["rpmalloc-caches", "asm-kernel"\]$/pure-rust-inflate = ["rpmalloc-caches"]/' Cargo.toml
grep -n '^pure-rust-inflate' Cargo.toml

echo "== BUILD asm-OFF (engine A on x86) =="
cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -3
cp "$TGT/release/gzippy" "$OUT/gz-asmoff"
git checkout Cargo.toml

echo "== Gate-0: build flavor + version =="
for b in gz-asmon gz-asmoff; do
  echo "--- $b ---"
  GZIPPY_DEBUG=1 "$OUT/$b" --version 2>&1 | head -3 || true
done

echo "== Gate-0: engine-A wire-in symbol presence (objdump) =="
# engine A bounded fastloop should be linked ONLY in asm-off; run_contig asm only in asm-on.
echo -n "gz-asmoff decode_huffman_fastloop_bounded: "; nm "$OUT/gz-asmoff" 2>/dev/null | grep -c fastloop_bounded || true
echo -n "gz-asmon  decode_huffman_fastloop_bounded: "; nm "$OUT/gz-asmon" 2>/dev/null | grep -c fastloop_bounded || true
echo -n "gz-asmoff run_contig: "; nm "$OUT/gz-asmoff" 2>/dev/null | grep -c run_contig || true
echo -n "gz-asmon  run_contig: "; nm "$OUT/gz-asmon" 2>/dev/null | grep -c run_contig || true

echo "IXV_BUILD_DONE"
