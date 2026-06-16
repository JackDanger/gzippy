#!/usr/bin/env sh
# _wa_build_guest.sh — build symboled NATIVE (pure-rust-inflate) gzippy on the
# guest for window-absent perf annotation. Production codegen (fat LTO, cgu=1,
# opt3) but RETAIN symbols (strip=none, debuginfo=2) — instruction count
# unchanged, sliceable by symbol/source-line.
set -u
SRC="${SRC:-/root/gz-wa}"
ART="${ART:-/dev/shm/wa}"
mkdir -p "$ART"
cd "$SRC" || { echo "WA_FAIL no-src:$SRC"; exit 5; }
echo "=== df before ==="; df -h / | tail -1
export CARGO_TARGET_DIR="$SRC/target"
RUSTFLAGS="-C target-cpu=native -C strip=none -C debuginfo=2" \
  cargo build --release --no-default-features --features pure-rust-inflate \
  > "$ART/build.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "WA_FAIL build rc=$rc"; grep -E 'error' "$ART/build.log" | head -25; exit 8; fi
grep -E 'Finished|Compiling gzippy ' "$ART/build.log" | tail -2 | sed 's/^/   /'
GZ="$SRC/target/release/gzippy"
cp -f "$GZ" "$ART/gz-native-sym"
echo "   symbols=$(nm "$GZ" 2>/dev/null | wc -l)"
file "$GZ" | sed 's/^/   /'
echo "=== df after ==="; df -h / | tail -1
echo "WA_BUILD_OK $ART/gz-native-sym"
