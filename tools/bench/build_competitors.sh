#!/usr/bin/env bash
# build_competitors.sh — reproducibly build every decompressor gzippy is
# benchmarked against, from pinned git submodules, plus the minimal C
# front-ends used by matrix.sh. Run from anywhere; operates on the repo.
#
# Competitors (all submodules under vendor/):
#   vendor/libdeflate  -> build/libdeflate.a  (ld_gunzip links it)
#   vendor/zlib-ng     -> build/libz.a + zlib.h (zng_gunzip links it)
#   rapidgzip          -> installed CLI (pip); version pinned in vendor/rapidgzip
#   pigz / gunzip      -> system CLIs (reference)
#
# Idempotent: re-running rebuilds only what changed.
set -eu
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
JOBS="$(nproc 2>/dev/null || echo 4)"

echo "== init submodules =="
git submodule update --init vendor/libdeflate vendor/zlib-ng

echo "== build libdeflate (static) =="
cmake -S vendor/libdeflate -B vendor/libdeflate/build \
  -DCMAKE_BUILD_TYPE=Release -DLIBDEFLATE_BUILD_SHARED_LIB=OFF \
  -DCMAKE_C_FLAGS="-O3 -march=native" >/dev/null
cmake --build vendor/libdeflate/build -j"$JOBS" >/dev/null

echo "== build zlib-ng (static, zlib-compat) =="
cmake -S vendor/zlib-ng -B vendor/zlib-ng/build \
  -DCMAKE_BUILD_TYPE=Release -DZLIB_COMPAT=ON -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_C_FLAGS="-O3 -march=native" >/dev/null
cmake --build vendor/zlib-ng/build -j"$JOBS" >/dev/null

echo "== compile front-ends =="
gcc -O3 -march=native tools/bench/ld_gunzip.c \
  "$ROOT/vendor/libdeflate/build/libdeflate.a" -o tools/bench/ld_gunzip
gcc -O3 -march=native -I vendor/zlib-ng/build -I vendor/zlib-ng \
  tools/bench/zng_gunzip.c "$ROOT/vendor/zlib-ng/build/libz.a" -o tools/bench/zng_gunzip

echo "== versions =="
echo "libdeflate: $(git -C vendor/libdeflate describe --tags --always 2>/dev/null)"
echo "zlib-ng:    $(git -C vendor/zlib-ng describe --tags --always 2>/dev/null)"
command -v rapidgzip >/dev/null && rapidgzip --version 2>&1 | head -1
echo "DONE — tools/bench/{ld_gunzip,zng_gunzip} built."
