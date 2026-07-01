#!/usr/bin/env bash
set -u
cd /root/gzippy || exit 1
B=target/release/gzippy
LIB="$(find target/release/build -name libisal.a 2>/dev/null | head -1)"
echo "=== static lib: $LIB ==="
nm "$LIB" 2>/dev/null | grep -iE 'isal_inflate|isal_read_header|decode_huffman' | head
echo "=== isal_inflate defined in lib (count) ==="
nm "$LIB" 2>/dev/null | grep -wc 'isal_inflate'
echo "=== binary stripped? ==="
file "$B" | tr ',' '\n' | grep -i strip
echo "=== runtime proof: per-chunk body log (phase should show isal/clean inflate) ==="
GZIPPY_DEBUG=1 GZIPPY_BODY_FAIL_LOG=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  "$B" -d -c -p 8 benchmark_data/silesia-large.gz 2>&1 >/dev/null \
  | grep -iE 'isal|phase|clean_window|inflate_us' | head -8
echo "=== sha vs gzip(1) ==="
A="$(GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 8 benchmark_data/silesia-large.gz | sha256sum | cut -d' ' -f1)"
R="$(gzip -dc benchmark_data/silesia-large.gz | sha256sum | cut -d' ' -f1)"
echo "gzippy=$A"
echo "gzip  =$R"
[ "$A" = "$R" ] && echo "SHA MATCH" || echo "SHA MISMATCH"
