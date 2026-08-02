#!/usr/bin/env bash
# Deterministic SIZE-only census at L1, T1: what would routing L1 to `ldx` do?
# Size is arch-invariant and load-immune, so this needs no frozen box.
set -uo pipefail
OURS=/Users/jackdanger/www/gzippy/target/release/gzippy
LDX=/private/tmp/wt-port/target/release/examples/ldxdump
CORPUS=/Users/jackdanger/www/gzippy-bench/corpus
printf '%-22s %10s %10s %10s %10s %10s\n' file ours ldx gzip pigz libdefl
for f in "$CORPUS"/*; do
  n=$(basename "$f")
  o=$("$OURS" -1 -c -p1 "$f" 2>/dev/null | wc -c | tr -d ' ')
  # ldx emits RAW deflate; a gzip container is a fixed 10-byte header + 8-byte trailer.
  l=$(( $("$LDX" 1 < "$f" | wc -c | tr -d ' ') + 18 ))
  g=$(gzip -1 -c "$f" | wc -c | tr -d ' ')
  p=$(pigz -1 -c -p1 "$f" | wc -c | tr -d ' ')
  d=$(libdeflate-gzip -1 -c "$f" | wc -c | tr -d ' ')
  printf '%-22s %10s %10s %10s %10s %10s\n' "$n" "$o" "$l" "$g" "$p" "$d"
done
