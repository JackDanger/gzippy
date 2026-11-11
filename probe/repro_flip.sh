#!/bin/bash
# Reproduce the 2026-11-10 flush-rss-probe receipt on THIS worktree's binaries:
#   L9  -p4: OFF 3,740,910 B sha 327b0835…   ON 3,740,925 B (+15) sha ff98ecd5…
#   L11 -p4: OFF 3,731,478 B sha 1d4aed15…   ON 3,731,429 B (−49) sha 86440d79…
set -u
IN="/Users/jackdanger/www/gzippy/.review-tmp/in16m.bin"
OFF=tgt-flush-off/release/gzippy
ON=tgt-flush-on/release/gzippy
for BIN in "$OFF" "$ON"; do
    [ -x "$BIN" ] || { echo "missing binary: $BIN" >&2; exit 1; }
done
for L in 9 11; do
    for tag in OFF ON; do
        BIN=$([ "$tag" = OFF ] && echo "$OFF" || echo "$ON")
        out=$( "$BIN" -"$L" -p4 -c < "$IN" 2>/dev/null | shasum -a 256 )
        size=$( "$BIN" -"$L" -p4 -c < "$IN" 2>/dev/null | wc -c )
        echo "L$L $tag size=$size sha=$out"
    done
done
