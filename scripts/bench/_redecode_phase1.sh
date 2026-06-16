#!/usr/bin/env bash
# PHASE 1 — mechanism A (redundant re-decode) via verbose slow-path/spec counters.
# Each (corpus,T) in its OWN clean invocation (nasa-zeroing artifact avoidance).
# Counters are UNFROZEN (event counts, not walls). Sink = /dev/null.
set -u
B=/dev/shm/gz-b22-target/release/gzippy
test -c /dev/null || { echo "FAIL devnull-not-char"; exit 2; }
echo "BIN_SHA=$(sha256sum "$B" | cut -d' ' -f1)"
run() {
  corpus="$1"; t="$2"
  echo "######## CORPUS=$corpus T=$t ########"
  GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 \
    "$B" -d -c -p "$t" "$corpus" 2>&1 >/dev/null \
    | grep -iE "Slow-path decode|coordinator spawns|Speculation failure|block.*count|Bootstrap pool|Early window publish|Stale confirmed|non_marker|marker_count|chunk" \
    | sed 's/^/  /'
}
for c in /root/silesia.gz /root/nasa.gz; do
  for t in 1 4; do run "$c" "$t"; done
done
test -c /dev/null && echo "devnull_char_ok_after"
