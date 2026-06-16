#!/usr/bin/env sh
# _wa_diff_guest.sh — Stage c real-corpus differential: gz (Stage c) output sha
# must equal the gzip(1) oracle sha for every corpus × thread-count.
set -u
GZ="${GZ:-/dev/shm/wa/gz-stagec-sym}"
for C in /root/silesia.gz /root/nasa.gz /root/monorepo.gz /root/bignasa.gz; do
  [ -f "$C" ] || { echo "MISS $C"; continue; }
  REF="$(gzip -dc "$C" 2>/dev/null | sha256sum | cut -c1-16)"
  for T in 1 4 8; do
    S="$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$C" 2>/dev/null | sha256sum | cut -c1-16)"
    if [ "$S" = "$REF" ]; then R=OK; else R="MISMATCH(ref=$REF got=$S)"; fi
    echo "  $(basename "$C") T=$T $R"
  done
done
echo "WA_DIFF_DONE"
