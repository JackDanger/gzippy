#!/bin/bash
# M5 gate: sha grid {silesia, model, bignasa, storedmix} x T{1,2,4,8,16} x both builds.
# Single-arm (production defaults): the deleted kill-switch (GZIPPY_MARKER_RING)
# no longer exists; M3/M4 switches left at production default (unset).
set -u
CORPORA="silesia model bignasa storedmix"
THREADS="1 2 4 8 16"
BINS="/root/bin-m5-native /root/bin-m5-isal"
declare -A REF
for c in $CORPORA; do
  f=/root/$c.gz
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  REF[$c]=$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)
  echo "ref $c $(stat -c%s "$f") bytes gz, sha=${REF[$c]}"
done
fails=0; cells=0
for bin in $BINS; do
  for c in $CORPORA; do
    for t in $THREADS; do
      s=$("$bin" -d -c -p"$t" /root/$c.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
      cells=$((cells+1))
      if [ "$s" = "${REF[$c]}" ]; then
        echo "OK   $(basename "$bin") $c T$t"
      else
        echo "FAIL $(basename "$bin") $c T$t got=$s want=${REF[$c]}"
        fails=$((fails+1))
      fi
    done
  done
done
echo "GRID: $((cells-fails))/$cells OK, $fails FAIL"
