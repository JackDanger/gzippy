#!/usr/bin/env bash
# Is the compression-level curve MONOTONE? -N must never be larger than -(N-1).
set -uo pipefail
B=${1:-/private/tmp/wt-realmain/target/release/gzippy}
C=/Users/jackdanger/www/gzippy-bench/corpus
printf '%-22s' file; for L in 1 2 3 4 5 6 7 8 9; do printf '%10s' "L$L"; done; printf '  %s\n' "SAGS AT"
for f in "$C"/*; do
  n=$(basename "$f"); printf '%-22s' "$n"
  prev=0; sag=""
  for L in 1 2 3 4 5 6 7 8 9; do
    s=$("$B" -$L -c -p1 "$f" | wc -c | tr -d ' ')
    printf '%10s' "$s"
    [ "$prev" -ne 0 ] && [ "$s" -gt "$prev" ] && sag="$sag L$L"
    prev=$s
  done
  printf '  %s\n' "${sag:-—}"
done
