#!/usr/bin/env bash
# Where does libdeflate ITSELF lose to gzip or pigz, per label? Deterministic size only.
# If this set is non-empty, "tie libdeflate everywhere" cannot reach zero failing cells:
# tying libdeflate on such a cell inherits its loss to the other vendor.
set -uo pipefail
CORPUS=/Users/jackdanger/www/gzippy-bench/corpus
for L in 1 2 3 4 5 6 7 8 9; do
  for f in "$CORPUS"/*; do
    n=$(basename "$f")
    d=$(libdeflate-gzip -$L -c "$f" | wc -c | tr -d ' ')
    g=$(gzip -$L -c "$f" | wc -c | tr -d ' ')
    p=$(pigz -$L -c -p1 "$f" | wc -c | tr -d ' ')
    [ "$d" -gt "$g" ] && printf 'L%s %-22s libdeflate=%s > gzip=%s   (+%s, %.3f%%)\n' "$L" "$n" "$d" "$g" "$((d-g))" "$(echo "scale=6; 100*($d-$g)/$g" | bc)"
    [ "$d" -gt "$p" ] && printf 'L%s %-22s libdeflate=%s > pigz=%s   (+%s, %.3f%%)\n' "$L" "$n" "$d" "$p" "$((d-p))" "$(echo "scale=6; 100*($d-$p)/$p" | bc)"
  done
done
echo "=== scan complete ==="
