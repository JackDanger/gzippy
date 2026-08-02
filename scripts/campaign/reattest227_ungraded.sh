#!/usr/bin/env bash
# The GATE IS BLIND outside levels 2/6/9 (CLAUDE.md). This covers the OTHER SIX levels
# at T4 so #227's "0 opened" can be checked where the promotion rule cannot see.
set -uo pipefail
MAIN=/private/tmp/wt-realmain/target/release/gzippy
PR227=/private/tmp/wt-227/target/release/gzippy
C=/Users/jackdanger/www/gzippy-bench/corpus
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' level file main pr227 gzip pigz libdeflate
for L in 1 3 4 5 7 8; do
  for f in "$C"/*; do
    n=$(basename "$f")
    m=$("$MAIN"  -$L -c -p4 "$f" | wc -c | tr -d ' ')
    p=$("$PR227" -$L -c -p4 "$f" | wc -c | tr -d ' ')
    g=$(gzip -$L -c "$f" | wc -c | tr -d ' ')
    z=$(pigz -$L -c -p4 "$f" | wc -c | tr -d ' ')
    d=$(libdeflate-gzip -$L -c "$f" | wc -c | tr -d ' ')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$n" "$m" "$p" "$g" "$z" "$d"
  done
done
echo "=== ungraded census complete ==="
