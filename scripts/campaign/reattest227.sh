#!/usr/bin/env bash
# Re-attest PR #227's SIZE claim at T4 against CURRENT main, deterministically.
# Size is arch-invariant and load-immune; no frozen box needed. Not a wall claim.
set -uo pipefail
MAIN=/private/tmp/wt-realmain/target/release/gzippy
PR227=/private/tmp/wt-227/target/release/gzippy
C=/Users/jackdanger/www/gzippy-bench/corpus
echo "main=$(shasum -a 256 $MAIN | cut -c1-16)  pr227=$(shasum -a 256 $PR227 | cut -c1-16)"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' level file main227_base pr227 gzip pigz libdeflate
for L in 2 6 9; do
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
echo "=== census complete ==="
