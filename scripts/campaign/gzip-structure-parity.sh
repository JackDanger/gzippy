#!/usr/bin/env bash
# gzip-structure-parity.sh — prove CLI gzip-member identity with libdeflate.
#
# The raw `ldx-differential.sh` gate proves the isolated 0-9 port. This gate
# exercises production routing, framing, trailer, and requested thread counts.
# A byte-for-byte cmp is deliberate: equal decoded bytes or equal sizes are not
# structure parity.
#
# usage: scripts/campaign/gzip-structure-parity.sh [level ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${CORPUS:-$HOME/www/gzippy-bench/corpus}"
OURS="${OURS:-$REPO/target/release/gzippy}"
VENDOR="${LIBDEFLATE_GZIP:-$(command -v libdeflate-gzip || echo /opt/homebrew/bin/libdeflate-gzip)}"
THREADS="${THREADS:-1,2,4,8,16}"
levels=("$@")
[ ${#levels[@]} -eq 0 ] && levels=(1 2 3 4 5 6 7 8 9 10 11 12)

[ -d "$CORPUS" ] || { echo "no corpus at $CORPUS (set CORPUS=)" >&2; exit 2; }
[ -x "$OURS" ] || { echo "no gzippy binary at $OURS (set OURS=)" >&2; exit 2; }
[ -x "$VENDOR" ] || { echo "no libdeflate-gzip at $VENDOR (set LIBDEFLATE_GZIP=)" >&2; exit 2; }

work="$(mktemp -d "${TMPDIR:-/tmp}/gzip-structure-parity.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "ours:   $OURS"
echo "vendor: $VENDOR"
echo "corpus: $CORPUS"
echo "levels: ${levels[*]}"
echo "threads: $THREADS"

checked=0
: > "$work/empty"
for level in "${levels[@]}"; do
  for thread in ${THREADS//,/ }; do
    for input in "$work/empty" "$CORPUS"/*; do
      [ -f "$input" ] || continue
      "$OURS" "-$level" "-p$thread" -c "$input" > "$work/ours.gz"
      "$VENDOR" "-$level" -c "$input" > "$work/vendor.gz"
      if ! cmp -s "$work/ours.gz" "$work/vendor.gz"; then
        printf 'DIFFER L%s T%s %s ours=%s vendor=%s\n' \
          "$level" "$thread" "$(basename "$input")" \
          "$(wc -c < "$work/ours.gz" | tr -d ' ')" \
          "$(wc -c < "$work/vendor.gz" | tr -d ' ')" >&2
        exit 1
      fi
      checked=$((checked + 1))
    done
  done
  echo "L$level: exact $checked members cumulative"
done
echo "PASS: $checked gzip members are byte-identical"
