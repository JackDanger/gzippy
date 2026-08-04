#!/usr/bin/env bash
# ldx-differential.sh — the `ldx` port's RUNG-3 GATE.
#
# Compresses every corpus file with BOTH our pure-Rust port and libdeflate's own
# `libdeflate_deflate_compress`, and compares sha256 of the raw DEFLATE bytes.
# `wc -c` never counts here; two streams of equal length can differ in every bit.
#
# Exists because this comparison was hand-rolled once and the hand-rolled C harness
# silently truncated a 90 MB input at 64 MiB (`1 << 26`), producing a "1 of 22 files
# differs" result that was entirely an artefact of the instrument. The harness below
# FAILS LOUDLY on truncation instead. (CLAUDE.md hard stop #6, and "implausibly-good
# is a provenance alarm" — implausibly-BAD is one too.)
#
#   usage: scripts/campaign/ldx-differential.sh [level ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${CORPUS:-$HOME/www/gzippy-bench/corpus}"
WORK="${TMPDIR:-/tmp}/ldx-differential.$$"
# `vendor/` is not populated in linked worktrees, so fall back to the main checkout.
# Override with LIBDEFLATE= if it lives somewhere else.
LIBDEFLATE="${LIBDEFLATE:-$REPO/vendor/libdeflate}"
if [ ! -f "$LIBDEFLATE/build/libdeflate.a" ]; then
  MAIN="$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
  [ -f "$MAIN/vendor/libdeflate/build/libdeflate.a" ] && LIBDEFLATE="$MAIN/vendor/libdeflate"
fi

levels=("$@")
[ ${#levels[@]} -eq 0 ] && levels=(0 1)

[ -d "$CORPUS" ] || { echo "no corpus at $CORPUS (set CORPUS=)" >&2; exit 1; }
[ -f "$LIBDEFLATE/build/libdeflate.a" ] || {
  echo "no $LIBDEFLATE/build/libdeflate.a — build the vendored libdeflate first" >&2; exit 1; }

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/harness.c" <<'EOF'
/* Dump raw DEFLATE from libdeflate itself: harness <level> < in > out */
#include <stdio.h>
#include <stdlib.h>
#include "libdeflate.h"
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: harness <level>\n"); return 2; }
    int level = atoi(argv[1]);
    size_t cap = 1UL << 31, n = 0, r;
    unsigned char *in = malloc(cap);
    if (!in) { fprintf(stderr, "malloc failed\n"); return 2; }
    while ((r = fread(in + n, 1, cap - n, stdin)) > 0) {
        n += r;
        /* Never silently truncate — that is the bug this script was written after. */
        if (n == cap) { fprintf(stderr, "INPUT TRUNCATED at %zu bytes\n", cap); return 2; }
    }
    size_t outcap = libdeflate_deflate_compress_bound(NULL, n) + 4096;
    unsigned char *out = malloc(outcap);
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
    if (!c) { fprintf(stderr, "level %d unsupported by libdeflate\n", level); return 2; }
    size_t m = libdeflate_deflate_compress(c, in, n, out, outcap);
    if (m == 0 && n != 0) { fprintf(stderr, "compress failed\n"); return 2; }
    fwrite(out, 1, m, stdout);
    return 0;
}
EOF

cc -O2 -o "$WORK/harness" "$WORK/harness.c" -I"$LIBDEFLATE" "$LIBDEFLATE/build/libdeflate.a"
cargo build --release --example ldxdump --manifest-path "$REPO/Cargo.toml" >/dev/null 2>&1
OURS="$REPO/target/release/examples/ldxdump"

# Provenance: say exactly which binaries produced the numbers (hard stop #7).
echo "ours:       $OURS"
echo "  sha256:   $(shasum -a 256 "$OURS" | cut -d' ' -f1)"
echo "  tree:     $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' +DIRTY')"
echo "libdeflate: $LIBDEFLATE/build/libdeflate.a"
echo "  sha256:   $(shasum -a 256 "$LIBDEFLATE/build/libdeflate.a" | cut -d' ' -f1)"
echo

rc=0
for level in "${levels[@]}"; do
  pass=0; fail=0
  # The empty input is its own case: it is the one the passthrough exists for.
  printf '' > "$WORK/empty"
  for f in "$WORK/empty" "$CORPUS"/*; do
    name=$(basename "$f")
    a=$("$OURS" "$level" < "$f" | shasum -a 256 | cut -d' ' -f1)
    b=$("$WORK/harness" "$level" < "$f" | shasum -a 256 | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then
      pass=$((pass+1))
    else
      fail=$((fail+1)); rc=1
      printf 'L%s DIFFER  %-24s ours=%s libdeflate=%s\n' \
        "$level" "$name" \
        "$("$OURS" "$level" < "$f" | wc -c | tr -d ' ')" \
        "$("$WORK/harness" "$level" < "$f" | wc -c | tr -d ' ')"
    fi
  done
  printf 'L%s: byte-identical %s/%s\n' "$level" "$pass" "$((pass+fail))"
done

exit $rc
