#!/usr/bin/env bash
# Cross-tool roundtrip smoke. Asserts gzippy can read every other tool's
# output and vice versa, at every level our CLI exposes.
#
# Tools that aren't available locally are skipped (printed as "skip"),
# not failed — keeps `make ship` portable across dev boxes.
#
# Used by `make ship` step 5.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GZIPPY="$ROOT/target/release/gzippy"
ALICE="$ROOT/test_data/alice.txt"

[ -x "$GZIPPY" ] || { echo "ERROR: $GZIPPY not built" >&2; exit 1; }
[ -f "$ALICE"  ] || { echo "ERROR: $ALICE missing" >&2; exit 1; }

# Tool discovery: each entry is "name|path". A missing path → skipped.
tools=()
[ -x /usr/bin/gzip                   ] && tools+=("gzip|/usr/bin/gzip")
[ -x "$ROOT/vendor/pigz/pigz"        ] && tools+=("pigz|$ROOT/vendor/pigz/pigz")
[ -x "$ROOT/vendor/isa-l/build/igzip" ] && tools+=("igzip|$ROOT/vendor/isa-l/build/igzip")

if [ ${#tools[@]} -eq 0 ]; then
  echo "  no peer tools available locally — skipping cross-tool smoke"
  echo "  (gzippy ↔ gzippy roundtrip already covered by 'cargo test')"
  exit 0
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# ── Direction A: gzippy compresses → peer decompresses → expect alice ────────
fail=0
for level in 1 6 9 11; do
  # gzippy compresses into a single-member gzip
  "$GZIPPY" -c -$level "$ALICE" > "$tmpdir/g$level.gz" 2>/dev/null
  for entry in "${tools[@]}"; do
    name="${entry%%|*}"; bin="${entry##*|}"
    # igzip only handles levels 0-3; skip beyond that.
    if [ "$name" = "igzip" ] && [ "$level" -gt 3 ]; then
      printf "  gzippy L%-2d → %-5s  skip (level out of range)\n" "$level" "$name"
      continue
    fi
    if "$bin" -dc "$tmpdir/g$level.gz" > "$tmpdir/out" 2>/dev/null; then
      if cmp -s "$ALICE" "$tmpdir/out"; then
        printf "  gzippy L%-2d → %-5s  ✓\n" "$level" "$name"
      else
        printf "  gzippy L%-2d → %-5s  ✗ MISMATCH\n" "$level" "$name"
        fail=1
      fi
    else
      printf "  gzippy L%-2d → %-5s  ✗ DECODE FAILED\n" "$level" "$name"
      fail=1
    fi
  done
done

# ── Direction B: peer compresses → gzippy decompresses → expect alice ────────
for entry in "${tools[@]}"; do
  name="${entry%%|*}"; bin="${entry##*|}"
  for level in 1 6 9; do
    if [ "$name" = "igzip" ] && [ "$level" -gt 3 ]; then
      printf "  %-5s L%-2d → gzippy   skip (level out of range)\n" "$name" "$level"
      continue
    fi
    "$bin" -c -$level "$ALICE" > "$tmpdir/peer.gz" 2>/dev/null
    if "$GZIPPY" -dc "$tmpdir/peer.gz" > "$tmpdir/out" 2>/dev/null; then
      if cmp -s "$ALICE" "$tmpdir/out"; then
        printf "  %-5s L%-2d → gzippy   ✓\n" "$name" "$level"
      else
        printf "  %-5s L%-2d → gzippy   ✗ MISMATCH\n" "$name" "$level"
        fail=1
      fi
    else
      printf "  %-5s L%-2d → gzippy   ✗ DECODE FAILED\n" "$name" "$level"
      fail=1
    fi
  done
done

if [ $fail -ne 0 ]; then
  echo ""
  echo "Cross-tool roundtrip FAILED" >&2
  exit 1
fi

echo ""
echo "  ✓ all available tool combinations round-tripped"
