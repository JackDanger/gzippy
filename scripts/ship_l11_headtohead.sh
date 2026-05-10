#!/usr/bin/env bash
# Local L11 head-to-head vs vendor C zopfli.
#
# Asserts: gzippy --ultra deflate is byte-identical to vendor zopfli.
# Reports: wall-clock for -p1 and -p8 (informational; no hard fail
# threshold — different machines vary).
#
# Used by `make ship` step 4.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GZIPPY="$ROOT/target/release/gzippy"
ZOPFLI="$ROOT/vendor/zopfli/zopfli"
ALICE="$ROOT/test_data/alice.txt"
ONE_MB="$ROOT/test_data/text-1MB.txt"

if [ ! -x "$GZIPPY" ]; then echo "ERROR: $GZIPPY not built" >&2; exit 1; fi

# Build vendor C zopfli on demand (idempotent; ~5s the first time).
if [ ! -x "$ZOPFLI" ]; then
  echo "  building vendor C zopfli (one-time)..."
  git -C "$ROOT" submodule update --init vendor/zopfli >/dev/null 2>&1
  make -C "$ROOT/vendor/zopfli" zopfli >/dev/null 2>&1
fi

if [ ! -f "$ALICE" ] || [ ! -f "$ONE_MB" ]; then
  echo "ERROR: missing test_data/{alice.txt,text-1MB.txt}" >&2
  exit 1
fi

# Strip gzip header (variable-length: 10 fixed bytes + optional FEXTRA,
# FNAME, FCOMMENT, FHCRC) and trailing 8 bytes (CRC32 + ISIZE) so we
# compare DEFLATE payloads, not header metadata.
strip_deflate() {
  python3 - "$1" <<'PY'
import sys
d = open(sys.argv[1], 'rb').read()
i = 10
flg = d[3]
if flg & 0x04: i += 2 + int.from_bytes(d[i:i+2], 'little')
if flg & 0x08: i = d.index(0, i) + 1
if flg & 0x10: i = d.index(0, i) + 1
if flg & 0x02: i += 2
sys.stdout.buffer.write(d[i:-8])
PY
}

run_one() {
  local name="$1" input="$2"
  local insize
  insize=$(wc -c < "$input")

  # Use temp dir for outputs so .gz files don't pollute test_data.
  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN

  # C zopfli output. The binary writes <input>.gz next to the input;
  # work on a copy in tmp so we don't pollute test_data/.
  cp "$input" "$tmpdir/in"
  local t_c_start t_c_end t_p1_start t_p1_end t_p8_start t_p8_end
  t_c_start=$(python3 -c 'import time; print(time.time())')
  "$ZOPFLI" --i15 "$tmpdir/in" 2>/dev/null
  t_c_end=$(python3 -c 'import time; print(time.time())')

  # gzippy --ultra -p1
  t_p1_start=$(python3 -c 'import time; print(time.time())')
  "$GZIPPY" --ultra --processes 1 -c "$input" > "$tmpdir/p1.gz"
  t_p1_end=$(python3 -c 'import time; print(time.time())')

  # gzippy --ultra -p8
  t_p8_start=$(python3 -c 'import time; print(time.time())')
  "$GZIPPY" --ultra --processes 8 -c "$input" > "$tmpdir/p8.gz"
  t_p8_end=$(python3 -c 'import time; print(time.time())')

  local t_c t_p1 t_p8
  t_c=$(python3 -c "print(f'{$t_c_end - $t_c_start:.3f}')")
  t_p1=$(python3 -c "print(f'{$t_p1_end - $t_p1_start:.3f}')")
  t_p8=$(python3 -c "print(f'{$t_p8_end - $t_p8_start:.3f}')")

  local c_def p1_def p8_def
  c_def=$(strip_deflate "$tmpdir/in.gz" | wc -c | tr -d ' ')
  p1_def=$(strip_deflate "$tmpdir/p1.gz" | wc -c | tr -d ' ')
  p8_def=$(strip_deflate "$tmpdir/p8.gz" | wc -c | tr -d ' ')

  # Bytewise compare deflate payloads (the ratio gate).
  local p1_match="✗" p8_match="✗"
  if cmp -s <(strip_deflate "$tmpdir/in.gz") <(strip_deflate "$tmpdir/p1.gz"); then p1_match="✓"; fi
  if cmp -s <(strip_deflate "$tmpdir/in.gz") <(strip_deflate "$tmpdir/p8.gz"); then p8_match="✓"; fi

  printf "  %-15s  %7d B  C %5ss  -p1 %5ss/%5dB %s  -p8 %5ss/%5dB %s\n" \
    "$name" "$insize" "$t_c" "$t_p1" "$p1_def" "$p1_match" "$t_p8" "$p8_def" "$p8_match"

  if [ "$p1_match" = "✗" ] || [ "$p8_match" = "✗" ]; then
    echo "    DEFLATE MISMATCH — gzippy is producing different bytes than C zopfli" >&2
    return 1
  fi
}

echo "  workload          input      C zopfli   gzippy -p1            gzippy -p8"
echo "  ────────          ─────      ────────   ──────────            ──────────"

fail=0
run_one "alice.txt"       "$ALICE"   || fail=1
run_one "text-1MB.txt"    "$ONE_MB"  || fail=1

if [ $fail -ne 0 ]; then
  echo ""
  echo "L11 ratio gate FAILED — see plan.md Phase 11 doctrine ('ratio is sacred')" >&2
  exit 1
fi

echo ""
echo "  ✓ L11 deflate byte-identical to C zopfli on both inputs (-p1 and -p8)"
