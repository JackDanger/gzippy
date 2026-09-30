#!/usr/bin/env bash
# z119 family sweep (ladder-tune probe adjudication).
# Every member, every arm, every level listed — no cherry-picking.
# Binaries verify-stamped inline (the b2eca3bc lesson).
set -uo pipefail
CORPUS="$HOME/www/gzippy-bench/corpus"
BASE=/tmp/z119-wt-branch/target/release/gzippy        # branch+mechanism, default features == functionally main
TUNE="$HOME/www/gzippy-bench/gzippy-tune-z119"
OUT="$HOME/www/gzippy-bench/z119-results"
mkdir -p "$OUT"
echo "base-binary $(shasum -a 256 "$BASE" | cut -d' ' -f1)"  > "$OUT/stamp.txt"
echo "tune  $(shasum -a 256 "$TUNE" | cut -d' ' -f1)"       >> "$OUT/stamp.txt"
MEMBERS=(access.log data.json data.sqlite minjs.min.js sil40 tool.bin weights.safetensors text-1MB movie.mp4 photo.jpg aozora.txt armexe.elf ecoli.fastq markup.xml monorepo.tar winexe.exe)
echo "member	input	bytes	base_L4	armA_L4	armB_L4	base_L5	armA_L5	armB_L5" > "$OUT/sizes.tsv"
for m in "${MEMBERS[@]}"; do
  row="$m	$(stat -f%z "$CORPUS/$m")"
  for L in 4 5; do
    for arm in base A B; do
      case "$arm" in
        base) env=""; BIN=$BASE;;
        A) env="GZIPPY_LDX_MIN3=1"; BIN=$TUNE;;
        B) env="GZIPPY_LDX_MIN3=1 GZIPPY_LDX_LEN_GRADE=1"; BIN=$TUNE;;
      esac
      out="$OUT/${m}.L${L}.${arm}.gz"
      env $env "$BIN" -c -p 1 "-$L" "$CORPUS/$m" > "$out" 2>/dev/null
      if ! gzip -t "$out" 2>/dev/null; then echo "ROUNDTRIP FAIL: $m L$L $arm" >&2; fi
      row="$row	$(stat -f%z "$out")"
    done
  done
  echo "$row" >> "$OUT/sizes.tsv"
done
echo "sizes sweep done"
