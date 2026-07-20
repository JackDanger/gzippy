#!/bin/bash
# SF7 byte-identical 54-cell grid: sf6 vs sf7, L{1,2,6,8,9,12} x {sil,text,bin} x T{1,4,16}
# Plus independent-oracle roundtrip (system gzip -dc) sha check on the sf7 output.
set -u
SF6=/Users/jackdanger/www/gzippy-sf6/target/release/gzippy
SF7=/Users/jackdanger/www/gzippy-sf7/target/release/gzippy
D=/tmp/sf7grid
mkdir -p $D
BD=/Users/jackdanger/www/gzippy-reimplement-isal/benchmark_data
# corpora (sizes mirroring solvency frontier-corpora: sil 60MB, text 58MB, bin 35MB)
[ -f $D/sil ] || head -c 60000000 $BD/silesia.tar > $D/sil
[ -f $D/text ] || head -c 58000000 $BD/logs.txt > $D/text
[ -f $D/bin ] || head -c 35000000 $BD/software.archive > $D/bin
fail=0
for c in sil text bin; do
  insha=$(shasum -a 256 $D/$c | cut -d' ' -f1)
  for L in 1 2 6 8 9 12; do
    for T in 1 4 16; do
      $SF6 -$L -p$T -c $D/$c > $D/a.gz 2>/dev/null
      $SF7 -$L -p$T -c $D/$c > $D/b.gz 2>/dev/null
      if ! cmp -s $D/a.gz $D/b.gz; then
        echo "DIVERGE $c L$L T$T"; fail=1; continue
      fi
      outsha=$(gzip -dc $D/b.gz 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
      if [ "$outsha" != "$insha" ]; then
        echo "ROUNDTRIP-FAIL $c L$L T$T"; fail=1
      else
        echo "OK $c L$L T$T size=$(stat -f%z $D/b.gz)"
      fi
    done
  done
done
[ $fail -eq 0 ] && echo "GRID-PASS: all 54 cells byte-identical sf6==sf7 + gzip-oracle roundtrip" || echo "GRID-FAIL"
