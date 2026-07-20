#!/bin/bash
# M1 directional: interleaved paired A/B sf6-vs-sf7, /dev/null both arms, N=15
# Cells: {sil,text} x L{1,6,9,12} x T1  (hash-chain matchfinder cells)
set -u
SF6=/Users/jackdanger/www/gzippy-sf6/target/release/gzippy
SF7=/Users/jackdanger/www/gzippy-sf7/target/release/gzippy
D=/tmp/sf7grid
N=15
for c in sil text; do
  for L in 1 6 9 12; do
    echo "== $c L$L T1 =="
    for i in $(seq 1 $N); do
      t6=$( { /usr/bin/time -p $SF6 -$L -p1 -c $D/$c > /dev/null; } 2>&1 | awk '/real/{print $2}' )
      t7=$( { /usr/bin/time -p $SF7 -$L -p1 -c $D/$c > /dev/null; } 2>&1 | awk '/real/{print $2}' )
      echo "pair $i sf6=$t6 sf7=$t7"
    done
  done
done
