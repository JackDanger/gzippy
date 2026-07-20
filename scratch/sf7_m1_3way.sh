#!/bin/bash
# M1 3-way directional: sf6 vs sf7-v1 vs sf7-B, interleaved paired, /dev/null.
# Cells: {sil,text} x L{6,9,12} x T1, N=15.
set -u
SF6=/Users/jackdanger/www/gzippy-sf6/target/release/gzippy
V1=/tmp/sf7grid/gzippy-sf7v1
VB=/Users/jackdanger/www/gzippy-sf7/target/release/gzippy
D=/tmp/sf7grid
N=15
for c in sil text; do
  for L in 6 9 12; do
    echo "== $c L$L T1 =="
    for i in $(seq 1 $N); do
      t6=$( { /usr/bin/time -p $SF6 -$L -p1 -c $D/$c > /dev/null; } 2>&1 | awk '/real/{print $2}' )
      t1=$( { /usr/bin/time -p $V1  -$L -p1 -c $D/$c > /dev/null; } 2>&1 | awk '/real/{print $2}' )
      tb=$( { /usr/bin/time -p $VB  -$L -p1 -c $D/$c > /dev/null; } 2>&1 | awk '/real/{print $2}' )
      echo "trio $i sf6=$t6 v1=$t1 vb=$tb"
    done
  done
done
echo 3WAY_DONE
