#!/usr/bin/env bash
# Interleaved best-of-N wall: gzippy vs rapidgzip, /dev/null sink, sha-verified.
set -uo pipefail
GZ=/dev/shm/gz_after
RG=/root/oracle_c/rapidgzip-native
N="${N:-8}"
now() { date +%s.%N; }
run_one() { # $1=cmd... -> prints ms
  local s e; s=$(now); "$@" -d -c >/dev/null 2>/dev/null; e=$(now)
  awk -v s="$s" -v e="$e" 'BEGIN{printf "%.1f",(e-s)*1000}'
}
# sha verify both produce identical bytes
for F in /tmp/corpora/silesia.gz /tmp/corpora/nasa.gz /tmp/corpora/monorepo.gz; do
  ga=$(GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p 4 "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
  ra=$($RG -d -c -P 4 "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "SHA $F gz=$ga rg=$ra $([ "$ga" = "$ra" ] && echo MATCH || echo MISMATCH)"
done
for T in 4 7; do
  for F in /tmp/corpora/silesia.gz /tmp/corpora/nasa.gz /tmp/corpora/monorepo.gz; do
    gmin=99999; rmin=99999; gsum=0; rsum=0
    declare -a garr rarr
    for i in $(seq 1 "$N"); do
      g=$(GZIPPY_FORCE_PARALLEL_SM=1 run_one $GZ -p $T "$F")
      r=$(run_one $RG -P $T "$F")
      garr+=("$g"); rarr+=("$r")
      awk -v a="$g" -v b="$gmin" 'BEGIN{exit !(a<b)}' && gmin=$g
      awk -v a="$r" -v b="$rmin" 'BEGIN{exit !(a<b)}' && rmin=$r
    done
    # spread = max-min over the N gz runs
    gmax=$(printf '%s\n' "${garr[@]}" | sort -n | tail -1)
    ratio=$(awk -v g="$gmin" -v r="$rmin" 'BEGIN{printf "%.3f", g/r}')
    spread=$(awk -v a="$gmax" -v b="$gmin" 'BEGIN{printf "%.1f",a-b}')
    echo "T=$T $(basename $F): gz_min=${gmin}ms rg_min=${rmin}ms ratio=${ratio} gz_spread=${spread}ms  gz=[${garr[*]}]"
    unset garr rarr
  done
done
