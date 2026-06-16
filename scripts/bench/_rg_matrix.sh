#!/usr/bin/env bash
# rg self-speedup matrix: best-of-N wall per (corpus,T). No verbose (clean wall).
set -u
RG="$1"; N="${2:-5}"
declare -A MASK=( [1]=0 [4]=0-3 [8]=0-7 )
for corpus in model silesia; do
  F=/root/$corpus.gz
  for T in 1 4 8; do
    m=${MASK[$T]}
    best=99999
    for ((i=0;i<N;i++)); do
      s=$(date +%s.%N)
      taskset -c "$m" "$RG" -d -c -P "$T" "$F" > /dev/null 2>/dev/null
      e=$(date +%s.%N)
      d=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}')
      awk -v d="$d" -v best="$best" 'BEGIN{exit !(d<best)}' && best=$d
    done
    echo "rg $corpus T=$T mask=$m best=${best}s"
  done
done
