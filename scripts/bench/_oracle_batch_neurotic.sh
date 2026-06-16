#!/usr/bin/env bash
# Neurotic (Intel i7-13700T) oracle batch. T4 physical cores 0,2,4,8 (avoid SMT/offline).
# arg1=gzippy bin, arg2=phaseAB, arg3=rg ELF
set -u
B="$1"; PHASE="$2"; RG="$3"
export GZIPPY_FORCE_PARALLEL_SM=1
MASK4="0,2,4,8"
MASK1="0"

echo "##### rg native ELF self-speedup (best-of-7) #####"
for corpus in model silesia; do
  for T in 1 4; do
    m=$MASK1; [ "$T" = 4 ] && m=$MASK4
    best=99999
    for i in $(seq 1 7); do
      s=$(date +%s.%N); taskset -c "$m" "$RG" -d -c -P "$T" /root/$corpus.gz >/dev/null 2>/dev/null; e=$(date +%s.%N)
      d=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}')
      awk -v d="$d" -v bb="$best" 'BEGIN{exit !(d<bb)}' && best=$d
    done
    echo "rg $corpus T=$T best=${best}s"
  done
done

echo "##### gzippy T1 baseline (best-of-7) #####"
for corpus in model silesia; do
  best=99999
  for i in $(seq 1 7); do
    s=$(date +%s.%N); taskset -c "$MASK1" "$B" -d -c -p 1 /root/$corpus.gz >/dev/null 2>/dev/null; e=$(date +%s.%N)
    d=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}')
    awk -v d="$d" -v bb="$best" 'BEGIN{exit !(d<bb)}' && best=$d
  done
  echo "gzippy $corpus T1 best=${best}s"
done

echo "##### ORACLE A/B  T4 (mask $MASK4) #####"
for corpus in model silesia; do
  echo "===== ORACLE corpus=$corpus T=4 mask=$MASK4 ====="
  bash "$PHASE" "$B" /root/$corpus.gz 4 9 "$MASK4" /root/wd_${corpus}_T4 2>&1
done
echo "##### BATCH DONE #####"
