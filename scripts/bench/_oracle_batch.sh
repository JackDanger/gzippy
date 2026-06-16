#!/usr/bin/env bash
# Oracle A/B batch driver. arg1=gzippy bin, arg2=phaseAB script path
set -u
B="$1"; PHASE="$2"
export GZIPPY_FORCE_PARALLEL_SM=1

echo "##### gzippy T1 baseline (best-of-7) #####"
for corpus in model silesia; do
  best=99999
  for i in $(seq 1 7); do
    s=$(date +%s.%N); taskset -c 0 "$B" -d -c -p 1 /root/$corpus.gz >/dev/null 2>/dev/null; e=$(date +%s.%N)
    d=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}')
    awk -v d="$d" -v bb="$best" 'BEGIN{exit !(d<bb)}' && best=$d
  done
  echo "gzippy $corpus T1 best=${best}s"
done

echo "##### ORACLE A/B #####"
for corpus in model silesia; do
  for T in 8 4; do
    mask=0-3; [ "$T" = 8 ] && mask=0-7
    echo "===== ORACLE corpus=$corpus T=$T mask=$mask ====="
    bash "$PHASE" "$B" /root/$corpus.gz "$T" 9 "$mask" /root/wd_${corpus}_T$T 2>&1
  done
done
echo "##### BATCH DONE #####"
