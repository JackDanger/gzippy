#!/usr/bin/env bash
# Thread-scaling interleaved A/B (run inside LXC 199 via pct exec).
# Answers: does rapidgzip's advantage grow or shrink with thread count?
# Pins to P-cores only (no E-core spill); interleaves so common-mode cancels.
set -u
cd /root/gzippy
F=benchmark_data/silesia-large.gz
RAW=503627776
RG=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
BIN=/tmp/bench-bin
N="${1:-12}"

run_one() { # $1=cpus $2=threads
  local CPUS="$1" T="$2"
  local TS="taskset -c $CPUS"
  declare -A C
  C[purerust]="$BIN/gzippy-purerust-sp -d -c -p $T $F"
  C[isal]="$BIN/gzippy-isal -d -c -p $T $F"
  C[rapidgzip]="$RG -d -P $T -c $F"
  local ORDER=(purerust isal rapidgzip)
  declare -A TT; for v in "${ORDER[@]}"; do TT[$v]=""; done
  for ((i=1;i<=N;i++)); do
    for v in "${ORDER[@]}"; do
      s=$(date +%s.%N); $TS ${C[$v]} >/dev/null 2>&1; e=$(date +%s.%N)
      TT[$v]="${TT[$v]} $(awk "BEGIN{printf \"%.4f\", $e-$s}")"
    done
  done
  echo "--- T=$T  (cpus $CPUS) ---"
  local med_pure=0 med_isal=0 med_rg=0
  for v in "${ORDER[@]}"; do
    read m mn mx < <(echo "${TT[$v]}" | RAW=$RAW awk '
      {for(i=1;i<=NF;i++)a[i]=$i;n=NF}
      END{raw=ENVIRON["RAW"]+0;
        for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){x=a[i];a[i]=a[j];a[j]=x}
        med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
        printf "%.0f %.0f %.0f", raw/med/1e6, raw/a[n]/1e6, raw/a[1]/1e6}')
    printf "  %-10s med=%5s MB/s  [%s-%s]\n" "$v" "$m" "$mn" "$mx"
    case $v in purerust) med_pure=$m;; isal) med_isal=$m;; rapidgzip) med_rg=$m;; esac
  done
  awk -v p=$med_pure -v s=$med_isal -v r=$med_rg 'BEGIN{
    printf "  ratios: rapidgzip/purerust=%.2fx  rapidgzip/isal=%.2fx  isal/purerust=%.2fx\n", r/p, r/s, s/p}'
}

echo "ref sha256=$(taskset -c 0,2,4,6 $BIN/gzippy-purerust-sp -d -c -p 4 $F 2>/dev/null | sha256sum | cut -d' ' -f1)"
run_one "0,2,4,6" 4
run_one "0-7"     8
run_one "0-15"    16
