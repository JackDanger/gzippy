#!/usr/bin/env bash
# Interleaved A/B pinned to 4 P-cores (run inside LXC 199 via pct exec).
# Variants alternate every rep so common-mode background load cancels.
set -u
cd /root/gzippy
F=benchmark_data/silesia-large.gz
RAW=503627776
RG=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
BIN=/tmp/bench-bin
TS="taskset -c 0,2,4,6"
N="${1:-20}"

declare -A CMD
CMD[purerust]="$BIN/gzippy-purerust-sp -d -c -p 4 $F"
CMD[pure_shared]="env GZIPPY_SHARED_POOL=1 $BIN/gzippy-purerust-sp -d -c -p 4 $F"
CMD[isal]="$BIN/gzippy-isal -d -c -p 4 $F"
CMD[rapidgzip]="$RG -d -P 4 -c $F"
ORDER=(purerust pure_shared isal rapidgzip)

# correctness once
H=$($TS $BIN/gzippy-purerust-sp -d -c -p 4 $F 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "ref sha256=$H"

declare -A TT
for v in "${ORDER[@]}"; do TT[$v]=""; done
for ((i=1;i<=N;i++)); do
  for v in "${ORDER[@]}"; do
    s=$(date +%s.%N); $TS ${CMD[$v]} >/dev/null 2>&1; e=$(date +%s.%N)
    TT[$v]="${TT[$v]} $(awk "BEGIN{printf \"%.4f\", $e-$s}")"
  done
done

echo "variant       median_MBps   min   max   (n=$N pinned 0,2,4,6)"
for v in "${ORDER[@]}"; do
  echo "${TT[$v]}" | RAW=$RAW V=$v awk '
    {for(i=1;i<=NF;i++)a[i]=$i;n=NF}
    END{raw=ENVIRON["RAW"]+0;
      for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){x=a[i];a[i]=a[j];a[j]=x}
      med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "%-12s  %9.0f  %4.0f  %4.0f\n", ENVIRON["V"], raw/med/1e6, raw/a[n]/1e6, raw/a[1]/1e6}'
done
