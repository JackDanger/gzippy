#!/usr/bin/env bash
# Runs INSIDE lxc/199 (via pct exec). Neighbors must already be frozen.
# Settles: is the T=16 gap stalls or work? Is pool-collapse a real lever?
# Is T=8-on-physical-cores faster than T=16 (HT)?
set -u
cd /root/gzippy
F=benchmark_data/silesia-large.gz
RAW=503627776            # output bytes
MB=$(awk "BEGIN{print $RAW/1048576}")
RG=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
B=/tmp/bench-bin/gzippy-purerust-sp
EV=task-clock,cpu_core/cycles/,cpu_core/instructions/,minor-faults,dTLB-load-misses

field() { echo "$1" | awk -v k="$2" '$0 ~ k {gsub(/,/,"",$1); print $1; exit}'; }

echo "### purerust scaling (instructions are spin-proof; faults/MB tests pool churn) ###"
printf "%-7s %14s %14s %12s %10s %9s %9s\n" label instructions cycles min-faults flt/MBout IPC wall_s
for spec in "T4:0,2,4,6:4" "T8phys:0,2,4,6,8,10,12,14:8" "T8ht:0-7:8" "T16:0-15:16"; do
  IFS=: read lbl cpus t <<< "$spec"
  M=$(taskset -c "$cpus" perf stat -r 5 -e "$EV" "$B" -d -c -p "$t" "$F" 2>&1 >/dev/null)
  ins=$(field "$M" instructions); cyc=$(field "$M" "cpu_core/cycles/")
  mf=$(field "$M" minor-faults); wall=$(echo "$M" | awk '/elapsed/{print $1;exit}')
  ipc=$(awk -v i="$ins" -v c="$cyc" 'BEGIN{printf (c>0)?"%.2f":"-", i/c}')
  fpm=$(awk -v m="$mf" -v mb="$MB" 'BEGIN{printf "%.1f", m/mb}')
  printf "%-7s %14s %14s %12s %10s %9s %9s\n" "$lbl" "$ins" "$cyc" "$mf" "$fpm" "$ipc" "$wall"
done

echo
echo "### rapidgzip T=16 (instruction comparison) ###"
M=$(taskset -c 0-15 perf stat -r 5 -e "$EV" $RG -d -P 16 -c "$F" 2>&1 >/dev/null)
ins=$(field "$M" instructions); cyc=$(field "$M" "cpu_core/cycles/"); mf=$(field "$M" minor-faults)
wall=$(echo "$M" | awk '/elapsed/{print $1;exit}')
printf "  rapidgzip ins=%s cyc=%s min-faults=%s flt/MBout=%.1f wall=%ss\n" \
  "$ins" "$cyc" "$mf" "$(awk -v m="$mf" -v mb="$MB" 'BEGIN{print m/mb}')" "$wall"

echo
echo "### T=8-physical vs T=16(HT) wall race (interleaved n=8) ###"
p8=""; p16=""
for i in $(seq 1 8); do
  s=$(date +%s.%N); taskset -c 0,2,4,6,8,10,12,14 "$B" -d -c -p 8  "$F" >/dev/null 2>&1; e=$(date +%s.%N); p8="$p8 $(awk "BEGIN{printf \"%.4f\",$e-$s}")"
  s=$(date +%s.%N); taskset -c 0-15                "$B" -d -c -p 16 "$F" >/dev/null 2>&1; e=$(date +%s.%N); p16="$p16 $(awk "BEGIN{printf \"%.4f\",$e-$s}")"
done
med() { echo "$1" | awk -v raw="$RAW" '{for(i=1;i<=NF;i++)a[i]=$i;n=NF;for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){x=a[i];a[i]=a[j];a[j]=x}m=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;printf "%.0f MB/s (median wall %.4fs)",raw/m/1e6,m}'; }
echo "  T8phys: $(med "$p8")"
echo "  T16   : $(med "$p16")"
