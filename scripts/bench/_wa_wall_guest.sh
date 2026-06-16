#!/usr/bin/env sh
# _wa_wall_guest.sh — interleaved wall: base vs stagec vs rg, silesia T4,
# /dev/null sink ALL arms, awk timing (guest has no bc). N=WN interleaved trials.
set -u
BASE="${BASE:-/dev/shm/wa/gz-base-sym}"
STC="${STC:-/dev/shm/wa/gz-stagec-sym}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
C="${C:-/root/silesia.gz}"
T="${T:-4}"; MASK="${MASK:-0-7}"; WN="${WN:-13}"
ART=/dev/shm/wa
: > "$ART/w.base"; : > "$ART/w.stagec"; : > "$ART/w.rg"
now() { date +%s.%N; }
i=0
while [ "$i" -lt "$WN" ]; do
  t0=$(now); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BASE" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.base"
  t0=$(now); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$STC" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.stagec"
  t0=$(now); taskset -c "$MASK" "$RG" -d -c -f -P"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.rg"
  i=$((i+1))
done
echo "=== interleaved wall (N=$WN, T=$T, /dev/null), ms ==="
for N in base stagec rg; do
  sort -n "$ART/w.$N" | awk -v n="$N" '{a[NR]=$1} END{
    min=a[1]; med=a[int((NR+1)/2)];
    s=0;for(i=1;i<=NR;i++)s+=a[i]; mean=s/NR;
    sd=0;for(i=1;i<=NR;i++)sd+=(a[i]-mean)^2; sd=sqrt(sd/NR);
    printf "  %-7s min=%.1f med=%.1f mean=%.1f sd=%.1f\n",n,min,med,mean,sd}'
done
# ratios on medians
awk 'function med(f,  a,n){n=0;while((getline l < f)>0){a[++n]=l};
  for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t};
  return a[int((n+1)/2)]}
BEGIN{ b=med("'"$ART"'/w.base"); s=med("'"$ART"'/w.stagec"); r=med("'"$ART"'/w.rg");
  printf "  median: base=%.1f stagec=%.1f rg=%.1f\n",b,s,r;
  printf "  stagec/base=%.4f  base/rg=%.4f  stagec/rg=%.4f\n", s/b, b/r, s/r }'
echo "WA_WALL_DONE"
