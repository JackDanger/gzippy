#!/usr/bin/env sh
# _fuse_measure.sh — frozen A/B for perf/fuse-window-absent (hoist sparsity
# dispatch). awk-only (guest has no bc). Freezes with a GUARANTEED no_turbo=0
# restore trap. perf stat N=REPS instr/cyc + interleaved wall N=WN, /dev/null
# both arms (SINK LAW), sha-verified. Optional 3rd binary CUM for cumulative.
set -u
BASE="${BASE:-/dev/shm/wa/gz-base-sym}"
STC="${STC:-/dev/shm/wa/gz-stagec-sym}"
CUM="${CUM:-}"                       # optional cumulative-base binary (bdcad07b)
RG="${RG:-/root/oracle_c/rapidgzip-native}"
C="${C:-/root/silesia.gz}"
T="${T:-4}"; MASK="${MASK:-0,2,4,8}"; REPS="${REPS:-9}"; WN="${WN:-13}"
ART=/dev/shm/wa; mkdir -p "$ART"

restore(){ echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null;
  echo "[restore] no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"; }
trap restore EXIT INT TERM
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null
echo "[freeze] gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"

REF="$(gzip -dc "$C" 2>/dev/null | sha256sum | cut -c1-16)"
NBYTES="$(GZIPPY_FORCE_PARALLEL_SM=1 "$STC" -d -c -p"$T" "$C" 2>/dev/null | wc -c)"
echo "=== corpus=$(basename "$C") ref=$REF nbytes=$NBYTES T=$T MASK=$MASK REPS=$REPS WN=$WN ==="

IEV="cpu_core/instructions/u"; CEV="cpu_core/cycles/u"
perf stat -e "$IEV" -- true >/dev/null 2>&1 || { IEV="instructions:u"; CEV="cycles:u"; }

# sha-verify each gz binary
ARMS="BASE:$BASE STAGEC:$STC"
[ -n "$CUM" ] && ARMS="CUM:$CUM $ARMS"
for L in $ARMS; do B="${L#*:}"; N="${L%%:*}"
  s="$(GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p"$T" "$C" 2>/dev/null | sha256sum | cut -c1-16)"
  [ "$s" = "$REF" ] && echo "  SHA OK $N" || echo "  !! SHA MISMATCH $N got=$s"
done

echo "############ perf stat instr/cyc (N=$REPS, /dev/null) ############"
for L in $ARMS; do N="${L%%:*}"; B="${L#*:}"
  perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/m.$N.stat" -- \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$B" -d -c -p"$T" "$C" >/dev/null 2>>"$ART/m.err"
  ins="$(grep instructions "$ART/m.$N.stat" | awk '{gsub(/,/,"",$1);print $1}')"
  cyc="$(grep cycles "$ART/m.$N.stat" | awk '{gsub(/,/,"",$1);print $1}')"
  awk -v n="$N" -v i="$ins" -v c="$cyc" -v b="$NBYTES" 'BEGIN{
    printf "  %-7s instr=%d cyc=%d  instr/byte=%.4f cyc/byte=%.4f\n", n,i,c,i/b,c/b}'
done

echo "############ interleaved wall (N=$WN, /dev/null), ms ############"
: > "$ART/w.base"; : > "$ART/w.stagec"; : > "$ART/w.rg"; [ -n "$CUM" ] && : > "$ART/w.cum"
now(){ date +%s.%N; }
i=0
while [ "$i" -lt "$WN" ]; do
  if [ -n "$CUM" ]; then
    t0=$(now); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$CUM" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(now)
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.cum"
  fi
  t0=$(now); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BASE" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.base"
  t0=$(now); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$STC" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.stagec"
  t0=$(now); taskset -c "$MASK" "$RG" -d -c -f -P"$T" "$C" >/dev/null 2>&1; t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n",(b-a)*1000}' >> "$ART/w.rg"
  i=$((i+1))
done
NAMES="base stagec rg"; [ -n "$CUM" ] && NAMES="cum base stagec rg"
for N in $NAMES; do
  sort -n "$ART/w.$N" | awk -v n="$N" '{a[NR]=$1} END{
    min=a[1]; med=a[int((NR+1)/2)];
    s=0;for(j=1;j<=NR;j++)s+=a[j]; mean=s/NR;
    sd=0;for(j=1;j<=NR;j++)sd+=(a[j]-mean)^2; sd=sqrt(sd/NR);
    printf "  %-7s min=%.1f med=%.1f mean=%.1f sd=%.1f\n",n,min,med,mean,sd}'
done
awk 'function med(f,  a,n,i,j,t){n=0;while((getline l < f)>0){a[++n]=l};
  for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t};
  return a[int((n+1)/2)]}
BEGIN{ b=med("'"$ART"'/w.base"); s=med("'"$ART"'/w.stagec"); r=med("'"$ART"'/w.rg");
  printf "  median: base=%.1f stagec=%.1f rg=%.1f\n",b,s,r;
  printf "  stagec/base=%.4f  base/rg=%.4f  stagec/rg=%.4f\n", s/b, b/r, s/r;
  cf="'"$ART"'/w.cum"; if((getline _ < cf)>0){ c=med(cf); printf "  cum=%.1f  stagec/cum=%.4f  cum/rg=%.4f\n", c, s/c, c/r } }'
echo "FUSE_MEASURE_DONE"
