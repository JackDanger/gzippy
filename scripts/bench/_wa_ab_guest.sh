#!/usr/bin/env sh
# _wa_ab_guest.sh — Stage c A/B: baseline vs #[inline(always)] emit_backref_ring,
# plus rg, silesia T4. /dev/null sink for ALL arms (SINK LAW). perf stat for
# cyc/instr (N=REPS) + interleaved wall (N=WN).
set -u
BASE="${BASE:-/dev/shm/wa/gz-base-sym}"
STC="${STC:-/dev/shm/wa/gz-stagec-sym}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
C="${C:-/root/silesia.gz}"
T="${T:-4}"; MASK="${MASK:-0-7}"; REPS="${REPS:-7}"; WN="${WN:-11}"
ART=/dev/shm/wa; mkdir -p "$ART"
REF="$(gzip -dc "$C" 2>/dev/null | sha256sum | cut -c1-16)"
NBYTES="$(GZIPPY_FORCE_PARALLEL_SM=1 "$STC" -d -c -p"$T" "$C" 2>/dev/null | wc -c)"
echo "=== ref=$REF nbytes=$NBYTES T=$T MASK=$MASK REPS=$REPS WN=$WN ==="

IEV="cpu_core/instructions/u"; CEV="cpu_core/cycles/u"
perf stat -e "$IEV" -- true >/dev/null 2>&1 || { IEV="instructions:u"; CEV="cycles:u"; }

# sha-verify once each (single run, true file sink — not the -r shared fd)
for B in "$BASE" "$STC"; do
  s="$(GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p"$T" "$C" 2>/dev/null | sha256sum | cut -c1-16)"
  [ "$s" = "$REF" ] && echo "  SHA OK $(basename "$B")" || echo "  !! SHA MISMATCH $(basename "$B") got=$s"
done

echo "############ perf stat cyc/instr (N=$REPS, /dev/null) ############"
for L in BASE:$BASE STAGEC:$STC; do
  N="${L%%:*}"; B="${L#*:}"
  perf stat -r "$REPS" -e "$IEV","$CEV" -o "$ART/ab.$N.stat" -- \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$B" -d -c -p"$T" "$C" >/dev/null 2>>"$ART/ab.err"
  ins="$(grep instructions "$ART/ab.$N.stat" | awk '{print $1}' | tr -d ',')"
  cyc="$(grep cycles "$ART/ab.$N.stat" | awk '{print $1}' | tr -d ',')"
  printf "  %-7s instr=%s cyc=%s  cyc/byte=%.4f instr/byte=%.4f\n" "$N" "$ins" "$cyc" \
    "$(echo "$cyc/$NBYTES" | bc -l)" "$(echo "$ins/$NBYTES" | bc -l)"
done

echo "############ interleaved wall (N=$WN, /dev/null, ms) ############"
i=0
: > "$ART/wall.base"; : > "$ART/wall.stagec"; : > "$ART/wall.rg"
while [ "$i" -lt "$WN" ]; do
  t0=$(date +%s.%N); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BASE" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(date +%s.%N)
  echo "($t1-$t0)*1000" | bc -l >> "$ART/wall.base"
  t0=$(date +%s.%N); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$STC" -d -c -p"$T" "$C" >/dev/null 2>&1; t1=$(date +%s.%N)
  echo "($t1-$t0)*1000" | bc -l >> "$ART/wall.stagec"
  t0=$(date +%s.%N); taskset -c "$MASK" "$RG" -d -c -f -P"$T" "$C" >/dev/null 2>&1; t1=$(date +%s.%N)
  echo "($t1-$t0)*1000" | bc -l >> "$ART/wall.rg"
  i=$((i+1))
done
for N in base stagec rg; do
  sort -n "$ART/wall.$N" | awk -v n="$N" '{a[NR]=$1} END{
    min=a[1]; med=a[int((NR+1)/2)];
    s=0; for(i=1;i<=NR;i++)s+=a[i]; mean=s/NR;
    sd=0; for(i=1;i<=NR;i++)sd+=(a[i]-mean)^2; sd=sqrt(sd/NR);
    printf "  %-7s min=%.1f med=%.1f mean=%.1f sd=%.1f (ms)\n", n, min, med, mean, sd}'
done
echo "WA_AB_DONE"
