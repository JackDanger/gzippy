#!/usr/bin/env bash
# Two advisor-owed measurements to convert the C2 localization "argued"->"removed":
#  (1) warm_miss CAUSE: verbose on ocl_cf+PERFECT_OVERLAP — Prefetch guard-rejects
#      (overshoot-tail discard, the COSTLY path) vs speculative_missing (cheap startup).
#  (2) SEEDFULL removal bound: ocl_cf vs ocl_cf+GZIPPY_SEED_WINDOWS — bounds the
#      marker-BOOTSTRAP pure-Rust cost (seedfull flips every chunk to the clean engine,
#      removing the bootstrap). seedfull is masks-binder => SHA-NOT-CHECKED for B; A is byte-exact.
set -u
GZ=/root/gzippy-bench/target/release/gzippy
CORPUS=/root/silesia.gz; MASK=0,2,4,6,8,10,12,14
N="${1:-13}"
ART=/dev/shm/c2-owed; mkdir -p $ART; SINK=$ART/s.bin
REF=$(gzip -dc "$CORPUS"|sha256sum|cut -d' ' -f1); REF16=$(echo $REF|cut -c1-16)

echo "===== (1) warm_miss CAUSE (ocl_cf + PERFECT_OVERLAP, verbose) ====="
env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_PERFECT_OVERLAP=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  taskset -c $MASK $GZ -d -c -p 8 "$CORPUS" >/dev/null 2>$ART/v.txt
echo "out checks: window routing + miss-cause counters:"
grep -E 'warm_hit_frac|warm_chunks|Prefetch guard-rejects|window_seeded|flip_to_clean|finished_no_flip|isal_oracle' $ART/v.txt || true

echo ""
echo "===== (2) SEEDFULL removal bound (A ocl_cf vs B ocl_cf+seedfull vs C rg), N=$N ====="
timed(){ local s e; s=$(date +%s.%N); taskset -c $MASK "$@" >"$SINK" 2>/dev/null; e=$(date +%s.%N)
  awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}'; echo " $(sha256sum "$SINK"|cut -c1-16)"; }
AT=""; BT=""; CT=""
for ((i=0;i<=N;i++)); do
  read as ash < <(timed env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p 8 "$CORPUS")
  read bs bsh < <(timed env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_SEED_WINDOWS=1 GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p 8 "$CORPUS")
  read cs csh < <(timed rapidgzip -d -c -f -P 8 "$CORPUS")
  [ "$i" -eq 0 ] && continue
  AT="$AT $as"; BT="$BT $bs"; CT="$CT $cs"
  [ "$ash" = "$REF16" ] || echo "## A sha-div i=$i $ash (A must be byte-exact!)"
  # B (seedfull) is masks-binder; sha not enforced (it MAY differ by design)
done
rm -f "$SINK"
st(){ echo "$1"|tr ' ' '\n'|grep -v '^$'|sort -n|awk '{v[NR]=$1}END{n=NR;min=v[1];max=v[n];mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;printf "%.4f %.4f %.0f",min,mid,(min>0)?(max-min)/min*100:0}'; }
read am amd asp < <(st "$AT"); read bm bmd bsp < <(st "$BT"); read cm cmd csp < <(st "$CT")
echo "A ocl_cf          min=$am med=$amd sp=$asp% ratio_vs_rg=$(awk -v g=$am -v r=$cm 'BEGIN{printf "%.3f",r/g}')"
echo "B ocl_cf+seedfull min=$bm med=$bmd sp=$bsp% ratio_vs_rg=$(awk -v g=$bm -v r=$cm 'BEGIN{printf "%.3f",r/g}') (masks-binder, SHA-NOT-CHECKED; CEILING)"
echo "C rapidgzip       min=$cm med=$cmd sp=$csp%"
echo "BOOTSTRAP-term bound (A-B) = $(awk -v a=$am -v b=$bm 'BEGIN{printf "%.4fs (%.1f%% of A)", a-b, (a>0)?(a-b)/a*100:0}')"
