#!/usr/bin/env bash
# Decisive removal-oracle bound for the 21ms NON-ENGINE residual on the ocl_cf path.
# Interleaves THREE contenders best-of-N on the SAME /dev/shm regular-file sink:
#   A) ocl_cf            = GZIPPY_ISAL_ENGINE_ORACLE=1                       (engine removed)
#   B) ocl_cf+overlap    = GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_PERFECT_OVERLAP=1
#                          (engine removed AND head-of-line dispatch gap removed)
#   C) rapidgzip         = the 1.0x target
# Falsifier: if B does NOT move toward C (ratio_B ~= ratio_A, Δ < spread) with
# warm_hit_frac ~= 1.0, the head-of-line dispatch gap is NOT the 21ms residual.
set -u
GZ=/root/gzippy-bench/target/release/gzippy
CORPUS=/root/silesia.gz
MASK=0,2,4,6,8,10,12,14
RG=rapidgzip
N="${1:-13}"
ART=/dev/shm/oclcf-bound
mkdir -p $ART
REF_SHA=$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)
RAW=$(gzip -dc "$CORPUS" | wc -c)
SINK=$ART/sink.bin

timed() { # <env-prefix...> -- <cmd...> ; echoes "secs sha"
  local s e secs sha rc
  s=$(date +%s.%N)
  taskset -c "$MASK" "$@" >"$SINK" 2>>$ART/run.stderr; rc=$?
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$SINK" | cut -d' ' -f1)
  echo "$secs $sha $rc"
}

AT=""; BT=""; CT=""; DIV=0
# capture warm_hit_frac once for the self-test (B run, verbose)
WHF=$(env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_PERFECT_OVERLAP=1 GZIPPY_FORCE_PARALLEL_SM=1 \
        taskset -c "$MASK" $GZ -d -c -p 8 "$CORPUS" 2>&1 >/dev/null | grep -oE 'warm_hit_frac=[0-9.]+' | head -1)

for ((i=0;i<=N;i++)); do
  read as ash arc < <(timed env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p 8 "$CORPUS")
  read bs bsh brc < <(timed env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_PERFECT_OVERLAP=1 GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p 8 "$CORPUS")
  read cs csh crc < <(timed $RG -d -c -f -P 8 "$CORPUS")
  [ "$i" -eq 0 ] && continue
  AT="$AT $as"; BT="$BT $bs"; CT="$CT $cs"
  [ "$ash" != "$REF_SHA" ] && { echo "!! SHA DIV A i=$i $ash"; DIV=1; }
  [ "$bsh" != "$REF_SHA" ] && { echo "!! SHA DIV B i=$i $bsh"; DIV=1; }
  [ "$csh" != "$REF_SHA" ] && { echo "!! SHA DIV C i=$i $csh"; DIV=1; }
done
rm -f "$SINK"

stats() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
  {v[NR]=$1} END{n=NR; if(n==0){print "0 0 0";exit}
   min=v[1];max=v[n];mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
   printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0}'; }

read amin amed asp < <(stats "$AT")
read bmin bmed bsp < <(stats "$BT")
read cmin cmed csp < <(stats "$CT")
ra=$(awk -v g="$amin" -v r="$cmin" 'BEGIN{printf "%.3f",(g>0)?r/g:0}')
rb=$(awk -v g="$bmin" -v r="$cmin" 'BEGIN{printf "%.3f",(g>0)?r/g:0}')

echo "================ OCL_CF OVERLAP-BOUND (T=8 N=$N) ================"
echo "self-test (oracle valid iff ~1.0): $WHF  sha_div=$DIV"
printf "A ocl_cf         min=%.4f med=%.4f spread=%s%%  ratio_vs_rg=%s\n" "$amin" "$amed" "$asp" "$ra"
printf "B ocl_cf+overlap min=%.4f med=%.4f spread=%s%%  ratio_vs_rg=%s\n" "$bmin" "$bmed" "$bsp" "$rb"
printf "C rapidgzip      min=%.4f med=%.4f spread=%s%%  ratio=1.000\n" "$cmin" "$cmed" "$csp"
echo "DELTA A->B (overlap removal): $(awk -v a="$amin" -v b="$bmin" 'BEGIN{printf "%.4fs (%.1f%%)", a-b, (a>0)?(a-b)/a*100:0}')"
echo "================ END ================"
