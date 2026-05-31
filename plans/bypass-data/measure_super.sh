#!/bin/bash
set -u
GZ=/root/gzippy/target/release/gzippy
RG=/root/gzippy/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
IN=/tmp/gzipcli-large.gz
MERGED=/tmp/jdmeta_1.bin:/tmp/jdmeta_2.bin:/tmp/jdmeta_4.bin:/tmp/jdmeta_8.bin:/tmp/jdmeta_16.bin
EXP=e114dd2baa2e7c4aa1ef72de54eda2ec698a8689c6e5ec12c9a9a5b2976bb092
N=${N:-7}

# wall in ms; verify sha for variants that produce correct output
run_v() { local label="$1" verify="$2"; shift 2
  local s e sha ms ok
  s=$(date +%s.%N); sha=$("$@" 2>/dev/null | sha256sum | cut -d' ' -f1); e=$(date +%s.%N)
  ms=$(python3 -c "print(f'{($e-$s)*1000:.0f}')")
  if [ "$verify" = "1" ]; then ok=$([ "$sha" = "$EXP" ] && echo OK || echo "BADSHA"); else ok="(garbage-ok)"; fi
  echo "$label $ms $ok"
}

for T in 1 2 4 8 16; do
  echo "##### T=$T #####"
  for i in $(seq 1 $N); do
    run_v "gz_normal_T${T}"   1 env GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p$T $IN
    run_v "gz_oracle_T${T}"   1 env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_CLEAN_WINDOW_ORACLE=1 $GZ -d -c -p$T $IN
    run_v "gz_sleep0_T${T}"   0 env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE=$MERGED GZIPPY_SLEEP_DECODE_NS=0 $GZ -d -c -p$T $IN
    run_v "gz_sleep2_T${T}"   0 env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE=$MERGED GZIPPY_SLEEP_DECODE_NS=2000000 $GZ -d -c -p$T $IN
    run_v "gz_sleep8_T${T}"   0 env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE=$MERGED GZIPPY_SLEEP_DECODE_NS=8000000 $GZ -d -c -p$T $IN
    run_v "rg_normal_T${T}"   1 $RG -d -c -P $T $IN
  done
done
