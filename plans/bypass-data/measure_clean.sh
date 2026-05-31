#!/bin/bash
set -u
GZ=/root/gzippy/target/release/gzippy
RG=/root/gzippy/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
IN=/tmp/gzipcli-large.gz
MERGED=/tmp/jdmeta_1.bin:/tmp/jdmeta_2.bin:/tmp/jdmeta_4.bin:/tmp/jdmeta_8.bin:/tmp/jdmeta_16.bin
N=${N:-7}
# wall in ms, output to /dev/null (NO sha hashing in the timed region)
run() { local label="$1"; shift; local s e ms
  s=$(date +%s.%N); "$@" >/dev/null 2>/dev/null; e=$(date +%s.%N)
  ms=$(python3 -c "print(f'{($e-$s)*1000:.0f}')"); echo "$label $ms"; }
for T in 1 2 4 8 16; do
  echo "##### T=$T #####"
  for i in $(seq 1 $N); do
    run "gz_normal_T${T}" env GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p$T $IN
    run "gz_sleep0_T${T}" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_BYPASS_DECODE=$MERGED GZIPPY_SLEEP_DECODE_NS=0 $GZ -d -c -p$T $IN
    run "rg_normal_T${T}" $RG -d -c -P $T $IN
  done
done
