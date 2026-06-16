#!/usr/bin/env bash
# Clean /dev/null walls: gzippy OFF matrix + oracle-ON at T8, interleaved best-of-N, sha-checked once.
set -u
B="$1"; N="${2:-9}"
export GZIPPY_FORCE_PARALLEL_SM=1
declare -A MASK=( [1]=0 [4]=0-3 [8]=0-7 )

# correctness gate (once per corpus)
for corpus in model silesia; do
  REF=$(gzip -dc /root/$corpus.gz | sha256sum | cut -d' ' -f1)
  GOT=$(taskset -c 0-7 "$B" -d -c -p 8 /root/$corpus.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
  [ "$REF" = "$GOT" ] && echo "$corpus sha OK" || echo "$corpus SHA MISMATCH"
done

echo "##### gzippy OFF matrix (/dev/null) #####"
for corpus in model silesia; do
  for T in 1 4 8; do
    m=${MASK[$T]}; best=99999
    for i in $(seq 1 $N); do
      s=$(date +%s.%N); taskset -c "$m" "$B" -d -c -p $T /root/$corpus.gz >/dev/null 2>/dev/null; e=$(date +%s.%N)
      d=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.4f",b-a}'); awk -v d=$d -v bb=$best 'BEGIN{exit !(d<bb)}' && best=$d
    done
    echo "OFF $corpus T=$T best=${best}s"
  done
done

echo "##### oracle ON vs OFF interleaved (/dev/null) T4 + T8 #####"
for corpus in model silesia; do
  for T in 4 8; do
    m=${MASK[$T]}; SEED=/root/wd_${corpus}_T$T/seed.bin
    [ -s "$SEED" ] || { echo "$corpus T=$T NO SEED FILE"; continue; }
    onb=99999; offb=99999; hits=NA
    for i in $(seq 1 $N); do
      s=$(date +%s.%N); h=$(GZIPPY_SEED_WINDOWS=$SEED taskset -c "$m" "$B" -d -c -p $T /root/$corpus.gz 2>/tmp/h.err >/dev/null; sed -n 's/.*replay: hits=\([0-9]*\).*/\1/p' /tmp/h.err); e=$(date +%s.%N)
      d=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.4f",b-a}'); awk -v d=$d -v bb=$onb 'BEGIN{exit !(d<bb)}' && onb=$d; hits=$h
      s=$(date +%s.%N); taskset -c "$m" "$B" -d -c -p $T /root/$corpus.gz >/dev/null 2>/dev/null; e=$(date +%s.%N)
      d=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.4f",b-a}'); awk -v d=$d -v bb=$offb 'BEGIN{exit !(d<bb)}' && offb=$d
    done
    delta=$(awk -v on=$onb -v off=$offb 'BEGIN{printf "%+.4f (%.1f%% of OFF)",off-on,(off-on)/off*100}')
    echo "$corpus T=$T  ON=${onb}s(hits=$hits)  OFF=${offb}s  DELTA(OFF-ON)=$delta"
  done
done
echo "##### DONE #####"
