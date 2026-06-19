#!/bin/bash
# By-function decomposition: gz-OLD vs igzip, instructions+cycles, T1 pinned cpu4.
# ONE perf instance at a time (sequential). Emits absolute totals (perf stat N=7
# interleaved) + per-symbol split (perf record/report).
set -u
GZ=/dev/shm/gzold/target/release/gzippy
IGZIP=/usr/bin/igzip
PIN=4
N=7
DD=/dev/shm/perfdec
rm -rf "$DD"; mkdir -p "$DD"

SIL=/root/silesia.gz;  SIL_BYTES=211968000
NASA=/root/nasa.gz;    NASA_BYTES=205242368

CSV="$DD/stat.csv"
echo "corpus,tool,rep,instructions,cycles" > "$CSV"

stat_one() { # corpus tool gzfile rep
  local corpus=$1 tool=$2 f=$3 rep=$4 line ins cyc
  if [ "$tool" = gz ]; then
    line=$(taskset -c $PIN perf stat -x, -e cpu_core/instructions/,cpu_core/cycles/ -- \
      env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZ" -d -c -p1 "$f" 2>&1 >/dev/null)
  else
    line=$(taskset -c $PIN perf stat -x, -e cpu_core/instructions/,cpu_core/cycles/ -- \
      taskset -c $PIN "$IGZIP" -d -c "$f" 2>&1 >/dev/null)
  fi
  ins=$(echo "$line" | awk -F, '/instructions/{print $1}' | head -1)
  cyc=$(echo "$line" | awk -F, '/cycles/{print $1}'       | head -1)
  echo "$corpus,$tool,$rep,$ins,$cyc" >> "$CSV"
}

echo "### PHASE 1: interleaved perf stat (gz,igzip) x (silesia,nasa) x N=$N"
for rep in $(seq 1 $N); do
  stat_one silesia gz    "$SIL"  $rep
  stat_one silesia igzip "$SIL"  $rep
  stat_one nasa    gz    "$NASA" $rep
  stat_one nasa    igzip "$NASA" $rep
done

echo "### PHASE 2: perf record/report per-symbol split (sequential)"
record_report() { # tag tool gzfile
  local tag=$1 tool=$2 f=$3
  if [ "$tool" = gz ]; then
    perf record -e cpu_core/instructions/,cpu_core/cycles/ -o "$DD/$tag.data" -- \
      taskset -c $PIN bash -c "for i in \$(seq 1 $N); do GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p1 $f >/dev/null 2>&1; done" \
      >/dev/null 2>"$DD/$tag.rec.log"
  else
    perf record -e cpu_core/instructions/,cpu_core/cycles/ -o "$DD/$tag.data" -- \
      taskset -c $PIN bash -c "for i in \$(seq 1 $N); do $IGZIP -d -c $f >/dev/null 2>&1; done" \
      >/dev/null 2>"$DD/$tag.rec.log"
  fi
  perf report -i "$DD/$tag.data" --stdio --percent-limit 0.3 -e cpu_core/instructions/ --sort symbol 2>/dev/null > "$DD/$tag.instr.rpt"
  perf report -i "$DD/$tag.data" --stdio --percent-limit 0.3 -e cpu_core/cycles/ --sort symbol 2>/dev/null > "$DD/$tag.cyc.rpt"
}

record_report gz_sil    gz    "$SIL"
record_report ig_sil    igzip "$SIL"
record_report gz_nasa   gz    "$NASA"
record_report ig_nasa   igzip "$NASA"

echo "### DONE. sizes: silesia=$SIL_BYTES nasa=$NASA_BYTES"
