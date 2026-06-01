#!/usr/bin/env bash
# Re-check the mechanism counters n=3 interleaved (the n=1 in the main run
# could be a one-off). Reuses the already-built binaries in /dev/shm.
set -u
WORK=/dev/shm/memmodel-ba
GZ=/root/gzippy/benchmark_data/silesia-gzip.tar.gz
BEFORE="$WORK/gzippy_before"; AFTER="$WORK/gzippy_after"
[ -x "$BEFORE" ] && [ -x "$AFTER" ] || { echo "binaries missing in $WORK"; exit 1; }

echo "## n=3 interleaved perf (T=8): dtlb_store_misses.walk_completed + minor-faults"
for i in 1 2 3; do
  for tag in before after; do
    bin="$WORK/gzippy_$tag"
    perf stat -e dtlb_store_misses.walk_completed,minor-faults \
      -- env GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p 8 "$GZ" > /dev/null 2> "$WORK/pr_${tag}_$i.txt"
    walk=$(grep "dtlb_store_misses.walk_completed" "$WORK/pr_${tag}_$i.txt" | grep cpu_core | awk '{gsub(/,/,"",$1); print $1}')
    mf=$(grep "minor-faults" "$WORK/pr_${tag}_$i.txt" | awk '{gsub(/,/,"",$1); print $1}')
    echo "  trial$i $tag: dtlb_store_walks=$walk minor_faults=$mf"
  done
done
echo "## DONE"
