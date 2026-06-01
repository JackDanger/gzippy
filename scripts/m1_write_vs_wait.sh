#!/usr/bin/env bash
# M1 — consumer WRITE-time vs NEXT-CHUNK-WAIT, T8, TMPFS file-output sink
# (the write MUST be real). RUN ON GUEST 199, inside the host freeze-lock.
# The [gzippy M1] stderr line reports per-chunk write_us, next_chunk_wait_us
# (recv_future + block_fetcher_get), and the VERDICT (write-bound w/ headroom
# vs starved). Decode output goes to a tmpfs FILE (not stdout-to-pipe) so the
# write is a real fd write. sha-verified each run.
set -u
B=/root/gzippy/target/release/gzippy
F=/root/gzippy/benchmark_data/silesia-large.gz
PIN="taskset -c 0,2,4,6,8,10,12,14"
N="${N:-11}"
RAW_EXPECT=503627776
export GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TIMELINE=1

# tmpfs scratch (assert it IS tmpfs so the write isn't disk-bound)
TMPD="${TMPD:-/dev/shm/m1}"
mkdir -p "$TMPD"
FSTYPE=$(stat -f -c %T "$TMPD" 2>/dev/null)
echo "## M1 tmpfs sink fstype=$FSTYPE (expect tmpfs/ramfs)  N=$N  T8 pinned 0,2,4,6,8,10,12,14"

REFSUM=$(gzip -dc "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "REF_SHA=$REFSUM"

diverged=0
for ((t=1;t<=N;t++)); do
  OUT="$TMPD/out.bin"
  # decode to a real tmpfs file; capture stderr (M1 line) ; time the wall too
  s=$(date +%s.%N)
  $PIN "$B" -d -c -p 8 "$F" >"$OUT" 2>"$TMPD/err.$t"
  rc=$?
  e=$(date +%s.%N)
  dt=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.4f", b-a}')
  sz=$(stat -c %s "$OUT" 2>/dev/null)
  sum=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$sum" = "$REFSUM" ] || { echo "!! SHA DIVERGENCE trial $t"; diverged=1; }
  [ "$sz" = "$RAW_EXPECT" ] || echo "!! size $sz != $RAW_EXPECT trial $t"
  m1=$(grep '\[gzippy M1\]' "$TMPD/err.$t" | head -1)
  echo "trial $t wall=${dt}s rc=$rc $m1"
done
echo "SHA_VERIFIED=$([ $diverged = 0 ] && echo OK || echo FAIL)"
rm -f "$TMPD"/out.bin "$TMPD"/err.*
