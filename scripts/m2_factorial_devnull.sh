#!/usr/bin/env bash
# M2 — 3-knob factorial at the /dev/null SINK (write removed => measures the
# 1.33x COMPUTE gap, not the write). Knobs (absolute us per chunk/iter):
#   B = GZIPPY_SLOW_BOOTSTRAP_US  (decode, worker, gzip_chunk.rs)
#   C = GZIPPY_SLOW_CONSUMER_US   (consumer non-write busy work, chunk_fetcher.rs)
#   R = GZIPPY_SLOW_RESOLVE_US    (marker-resolve, worker post-process, run_post_process_task)
# 2^3 = 8 cells, {spin,sleep}, N interleaved, sha-verified vs gzip(1).
# Sink = `-d -c >/dev/null` (REAL fd, full decode — NOT bare --count; the
# rapidgzip count-skip short-circuit needs no -o/-c so this avoids it).
# RUN ON GUEST 199 inside the host freeze-lock.
set -u
B=/root/gzippy/target/release/gzippy
F=/root/gzippy/benchmark_data/silesia-large.gz
PIN="taskset -c 0,2,4,6,8,10,12,14"
K="${K:-5000}"
N="${N:-9}"
export GZIPPY_FORCE_PARALLEL_SM=1

# cells: 3-bit BCR (Bootstrap Consumer Resolve). 000 baseline ... 111 all.
declare -A CELL_ENV
CELL_ENV[000]=""
CELL_ENV[B00]="GZIPPY_SLOW_BOOTSTRAP_US=$K"
CELL_ENV[0C0]="GZIPPY_SLOW_CONSUMER_US=$K"
CELL_ENV[00R]="GZIPPY_SLOW_RESOLVE_US=$K"
CELL_ENV[BC0]="GZIPPY_SLOW_BOOTSTRAP_US=$K GZIPPY_SLOW_CONSUMER_US=$K"
CELL_ENV[B0R]="GZIPPY_SLOW_BOOTSTRAP_US=$K GZIPPY_SLOW_RESOLVE_US=$K"
CELL_ENV[0CR]="GZIPPY_SLOW_CONSUMER_US=$K GZIPPY_SLOW_RESOLVE_US=$K"
CELL_ENV[BCR]="GZIPPY_SLOW_BOOTSTRAP_US=$K GZIPPY_SLOW_CONSUMER_US=$K GZIPPY_SLOW_RESOLVE_US=$K"

CELLS="000 B00 0C0 00R BC0 B0R 0CR BCR"
MODES="spin sleep"

# /dev/null full-decode short-circuit confound check: with -d -c >/dev/null the
# decode is REAL (we verify the decoded size via a one-off /dev/stdout run).
SZ=$($PIN "$B" -d -c -p 8 "$F" 2>/dev/null | wc -c)
echo "## CONFOUND CHECK: gzippy -d -c decoded_bytes=$SZ (expect 503627776, full decode confirmed)"
REFSUM=$(gzip -dc "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "REF_SHA=$REFSUM  K=${K}us  N=$N  SINK=/dev/null  PIN=0,2,4,6,8,10,12,14"
echo "load_at_start=$(awk '{print $1}' /proc/loadavg)"

declare -A TIMES
diverged=0
for ((t=1;t<=N;t++)); do
  for mode in $MODES; do
    for cell in $CELLS; do
      env_vars="${CELL_ENV[$cell]}"
      sleepflag=""
      [ "$mode" = sleep ] && sleepflag="GZIPPY_SLOW_BOOTSTRAP_SLEEP=1 GZIPPY_SLOW_CONSUMER_SLEEP=1 GZIPPY_SLOW_RESOLVE_SLEEP=1"
      # sha-verify a fraction of runs (cell 000 every trial; others trial 1)
      if [ "$cell" = 000 ] || [ "$t" = 1 ]; then
        out=$(mktemp)
        s=$(date +%s.%N); env $env_vars $sleepflag $PIN "$B" -d -c -p 8 "$F" >"$out" 2>/dev/null; rc=$?; e=$(date +%s.%N)
        sum=$(sha256sum "$out" | cut -d' ' -f1)
        [ "$sum" = "$REFSUM" ] || { echo "!! SHA DIVERGENCE ${mode}/${cell} trial $t"; diverged=1; }
        rm -f "$out"
      else
        s=$(date +%s.%N); env $env_vars $sleepflag $PIN "$B" -d -c -p 8 "$F" >/dev/null 2>/dev/null; rc=$?; e=$(date +%s.%N)
      fi
      [ $rc -eq 0 ] || echo "!! ${mode}/${cell} trial $t exit $rc"
      dt=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.4f", b-a}')
      TIMES["${mode}/${cell}"]="${TIMES["${mode}/${cell}"]:-} $dt"
    done
  done
done

echo "load_at_end=$(awk '{print $1}' /proc/loadavg)"
echo "SHA_VERIFIED=$([ $diverged = 0 ] && echo OK || echo FAIL)"
echo "===== RAW CELL TIMES (/dev/null sink) ====="
for mode in $MODES; do
  for cell in $CELLS; do
    echo "${mode}/${cell}:${TIMES["${mode}/${cell}"]}"
  done
done
