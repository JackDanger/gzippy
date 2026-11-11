#!/bin/bash
# Interleaved wall film: SAFE (flipsafe-ON = flush+flipsafe) vs OFF (no
# features) at the prize coordinates. Strict alternation, medians over the
# armwise samples; warmup uncounted. Same discipline as the 2026-11-10 probe
# (interleaved medians, both arms same box, sequential).
#
# usage: wall_film.sh REPS LEVEL THREADS INPUT > film.tsv
# output: seq<TAB>tag<TAB>ms ; median computed by the caller.
set -u
WT=/Users/jackdanger/www/gzippy/.review-tmp/wA-flipfix
REPS=$1; LEVEL=$2; THREADS=$3; INPUT=$4
SAFE=$WT/tgt-flipsafe/release/gzippy
OFFS=$WT/tgt-flush-off/release/gzippy
for b in "$SAFE" "$OFFS"; do [ -x "$b" ] || { echo "missing $b" >&2; exit 1; }; done

# warmup both arms (cache/allocator state), not counted
"$OFFS" -"$LEVEL" -p"$THREADS" -c < "$INPUT" >/dev/null 2>&1
"$SAFE" -"$LEVEL" -p"$THREADS" -c < "$INPUT" >/dev/null 2>&1

seq=0
for i in $(seq 1 $((REPS*2))); do
    if [ $((i % 2)) -eq 1 ]; then BIN=$OFFS; tag=off; else BIN=$SAFE; tag=safe; fi
    t0=$(date +%s%N)
    "$BIN" -"$LEVEL" -p"$THREADS" -c < "$INPUT" >/dev/null 2>/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    seq=$((seq+1))
    echo -e "$seq\t$tag\t$ms"
done
