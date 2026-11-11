#!/bin/bash
# Envelope study: FLUSHSTAT capture across the staged corpus at L9-12 -p4.
# Purpose: measure where (block_len, matched, d=best-only_lits) sits for EVERY
# verdict=1 flip across content classes, and how often a candidate predicate
# P(matched<=K && len<=B) would fire. The streams are thrown away (this is a
# stats run, not a byte census).
set -u
WT=/Users/jackdanger/www/gzippy/.review-tmp/wA-flipfix
BIN=$WT/tgt-flipprobe/release/gzippy
OUT=$WT/probe/envelope
mkdir -p "$OUT"

MEMBERS=$WT/probe/envelope_members.tsv
{
    # 12 silesia members (staged by the wave-B census)
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/silesia/*; do
        printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    # 8+8 slices of the big files (same content classes as logs.txt/software.archive)
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/logs.part-a? \
             /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/sw.part-a?; do
        [ -e "$f" ] && printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    # the split-corpus members (16)
    for f in "$HOME"/www/gzippy-bench/corpus/*; do
        [ -e "$f" ] && printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    # wall-census stand-ins
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-walls/corpus/*.bin; do
        printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    # test_data
    printf 'alice.txt\t/Users/jackdanger/www/gzippy/test_data/alice.txt\n'
    printf 'text-1MB.txt\t/Users/jackdanger/www/gzippy/test_data/text-1MB.txt\n'
} | sort -u > "$MEMBERS"

echo "members: $(wc -l < "$MEMBERS")"
run_one() {
    local m=$1 path=$2 level=$3
    local tag="${m}_L${level}"
    [ -s "$OUT/$tag.stat" ] && return 0
    FLIP_PROBE=1 timeout 1200 "$BIN" -"$level" -p4 -c < "$path" 2>"$OUT/$tag.stat" >/dev/null
    echo "$(wc -l < "$OUT/$tag.stat") $tag"
}
export -f run_one
export BIN OUT
: > "$OUT/jobs.log"
while IFS=$'\t' read -r m path; do
    for lvl in 9 10 11 12; do
        echo -e "$m\t$path\t$lvl" >> "$OUT/jobs.log"
    done
done < "$MEMBERS"

xargs -P 4 -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$OUT/jobs.log" >> "$OUT/jobs.log" 2>&1
echo DONE
