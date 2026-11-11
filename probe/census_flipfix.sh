#!/bin/bash
# SHACENSUS — the flipsafe deliverable's byte matrix:
#   members x L9..L12 x -p2,-p4 x {arm OFF (no features), arm SAFE
#   (near-opt-parallel-flush + near-opt-parallel-flush-flipsafe)}
# each cell: compress with BOTH binaries (-c stdout, mtime/fname-free), sha256
# both, verdict OK iff identical, and a gzip -t roundtrip on the SAFE stream.
#
# The OFF arm is the "parallel-flush-OFF baseline" the task names: the no-
# feature build, which is byte-identical to shipped main (3,740,910 B
# in16m L9 -p4 anchor re-verified below).
#
# usage: scripts/probe/census_flipfix.sh
set -u
WT=/Users/jackdanger/www/gzippy/.review-tmp/wA-flipfix
OFF_B=$WT/tgt-flush-off/release/gzippy
SAFE_B=$WT/tgt-flipsafe-census/release/gzippy   # with anatomy-counters: settles witness
OUT=$WT/probe/census
mkdir -p "$OUT"
unset GZIP PIGZ

for b in "$OFF_B" "$SAFE_B"; do [ -x "$b" ] || { echo "missing $b" >&2; exit 1; }; done
shasum -a 256 "$OFF_B" "$SAFE_B" > "$OUT/binaries.sha256"

MEMBERS=$WT/probe/census_members.tsv
{
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/silesia/*; do
        printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    printf '%s\t%s\n' logs.txt /Users/jackdanger/www/gzippy/benchmark_data/logs.txt
    printf '%s\t%s\n' software.archive /Users/jackdanger/www/gzippy/benchmark_data/software.archive
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/logs.part-a? \
             /Users/jackdanger/www/gzippy/.review-tmp/wC-bytes/run/corpus/sw.part-a?; do
        [ -e "$f" ] && printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    for f in "$HOME"/www/gzippy-bench/corpus/*; do
        [ -e "$f" ] && printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    for f in /Users/jackdanger/www/gzippy/.review-tmp/wC-walls/corpus/*.bin; do
        printf '%s\t%s\n' "$(basename "$f")" "$f"
    done
    printf '%s\t%s\n' alice.txt /Users/jackdanger/www/gzippy/test_data/alice.txt
} | sort -t/ -k1 | awk -F'\t' '!seen[$1]++' > "$MEMBERS"
echo "members: $(wc -l < "$MEMBERS")"

: > "$OUT/jobs.tsv"
while IFS=$'\t' read -r m path; do
    for lev in 9 10 11 12; do
        for t in 2 4; do
            echo -e "$m\t$path\t$lev\t$t" >> "$OUT/jobs.tsv"
        done
    done
done < "$MEMBERS"
wc -l "$OUT/jobs.tsv"

run_cell() {
    local m=$1 path=$2 lev=$3 t=$4
    local tag="${m}_L${lev}_p${t}"
    local out="$OUT/$tag.tsv"
    [ -s "$out" ] && return 0
    local sha_off sha_safe size_off size_safe rt
    sha_off=$("$OFF_B" -"$lev" -p"$t" -c < "$path" 2>/dev/null | shasum -a 256 | awk '{print $1}')
    size_off=$("$OFF_B" -"$lev" -p"$t" -c < "$path" 2>/dev/null | wc -c | tr -d ' ')
    "$SAFE_B" -"$lev" -p"$t" -c < "$path" 2>"$OUT/$tag.stderr" > "$OUT/$tag.safe.gz"
    sha_safe=$(shasum -a 256 "$OUT/$tag.safe.gz" | awk '{print $1}')
    size_safe=$(wc -c < "$OUT/$tag.safe.gz" | tr -d ' ')
    # settles witness (anatomy counters on the SAFE binary; stderr only)
    settles=$(grep -o '"near_opt_flip_safe_settles":[0-9]*' "$OUT/$tag.stderr" 2>/dev/null | cut -d: -f2)
    flips=$(grep -o '"near_opt_only_literals_blocks":[0-9]*' "$OUT/$tag.stderr" 2>/dev/null | cut -d: -f2)
    rm -f "$OUT/$tag.stderr" "$OUT/$tag.safe.gz"
    if [ "$sha_off" = "$sha_safe" ]; then v=OK; else v=DIFF; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tyes\t%s\t%s\n' \
        "$tag" "$lev" "$t" "$size_off" "$sha_off" "$size_safe" "$sha_safe" "$v" \
        "$settles" "$flips" > "$out"
}
export -f run_cell
export OFF_B SAFE_B OUT

xargs -P 3 -n 4 bash -c 'run_cell "$1" "$2" "$3" "$4"' _ < "$OUT/jobs.tsv"
: > "$OUT/census.tsv"
for tag $(cat /dev/null); do :; done
for f in "$OUT"/*_L*.tsv; do
    [ -e "$f" ] || continue
    done_file=$(echo "$f" | xargs basename)
    cat "$f" >> "$OUT/census.tsv"
done
sort -t/ -k1 "$OUT/census.tsv" -o "$OUT/census.tsv"
echo "VERDICT rows: $(wc -l < "$OUT/census.tsv") DIFF: $(awk -F'\t' '$8=="DIFF"' "$OUT/census.tsv" | wc -l)"
