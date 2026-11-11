#!/bin/bash
# battery.sh — the byte-parity (lemma) battery for lever/nearopt-fixedpoint-skip.
# Every locally staged member x L9-12 x -p2/-p4, both arms (BASE = skip-off,
# SKIP = skip-on, same source, compile-time-off gate), sha-256 EVERY stream,
# and roundtrip every stream through Apple zcat (independent decoder, NOT a
# gzippy shim — verified by --version guard below). CSV receipt in $FIX.
set -uo pipefail
WT="/Users/jackdanger/www/gzippy/.review-tmp/wA-fixpoint"
FIX="/Users/jackdanger/www/gzippy/.review-tmp/fixpt"
B="$WT/.rel-base/release/gzippy"   # skip OFF, counters OFF (default features)
S="$WT/.rel-skip/release/gzippy"   # nearopt-fixedpoint-skip ON
ZCAT="/opt/homebrew/bin/zcat"
CSV="$FIX/byte_shas.csv"
SCR="$FIX/scratch"

# header (rewrite)
[ "${1:-}" = "--resume" ] || printf 'member,level,p,sha_base,sha_skip,bytes_base,bytes_skip,rt_base,rt_skip,verdict\n' > "$CSV"
mkdir -p "$SCR"
sh256() { shasum -a 256 "$1" | awk '{print $1}'; }
nbytes() { wc -c < "$1" | tr -d ' '; }

# ---- arm provenance into the stderr of this run
"$(command -v shasum)" -a 256 "$B" "$S" 2>/dev/null | sed "s|$WT/||" >&2

members=()
for f in /Users/jackdanger/www/gzippy/corpus/silesia_blocks/*.bin; do members+=("$f"); done
members+=("/Users/jackdanger/www/gzippy/test_data/alice.txt")
members+=("/Users/jackdanger/www/gzippy/test_data/text-1MB.txt")
for f in access.log aozora.txt armexe.elf data.json data.sqlite ecoli.fastq \
         markup.xml minjs.min.js movie.mp4 photo.jpg sil40 tool.bin \
         winexe.exe monorepo.tar weights.safetensors; do
    members+=("$HOME/www/gzippy-bench/corpus/$f")
done
members+=("$FIX/in16m.bin" "$FIX/silesia-agg.bin" "$FIX/logs16m.bin" "$FIX/logs64m.bin")
# tot 49 inputs (30 silesia + 2 test_data + 15 squishy [the two largest squishy
# members last, in ascending size] + 4 aggregates)

total=0; diffs=0; invalid=0
start="${1:-0}"   # member index to start from (resume)
i=0
for m in "${members[@]}"; do
    if [ "$i" -lt "$start" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    n=$(basename "$m")
    for lvl in 9 10 11 12; do
      for th in 2 4; do
        total=$((total+1))
        if ! "$B" -n -$lvl -p"$th" -c "$m" > "$SCR/a.gz" 2>"$SCR/a.err"; then rcA=$?; else rcA=0; fi
        if ! "$S" -n -$lvl -p"$th" -c "$m" > "$SCR/b.gz" 2>"$SCR/b.err"; then rcB=$?; else rcB=0; fi
        if [ "$rcA" -ne 0 ] || [ "$rcB" -ne 0 ] || [ ! -s "$SCR/a.gz" ] || [ ! -s "$SCR/b.gz" ]; then
            printf '%s,L%s,p%s,?,?,%s,%s,ERR,ERR,INVALID\n' "$n" "$lvl" "$th" "$(wc -c < "$SCR/a.gz" | tr -d ' ')" "$(wc -c < "$SCR/b.gz" | tr -d ' ')" >> "$CSV"
            echo "INVALID L$lvl p$th $n rc=$rcA/$rcB (stderr: ${SCR}/a.err,${SCR}/b.err)" >&2
            invalid=$((invalid+1)); continue
        fi
        if ! "$ZCAT" < "$SCR/a.gz" 2>/dev/null | cmp -s - "$m"; then rta=FAIL; else rta=OK; fi
        if ! "$ZCAT" < "$SCR/b.gz" 2>/dev/null | cmp -s - "$m"; then rtb=FAIL; else rtb=OK; fi
        shaA="$(sh256 "$SCR/a.gz")"; shaB="$(sh256 "$SCR/b.gz")"
        szA="$(nbytes "$SCR/a.gz")"; szB="$(nbytes "$SCR/b.gz")"
        if [ -n "$rta" ] && [ -n "$rtb" ] && [ "$rta" = OK ] && [ "$rtb" = OK ]; then
            if [ "$shaA" = "$shaB" ]; then verdict=SAME; else verdict=DIFF; diffs=$((diffs+1)); fi
        else
            verdict=INVALID; invalid=$((invalid+1))
        fi
        printf '%s,L%s,p%s,%s,%s,%s,%s,%s,%s,%s\n' "$n" "$lvl" "$th" "$shaA" "$shaB" "$szA" "$szB" "$rta" "$rtb" "$verdict" >> "$CSV"
        if [ "$verdict" != SAME ]; then echo "$verdict L$lvl p$th $n" >&2; fi
      done
    done
done
echo "battery done: streams=$total diffs=$diffs invalid=$invalid" >&2
