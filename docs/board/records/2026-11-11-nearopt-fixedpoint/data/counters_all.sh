#!/bin/bash
# counters_all.sh — the instrument phase, FINAL counter semantics (the
# exhaustion-boundary split): runs the counters-only arm (.rel-count) and the
# counters+skip arm (.rel-skipc) over every battery member. Two CSVs:
#   counters_cntr.csv  (skip OFF)  — the no-skip observable
#   counters_skipc.csv (skip ON)   — pass_entries minus the actually-skipped
# ══ cells: every member x {L9,L10,L11,L12}-p4 where the DP runs (-p4), plus
#   the -p2 route-flip inert rows (expect all nearopt counters == 0).
set -uo pipefail
WT="/Users/jackdanger/www/gzippy/.review-tmp/wA-fixpoint"
FIX="/Users/jackdanger/www/gzippy/.review-tmp/fixpt"
C="$WT/.rel-count/release/gzippy"    # anatomy-counters ON, skip OFF
CS="$WT/.rel-skipc/release/gzippy"   # anatomy-counters + skip ON
SCR="$FIX/scratch"
OUT1="$FIX/counters_cntr.csv"
OUT2="$FIX/counters_skipc.csv"
keys='nearopt_pass_entries set_from_codes_calls cost_sheet_equal_flags nearopt_finite_skip_would_save cost_sheet_equal_at_last_pass near_opt_flush_blocks near_opt_only_literals_blocks'
mkdir -p "$SCR"

printf 'member,level,p,%s\n' "$(echo $keys | tr ' ' ',')" > "$OUT1"
printf 'member,level,p,%s\n' "$(echo $keys | tr ' ' ',')" > "$OUT2"

read_counters() {
    local bin="$1" m="$2" lvl="$3" th="$4"
    "$bin" -n -$lvl -p"$th" -c "$m" > /dev/null 2> "$SCR/cnt.tmp"
    local line; line=$(grep '^ANATOMY_COUNTERS=' "$SCR/cnt.tmp" | tail -1)
    [ -n "$line" ] || { echo "NO-COUNTERS $m L$lvl p$th" >&2; return 1; }
    printf '%s' "$line"
}
flatline() { # out m lvl th [bin]
    local out="$1" m="$2" lvl="$3" th="$4" bin="${5:-$C}"
    local json vals v k
    json="$(read_counters "$bin" "$m" "$lvl" "$th")" || { echo "$(basename "$m"),L$lvl,p$th,RUNFAIL" >> "$out"; return; }
    vals=""
    for k in $keys; do
        v=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('$k','NA'))" "${json#ANATOMY_COUNTERS=}")
        vals="$vals,$v"
    done
    printf '%s,L%s,p%s,%s\n' "$(basename "$m")" "$lvl" "$th" "${vals#,}" >> "$out"
}

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

for m in "${members[@]}"; do
    for lvl in 9 10 11 12; do
        flatline "$OUT1" "$m" "$lvl" 4
        flatline "$OUT2" "$m" "$lvl" 4 "$CS"
    done
done
echo "members done: $(wc -l < "$OUT1") rows" >&2

echo "--- p2 inert rows ---" >&2
for m in "$FIX/in16m.bin" "$FIX/silesia-agg.bin" "$HOME/www/gzippy-bench/corpus/weights.safetensors"; do
    for lvl in 9 10 11 12; do
        flatline "$OUT1" "$m" "$lvl" 2
        flatline "$OUT2" "$m" "$lvl" 2 "$CS"
    done
done
echo "counters_all done: $OUT1 / $OUT2" >&2

# SKIPC interaction receipt: the skip-on stream shas must equal the byte
# battery's SKIP-arm shas (instrument inertness under the break).
for lvl in 9 10 11 12; do
    "$CS" -n -$lvl -p4 -c "$FIX/in16m.bin" > "$SCR/csi_$lvl.gz" 2>/dev/null
    "$C"  -n -$lvl -p4 -c "$FIX/in16m.bin" > "$SCR/cci_$lvl.gz" 2>/dev/null
    echo "instrument-arm streams L$lvl p4: cntr=$(shasum -a 256 "$SCR/cci_$lvl.gz" | awk '{print $1}') skipc=$(shasum -a 256 "$SCR/csi_$lvl.gz" | awk '{print $1}')" >&2
done
