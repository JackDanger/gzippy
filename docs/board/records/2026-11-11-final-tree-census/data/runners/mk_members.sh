#!/bin/bash
# Stage the wave-B 30-member corpus table for the final-tree census:
#   12 silesia tar members + logs.txt + software.archive
#   + 8 even slices of each big file (the wave-B byte-census recipe).
# Output: run/members.tsv  (name, path, bytes, sha256-of-INPUT)
# Each member is recorded fresh here (no trust in old tables); smallest member
# slice reconstruction is asserted (7 equal slices + tail == whole-file sha).
set -euo pipefail
R=/Users/jackdanger/www/gzippy
RUN=$R/.review-tmp/wC-final/run
SIL=$R/.review-tmp/wC-bytes/run/corpus/silesia

: > "$RUN/members.tsv"
add_member() { # name path
    local bytes sha
    bytes=$(stat -f%z "$2" | tr -d ' ')
    sha=$(shasum -a 256 "$2" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$bytes" "$sha" >> "$RUN/members.tsv"
}

# 12 silesia members (staged by the wave-B byte census; sizes match its table)
for f in "$SIL"/*; do
    add_member "silesia-$(basename "$f")" "$f"
done

# the two 221 MB members
add_member logs.txt "$R/benchmark_data/logs.txt"
add_member software.archive "$R/benchmark_data/software.archive"

# 8+8 even slices, exactly as staged by the wave-B census
for f in "$R"/.review-tmp/wC-bytes/run/corpus/logs.part-a?; do
    add_member "$(basename "$f")" "$f"
done
for f in "$R"/.review-tmp/wC-bytes/run/corpus/sw.part-a?; do
    add_member "$(basename "$f")" "$f"
done

# slice-reconstruction proof for each big file: cat part-aa..ah sha == whole sha
for stem in logs sw; do
    files=("$R"/.review-tmp/wC-bytes/run/corpus/${stem}.part-a?)
    big="$R/benchmark_data/logs.txt"
    [ "$stem" = sw ] && big="$R/benchmark_data/software.archive"
    whole=$(shasum -a 256 "$big" | awk '{print $1}')
    re=$(cat "${files[@]}" | shasum -a 256 | awk '{print $1}')
    if [ "$whole" = "$re" ]; then echo "reconstruction OK: ${stem}.part-a? == $(basename "$big") ($re)"; else
        echo "FATAL: slice reconstruction sha mismatch for $stem" >&2; exit 1; fi
done

wc -l "$RUN/members.tsv"
