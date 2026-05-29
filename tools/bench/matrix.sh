#!/usr/bin/env bash
# matrix.sh — competitive decompression matrix for gzippy.
#
# Runs INSIDE LXC 199 (invoke via the host freeze wrapper for clean walls).
# For each archive x thread-count x tool: best-of-N wall -> MB/s of
# DECOMPRESSED output, prints a table and flags every cell where gzippy
# loses to rapidgzip / libdeflate / zlib-ng (the zero-tolerance goal).
#
# Tools (all are processes: read file -> decompress -> write /dev/null):
#   gzippy     ./target/release/gzippy -d -c -p N
#   rapidgzip  rapidgzip -d -c -f -P N        (MT)
#   libdeflate tools/bench/ld_gunzip          (ST, constant across N)
#   zlib-ng    tools/bench/zng_gunzip         (ST, constant across N)
#   pigz       pigz -d -c                     (ST decode; reference)
#
# Pinning per thread count (P-cores, HT pairs 0/1..14/15):
#   T1=0  T2=0,2  T4=0,2,4,6  T8=0,2,4,6,8,10,12,14  T16=0-15
# ST tools pinned to core 0.
#
# Usage:  matrix.sh [N_TRIALS]   (default 3)
set -u
cd "$(dirname "$0")/../.." || exit 1
GZ=./target/release/gzippy
RG=/usr/local/bin/rapidgzip
LD=tools/bench/ld_gunzip
ZNG=tools/bench/zng_gunzip
TRIALS="${1:-3}"
THREADS=(1 2 4 8 16)
declare -A PIN=( [1]=0 [2]=0,2 [4]=0,2,4,6 [8]=0,2,4,6,8,10,12,14 [16]=0-15 )

# Corpus: label -> path  (edit to match what is built on the box)
CORPUS=(
  "single-silesia:benchmark_data/silesia-large.gz"
  "single-small:benchmark_data/silesia-gzip9.gz"
  "bgzf:benchmark_data/silesia-bgzf.gz"
  "multimember:benchmark_data/silesia-multimember.gz"
  "incompressible:benchmark_data/random.gz"
)

# best-of-TRIALS wall in ns for: $1=pin  rest=command
best() { local pin=$1; shift; local b=999999999999 s e d i
  for i in $(seq 1 "$TRIALS"); do
    s=$(date +%s%N); taskset -c "$pin" "$@" >/dev/null 2>&1; e=$(date +%s%N)
    d=$((e-s)); ((d<b)) && b=$d
  done; echo "$b"; }

mbps() { awk -v ns="$1" -v by="$2" 'BEGIN{printf "%.0f", by/(ns/1e9)/1e6}'; }

for entry in "${CORPUS[@]}"; do
  label="${entry%%:*}"; path="${entry#*:}"
  [ -f "$path" ] || { printf "## %-16s MISSING (%s)\n" "$label" "$path"; continue; }
  usize=$(gunzip -c "$path" 2>/dev/null | wc -c)
  printf "\n## %s  (%s, decompressed %d bytes)\n" "$label" "$path" "$usize"
  printf "   %-4s %-10s %-10s %-10s %-10s %-10s  %s\n" "T" "gzippy" "rapidgzip" "libdefl" "zlib-ng" "pigz" "VERDICT"
  for t in "${THREADS[@]}"; do
    p="${PIN[$t]}"
    g=$(mbps "$(best "$p" $GZ -d -c -p "$t" "$path")" "$usize")
    r=$(mbps "$(best "$p" $RG -d -c -f -P "$t" "$path")" "$usize")
    l=$(mbps "$(best 0 $LD "$path")" "$usize")
    z=$(mbps "$(best 0 $ZNG "$path")" "$usize")
    pg=$(mbps "$(best 0 pigz -d -c "$path")" "$usize")
    verdict=$(awk -v g="$g" -v r="$r" -v l="$l" -v z="$z" 'BEGIN{
      loses=""; if(g<r)loses=loses" rapidgzip"; if(g<l)loses=loses" libdeflate"; if(g<z)loses=loses" zlib-ng";
      print (loses==""?"WIN":"LOSES:"loses) }')
    printf "   %-4s %-10s %-10s %-10s %-10s %-10s  %s\n" "$t" "$g" "$r" "$l" "$z" "$pg" "$verdict"
  done
done
