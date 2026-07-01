#!/usr/bin/env bash
# PRE (main_start->decode_entry) + POST (crc_verified->main_end) medians, N runs.
set -u
GZ="${GZ:-/dev/shm/gztgt/release/gzippy}"
F="$1"; T="$2"; MASK="$3"; N="${4:-11}"
OUT=$(mktemp -d)
med() { sort -n | awk '{a[NR]=$1} END{m=int((NR+1)/2); if(NR%2){print a[m]} else {printf "%.3f\n",(a[m]+a[m+1])/2}}'; }
for r in $(seq 1 "$N"); do
  GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/$r.txt"
done
grep_phase() { for r in $(seq 1 "$N"); do grep -E " -> $1 " "$OUT/$r.txt" | head -1 | grep -oE '[0-9]+\.[0-9]+ms' | head -1 | sed 's/ms//'; done | med; }
echo "corpus=$F T=$T mask=$MASK N=$N"
echo "PRE  main_start->decode_entry ms : $(grep_phase decode_entry)"
echo "POST crc_verified->main_end  ms : $(grep_phase main_end)"
echo "scaffold->first_output       ms : $(grep_phase first_output)"
echo "first_output->consumer_done  ms : $(grep_phase consumer_done)"
rm -rf "$OUT"
