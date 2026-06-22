#!/usr/bin/env bash
# Phase-timing batch: run gz (phase-timing) N times pinned, collect per-phase
# median (ms). Also interleaves a gz-vs-rg wall best-of-N for Gate-1.
set -u
GZ="${GZ:-/dev/shm/gztgt/release/gzippy}"
RG="${RG:-/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"
F="$1"; T="$2"; MASK="$3"; N="${4:-9}"
OUT=$(mktemp -d)

# ---- phase-timing N runs ----
for r in $(seq 1 "$N"); do
  GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
    "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/pt.$r.txt"
done
med() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "NA"; exit} m=int((NR+1)/2); if(NR%2){print a[m]} else {printf "%.3f\n",(a[m]+a[m+1])/2} }'; }
phase() { # $1 = "from->to" arrow target name to grep
  for r in $(seq 1 "$N"); do
    grep -E " -> $1 " "$OUT/pt.$r.txt" | head -1 | grep -oE '[0-9]+\.[0-9]+ms' | head -1 | sed 's/ms//'
  done | med
}
walltot() {
  for r in $(seq 1 "$N"); do
    grep "decode_wall" "$OUT/pt.$r.txt" | grep -oE '=[0-9]+\.[0-9]+ms' | grep -oE '[0-9]+\.[0-9]+'
  done | med
}
echo "### PHASE-TIMING medians  corpus=$F T=$T mask=$MASK N=$N"
echo "decode_wall_total_ms : $(walltot)"
echo "envelope_parsed_ms   : $(phase envelope_parsed)"
echo "scaffold_built_ms    : $(phase scaffold_built)"
echo "first_output_ms      : $(phase first_output)"
echo "consumer_done_ms     : $(phase consumer_done)"
echo "finalize_done_ms     : $(phase finalize_done)"
echo "crc_verified_ms      : $(phase crc_verified)"
echo "conservation: $(grep -h conservation $OUT/pt.1.txt | sed 's/^ *//')"
echo "non-inert:    $(grep -h non-inert $OUT/pt.1.txt | head -1 | sed 's/^ *//')"

# ---- interleaved wall best-of-N gz vs rg + A/A (Gate-1) ----
extwall() {
  for r in $(seq 1 "$N"); do grep ",duration_time" "$OUT/$1.$r.csv" | head -1 | cut -d, -f1; done | med
}
for r in $(seq 1 "$N"); do
  taskset -c "$MASK" perf stat -x, -e duration_time -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/GZ.$r.csv"
  taskset -c "$MASK" perf stat -x, -e duration_time -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/GZA.$r.csv"
  taskset -c "$MASK" perf stat -x, -e duration_time -- taskset -c "$MASK" "$RG" -d -c -P"$T" "$F" >/dev/null 2>"$OUT/RG.$r.csv"
done
GZW=$(extwall GZ); GZAW=$(extwall GZA); RGW=$(extwall RG)
echo "### WALL medians (ns)  GZ=$GZW  GZ_AA=$GZAW  RG=$RGW"
echo "AA_ratio(GZ/GZ): $(awk -v a=$GZW -v b=$GZAW 'BEGIN{printf "%.4f", a/b}')"
echo "gz/rg wall:      $(awk -v a=$GZW -v b=$RGW 'BEGIN{printf "%.4f", a/b}')"
rm -rf "$OUT"
