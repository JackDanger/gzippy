#!/usr/bin/env bash
# Per-run: process-wall (perf duration_time) vs instrument decode-wall, to isolate
# any serial wrapper OUTSIDE decode_entry..crc_verified (CLI init / mmap / routing
# before decode_entry; process teardown after crc_verified). Same run, no cross-run
# noise. Also reports rg process-wall for comparison.
set -u
GZ="${GZ:-/dev/shm/gztgt/release/gzippy}"
RG="${RG:-/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"
F="$1"; T="$2"; MASK="$3"; N="${4:-9}"
OUT=$(mktemp -d)
med() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "NA"; exit} m=int((NR+1)/2); if(NR%2){print a[m]} else {printf "%.3f\n",(a[m]+a[m+1])/2} }'; }
for r in $(seq 1 "$N"); do
  taskset -c "$MASK" perf stat -x, -e duration_time -- env GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
    "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/g.$r.txt"
  # process wall ns -> ms
  pw=$(grep ",duration_time" "$OUT/g.$r.txt" | head -1 | cut -d, -f1)
  pw_ms=$(awk -v x=$pw 'BEGIN{printf "%.3f", x/1e6}')
  dw=$(grep decode_wall "$OUT/g.$r.txt" | grep -oE '=[0-9]+\.[0-9]+ms' | grep -oE '[0-9]+\.[0-9]+')
  echo "$pw_ms $dw" | awk '{printf "%.3f\n", $1-$2}' >> "$OUT/gap.txt"
  echo "$pw_ms" >> "$OUT/pw.txt"
  echo "$dw"   >> "$OUT/dw.txt"
done
for r in $(seq 1 "$N"); do
  taskset -c "$MASK" perf stat -x, -e duration_time -- taskset -c "$MASK" "$RG" -d -c -P"$T" "$F" >/dev/null 2>"$OUT/r.$r.txt"
  rw=$(grep ",duration_time" "$OUT/r.$r.txt" | head -1 | cut -d, -f1)
  awk -v x=$rw 'BEGIN{printf "%.3f\n", x/1e6}' >> "$OUT/rw.txt"
done
echo "corpus=$F T=$T mask=$MASK N=$N"
echo "GZ process-wall ms (med): $(cat $OUT/pw.txt | med)"
echo "GZ decode-wall  ms (med): $(cat $OUT/dw.txt | med)"
echo "GZ PRE/POST gap ms (med): $(cat $OUT/gap.txt | med)   <- wall outside decode_entry..crc_verified"
echo "RG process-wall ms (med): $(cat $OUT/rw.txt | med)"
rm -rf "$OUT"
