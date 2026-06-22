#!/usr/bin/env bash
# Truly interleaved per-rep: gz(perf+phase), gz(perf+phase A/A), rg(perf). Captures
# gz process-wall vs gz instrument-wall (main_start->main_end) vs phase split vs rg
# process-wall, all under the SAME contention pattern. Medians + excess breakdown.
set -u
GZ="${GZ:-/dev/shm/gztgt/release/gzippy}"
RG="${RG:-/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"
F="$1"; T="$2"; MASK="$3"; N="${4:-11}"
OUT=$(mktemp -d)
med() { sort -n | awk '{a[NR]=$1} END{m=int((NR+1)/2); if(NR%2){print a[m]} else {printf "%.3f\n",(a[m]+a[m+1])/2}}'; }
pw() { grep ",duration_time" "$1" | head -1 | cut -d, -f1 | awk '{printf "%.3f", $1/1e6}'; }
iw() { grep "wall(first->last)" "$1" | grep -oE '=[0-9]+\.[0-9]+ms' | grep -oE '[0-9]+\.[0-9]+'; }
ph() { grep -E " -> $2 " "$1" | head -1 | grep -oE '[0-9]+\.[0-9]+ms' | head -1 | sed 's/ms//'; }
for r in $(seq 1 "$N"); do
  taskset -c "$MASK" perf stat -x, -e duration_time -- env GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/g.$r"
  taskset -c "$MASK" perf stat -x, -e duration_time -- env GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/a.$r"
  taskset -c "$MASK" perf stat -x, -e duration_time -- taskset -c "$MASK" "$RG" -d -c -P"$T" "$F" >/dev/null 2>"$OUT/r.$r"
  pw "$OUT/g.$r" >>"$OUT/gpw"; echo >>"$OUT/gpw"
  pw "$OUT/a.$r" >>"$OUT/apw"; echo >>"$OUT/apw"
  pw "$OUT/r.$r" >>"$OUT/rpw"; echo >>"$OUT/rpw"
  iw "$OUT/g.$r" >>"$OUT/giw"
  ph "$OUT/g.$r" decode_entry >>"$OUT/pre"
  ph "$OUT/g.$r" main_end     >>"$OUT/post"
done
GPW=$(med<"$OUT/gpw"); APW=$(med<"$OUT/apw"); RPW=$(med<"$OUT/rpw"); GIW=$(med<"$OUT/giw"); PRE=$(med<"$OUT/pre"); POST=$(med<"$OUT/post")
echo "corpus=$F T=$T mask=$MASK N=$N (load-contended; interleaved A/A-gated)"
echo "gz process-wall ms : $GPW"
echo "gz A/A      wall ms : $APW   (A/A ratio $(awk -v a=$GPW -v b=$APW 'BEGIN{printf "%.4f",a/b}'))"
echo "rg process-wall ms : $RPW"
echo "gz/rg wall         : $(awk -v a=$GPW -v b=$RPW 'BEGIN{printf "%.4f",a/b}')  excess_ms=$(awk -v a=$GPW -v b=$RPW 'BEGIN{printf "%.2f",a-b}')"
echo "gz instrument-wall ms (main_start->main_end) : $GIW"
echo "  PRE  main_start->decode_entry ms : $PRE"
echo "  POST crc_verified->main_end   ms : $POST"
echo "gz harness+kernel-teardown ms (pw-iw) : $(awk -v a=$GPW -v b=$GIW 'BEGIN{printf "%.2f",a-b}')  (shared with rg harness)"
rm -rf "$OUT"
