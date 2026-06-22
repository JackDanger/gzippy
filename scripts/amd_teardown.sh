#!/usr/bin/env bash
# Gate-2 proportionality: teardown = full process-wall - instrument userspace-wall
# (main_start->main_end), measured WITHOUT perf (bash time, %e) to avoid perf
# teardown overhead. If teardown scales monotonically+proportionally with peak RSS
# across corpus sizes, the kernel address-space teardown of gz's RSS is the
# gz-vs-rg excess (perturbation knob = output/RSS size).
set -u
GZ="${GZ:-/dev/shm/gztgt/release/gzippy}"
RG="${RG:-/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"
MASK="${MASK:-8,10,12,14}"; N="${N:-11}"
med() { sort -n | awk '{a[NR]=$1} END{m=int((NR+1)/2); if(NR%2){print a[m]} else {printf "%.3f\n",(a[m]+a[m+1])/2}}'; }
run() {
  local F="$1"; local T="$2"; local label="$3"
  local OUT=$(mktemp -d)
  for r in $(seq 1 "$N"); do
    # full wall via /usr/bin/time -f %e (seconds); iw from instrument stderr
    /usr/bin/time -f "%e" env GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" \
      "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$OUT/$r"
    fw=$(tail -1 "$OUT/$r" | awk '{printf "%.3f", $1*1000}')
    iw=$(grep "wall(first" "$OUT/$r" | grep -oE '=[0-9]+\.[0-9]+ms' | grep -oE '[0-9]+\.[0-9]+')
    awk -v a=$fw -v b=$iw 'BEGIN{printf "%.3f\n", a-b}' >>"$OUT/td"
    echo "$iw" >>"$OUT/iw"
    echo "$fw" >>"$OUT/fw"
  done
  local rss=$(/usr/bin/time -v env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p"$T" "$F" 2>&1 >/dev/null | grep "Maximum resident" | grep -oE '[0-9]+')
  echo "$label F=$F T=$T : full_wall=$(med<"$OUT/fw")ms iw=$(med<"$OUT/iw")ms teardown(fw-iw)=$(med<"$OUT/td")ms peakRSS=${rss}KB"
  rm -rf "$OUT"
}
run /root/corpora/monorepo.tar.gz 2 monorepo
run /root/corpora/silesia.gz 2 silesia
# rg exec floors for comparison (no instrument; full wall on tiny + monorepo)
printf "tiny\n"|gzip -c>/tmp/tiny.gz
rtiny=$(for r in $(seq 1 "$N"); do /usr/bin/time -f "%e" taskset -c "$MASK" "$RG" -d -c -P2 /tmp/tiny.gz 2>&1 >/dev/null | tail -1 | awk '{printf "%.3f\n",$1*1000}'; done | med)
gtiny=$(for r in $(seq 1 "$N"); do /usr/bin/time -f "%e" env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p2 /tmp/tiny.gz 2>&1 >/dev/null | tail -1 | awk '{printf "%.3f\n",$1*1000}'; done | med)
echo "tiny full-wall: gz=${gtiny}ms rg=${rtiny}ms (exec floor incl runtime init+teardown of small RSS)"
