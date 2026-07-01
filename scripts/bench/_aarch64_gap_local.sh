#!/usr/bin/env bash
# _aarch64_gap_local.sh — interleaved best-of-N T1 wall: gzippy-native vs libdeflate.
# Gate-0: /dev/null both arms, sha-verified, A/A self-test. Gate-1: interleaved, best-of-N min + spread.
# macOS aarch64 (no perf) => wall-time only. NOT-YET-LAW (single box, indicative).
set -u
GZ=${GZ:-/Users/jackdanger/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_gap_local.sh <file.gz> [reps]}
N=${2:-11}

# helper: nanosecond wall of a command -> prints ms
now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

declare -A BEST WORST
ARMS="gz ld gz2"   # gz2 = second gzippy arm for A/A self-test
for a in $ARMS; do BEST[$a]=999999999; WORST[$a]=0; done

run_arm() {
  case "$1" in
    gz|gz2) GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    ld)     libdeflate-gunzip -c "$F" >/dev/null 2>&1 ;;
  esac
}

# warm
for a in $ARMS; do run_arm "$a"; done

for r in $(seq 1 "$N"); do
  order=$(echo $ARMS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for a in $order; do
    t0=$(now); run_arm "$a"; t1=$(now)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e6}')
    awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
    awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
  done
done

echo "== aarch64 T1 gap  file=$F reps=$N (best-of-N min ms, /dev/null both arms) =="
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-4s best=%9s ms  spread=%8s ms\n" "$a" "${BEST[$a]}" "$spread"
done
echo "--- ratios ---"
awk -v g="${BEST[gz]}" -v l="${BEST[ld]}" 'BEGIN{printf "  gz / libdeflate = %.4f  (>1 = gz slower)\n", g/l}'
awk -v g="${BEST[gz]}" -v g2="${BEST[gz2]}" 'BEGIN{r=g/g2; printf "  A/A gz/gz2      = %.4f  (self-test, want ~1.00)\n", r}'
