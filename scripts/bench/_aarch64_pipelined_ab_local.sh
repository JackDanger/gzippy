#!/usr/bin/env bash
# _aarch64_pipelined_ab_local.sh — ROUTE-B gated AB: the production pipelined
# fastloop (default) vs the legacy pack-8 baseline (GZIPPY_BASELINE_KERNEL=1),
# both vs libdeflate-gunzip, on ONE binary (env toggles the kernel — controls for
# build variance). Gate-0: /dev/null all arms, sha-verified up front, A/A self-test
# (pipe vs pipe2). Gate-1: interleaved, randomized per-rep order, best-of-N min + spread.
# macOS aarch64 (no perf) => wall-time only. NOT-YET-LAW (single box, indicative).
set -u
GZ=${GZ:-/Users/jackdanger/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_pipelined_ab_local.sh <file.gz> [reps]}
N=${2:-15}

now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

# Gate-0 sha self-check (all arms must equal libdeflate).
REF=$(libdeflate-gunzip -c "$F" | shasum | cut -d' ' -f1)
P=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
B=$(GZIPPY_BASELINE_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
if [ "$REF" != "$B" ] || [ "$REF" != "$P" ]; then
  echo "GATE-0 FAIL: sha mismatch ref=$REF base=$B pipe=$P"; exit 1
fi

declare -A BEST WORST
ARMS="pipe base ld pipe2"
for a in $ARMS; do BEST[$a]=999999999; WORST[$a]=0; done

run_arm() {
  case "$1" in
    pipe|pipe2) GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    base)       GZIPPY_BASELINE_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    ld)         libdeflate-gunzip -c "$F" >/dev/null 2>&1 ;;
  esac
}

for a in $ARMS; do run_arm "$a"; done   # warm

for r in $(seq 1 "$N"); do
  order=$(echo $ARMS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for a in $order; do
    t0=$(now); run_arm "$a"; t1=$(now)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e6}')
    awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
    awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
  done
done

echo "== ROUTE-B pipelined AB  file=$F reps=$N (best-of-N min ms, /dev/null all arms) =="
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-6s best=%9s ms  spread=%8s ms\n" "$a" "${BEST[$a]}" "$spread"
done
echo "--- ratios ---"
awk -v p="${BEST[pipe]}" -v b="${BEST[base]}" 'BEGIN{printf "  pipe / base       = %.4f  (<1 = pipelined faster)\n", p/b}'
awk -v b="${BEST[base]}" -v l="${BEST[ld]}" 'BEGIN{printf "  base / libdeflate = %.4f\n", b/l}'
awk -v p="${BEST[pipe]}" -v l="${BEST[ld]}" 'BEGIN{printf "  pipe / libdeflate = %.4f  (PRODUCTION)\n", p/l}'
awk -v p="${BEST[pipe]}" -v p2="${BEST[pipe2]}" 'BEGIN{printf "  A/A pipe/pipe2    = %.4f  (self-test, want ~1.00)\n", p/p2}'
