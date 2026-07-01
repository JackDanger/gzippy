#!/usr/bin/env bash
# _aarch64_thint1_ab_local.sh — ROUTE-A gated AB: the production thin-T1 driver
# (scaffold-shed, DEFAULT) vs the legacy parallel-pipeline T1 (GZIPPY_NO_THIN_T1=1),
# both vs libdeflate-gunzip. Single production binary; env toggles the T1 spine.
# Both arms include CRC32+ISIZE verification (production correctness) and the
# pipelined kernel (Task 1). thin/ld is the COMBINED Task1+Task2 production number.
# Gate-0: /dev/null all arms, sha-verified up front, A/A self-test. Gate-1:
# interleaved best-of-N min + spread AND a paired sign test (cancels per-rep
# contention). macOS aarch64 (no perf) => wall-only. NOT-YET-LAW (single box).
set -u
GZ=${GZ:-/Users/jackdanger/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_thint1_ab_local.sh <file.gz> [reps]}
N=${2:-15}
now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

REF=$(libdeflate-gunzip -c "$F" | shasum | cut -d' ' -f1)
T=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
B=$(GZIPPY_NO_THIN_T1=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
[ "$REF" = "$T" ] && [ "$REF" = "$B" ] || { echo "GATE-0 FAIL sha ref=$REF thin=$T base=$B"; exit 1; }

run_thin() { GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1; }
run_base() { GZIPPY_NO_THIN_T1=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1; }
run_ld()   { libdeflate-gunzip -c "$F" >/dev/null 2>&1; }
ms() { local t0 t1; t0=$(now); "$1"; t1=$(now); awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",(b-a)/1e6}'; }

declare -A BEST WORST
ARMS="thin base ld thin2"
for a in $ARMS; do BEST[$a]=999999999; WORST[$a]=0; done
runarm() { case "$1" in thin|thin2) run_thin;; base) run_base;; ld) run_ld;; esac; }
for a in $ARMS; do runarm "$a"; done   # warm

DELTAS=""
for r in $(seq 1 "$N"); do
  order=$(echo $ARMS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for a in $order; do
    t0=$(now); runarm "$a"; t1=$(now)
    t=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",(b-a)/1e6}')
    awk -v m="$t" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$t
    awk -v m="$t" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$t
  done
  # paired delta base-thin in the same rep
  if [ $((RANDOM % 2)) -eq 0 ]; then b=$(ms run_base); t=$(ms run_thin); else t=$(ms run_thin); b=$(ms run_base); fi
  DELTAS="$DELTAS $(awk -v b="$b" -v t="$t" 'BEGIN{printf "%.3f", b-t}')"
done

echo "== ROUTE-A thin-T1 AB  file=$F reps=$N (best-of-N min ms, /dev/null all arms) =="
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-6s best=%9s ms  spread=%8s ms\n" "$a" "${BEST[$a]}" "$spread"
done
echo "--- ratios ---"
awk -v t="${BEST[thin]}" -v b="${BEST[base]}" 'BEGIN{printf "  thin / base       = %.4f  (<1 = thin faster)\n", t/b}'
awk -v b="${BEST[base]}" -v l="${BEST[ld]}" 'BEGIN{printf "  base / libdeflate = %.4f  (legacy parallel T1)\n", b/l}'
awk -v t="${BEST[thin]}" -v l="${BEST[ld]}" 'BEGIN{printf "  thin / libdeflate = %.4f  (PRODUCTION Task1+Task2)\n", t/l}'
awk -v t="${BEST[thin]}" -v t2="${BEST[thin2]}" 'BEGIN{printf "  A/A thin/thin2    = %.4f  (self-test, want ~1.00)\n", t/t2}'
echo "$DELTAS" | tr ' ' '\n' | grep . | python3 -c '
import sys
d=[float(x) for x in sys.stdin if x.strip()]; d.sort(); n=len(d)
med=d[n//2] if n%2 else (d[n//2-1]+d[n//2])/2
pos=sum(1 for x in d if x>0)
print(f"  paired base-thin: median={med:.3f} ms  thin-faster {pos}/{n} reps  min={d[0]:.3f} max={d[-1]:.3f}")
'
