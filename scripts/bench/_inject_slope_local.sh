#!/usr/bin/env bash
# _inject_slope_local.sh — Gate-2 sub-region wall-slope injector sweep (aarch64).
# For one region knob, sweep N values, interleaved best-of-N min wall vs libdeflate.
# Proportional slope (wall rises with N) => region ON the T1 critical path; flat => slack.
# Gate-0: sha-verified byte-exact at every N (done separately); /dev/null sink.
set -u
GZ=${GZ:-/home/user/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _inject_slope_local.sh <file.gz> <KNOB> [reps] [Nvalues...]}
KNOB=${2:?need knob e.g. GZIPPY_INJ_LIT}
REPS=${3:-9}
shift 3 || true
NVALS=${*:-0 4 8 16 32}

now() { python3 -c 'import time;print(time.perf_counter_ns())'; }
run() { env "$KNOB=$1" GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1; }

declare -A BEST WORST
for n in $NVALS; do BEST[$n]=999999999; WORST[$n]=0; done
# warm
for n in $NVALS; do run "$n"; done

for r in $(seq 1 "$REPS"); do
  order=$(echo $NVALS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for n in $order; do
    t0=$(now); run "$n"; t1=$(now)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",(b-a)/1e6}')
    awk -v m="$ms" -v b="${BEST[$n]}" 'BEGIN{exit !(m<b)}' && BEST[$n]=$ms
    awk -v m="$ms" -v w="${WORST[$n]}" 'BEGIN{exit !(m>w)}' && WORST[$n]=$ms
  done
done

echo "== inject slope  file=$F knob=$KNOB reps=$REPS =="
b0=""
for n in $NVALS; do
  sp=$(awk -v b="${BEST[$n]}" -v w="${WORST[$n]}" 'BEGIN{printf "%.2f",w-b}')
  [ -z "$b0" ] && b0=${BEST[$n]}
  d=$(awk -v b="${BEST[$n]}" -v z="$b0" 'BEGIN{printf "%+.2f",b-z}')
  printf "  N=%-3s best=%9s ms  spread=%6s ms  Δvs(N=%s)=%s ms\n" "$n" "${BEST[$n]}" "$sp" "${NVALS%% *}" "$d"
done
