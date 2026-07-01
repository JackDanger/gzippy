#!/usr/bin/env bash
# _aarch64_pipelined_paired_local.sh — PAIRED significance for the ROUTE-B AB.
# Per rep, run base and pipe (randomized order within the rep) back-to-back and
# record delta = base_ms - pipe_ms. Pairing cancels per-rep OS contention (both
# arms see the same machine state in the same rep), which the best-of-N spread
# cannot. Reports: median delta, % reps pipe-faster (sign test), and a PAIRED
# A/A control (base vs base2) for the noise floor. Gate-0 sha checked up front.
set -u
GZ=${GZ:-/Users/jackdanger/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_pipelined_paired_local.sh <file.gz> [reps]}
N=${2:-21}
now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

REF=$(libdeflate-gunzip -c "$F" | shasum | cut -d' ' -f1)
B=$(GZIPPY_BASELINE_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
P=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
[ "$REF" = "$B" ] && [ "$REF" = "$P" ] || { echo "GATE-0 FAIL sha ref=$REF base=$B pipe=$P"; exit 1; }

run_base() { GZIPPY_BASELINE_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1; }
run_pipe() { GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1; }
ms() { local t0 t1; t0=$(now); "$1"; t1=$(now); awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",(b-a)/1e6}'; }

run_base; run_pipe   # warm
DELTAS=""; AA=""
for r in $(seq 1 "$N"); do
  if [ $((RANDOM % 2)) -eq 0 ]; then b=$(ms run_base); p=$(ms run_pipe); else p=$(ms run_pipe); b=$(ms run_base); fi
  DELTAS="$DELTAS $(awk -v b="$b" -v p="$p" 'BEGIN{printf "%.3f", b-p}')"
  # paired A/A control: base vs base again
  if [ $((RANDOM % 2)) -eq 0 ]; then a1=$(ms run_base); a2=$(ms run_base); else a2=$(ms run_base); a1=$(ms run_base); fi
  AA="$AA $(awk -v a="$a1" -v b="$a2" 'BEGIN{printf "%.3f", a-b}')"
done

echo "== ROUTE-B paired AB (base-pipe per rep)  file=$F reps=$N =="
echo "$DELTAS" | tr ' ' '\n' | grep . | python3 -c '
import sys
d=[float(x) for x in sys.stdin if x.strip()]
d.sort()
n=len(d); med=d[n//2] if n%2 else (d[n//2-1]+d[n//2])/2
pos=sum(1 for x in d if x>0)
print(f"  base-pipe: median={med:.3f} ms  pipe-faster {pos}/{n} reps  min={d[0]:.3f} max={d[-1]:.3f}")
'
echo "$AA" | tr ' ' '\n' | grep . | python3 -c '
import sys
d=[float(x) for x in sys.stdin if x.strip()]
d.sort()
n=len(d); med=d[n//2] if n%2 else (d[n//2-1]+d[n//2])/2
absmed=sorted(abs(x) for x in d)[n//2]
print(f"  A/A ctrl : median={med:.3f} ms  |median|={absmed:.3f} ms (noise floor)")
'
