#!/usr/bin/env bash
# whole_view.sh — FULL-SYSTEM view of the gzippy-vs-rapidgzip decode gap.
#
# Optimize the WHOLE, not parts. This prints, in ONE shot, the complete wall +
# cycle budget of BOTH tools so every lever is placed in the global picture and
# every change is verified on the whole wall — never on a single component %.
# Re-run after each lever; the gap is closed only when the WHOLE budget matches.
#
# Run on the bench host (x86 + perf). Usage:
#   whole_view.sh <gzippy-bin> <rapidgzip-bin> <file.gz>
# Env: PIN (taskset spec), ROUNDS (interleaved rounds, default 5).
set -u
G="${1:?gzippy bin}"; RG="${2:?rapidgzip bin}"; SLG="${3:?file.gz}"
PIN="${PIN:-taskset -c 0,2,4,6,8,10,12,14}"
ROUNDS="${ROUNDS:-5}"
GZ="$G -d -c -p 8 $SLG"; RGC="$RG -d -c -f -P 8 $SLG"

echo "############ THE WHOLE — gzippy vs rapidgzip — $(basename "$SLG") T8 ############"

echo "== 1. WALL (interleaved best-of-$ROUNDS, sha-verified — the ONLY verdict) =="
sg=$($PIN $GZ 2>/dev/null | sha256sum | cut -d' ' -f1)
sr=$($PIN $RGC 2>/dev/null | sha256sum | cut -d' ' -f1)
[ "$sg" = "$sr" ] && echo "   sha: MATCH (byte-identical) OK" || echo "   sha: MISMATCH gz=$sg rg=$sr  <<< CORRECTNESS BROKEN"
gb=99; rb=99
for t in $(seq 1 "$ROUNDS"); do
  s=$(date +%s.%N); $PIN $GZ  >/dev/null 2>&1; e=$(date +%s.%N); g=$(awk "BEGIN{print $e-$s}")
  s=$(date +%s.%N); $PIN $RGC >/dev/null 2>&1; e=$(date +%s.%N); r=$(awk "BEGIN{print $e-$s}")
  gb=$(awk -v a=$gb -v b=$g 'BEGIN{print (b<a)?b:a}'); rb=$(awk -v a=$rb -v b=$r 'BEGIN{print (b<a)?b:a}')
done
awk -v g=$gb -v r=$rb 'BEGIN{printf "   gzippy=%.3fs  rapidgzip=%.3fs  RATIO=%.3fx  (parity=1.000)\n",g,r,g/r}'

echo "== 2. GAP STRUCTURE (instr / IPC / parallelism — is the gap WORK or STALLS?) =="
for tag in "gzippy :$GZ" "rapidgz :$RGC"; do
  nm=${tag%%:*}; cmd=${tag#*:}
  perf stat -e instructions,task-clock $PIN $cmd >/dev/null 2>/tmp/wv.txt
  ins=$(grep -E "instructions" /tmp/wv.txt | grep -oE "[0-9,]{6,}" | head -1 | tr -d ,)
  ipc=$(grep -oE "[0-9]+\.[0-9]+  insn per cycle" /tmp/wv.txt | grep -oE "^[0-9.]+")
  cpu=$(grep -oE "[0-9]+\.[0-9]+ CPUs utilized" /tmp/wv.txt | grep -oE "^[0-9.]+")
  tc=$(grep -oE "[0-9,]+\.[0-9]+ msec task-clock" /tmp/wv.txt | grep -oE "[0-9,]+\.[0-9]+" | head -1 | tr -d ,)
  printf "   %s instr=%-13s IPC=%-5s CPUs=%-5s task-clock=%sms\n" "$nm" "${ins:-?}" "${ipc:-?}" "${cpu:-?}" "${tc:-?}"
done

echo "== 3. CYCLE BUDGET (where every cycle goes, both tools — the global map) =="
echo "   --- gzippy ---"
perf record -F 2500 -e cycles:pp -o /tmp/wvg.data $PIN $GZ  >/dev/null 2>/dev/null
perf report -i /tmp/wvg.data --stdio --percent-limit 1.5 2>/dev/null | grep -vE "^#|^$" | head -9 | sed 's/^/   /'
echo "   --- rapidgzip ---"
perf record -F 2500 -e cycles:pp -o /tmp/wvr.data $PIN $RGC >/dev/null 2>/dev/null
perf report -i /tmp/wvr.data --stdio --percent-limit 1.5 2>/dev/null | grep -vE "^#|^$" | head -9 | sed 's/^/   /'

echo "== 4. gzippy PHASE WALL (trace_v2 spans — gzippy's own whole) =="
GZIPPY_LOG_FILE=/tmp/wv.log $PIN $GZ >/dev/null 2>&1
if [ -s /tmp/wv.log ]; then
  python3 scripts/parallel_sm_log_summary.py /tmp/wv.log 2>&1 | head -22 | sed 's/^/   /'
else
  echo "   (no trace; build emits GZIPPY_LOG_FILE only with trace_v2 enabled)"
fi
echo "##############################################################################"
