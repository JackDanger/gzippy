#!/usr/bin/env bash
# whole_view.sh — FULL-SYSTEM view of the gzippy-vs-rapidgzip decode gap.
#
# Optimize the WHOLE, not parts. One command, both tools, the complete budget,
# verified on the WHOLE WALL — never a single component %.
#
# *** READ THIS (advisor review a54331c, 2026-05-30) ***
# This pipeline is CONSUMER-CP_dep-BOUND and pipeline-OVERLAPPED (falsification
# ledger): decode RATE is wall-DEAD. Therefore SECTION 3 (cycle budget) ranks
# CPU-TIME, which is NOT wall-time here — a sink can be 33% of cycles and 0% of
# the wall. Use section 3 only to find *candidate* code (e.g. the back-ref
# memmove); CONFIRM every lever on section 1 (WALL) + section 4 (CP_dep). A high
# % in section 3 is a HYPOTHESIS, not a verdict.
#
# TODO (advisor punch-list, in priority order — not yet implemented here):
#  1. Section 4 -> trace_v2/GZIPPY_TIMELINE + timeline_analyze.py: emit the
#     consumer-tid critical chain = CP_dep (the law that's live for this gap).
#  3. Reconciliation: WALL_GAP = CP_dep_delta + CP_res_delta + residual + e_model.
#  5. CP_res roofline: add LLC-load-misses / IMC cas_count + host B_DRAM constant.
#  (6 symbolize gzippy, 7 headroom projection — lower priority; section 3 is
#   CPU-time so naming a sink still doesn't make it wall.)
#
# Run on the x86 bench host (perf). Usage:
#   whole_view.sh <gzippy-bin> <rapidgzip-bin> <file.gz>
# Env: PIN (taskset spec), ROUNDS (interleaved rounds, default 5).
set -u
G="${1:?gzippy bin}"; RG="${2:?rapidgzip bin}"; SLG="${3:?file.gz}"
PIN="${PIN:-taskset -c 0,2,4,6,8,10,12,14}"
ROUNDS="${ROUNDS:-5}"
GZ="$G -d -c -p 8 $SLG"; RGC="$RG -d -c -f -P 8 $SLG"

echo "############ THE WHOLE — gzippy vs rapidgzip — $(basename "$SLG") T8 ############"
echo "REGIME (known, ledger): CONSUMER-CP_dep-bound + overlapped => decode-rate wall-DEAD."
echo "       => Section 3 is CPU-TIME (hypotheses), NOT wall. Verdict = Section 1 (WALL)."

echo "== 1. WALL (interleaved best-of-$ROUNDS, sha-verified — the ONLY verdict) =="
sg=$($PIN $GZ 2>/dev/null | sha256sum | cut -d' ' -f1)
sr=$($PIN $RGC 2>/dev/null | sha256sum | cut -d' ' -f1)
[ "$sg" = "$sr" ] && echo "   sha: MATCH (byte-identical) OK" || echo "   sha: MISMATCH gz=$sg rg=$sr  <<< CORRECTNESS BROKEN"
# Positive control: a timer that doesn't bracket the process is the oracle-class
# failure. Assert a known +0.3s sleep shows up in the measured wall.
s=$(date +%s.%N); $PIN $GZ >/dev/null 2>&1; e=$(date +%s.%N); base=$(awk "BEGIN{print $e-$s}")
s=$(date +%s.%N); sleep 0.3; $PIN $GZ >/dev/null 2>&1; e=$(date +%s.%N); slow=$(awk "BEGIN{print $e-$s}")
awk -v a=$base -v b=$slow 'BEGIN{d=b-a; if(d>0.2 && d<0.45) print "   timer self-test: OK (+0.3s sleep -> +"d"s)"; else print "   timer self-test: SUSPECT (+0.3s sleep -> +"d"s) <<< section 1 timing unreliable"}'
gb=99; rb=99
for t in $(seq 1 "$ROUNDS"); do
  s=$(date +%s.%N); $PIN $GZ  >/dev/null 2>&1; e=$(date +%s.%N); g=$(awk "BEGIN{print $e-$s}")
  s=$(date +%s.%N); $PIN $RGC >/dev/null 2>&1; e=$(date +%s.%N); r=$(awk "BEGIN{print $e-$s}")
  gb=$(awk -v a=$gb -v b=$g 'BEGIN{print (b<a)?b:a}'); rb=$(awk -v a=$rb -v b=$r 'BEGIN{print (b<a)?b:a}')
done
awk -v g=$gb -v r=$rb 'BEGIN{printf "   gzippy=%.3fs  rapidgzip=%.3fs  RATIO=%.3fx  (parity=1.000)\n",g,r,g/r}'

echo "== 2. GAP STRUCTURE (instr / IPC / parallelism — work vs stalls vs cores) =="
for which in gz rg; do
  if [ "$which" = gz ]; then nm="gzippy "; cmd="$GZ"; bw=$gb; else nm="rapidgz"; cmd="$RGC"; bw=$rb; fi
  perf stat -e instructions,cycles,task-clock $PIN $cmd >/dev/null 2>/tmp/wv.txt
  ins=$(grep -E "^[[:space:]]*[0-9,]+[[:space:]]+instructions" /tmp/wv.txt | grep -oE "^[[:space:]]*[0-9,]+" | head -1 | tr -d ', ')
  ipc=$(grep -oE "#[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+insn per cycle" /tmp/wv.txt | grep -oE "[0-9]+\.[0-9]+" | head -1)
  tc=$(grep -oE "[0-9,]+\.[0-9]+ msec task-clock" /tmp/wv.txt | grep -oE "[0-9,]+\.[0-9]+" | head -1 | tr -d ,)
  # parallelism from the STABLE best-of-N wall (gb/rb), not perf's noisy single elapsed.
  cpu=$(awk -v t="${tc:-0}" -v w="$bw" 'BEGIN{ if(w>0) printf "%.2f",(t/1000)/w; else print "?" }')
  # plausibility guard (oracle defense): a parsed number that is insane prints SUSPECT.
  ok=$(awk -v i="${ins:-0}" -v t="${tc:-0}" 'BEGIN{ if(i>1e8 && t>1) print "ok"; else print "SUSPECT" }')
  printf "   %s instr=%-13s IPC=%-5s CPUs(best-wall)=%-5s task-clock=%sms  [%s]\n" "$nm" "${ins:-?}" "${ipc:-?}" "${cpu:-?}" "${tc:-?}" "$ok"
done

echo "== 3. CPU-TIME SINKS (NOT WALL — hypotheses only; overlapped work invisible) =="
echo "   (gzippy decode is fat-LTO -> hex frames; only libc/kernel are named. rapidgzip"
echo "    symbolizes fully. Do NOT read 'gzippy has no hot fn' from the hex — it's unsymbolized.)"
echo "   --- gzippy ---"
perf record -F 2500 -e cycles:pp -o /tmp/wvg.data $PIN $GZ  >/dev/null 2>/dev/null
perf report -i /tmp/wvg.data --stdio --percent-limit 1.5 2>/dev/null | grep -vE "^#|^$" | head -9 | sed 's/^/   /'
echo "   --- rapidgzip ---"
perf record -F 2500 -e cycles:pp -o /tmp/wvr.data $PIN $RGC >/dev/null 2>/dev/null
perf report -i /tmp/wvr.data --stdio --percent-limit 1.5 2>/dev/null | grep -vE "^#|^$" | head -9 | sed 's/^/   /'

echo "== 4. CP_dep — gzippy consumer critical path (TODO: trace_v2/timeline_analyze.py) =="
echo "   [advisor item #1] Until wired, this is the WEAK trace (counts, not phase-wall)."
GZIPPY_LOG_FILE=/tmp/wv.log $PIN $GZ >/dev/null 2>&1
if [ -s /tmp/wv.log ]; then
  python3 scripts/parallel_sm_log_summary.py /tmp/wv.log 2>&1 | head -14 | sed 's/^/   /'
else
  echo "   (no trace emitted by this build)"
fi
echo "##############################################################################"
