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
ROUNDS="${ROUNDS:-7}"
GZ="$G -d -c -p 8 $SLG"; RGC="$RG -d -c -f -P 8 $SLG"

echo "############ THE WHOLE — gzippy vs rapidgzip — $(basename "$SLG") T8 ############"
echo "REGIME (known, ledger): CONSUMER-CP_dep-bound + overlapped => decode-rate wall-DEAD."
echo "       => Section 3 is CPU-TIME (hypotheses), NOT wall. Verdict = Section 1 (WALL)."

echo "== 1. WALL — sha-verified, interleaved best-of-$ROUNDS, ABSOLUTE+ratio (the ONLY verdict) =="
sg=$($PIN $GZ 2>/dev/null | sha256sum | cut -d' ' -f1)
sr=$($PIN $RGC 2>/dev/null | sha256sum | cut -d' ' -f1)
[ "$sg" = "$sr" ] && echo "   sha: MATCH (byte-identical) OK" || echo "   sha: MISMATCH gz=$sg rg=$sr  <<< CORRECTNESS BROKEN"
OUT=$($PIN $GZ 2>/dev/null | wc -c)   # decompressed bytes -> absolute MB/s
# self-test 1: the timer must bracket the process (a known +0.3s sleep must appear).
s=$(date +%s.%N); $PIN $GZ >/dev/null 2>&1; e=$(date +%s.%N); base=$(awk "BEGIN{print $e-$s}")
s=$(date +%s.%N); sleep 0.3; $PIN $GZ >/dev/null 2>&1; e=$(date +%s.%N); slow=$(awk "BEGIN{print $e-$s}")
awk -v a=$base -v b=$slow 'BEGIN{d=b-a; if(d>0.2 && d<0.45) print "   timer self-test: OK (+0.3s -> +"d"s)"; else print "   timer self-test: SUSPECT (+0.3s -> +"d"s) <<< timing unreliable"}'
# self-test 2: a binary vs ITSELF must read ~1.0, else the box is too noisy to trust ANY ratio.
s=$(date +%s.%N); $PIN $GZ >/dev/null 2>&1; e=$(date +%s.%N); x=$(awk "BEGIN{print $e-$s}")
awk -v a=$base -v b=$x 'BEGIN{r=(a>b)?a/b:b/a; if(r<1.15) printf "   self-test (gz vs gz): %.3f OK\n",r; else printf "   self-test (gz vs gz): %.3f <<< BOX TOO NOISY (>15%%) — rows INCONCLUSIVE; freeze host\n",r}'
gb=99; gx=0; rb=99
for t in $(seq 1 "$ROUNDS"); do
  s=$(date +%s.%N); $PIN $GZ  >/dev/null 2>&1; e=$(date +%s.%N); g=$(awk "BEGIN{print $e-$s}")
  s=$(date +%s.%N); $PIN $RGC >/dev/null 2>&1; e=$(date +%s.%N); r=$(awk "BEGIN{print $e-$s}")
  gb=$(awk -v a=$gb -v b=$g 'BEGIN{print (b<a)?b:a}'); gx=$(awk -v a=$gx -v b=$g 'BEGIN{print (b>a)?b:a}'); rb=$(awk -v a=$rb -v b=$r 'BEGIN{print (b<a)?b:a}')
done
awk -v g=$gb -v gx=$gx -v r=$rb -v out=$OUT 'BEGIN{
  gmb=out/g/1e6; rmb=out/r/1e6; sp=(gx-g)/g*100;
  printf "   gzippy=%.0f MB/s (best %.3fs, spread %.1f%%)   rapidgzip=%.0f MB/s   RATIO=%.3fx (parity=1.0)\n",gmb,g,sp,rmb,g/r;
  printf "   PROGRESS RULE: WIN only if gzippy MB/s ROSE vs last row AND ratio fell by > spread(%.1f%%); else TIE / rival-regressed.\n",sp
}'
# append-only trajectory log (emitted by the instrument, not hand-edited)
echo "$(date +%FT%H:%M),$(git rev-parse --short HEAD 2>/dev/null),$gb,$rb,$(awk -v g=$gb -v r=$rb 'BEGIN{printf "%.3f",g/r}'),$(awk -v g=$gb -v gx=$gx 'BEGIN{printf "%.1f",(gx-g)/g*100}')" >> plans/wall-progress.csv 2>/dev/null

echo "== 2. GAP STRUCTURE (instr / IPC / parallelism — work vs stalls vs cores) =="
for which in gz rg; do
  if [ "$which" = gz ]; then nm="gzippy "; cmd="$GZ"; bw=$gb; else nm="rapidgz"; cmd="$RGC"; bw=$rb; fi
  perf stat -e instructions,cycles,task-clock $PIN $cmd >/dev/null 2>/tmp/wv.txt
  ins=$(grep -E "instructions" /tmp/wv.txt | grep -oE "[0-9,]{8,}" | head -1 | tr -d ',')
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
