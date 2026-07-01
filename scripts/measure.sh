#!/usr/bin/env bash
# measure.sh — the ONE trustworthy way to compare decode throughput.
#
# Encodes the methodology rules the x86 post-mortem found we violated repeatedly:
#   1. NEVER report a bare absolute MB/s — they swing 2x with box load and have
#      fooled this project into opposite conclusions from the same binary.
#      Report only the INTERLEAVED RELATIVE delta (jitter-immune: both tools see
#      the same per-trial contention, so the ratio is stable even on a loaded box).
#   2. ALWAYS output-sha256-verify every run; a speed "win" with wrong bytes is a loss.
#   3. A delta inside the trial-to-trial spread is a TIE, not a win/loss.
#   4. Flag (don't hide) a contended box so absolutes aren't mistaken for signal.
#
# Usage:
#   RAW=<uncompressed_bytes> N=11 CPUS=0,2,4,6,8,10,12,14 \
#     scripts/measure.sh "gzippy=/path/gzippy -d -c -p 8 file.gz" \
#                        "rapidgzip=/path/rapidgzip -d -c -f -P 8 file.gz"
# First contender is the correctness REFERENCE; every run's stdout sha256 must match it.
set -u
N="${N:-11}"; RAW="${RAW:-0}"
TS=""; [ -n "${CPUS:-}" ] && TS="taskset -c $CPUS"

[ "$#" -ge 2 ] || { echo "need >=2 'label=command' contenders" >&2; exit 2; }
declare -a L; declare -A C
for s in "$@"; do case "$s" in *=*) L+=("${s%%=*}"); C["${s%%=*}"]="${s#*=}";; *) echo "bad: $s">&2; exit 2;; esac; done

load=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
ncpu=$(nproc 2>/dev/null || echo 1)
busy=$(awk -v l="$load" -v n="$ncpu" 'BEGIN{print (l > n*0.5)?1:0}')
[ "$busy" = 1 ] && echo "## WARN: loadavg $load on $ncpu cpus — absolutes are noise; trust ONLY the relative delta below." >&2

declare -A T; for l in "${L[@]}"; do T[$l]=""; done
ref="${L[0]}"; refsum=""; diverged=0
for ((i=1;i<=N;i++)); do for l in "${L[@]}"; do
  out=$(mktemp)
  s=$(date +%s.%N); $TS ${C[$l]} >"$out" 2>/dev/null; rc=$?; e=$(date +%s.%N)
  [ $rc -eq 0 ] || echo "!! $l trial $i exit $rc" >&2
  T[$l]="${T[$l]} $(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')"
  sum=$(sha256sum "$out" | cut -d' ' -f1)
  [ "$l" = "$ref" ] && [ -z "$refsum" ] && refsum="$sum"
  [ "$sum" != "$refsum" ] && { echo "!! CORRECTNESS DIVERGENCE: $l trial $i" >&2; diverged=1; }
  rm -f "$out"
done; done

echo "===== measure  N=$N  CPUS=${CPUS:-unpinned}  load=$load  output-verified=$([ $diverged = 0 ] && echo OK || echo FAIL) ====="
declare -A BEST SPREAD
for l in "${L[@]}"; do
  read best worst < <(echo "${T[$l]}" | tr ' ' '\n' | grep -v '^$' | sort -n | awk 'NR==1{b=$1} {w=$1} END{print b, w}')
  BEST[$l]=$best; SPREAD[$l]=$(awk -v b="$best" -v w="$worst" 'BEGIN{printf "%.0f", (w-b)/b*100}')
  mb=$(awk -v r="$RAW" -v t="$best" 'BEGIN{print (t>0 && r>0)?sprintf("%.0f MB/s", r/t/1e6):"n/a"}')
  printf "  %-12s best=%ss  %s  (spread %s%%)\n" "$l" "$best" "$mb" "${SPREAD[$l]}"
done
echo "--- RELATIVE (the trustworthy signal) ---"
rb="${BEST[$ref]}"; rsp="${SPREAD[$ref]}"
for l in "${L[@]}"; do [ "$l" = "$ref" ] && continue
  ratio=$(awk -v a="$rb" -v b="${BEST[$l]}" 'BEGIN{printf "%.3f", a/b}')   # ref_time/other_time = other_throughput/ref_throughput
  # within max(spread) => TIE
  margin=$(awk -v a="$rsp" -v b="${SPREAD[$l]}" 'BEGIN{m=(a>b)?a:b; print m/100.0}')
  verdict=$(awk -v r="$ratio" -v m="$margin" 'BEGIN{d=r-1; if(d>m)print "WIN (" l ")"; else if(d<-m)print "LOSS"; else print "TIE"}' l="$l")
  printf "  %-12s %sx vs %s   => %s\n" "$l" "$ratio" "$ref" "$verdict"
done
[ $diverged = 0 ] || echo "  !! RESULT INVALID: output diverged — fix correctness before trusting speed."
