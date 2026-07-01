#!/usr/bin/env bash
# =============================================================================
# interleaved_ab.sh — THE ONLY VALID WALL METRIC on the neurotic bench box.
# =============================================================================
# The box (i7-13700T inside Proxmox LXC 199) has ~15-20% run-to-run WALL jitter
# from bursty co-tenants. Absolute MB/s numbers are MEANINGLESS — they swing
# 1693/1576 -> 1416/1324 across batches. The per-trial-INTERLEAVED DELTA is
# rock-stable to <1% (proven: 3 batches, abs swung 20%, delta held +7.4/7.0/6.8%).
#
# So: run N>=5 trials; in EACH trial run every contender ONCE in a fixed order
# so they share identical instantaneous contention; take best-of-N (min wall =
# max MB/s) per contender; report the median + the pairwise THROUGHPUT RATIO vs
# the first (baseline) contender. A delta within the jitter floor is NOISE, not
# a win — banked numbers must clear jitter and reproduce in a SECOND run.
#
# This is a GENERAL harness: it races any set of "label=command" contenders, so
# it serves the 3-way prize-sizing baseline (pure/isal/rapidgzip), the
# decode-loop race, structural A/Bs, and per-platform C-deletion parity checks.
#
# ---- How to run (composition) ----------------------------------------------
# Run the loop INSIDE the bench LXC (199). Freeze neighbors + pin P-cores ON THE
# HOST (neurotic) by wrapping with scripts/freeze_wrapper.sh:
#
#   ssh neurotic 'bash /root/gzippy/scripts/freeze_wrapper.sh \
#     pct exec 199 -- env \
#       RAW=503627776 N=15 CPUS=0,2,4,6,8,10,12,14 REF=pure \
#       bash /root/gzippy/scripts/interleaved_ab.sh \
#         "pure=/tmp/bench-bin/gzippy-purerust-sp -d -c -p8 benchmark_data/silesia-large.gz" \
#         "isal=/tmp/bench-bin/gzippy-isal -d -c -p8 benchmark_data/silesia-large.gz" \
#         "rapidgzip=rapidgzip -d -P8 -c benchmark_data/silesia-large.gz"'
#
#   CPUS: physical P-cores are 0,2,4,6,8,10,12,14 (even logical = one thread per
#   P-core; 0-7 HT-doubles 4 cores; 0-15 = all 8 P-cores' 16 HT threads). Pin to
#   match the thread count you pass to -p/-P, or the topology confounds the run
#   (see memory: the "T16 plateau" was a pinning confound).
#
# ---- Env knobs --------------------------------------------------------------
#   N     trials per contender              (default 15)
#   CPUS  taskset -c cpu-list               (default: unpinned — DON'T for banking)
#   RAW   uncompressed bytes -> MB/s        (default 0 = report seconds)
#   REF   contender label whose first run   (default: none = no correctness check)
#         is the correctness reference; EVERY run's stdout sha256 is compared.
# =============================================================================
set -u
N="${N:-15}"
CPUS="${CPUS:-}"
RAW="${RAW:-0}"
REF="${REF:-}"
TS=""; [ -n "$CPUS" ] && TS="taskset -c $CPUS"

declare -a LABELS; declare -A CMD
for spec in "$@"; do
  case "$spec" in
    *=*) LABELS+=("${spec%%=*}"); CMD["${spec%%=*}"]="${spec#*=}" ;;
    *)   echo "bad contender (need label=command): $spec" >&2; exit 2 ;;
  esac
done
[ "${#LABELS[@]}" -ge 2 ] || { echo "need >=2 'label=command' contenders" >&2; exit 2; }

declare -A TIMES; for l in "${LABELS[@]}"; do TIMES[$l]=""; done
REFSUM=""; DIVERGED=0
for ((i=1;i<=N;i++)); do
  for l in "${LABELS[@]}"; do
    out=$(mktemp)
    s=$(date +%s.%N); $TS ${CMD[$l]} >"$out" 2>/dev/null; rc=$?; e=$(date +%s.%N)
    [ $rc -eq 0 ] || echo "!! $l trial $i exited $rc" >&2
    TIMES[$l]="${TIMES[$l]} $(awk "BEGIN{printf \"%.4f\", $e-$s}")"
    if [ -n "$REF" ]; then
      sum=$(sha256sum "$out" | cut -d' ' -f1)
      [ "$l" = "$REF" ] && [ -z "$REFSUM" ] && REFSUM="$sum"
      if [ -n "$REFSUM" ] && [ "$sum" != "$REFSUM" ]; then
        echo "!! CORRECTNESS DIVERGENCE: $l trial $i sha256=$sum != ref($REF) $REFSUM" >&2
        DIVERGED=1
      fi
    fi
    rm -f "$out"
  done
done

echo "============ interleaved A/B  N=$N  CPUS=${CPUS:-UNPINNED}  RAW=$RAW ============"
[ -n "$REF" ] && echo "ref($REF) sha256=$REFSUM  diverged=$DIVERGED"
declare -A MEDT
for l in "${LABELS[@]}"; do
  read medt mint maxt < <(echo "${TIMES[$l]}" | awk '
    {n=NF; for(i=1;i<=NF;i++)a[i]=$i}
    END{ for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}
         med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2; printf "%.4f %.4f %.4f", med, a[1], a[n] }')
  MEDT[$l]=$medt
  if awk "BEGIN{exit !($RAW>0)}"; then
    printf "  %-16s %8.1f MB/s (med)   [worst %.1f .. best %.1f]\n" "$l" \
      "$(awk "BEGIN{print $RAW/$medt/1e6}")" \
      "$(awk "BEGIN{print $RAW/$maxt/1e6}")" \
      "$(awk "BEGIN{print $RAW/$mint/1e6}")"
  else
    printf "  %-16s %8.4fs (med)   [best %.4f .. worst %.4f]\n" "$l" "$medt" "$mint" "$maxt"
  fi
done
echo "---- deltas vs baseline (${LABELS[0]}) ----"
base="${LABELS[0]}"
for l in "${LABELS[@]}"; do
  [ "$l" = "$base" ] && continue
  awk -v tb="${MEDT[$base]}" -v tl="${MEDT[$l]}" -v bl="$base" -v ll="$l" 'BEGIN{
    r=tb/tl; printf "  %-16s vs %-12s %.3fx throughput  (%+.1f%%)\n", ll, bl, r, (r-1)*100}'
done
[ "$DIVERGED" -eq 0 ] || { echo "FAIL: correctness divergence detected" >&2; exit 1; }
