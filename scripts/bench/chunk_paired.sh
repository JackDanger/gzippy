#!/usr/bin/env bash
# chunk_paired.sh — PAIRED cyc/byte A/B for the T1 chunk-size lever.
#   arm A (A1,A2) = baseline   : default chunk (no override)
#   arm B         = candidate  : GZIPPY_CHUNK_KIB=$BKIB  (default 1024)
# SAME binary (t1prod), per-arm ENV is the only difference. Interleaved A1,A2,B
# per rep -> paired diffs via the committed analyzer. medΔ=(B-A1): NEGATIVE => B faster.
set -u
GZIPPY=${GZIPPY:-/root/bin/gzippy-t1prod}
BKIB=${BKIB:-1024}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"nasa silesia"}
HERE=/root/distpreload-harness
ANALYZE="$HERE/_distpreload_paired_analyze.py"
OUT_BASE=/tmp/chunk_paired
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock,page-faults"
declare -A BYTES=( [nasa]=205242368 [silesia]=211968000 )

ENVB="GZIPPY_CHUNK_KIB=$BKIB"
echo "==== CHUNK PAIRED A=default B=$ENVB  pin=cpu$PIN reps=$REPS  $(date) ===="
echo "GZIPPY=$GZIPPY sha=$(sha256sum "$GZIPPY"|cut -c1-12)  load:$(cat /proc/loadavg)"

# GATE-0: both arms non-inert + byte-exact + KERN fired
GATE_FAIL=0
for corp in $CORPORA; do
  F=/root/$corp.gz; REF=$(zcat "$F"|sha256sum|cut -c1-16)
  ENTR=$(GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>&1 >/dev/null | grep "asm-kernel:c" | tail -1 | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
  ASHA=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  BSHA=$(env $ENVB GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  K=$([ "${ENTR:-0}" -gt 0 ] && echo "KERN_OK($ENTR)" || { echo KERN_ZERO; GATE_FAIL=1; })
  AS=$([ "$ASHA" = "$REF" ] && echo A_OK || { echo A_MISMATCH; GATE_FAIL=1; })
  BS=$([ "$BSHA" = "$REF" ] && echo B_OK || { echo B_MISMATCH; GATE_FAIL=1; })
  echo "  $corp ref=$REF $K $AS $BS"
done
[ "$GATE_FAIL" != 0 ] && { echo "!!! GATE-0 FAILED"; exit 2; }
echo "--- GATE0 PASS"

OUT="$OUT_BASE"; rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/bytes.txt"
for corp in $CORPORA; do
  F=/root/$corp.gz; echo "$corp ${BYTES[$corp]}" >> "$OUT/bytes.txt"
  for r in $(seq 1 $REPS); do
    for arm in A1 A2 B; do
      case $arm in
        A1|A2) PRE="";;
        B)     PRE="$ENVB";;
      esac
      taskset -c $PIN perf stat -x, -e "$EVENTS" \
        -- env $PRE GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" \
        >/dev/null 2>"$OUT/$corp.$arm.$r.csv"
    done
  done
done
python3 "$ANALYZE" "$OUT" --tag "chunk default-vs-${BKIB}KiB" $CORPORA

# faults summary (mean per arm)
echo "--- page-faults mean per arm ---"
for corp in $CORPORA; do
  for arm in A1 B; do
    m=$(for f in "$OUT"/$corp.$arm.*.csv; do awk -F, '/page-faults/{print $1}' "$f"; done | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}')
    echo "  $corp $arm faults_mean=$m"
  done
done
echo "DONE_CHUNK_PAIRED"
