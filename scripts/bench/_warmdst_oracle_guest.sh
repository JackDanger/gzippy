#!/usr/bin/env bash
# Warm-output-buffer removal-oracle (TASK 2/3).
# COLD  = production default (manual buffer pool OFF -> per-chunk output buffer
#         dropped after writev -> global rpmalloc re-faults every page).
# WARM  = GZIPPY_MANUAL_BUFFER_POOL=1 (LIFO pool RETAINS the freed output Vecs;
#         pages stay resident -> next chunk's take_u8 reuses warm -> ~0 re-fault).
# Both arms byte-identical output (sha verified). Non-inert proof: pool u8 hits
# must be >0 in WARM and 0 in COLD, AND page-faults must drop. cpu4 pinned,
# /dev/null sink, perf -r N freq-invariant cyc/B.
set -u
BIN=${BIN:-/root/bin/gzippy-new-native}
IGZIP=${IGZIP:-/usr/bin/igzip}
PIN=${PIN:-4}
REPS=${REPS:-11}
CORPORA=${CORPORA:-"silesia nasa"}
EV="page-faults,minor-faults,cpu_core/cycles/,cpu_core/instructions/"

echo "## bin=$BIN reps=$REPS pin=$PIN"
echo "## $($BIN --version 2>&1 | head -1)"
for c in $CORPORA; do
  GZ=/root/$c.gz
  OBYTES=$($IGZIP -d -c "$GZ" | wc -c)
  REF=$($IGZIP -d -c "$GZ" | sha256sum | cut -d" " -f1)
  echo
  echo "### corpus=$c output_bytes=$OBYTES ref_sha=${REF:0:12}"

  for MODE in cold warm; do
    if [ "$MODE" = warm ]; then EXTRA="GZIPPY_MANUAL_BUFFER_POOL=1"; else EXTRA=""; fi
    # sha + pool-hit proof (one verbose run)
    SHA=$(env $EXTRA GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN $BIN -d -c -p1 "$GZ" 2>/dev/null | sha256sum | cut -d" " -f1)
    POOL=$(env $EXTRA GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 taskset -c $PIN $BIN -d -c -p1 "$GZ" 2>&1 >/dev/null | grep -E "Buffer pool u8|in-flight depth" | tr "\n" " ")
    OK="MISMATCH"; [ "$SHA" = "$REF" ] && OK="sha-OK"
    # perf stat faults + cycles
    PERF=$(env $EXTRA GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN perf stat -r $REPS -e $EV \
            -- taskset -c $PIN $BIN -d -c -p1 "$GZ" >/dev/null 2>"/tmp/perf_${c}_${MODE}.txt"; cat "/tmp/perf_${c}_${MODE}.txt")
    PF=$(echo "$PERF"   | grep -E "page-faults"  | head -1 | awk "{gsub(/,/,\"\",\$1); print \$1}")
    CYC=$(echo "$PERF"  | grep -E "cycles"       | head -1 | awk "{gsub(/,/,\"\",\$1); print \$1}")
    INS=$(echo "$PERF"  | grep -E "instructions" | head -1 | awk "{gsub(/,/,\"\",\$1); print \$1}")
    CPB=$(awk -v c="$CYC" -v b="$OBYTES" "BEGIN{printf \"%.4f\", c/b}")
    IPB=$(awk -v i="$INS" -v b="$OBYTES" "BEGIN{printf \"%.4f\", i/b}")
    printf "  %-4s %-9s faults=%-9s cyc=%-12s cyc/B=%-7s instr/B=%-7s\n" "$MODE" "$OK" "$PF" "$CYC" "$CPB" "$IPB"
    echo "       pool: $POOL"
  done

  # igzip reference arm (same sink, same pin)
  PERF=$(taskset -c $PIN perf stat -r $REPS -e $EV -- taskset -c $PIN $IGZIP -d -c "$GZ" >/dev/null 2>/tmp/perf_${c}_ig.txt; cat /tmp/perf_${c}_ig.txt)
  PF=$(echo "$PERF"  | grep -E "page-faults"  | head -1 | awk "{gsub(/,/,\"\",\$1); print \$1}")
  CYC=$(echo "$PERF" | grep -E "cycles"       | head -1 | awk "{gsub(/,/,\"\",\$1); print \$1}")
  CPB=$(awk -v c="$CYC" -v b="$OBYTES" "BEGIN{printf \"%.4f\", c/b}")
  printf "  %-4s %-9s faults=%-9s cyc=%-12s cyc/B=%-7s\n" "igzip" "ref" "$PF" "$CYC" "$CPB"
done
