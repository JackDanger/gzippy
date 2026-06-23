#!/usr/bin/env bash
# Isolate PURE teardown ∝ RSS (replicates the AMD-T2 method on a 2nd arch):
#   teardown = external process-wall − internal(main_start->main_end) span.
# Sweep GZIPPY_RSS_INFLATE_MIB so the touch cost lands in the INTERNAL span
# (engage() runs before main_end) and only the resident footprint persists into
# teardown. (external−internal) vs RSS slope = pure teardown $/MiB; the internal
# slope = first-touch cost (the part real RSS-reduction also saves, but separate).
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
C=${CORPUS:-/root/monorepo.gz}
T=${T:-2}
PIN=${PIN:-4-5}
N=${N:-12}
echo "cell,level,rep,external_ms,internal_ms,teardown_ms"
for lv in 0 40 60; do
  envv=""
  [ "$lv" != "0" ] && envv="GZIPPY_RSS_INFLATE_MIB=$lv"
  for rep in $(seq 1 "$N"); do
    t0=$(date +%s.%N)
    intl=$(env $envv GZIPPY_PHASE_TIMING=1 GZIPPY_FORCE_PARALLEL_SM=1 \
            taskset -c "$PIN" "$B" -d -c -p "$T" "$C" 2>/tmp/pt.txt >/dev/null; \
            grep -oE 'first->last\)=[0-9.]+ms' /tmp/pt.txt | grep -oE '[0-9.]+' | head -1)
    t1=$(date +%s.%N)
    ext=$(awk "BEGIN{printf \"%.3f\", ($t1-$t0)*1000}")
    td=$(awk "BEGIN{printf \"%.3f\", $ext - ${intl:-0}}")
    echo "T2,$lv,$rep,$ext,${intl:-NA},$td"
  done
done
