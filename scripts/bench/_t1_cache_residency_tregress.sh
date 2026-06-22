#!/usr/bin/env bash
# _t1_cache_residency_tregress.sh — T>1 NO-REGRESSION guard (same methodology
# both arms). The T1 cache-residency levers are T1-GATED (decode_chunk_thin_t1 /
# take_data are reached ONLY from drive_thin_t1_oracle at T==1), so the T>1
# parallel path is byte-identical code. This confirms it at the wall: NEW-build
# prodt vs OLD-build (pre-bake) prodt at T1/T4/T8, /dev/null both arms,
# interleaved best-of-N, pinned to the same core set. Within spread = no regress.
set -u
BIN_NEW=${BIN_NEW:-/dev/shm/tn/release/examples/streaming_thin}
BIN_OLD=${BIN_OLD:-/dev/shm/told/release/examples/streaming_thin}
REPS=${REPS:-9}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260622}
THREADS=${THREADS:-"1 4 8"}
PINSET=${PINSET:-0-15}

echo "== T>1 no-regression NEW vs OLD prodt (pinset=$PINSET reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "NEW sha=$(sha256sum "$BIN_NEW" | cut -c1-12)  OLD sha=$(sha256sum "$BIN_OLD" | cut -c1-12)"
for T in $THREADS; do
  echo "##### T=$T #####"
  for corp in $CORPORA; do
    F=$GZDIR/$corp.gz
    [ -f "$F" ] || { echo "  $corp: NO FILE"; continue; }
    REF=$(zcat "$F" | wc -c)
    declare -A BEST BYTES
    BEST[new]=999999; BEST[old]=999999; BYTES[new]=0; BYTES[old]=0
    declare -A BINOF; BINOF[new]=$BIN_NEW; BINOF[old]=$BIN_OLD
    for r in $(seq 1 "$REPS"); do
      for a in $(echo "new old" | tr ' ' '\n' | shuf --random-source=<(yes "$SEED$r")); do
        line=$(taskset -c $PINSET "${BINOF[$a]}" prodt "$F" "$T" 2>/dev/null | grep '^RESULT')
        ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
        by=$(echo "$line" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
        [ -z "$ms" ] && continue
        BYTES[$a]=$by
        awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
      done
    done
    bok=OK; [ "${BYTES[new]}" = "$REF" ] || bok="BAD(${BYTES[new]})"
    printf "  %-9s new=%9s ms  old=%9s ms  new/old=%s  bytes=%s\n" "$corp" "${BEST[new]}" "${BEST[old]}" \
      "$(awk -v n="${BEST[new]}" -v o="${BEST[old]}" 'BEGIN{printf "%.3f",n/o}')" "$bok"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
