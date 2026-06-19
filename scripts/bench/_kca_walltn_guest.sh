#!/usr/bin/env bash
# _kca_walltn_guest.sh — NIGHT22 T4/T8 wall + peak-RSS regression check: A vs OLD.
# Confirms element A (inner-kernel single-state-base) did NOT regress the T>1
# parallel pipeline. Best-of-N wall (interleaved A,OLD per rep) + peak RSS via
# /usr/bin/time -v %M. Sink = /dev/null both arms. sha-verified per arm.
set -u
LOG=/dev/shm/kca_walltn.log
DONE=/dev/shm/kca_walltn.DONE
rm -f "$DONE"
exec > "$LOG" 2>&1
A=/root/gz-A-native
OLD=/root/bin/gzippy-chunkt1
REPS=${REPS:-9}
CORPORA=${CORPORA:-"silesia"}
THREADS=${THREADS:-"4 8"}
TIME=/usr/bin/time
echo "== NIGHT22 T4/T8 wall+RSS  A vs OLD =="
echo "load: $(cat /proc/loadavg)"
echo "A sha=$(sha256sum $A|cut -c1-12)  OLD sha=$(sha256sum $OLD|cut -c1-12)"
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "NO FILE $F"; continue; }
  REF=$(zcat "$F" | sha256sum | cut -c1-16)
  for T in $THREADS; do
    for arm in A OLD; do
      eval "BIN=\$$arm"
      # sha check
      S=$(GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p$T "$F" 2>/dev/null | sha256sum | cut -c1-16)
      SOK=$([ "$S" = "$REF" ] && echo OK || echo BAD)
      best=99999; rss=0
      for r in $(seq 1 $REPS); do
        t0=$(date +%s.%N)
        $TIME -v env GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p$T "$F" >/dev/null 2>/tmp/wt.err
        t1=$(date +%s.%N)
        el=$(echo "$t1 - $t0" | bc -l)
        m=$(grep "Maximum resident" /tmp/wt.err | grep -o '[0-9]*')
        awk "BEGIN{exit !($el < $best)}" && best=$el
        [ "${m:-0}" -gt "$rss" ] && rss=$m
      done
      printf "  %-8s T%-2s %-4s best=%.4fs  peakRSS=%sKB (%.1fMB)\n" "$corp" "$T" "$arm/$SOK" "$best" "$rss" "$(awk "BEGIN{print $rss/1024}")"
    done
  done
done
echo PASS > "$DONE"
echo "== DONE ($(date +%T)) =="
