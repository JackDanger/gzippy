#!/bin/bash
# _q2_split_guest.sh — Q2 decode-compute vs store-bandwidth binder, guest-side.
# SLOW_DECODE vs SLOW_STORE dose-response, frequency-neutral sleep kind.
# Usage: _q2_split_guest.sh CORPUS T MASK N
# Arms interleaved per rep (ALL sink /dev/null):
#   base (dose 0), dec50/dec100/dec200 (GZIPPY_SLOW_DECODE), st50/st100/st200 (GZIPPY_SLOW_STORE)
#   all non-base arms set GZIPPY_SLOW_KIND=sleep (turbo-neutral control)
# CSV: ARM,corpus,T,knob,dose,rep,wall_ms,rc
set -u
C=${1:?corpus}; T=${2:?threads}; MASK=${3:?mask}; N=${4:?reps}
BIN=/dev/shm/gz-head-native
ERR=/tmp/q2-arm.err
GZ=/root/$C.gz
test -c /dev/null || { echo "FATAL /dev/null not char-special"; exit 9; }

# fired-proof: one spin-kind run of each knob @200 (heavier than sleep) to prove
# the inject site executes (large wall jump vs base => site live, not inert).
echo "## fired-proof (spin kind, untimed-info):"
for k in DECODE STORE; do
  t0=$(date +%s%N); env GZIPPY_SLOW_$k=200 GZIPPY_SLOW_KIND=spin taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>/dev/null; t1=$(date +%s%N)
  echo "##   SLOW_$k=200 spin wall_ms=$(( (t1-t0)/1000000 ))"
done
t0=$(date +%s%N); taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>/dev/null; t1=$(date +%s%N)
echo "##   base(no knob) wall_ms=$(( (t1-t0)/1000000 ))"

echo "ARM,corpus,T,knob,dose,rep,wall_ms,rc"
for i in $(seq 1 "$N"); do
  for spec in base:0 decode:50 decode:100 decode:200 store:50 store:100 store:200; do
    knob=${spec%%:*}; dose=${spec##*:}
    t0=$(date +%s%N)
    case $knob in
      base)   taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      decode) env GZIPPY_SLOW_DECODE=$dose GZIPPY_SLOW_KIND=sleep taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
      store)  env GZIPPY_SLOW_STORE=$dose  GZIPPY_SLOW_KIND=sleep taskset -c "$MASK" "$BIN" -d -c -p"$T" "$GZ" >/dev/null 2>"$ERR" ;;
    esac
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    echo "ARM,$C,$T,$knob,$dose,$i,$wall_ms,$rc"
  done
done
test -c /dev/null || { echo "FATAL /dev/null clobbered post-run"; exit 9; }
