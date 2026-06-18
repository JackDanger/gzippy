#!/usr/bin/env bash
# End-to-end WALL, best-of-N, per sink, sha-verified. Interleaved over binaries.
set -u
PIN=${PIN:-4}; N=${N:-9}; CORPORA=${CORPORA:-"silesia nasa"}
IG=/usr/bin/igzip
declare -A BINS=( [new-native]=/root/bin/gzippy-new-native [bigreserve]=/root/bin/gzippy-bigreserve )
now(){ date +%s.%N; }
run_g(){ env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$1" -d -c -p1 "$2"; }
run_ig(){ taskset -c $PIN $IG -d -c "$2"; }
best(){ # $1=cmdfn $2=bin $3=gz $4=sink ; echo best wall seconds
  local b=99; for r in $(seq 1 $N); do
    local s=$(now); $1 "$2" "$3" >"$4" 2>/dev/null; local e=$(now)
    local d=$(awk -v a=$s -v b=$e "BEGIN{print b-a}")
    awk -v d=$d -v b=$b "BEGIN{exit !(d<b)}" && b=$d
  done; echo $b; }
for c in $CORPORA; do
  GZ=/root/$c.gz; REF=$($IG -d -c "$GZ"|sha256sum|cut -d' ' -f1)
  echo; echo "### $c  ref=${REF:0:12}"
  for sink in /dev/null /tmp/out.$c; do
    echo "  -- sink=$sink"
    for name in new-native bigreserve; do
      run_g "${BINS[$name]}" "$GZ" >"$sink" 2>/dev/null
      SHA=$(sha256sum "$sink"|cut -d' ' -f1); [ "$sink" = /dev/null ] && SHA=$(run_g "${BINS[$name]}" "$GZ" 2>/dev/null|sha256sum|cut -d' ' -f1)
      OK=BAD; [ "$SHA" = "$REF" ] && OK=ok
      W=$(best run_g "${BINS[$name]}" "$GZ" "$sink")
      printf "     %-12s %-3s wall=%ss\n" "$name" "$OK" "$W"
    done
    run_ig x "$GZ" >"$sink" 2>/dev/null
    SHA=$(sha256sum "$sink"|cut -d' ' -f1); [ "$sink" = /dev/null ] && SHA=$(run_ig x "$GZ" 2>/dev/null|sha256sum|cut -d' ' -f1)
    OK=BAD; [ "$SHA" = "$REF" ] && OK=ok
    W=$(best run_ig x "$GZ" "$sink")
    printf "     %-12s %-3s wall=%ss\n" "igzip" "$OK" "$W"
    rm -f /tmp/out.$c
  done
done
