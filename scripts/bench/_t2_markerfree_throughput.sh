#!/usr/bin/env bash
# Interleaved best-of-N throughput A/B for the GZIPPY_FREE_MARKERS fix.
# Both arms use the same /dev/null sink (SINK LAW). Reports per-arm min wall
# (best-of-N) per thread count for OFF (baseline) vs ON (madvise free).
set -u
B=/dev/shm/rss_build/target/release/gzippy
N=9
run_one() { # $1=free(0/1) $2=threads $3=corpus
  local env_free=""
  [ "$1" = "1" ] && env_free="GZIPPY_FREE_MARKERS=1"
  local t0 t1
  t0=$(date +%s.%N)
  env $env_free GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p "$2" "$3" >/dev/null 2>&1
  t1=$(date +%s.%N)
  echo "$t1 - $t0" | bc -l
}
bestof() { # $1=free $2=threads $3=corpus -> min of N
  local best="" v
  for _ in $(seq 1 "$N"); do
    v=$(run_one "$1" "$2" "$3")
    if [ -z "$best" ] || (( $(echo "$v < $best" | bc -l) )); then best=$v; fi
  done
  echo "$best"
}
for entry in "$@"; do
  T="${entry%%:*}"; C="${entry##*:}"
  name=$(basename "$C")
  # interleave: alternate off/on across reps to cancel drift
  offs=""; ons=""
  for _ in $(seq 1 "$N"); do
    offs="$offs $(run_one 0 "$T" "$C")"
    ons="$ons $(run_one 1 "$T" "$C")"
  done
  omin=$(printf '%s\n' $offs | sort -g | head -1)
  nmin=$(printf '%s\n' $ons  | sort -g | head -1)
  ratio=$(echo "$nmin / $omin" | bc -l)
  printf "[%s T%s] OFF_min=%.4fs ON_min=%.4fs  ON/OFF=%.4f\n" "$name" "$T" "$omin" "$nmin" "$ratio"
  printf "    OFF:%s\n    ON :%s\n" "$offs" "$ons"
done
