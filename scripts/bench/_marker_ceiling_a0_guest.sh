#!/usr/bin/env bash
# A0 — marker-decode speed-up CEILING (cursor-directed; cheapest port-kill test).
# REMOVAL oracle (Gate-2): seed a zeroed window so window-absent chunks decode via
# the FAST clean asm run_contig instead of the slow 11.7 cyc/B marker fast-loop.
#   * BASE  = correct decode (sha==zcat)            -> the wall to beat
#   * U16   = clean decode + u16 write + resolve, CONSUMER-serial (PESSIMISTIC loc)
#   * U16W  = clean decode + u16 write + resolve, WORKER-parallel (OPTIMISTIC loc)
#   * U8    = clean decode, resolve DELETED (over-generous absolute ceiling)
# Truth between U16 and U16W. (baseline-ceiling)/baseline = marker speed-up CEILING.
# Δ<spread on the U16 arms ⇒ port DEAD (stop). Ceiling arms output WRONG bytes by
# design (sha mismatch EXPECTED) — non-inert proof = the HITS/RESOLVE_BYTES counters
# captured in the Gate-0 pass below, NOT sha. Paired-interleaved, /dev/null both.
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
GZDIR=${GZDIR:-/root}
N=${N:-15}
PIN_T2=${PIN_T2:-4-5}; PIN_T4=${PIN_T4:-4-7}
TS() { date +%s.%N; }
el() { awk "BEGIN{printf \"%.6f\", $1 - $2}"; }
run() { # $1=env $2=threads $3=pin $4=corpus -> wall s (process may exit nonzero on ceiling arms; we time it anyway)
  local t0 t1; t0=$(TS)
  env $1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$3" "$B" -d -c -p "$2" "$4" >/dev/null 2>/dev/null
  t1=$(TS); el "$t1" "$t0"
}

declare -A CORP=( [T4]="$GZDIR/silesia.gz" [T2]="$GZDIR/monorepo.gz" )
declare -A THR=(  [T4]=4 [T2]=2 )
declare -A PIN=(  [T4]="$PIN_T4" [T2]="$PIN_T2" )

# ---- Gate-0: non-inert proof (HITS / RESOLVE_BYTES) for each ceiling arm ----
echo "# A0 marker-ceiling  BIN=$B  bin-sha=$(sha256sum "$B"|cut -d' ' -f1)  N=$N" 1>&2
for cell in T4 T2; do
  c="${CORP[$cell]}"; t="${THR[$cell]}"; p="${PIN[$cell]}"
  # BASE sha gate (correctness of the binary)
  s1=$("$B" -d -c "$c" 2>/dev/null | sha256sum | cut -d' ' -f1)
  s2=$(zcat "$c" | sha256sum | cut -d' ' -f1)
  echo "# SHA[$cell]=$([ "$s1" = "$s2" ] && echo 1 || echo 0)" 1>&2
  for arm in MARKER_CEILING_U16 MARKER_CEILING_U16W MARKER_CEILING; do
    h=$(env GZIPPY_$arm=1 GZIPPY_SLOW_HITS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
         taskset -c "$p" "$B" -d -c -p "$t" "$c" 2>&1 >/dev/null \
         | grep -iE "ceiling|resolve|hits" | tr '\n' ' ')
    echo "# NONINERT[$cell $arm]: $h" 1>&2
  done
done

# ---- timing sweep (paired-interleaved) ----
echo "cell,corpus,threads,arm,rep,wall_s"
for cell in T4 T2; do
  c="${CORP[$cell]}"; t="${THR[$cell]}"; p="${PIN[$cell]}"; bn=$(basename "$c")
  for rep in $(seq 1 "$N"); do
    w=$(run "" "$t" "$p" "$c");                          echo "$cell,$bn,$t,BASE,$rep,$w"
    w=$(run "GZIPPY_MARKER_CEILING_U16=1" "$t" "$p" "$c"); echo "$cell,$bn,$t,U16,$rep,$w"
    w=$(run "GZIPPY_MARKER_CEILING_U16W=1" "$t" "$p" "$c");echo "$cell,$bn,$t,U16W,$rep,$w"
    w=$(run "GZIPPY_MARKER_CEILING=1" "$t" "$p" "$c");    echo "$cell,$bn,$t,U8,$rep,$w"
  done
done
echo "# done $(date -u +%FT%TZ)" 1>&2
