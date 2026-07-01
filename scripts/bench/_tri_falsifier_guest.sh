#!/usr/bin/env bash
# T>=2 TRI-FALSIFIER + ORTHOGONALITY + CEILING ORACLE (guest-side collector).
#
# ONE gzippy-native binary, two cells, two ORTHOGONAL perturbations:
#   * cell T4 silesia  = the marker-loop cell (AMD-T4 finding)
#   * cell T2 monorepo = the RSS/teardown cell (AMD-T2 finding)
#   * pert MARKER  = GZIPPY_SLOW_MARKER_MODE (decode-only; non-inert MARKER_HITS)
#   * pert RSS     = GZIPPY_RSS_INFLATE_MIB  (RSS-only; decode bit-identical)
#
# FALSIFIES the lead's "one unifying decoupled-marker port fixes BOTH T2-RSS and
# T4-marker" hypothesis: if MARKER moves T4 but not T2, and RSS moves T2 but not
# T4, the two are orthogonal mechanisms (no single port closes both). Also gives
# each prize CEILING (Leg-1 marker-slow @100%; Leg-2 RSS-inflate slope).
#
# Paired-interleaved within each rep so turbo/drift cancels in the paired Δ.
# /dev/null both arms (SINK LAW). sha-verified. Emits CSV to stdout for the
# python analyzer. Gate-0 non-inert proof captured per perturbed run.
set -u
BIN=${BIN:-/dev/shm/tri-target/release/gzippy}
RG=${RG:-/usr/local/bin/rapidgzip}
GZDIR=${GZDIR:-/root}
N=${N:-15}
# Core pins: T2 uses 2 cores, T4 uses 4. Away from core 0 (host/IRQ).
PIN_T2=${PIN_T2:-4-5}
PIN_T4=${PIN_T4:-4-7}

TS() { date +%s.%N; }
# awk (not bc — bc is absent on the guest) for the float subtraction.
elapsed() { awk "BEGIN{printf \"%.6f\", $1 - $2}"; }

# sha gate once per (corpus) — proves the binary is correct before timing.
sha_ok() { # $1=corpus
  local s1 s2
  s1=$("$BIN" -d -c "$1" 2>/dev/null | sha256sum | cut -d' ' -f1)
  s2=$(zcat "$1" | sha256sum | cut -d' ' -f1)
  [ "$s1" = "$s2" ] && echo 1 || echo 0
}

# non-inert hits for a MARKER perturbed run (must be >0 to trust the arm).
marker_hits() { # $1=threads $2=pin $3=corpus
  env GZIPPY_SLOW_MARKER_MODE=100 GZIPPY_SLOW_MARKER_HITS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
    taskset -c "$2" "$BIN" -d -c -p "$1" "$3" >/dev/null 2>/tmp/mh.txt
  grep -oiE 'marker[_-]?hit[s]?[^0-9]*[0-9]+' /tmp/mh.txt | grep -oE '[0-9]+' | tail -1 | sed 's/^$/0/'
  grep -oiE 'hits[^0-9]*[0-9]+' /tmp/mh.txt | grep -oE '[0-9]+' | tail -1
}

# one timed decode (env, threads, pin, corpus) -> wall seconds, /dev/null sink.
run() { # $1=env $2=threads $3=pin $4=corpus
  local t0 t1
  t0=$(TS)
  env $1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$3" "$BIN" -d -c -p "$2" "$4" >/dev/null 2>/dev/null
  t1=$(TS)
  elapsed "$t1" "$t0"
}

# peak RSS (KiB) of gz vs rg on a cell (a few reps, min taken in python).
rss_run() { # $1=cmdline... ; prints max-RSS KiB
  /usr/bin/env time -v "$@" >/dev/null 2>/tmp/rss.txt || /usr/bin/time -v "$@" >/dev/null 2>/tmp/rss.txt
  grep -i 'Maximum resident' /tmp/rss.txt | grep -oE '[0-9]+' | tail -1
}

echo "# tri-falsifier  BIN=$BIN  N=$N  $(date -u +%FT%TZ)"
echo "# bin-sha: $(sha256sum "$BIN" | cut -d' ' -f1)"
echo "cell,corpus,threads,pert,level,kind,rep,wall_s,sha,hits"

# ---- cell defs: name corpus threads pin ----
declare -A CORP=( [T4]="$GZDIR/silesia.gz" [T2]="$GZDIR/monorepo.gz" )
declare -A THR=(  [T4]=4 [T2]=2 )
declare -A PIN=(  [T4]="$PIN_T4" [T2]="$PIN_T2" )

# sha + non-inert checks up front (Gate-0/4)
for cell in T4 T2; do
  c="${CORP[$cell]}"
  echo "# SHA[$cell $(basename "$c")]=$(sha_ok "$c")" 1>&2
  mh=$(env GZIPPY_SLOW_MARKER_MODE=100 GZIPPY_SLOW_MARKER_HITS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
        taskset -c "${PIN[$cell]}" "$BIN" -d -c -p "${THR[$cell]}" "$c" 2>&1 >/dev/null \
        | grep -oiE 'hits[^0-9]*[0-9]+' | grep -oE '[0-9]+' | tail -1)
  echo "# MARKER_HITS[$cell]=${mh:-0}  (must be >0 to trust the marker arm)" 1>&2
done

# ---- the sweep ----
# MARKER perturbation levels (spin) + sleep control @100. RSS levels.
MARKER_SPIN=(0 50 100)
RSS_LVL=(0 20 40 60)

for cell in T4 T2; do
  c="${CORP[$cell]}"; t="${THR[$cell]}"; p="${PIN[$cell]}"
  for rep in $(seq 1 "$N"); do
    # baseline (shared level-0 for both perts)
    w=$(run "" "$t" "$p" "$c"); echo "$cell,$(basename "$c"),$t,BASE,0,base,$rep,$w,-,-"
    # MARKER spin sweep
    for lv in "${MARKER_SPIN[@]}"; do
      [ "$lv" = "0" ] && continue
      w=$(run "GZIPPY_SLOW_MARKER_MODE=$lv" "$t" "$p" "$c")
      echo "$cell,$(basename "$c"),$t,MARKER,$lv,spin,$rep,$w,-,-"
    done
    # MARKER sleep control @100 (freq-neutral)
    w=$(run "GZIPPY_SLOW_MARKER_MODE=100 GZIPPY_SLOW_KIND=sleep" "$t" "$p" "$c")
    echo "$cell,$(basename "$c"),$t,MARKER,100,sleep,$rep,$w,-,-"
    # RSS inflate sweep
    for lv in "${RSS_LVL[@]}"; do
      [ "$lv" = "0" ] && continue
      w=$(run "GZIPPY_RSS_INFLATE_MIB=$lv" "$t" "$p" "$c")
      echo "$cell,$(basename "$c"),$t,RSS,$lv,mmap,$rep,$w,-,-"
    done
  done
done

# ---- peak RSS gz vs rg (the scoreboard fact) ----
for cell in T4 T2; do
  c="${CORP[$cell]}"; t="${THR[$cell]}"; p="${PIN[$cell]}"
  for rep in 1 2 3; do
    gzr=$(rss_run env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$p" "$BIN" -d -c -p "$t" "$c")
    rgr=$(rss_run taskset -c "$p" "$RG" -d -c -P "$t" "$c")
    echo "$cell,$(basename "$c"),$t,PEAKRSS,gz,kib,$rep,$gzr,-,-"
    echo "$cell,$(basename "$c"),$t,PEAKRSS,rg,kib,$rep,$rgr,-,-"
  done
done
echo "# done $(date -u +%FT%TZ)" 1>&2
