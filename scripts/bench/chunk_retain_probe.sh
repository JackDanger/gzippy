#!/usr/bin/env bash
# chunk_retain_probe.sh — STAGE-A discriminator for the CHUNK_KIB x pool-retain cross.
# T1 (-p1), pin cpu4, /dev/null sink. Reports mean page-faults + cyc/B per config
# (perf stat -r N). The HYPOTHESIS under test: small chunk + retain (manual pool)
# drops faults toward igzip's ~666 AND drops cyc/B. If faults DON'T move, lever dead.
set -u
GZIPPY=${GZIPPY:-/root/bin/gzippy-t1prod}
IGZIP=${IGZIP:-/usr/bin/igzip}
PIN=${PIN:-4}
N=${N:-5}
CORPORA=${CORPORA:-"nasa silesia"}
declare -A BYTES=( [nasa]=205242368 [silesia]=211968000 [monorepo]=0 )

echo "==== CHUNK x RETAIN PROBE  pin=cpu$PIN  N=$N  reps  $(date) ===="
echo "GZIPPY=$GZIPPY sha=$(sha256sum "$GZIPPY"|cut -c1-12)  load:$(cat /proc/loadavg)"
echo "igzip: $($IGZIP --version 2>&1|head -1)"

# config list: label|envprefix
CONFIGS=(
  "igzip|IGZIP"
  "c_def_roff|"
  "c1024_roff|GZIPPY_CHUNK_KIB=1024"
  "c512_roff|GZIPPY_CHUNK_KIB=512"
  "c256_roff|GZIPPY_CHUNK_KIB=256"
  "c_def_ron|GZIPPY_MANUAL_BUFFER_POOL=1"
  "c1024_ron|GZIPPY_CHUNK_KIB=1024 GZIPPY_MANUAL_BUFFER_POOL=1"
  "c512_ron|GZIPPY_CHUNK_KIB=512 GZIPPY_MANUAL_BUFFER_POOL=1"
  "c256_ron|GZIPPY_CHUNK_KIB=256 GZIPPY_MANUAL_BUFFER_POOL=1"
)

run_perf() { # $1=corpus $2=label $3=envprefix  -> prints "faults cyc"
  local F=/root/$1.gz label=$2 envp=$3 out
  if [ "$envp" = "IGZIP" ]; then
    out=$(taskset -c $PIN perf stat -r $N -x, -e page-faults,cpu_core/cycles/ \
       -- taskset -c $PIN "$IGZIP" -d -c "$F" 2>&1 >/dev/null)
  else
    out=$(env $envp GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN perf stat -r $N -x, \
       -e page-faults,cpu_core/cycles/ \
       -- env $envp GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 "$F" 2>&1 >/dev/null)
  fi
  local faults=$(echo "$out" | awk -F, '/page-faults/{print $1; exit}')
  local cyc=$(echo "$out" | awk -F, '/cycles/{print $1; exit}')
  echo "$faults $cyc"
}

# Gate-0: pool non-inert + sha for the retain configs (one VERBOSE run on nasa)
echo "--- GATE0 non-inert + byte-exact (nasa) ---"
REF=$(zcat /root/nasa.gz | sha256sum | cut -c1-16)
for label in c256_ron c_def_ron; do
  envp=$(for c in "${CONFIGS[@]}"; do [ "${c%%|*}" = "$label" ] && echo "${c#*|}"; done)
  V=$(env $envp GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 /root/nasa.gz 2>&1 >/tmp/g0nasa.out)
  SHA=$(env $envp GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$GZIPPY" -d -c -p1 /root/nasa.gz 2>/dev/null | sha256sum | cut -c1-16)
  POOL=$(echo "$V" | grep -i "pool u8" | tail -1)
  LIVE=$(echo "$V" | grep -i "concurrently-live\|MAX_LIVE\|in-flight depth" | tail -1)
  CHK=$(echo "$V" | grep -iE "chunks|chunk count" | tail -1)
  echo "  $label sha=$SHA $([ "$SHA" = "$REF" ] && echo SHA_OK || echo SHA_MISMATCH)"
  echo "    pool: $POOL"
  echo "    live: $LIVE"
  echo "    chunks: $CHK"
done
echo "  ref(nasa)=$REF"

printf "\n%-12s %-8s %12s %14s %10s\n" corpus config faults cyc cyc/B
for corp in $CORPORA; do
  B=${BYTES[$corp]}
  for c in "${CONFIGS[@]}"; do
    label=${c%%|*}; envp=${c#*|}
    read fa cy < <(run_perf "$corp" "$label" "$envp")
    cb=$(awk -v c="$cy" -v b="$B" 'BEGIN{ if(b>0&&c!="") printf "%.4f", c/b; else print "NA"}')
    printf "%-12s %-8s %12s %14s %10s\n" "$corp" "$label" "$fa" "$cy" "$cb"
  done
done
echo "DONE_CHUNK_RETAIN_PROBE"
