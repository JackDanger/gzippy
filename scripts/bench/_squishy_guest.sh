#!/usr/bin/env bash
# _squishy_guest.sh — 3-way matrix measurement (rg vs gz1-isal vs gz2-native)
# for the squishy corpus suite. Pre-built binaries only; no stale-binary guard.
#
# Env (all required):
#   GZ1      — path to gzippy-isal binary (pre-built)
#   GZ2      — path to gzippy-native binary (pre-built)
#   RG       — path to rapidgzip binary
#   CORPORA  — space-sep list of "name|path" pairs
#   TS       — space-sep thread counts (default "1 4 8 16")
#   N        — best-of-N trials (default 9; warmup iter0 dropped)
#   HOST_FROZEN — 1 to ack freeze unreadable from LXC sysfs
#   GOV / NO_TURBO — expected governor/no_turbo values
set -u

fail() { echo "SQUISHY_FAIL=$1"; exit "${2:-1}"; }
: "${GZ1:?need GZ1}"; : "${GZ2:?need GZ2}"; : "${RG:?need RG}"; : "${CORPORA:?need CORPORA}"
TS="${TS:-1 4 8 16}"; N="${N:-9}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"; HOST_FROZEN="${HOST_FROZEN:-1}"
ART="${ART:-/dev/shm/squishy-art}"; mkdir -p "$ART"

pin_mask() {
  case "$1" in
    1)  echo "0";;
    4)  echo "0,2,4,6";;
    8)  echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *)  echo "0";;
  esac
}

# ---- contamination guard (allowlist scrub) -----------------------------------
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    GZIPPY_FORCE_PARALLEL_SM|GZIPPY_DEBUG) ;;
    *) SCRUBBED="$SCRUBBED $v"; unset "$v" 2>/dev/null || true;;
  esac
done
if [ -n "$SCRUBBED" ]; then
  echo "## SCRUBBED non-production GZIPPY_* env:$SCRUBBED"
  case "$SCRUBBED" in
    *SEED*|*ORACLE*|*BYPASS*|*SLOW*|*SLEEP*)
      fail "contaminated-env (seeding/oracle var present)" 2;;
  esac
fi

# ---- host freeze readback ---------------------------------------------------
AG="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
AT="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
if [ "$AG" = "$GOV" ] && [ "$AT" = "$NO_TURBO" ]; then
  echo "## freeze OK: gov=$AG no_turbo=$AT"
elif [ "$AG" != NA ] && [ "$AG" != "$GOV" ]; then
  fail "host-not-frozen gov=$AG (want $GOV)" 13
elif [ "$AT" != NA ] && [ "$AT" != "$NO_TURBO" ]; then
  fail "host-not-frozen no_turbo=$AT (want $NO_TURBO)" 13
elif [ "$HOST_FROZEN" = 1 ]; then
  echo "## WARN freeze unreadable (gov=$AG no_turbo=$AT) HOST_FROZEN=1 ack"
else
  fail "host-freeze-unreadable (gov=$AG no_turbo=$AT) — pass HOST_FROZEN=1" 13
fi

# ---- quiet gate (instantaneous runnable avg) --------------------------------
RUN_SUM=0; RUN_CNT=0
for _qs in 1 2 3 4; do
  _r="$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA)"
  [ "$_r" = NA ] && break
  RUN_SUM=$((RUN_SUM + _r)); RUN_CNT=$((RUN_CNT + 1))
  [ "$_qs" -lt 4 ] && sleep 1
done
if [ "$RUN_CNT" -gt 0 ]; then
  RUN_AVG="$(awk -v s="$RUN_SUM" -v c="$RUN_CNT" 'BEGIN{printf "%.2f", s/c}')"
  echo "## quiet-gate: runnable_avg=$RUN_AVG"
  hot="$(awk -v a="$RUN_AVG" 'BEGIN{print (a+0>2.5)?1:0}')"
  if [ "$hot" = 1 ]; then
    echo "## WARN: box loaded ($RUN_AVG > 2.5) — ratios still jitter-immune; abs numbers may be inflated"
  fi
else
  echo "## quiet-gate: procs_running unreadable (context only)"
fi

# ---- binary identity --------------------------------------------------------
GZ1_SHA="$(sha256sum "$GZ1" | cut -c1-16)"
GZ2_SHA="$(sha256sum "$GZ2" | cut -c1-16)"
echo "================ SQUISHY PROVENANCE ================"
echo "gz1=$GZ1  sha16=$GZ1_SHA"
echo "gz2=$GZ2  sha16=$GZ2_SHA"
echo "rg=$("$RG" --version 2>&1 | head -1)"
echo "TS='$TS'  N=$N  HOST_FROZEN=$HOST_FROZEN"
echo "gov=$AG  no_turbo=$AT"
echo "====================================================="

# ---- sinks (regular files on /dev/shm, never pipes) -------------------------
SINK1="$ART/s1.bin"; SINK2="$ART/s2.bin"; SINKR="$ART/sr.bin"
for s in "$SINK1" "$SINK2" "$SINKR"; do
  rm -f "$s"
  : > "$s"
  [ -f "$s" ] && [ ! -L "$s" ] && [ ! -p "$s" ] || fail "sink-not-regular:$s" 14
done

timed() { # <sink> <mask> <cmd...> -> echoes "secs sha"
  local sink="$1" mask="$2"; shift 2
  local s e secs sha rc
  rm -f "$sink"; : > "$sink"
  s=$(date +%s.%N)
  set +e; taskset -c "$mask" "$@" > "$sink" 2>/dev/null; rc=$?; set -e 2>/dev/null || true
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$sink" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  echo "$secs $sha"
}

stats() { # space-sep list of floats -> "min med spread%"
  printf '%s\n' $1 | grep -v '^$' | sort -n | awk '
    {v[NR]=$1} END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'
}

mbps() { awk -v r="$1" -v t="$2" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

echo ""
echo "================ SQUISHY MATRIX ================"

for entry in $CORPORA; do
  name="${entry%%|*}"; path="${entry#*|}"
  if [ ! -f "$path" ]; then
    echo "## $name: MISSING $path — skip"; continue
  fi

  # Compute reference sha + raw size for this corpus (correctness oracle)
  echo "## computing ref sha for $name ($(stat -c %s "$path") bytes gz)..."
  gz_sz=$(stat -c %s "$path")
  REF="$(gzip -dc "$path" | sha256sum | cut -d' ' -f1)"
  raw_bytes="$(gzip -dc "$path" | wc -c)"
  ratio="$(awk -v r="$raw_bytes" -v g="$gz_sz" 'BEGIN{printf "%.2f", r/g}')"
  echo ""
  echo "########## CORPUS $name  gz=${gz_sz}  raw=${raw_bytes}  ratio=${ratio}x  ref16=${REF:0:16} ##########"

  for T in $TS; do
    MASK="$(pin_mask "$T")"

    # Record routing (don't abort on mismatch — just report)
    DBG1="$(GZIPPY_DEBUG=1 "$GZ1" -d -c -p "$T" "$path" 2>&1 >/dev/null | grep -m1 'path=' || true)"
    DBG2="$(GZIPPY_DEBUG=1 "$GZ2" -d -c -p "$T" "$path" 2>&1 >/dev/null | grep -m1 'path=' || true)"
    echo "## routing T$T: gz1=[${DBG1:-none}]  gz2=[${DBG2:-none}]"

    T1T=""; T2T=""; TRT=""; DIVERGED=0

    for i in $(seq 0 "$N"); do
      read -r g1s g1h <<< "$(timed "$SINK1" "$MASK" "$GZ1" -d -c -p "$T" "$path")"
      read -r g2s g2h <<< "$(timed "$SINK2" "$MASK" "$GZ2" -d -c -p "$T" "$path")"
      read -r rgs rgh <<< "$(timed "$SINKR" "$MASK" "$RG"  -d -c -f -P "$T" "$path")"
      [ "$i" -eq 0 ] && continue  # drop warmup
      T1T="$T1T $g1s"; T2T="$T2T $g2s"; TRT="$TRT $rgs"
      if [ "$g1h" != "$REF" ]; then echo "!! SHA_MISMATCH gz1 $name T$T i=$i sha16=${g1h:0:16}"; DIVERGED=1; fi
      if [ "$g2h" != "$REF" ]; then echo "!! SHA_MISMATCH gz2 $name T$T i=$i sha16=${g2h:0:16}"; DIVERGED=1; fi
      if [ "$rgh" != "$REF" ]; then echo "!! SHA_MISMATCH rg  $name T$T i=$i sha16=${rgh:0:16}"; DIVERGED=1; fi
    done

    if [ "$DIVERGED" -ne 0 ]; then
      echo "!! SHA MISMATCH — number VOID for $name T$T"
      continue
    fi

    read -r g1min g1med g1sp <<< "$(stats "$T1T")"
    read -r g2min g2med g2sp <<< "$(stats "$T2T")"
    read -r rgmin rgmed rgsp <<< "$(stats "$TRT")"

    RR1="$(awk -v g="$g1min" -v r="$rgmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
    RR2="$(awk -v g="$g2min" -v r="$rgmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
    # Verdicts using spread-aware margin
    V1="$(awk -v x="$RR1" -v m="$(awk -v a="$g1sp" -v b="$rgsp" 'BEGIN{m=(a>b)?a:b; printf "%.4f",m/100}')" \
          'BEGIN{d=x-1; if(d>m)print "WIN(gz1)"; else if(d<-m)print "LOSS"; else print "TIE"}')"
    V2="$(awk -v x="$RR2" -v m="$(awk -v a="$g2sp" -v b="$rgsp" 'BEGIN{m=(a>b)?a:b; printf "%.4f",m/100}')" \
          'BEGIN{d=x-1; if(d>m)print "WIN(gz2)"; else if(d<-m)print "LOSS"; else print "TIE"}')"

    echo "RESULT corpus=$name T=$T gz1=${g1min}s($(mbps "$raw_bytes" "$g1min")MB/s,sp=${g1sp}%) gz2=${g2min}s($(mbps "$raw_bytes" "$g2min")MB/s,sp=${g2sp}%) rg=${rgmin}s($(mbps "$raw_bytes" "$rgmin")MB/s,sp=${rgsp}%) rg/gz1=${RR1}[${V1}] rg/gz2=${RR2}[${V2}] sha=OK"
  done
done

rm -f "$SINK1" "$SINK2" "$SINKR"
echo ""
echo "================ END SQUISHY MATRIX ================"
echo "SQUISHY_DONE"
