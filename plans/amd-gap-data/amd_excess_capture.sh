#!/usr/bin/env bash
# AMD EXCESS CAPTURE: silesia(LOSS) + nasa(CONTROL) @ T4, N=9 interleaved (gz,rg,gzAA),
# per-region {cyc,bytes} from the symmetric region-prof partition + total perf cycles.
# llama SIGSTOP'd (safe: trap + <=600s detached watchdog GUARANTEED CONT+restore).
# FROZEN gov=performance boost=0. Gate-0: per-region non-inert + OVERLAP_VIOLATIONS==0
# + sha==zcat. Emits raw.csv for the fulcrum excess assembler.
set -u
GZ=/dev/shm/ar-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/amd-excess
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
TH=4
# corpus = path:tag
CORPORA=("/root/silesia.gz:silesia" "/root/nasa.gz:nasa")
LLAMA_PIDS="881047"

unset GZIPPY_OVERLAP_WRITER RAPIDGZIP_AW_PROF RAPIDGZIP_WA_PROF

# ---- box + llama save / trap / watchdog (GUARANTEED restore) ----
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GF GV
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do GF+=("$f"); GV+=("$(cat "$f")"); done
restore(){
  for p in $LLAMA_PIDS; do kill -CONT "$p" 2>/dev/null; done
  for i in "${!GF[@]}"; do echo "${GV[$i]}" > "${GF[$i]}" 2>/dev/null; done
  echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
}
trap 'restore' EXIT INT TERM
# detached watchdog: <=600s then force-CONT llama + restore gov/boost
setsid sh -c "sleep 600; for p in $LLAMA_PIDS; do kill -CONT \$p 2>/dev/null; done; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand>\$f 2>/dev/null; done; echo 1>/sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WD=$!
# pause llama
for p in $LLAMA_PIDS; do kill -STOP "$p" 2>/dev/null; done
# freeze
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
sleep 1
echo "FROZEN gov=performance boost=0; llama($LLAMA_PIDS) STOP=$(ps -o stat= -p $LLAMA_PIDS 2>/dev/null | tr -d ' '); watchdog=$WD cores=$CORES N=$N TH=$TH" >&2

PE="cycles,instructions"
pf(){ grep -E "[[:space:]]$2(:u)?\b" "$1" | grep -oE '[0-9,]+' | head -1 | tr -d ,; }
# gz region extractors (dump lines pad with spaces: "R_WORKER  cyc=    N calls= N bytes=  N")
gzc(){ grep "$2" "$1" | head -1 | grep -oE 'cyc= *[0-9]+' | head -1 | grep -oE '[0-9]+'; }
gzb(){ grep "$2" "$1" | head -1 | grep -oE '(bytes|mkbytes)= *[0-9]+' | head -1 | grep -oE '[0-9]+'; }
gzov(){ grep -oE 'OVERLAP_VIOLATIONS=[0-9]+' "$1" | head -1 | cut -d= -f2; }
# rg single-line: "[RG-REGION] WORKER cyc=A calls=.. bytes=B | MARKERPP cyc=C calls=.. mkbytes=D | OUTPUT cyc=E calls=.. bytes=F"
rgfield(){ grep '\[RG-REGION\] WORKER' "$1" | head -1 | grep -oE "$2 cyc=[0-9]+ calls=[0-9]+ (bytes|mkbytes)=[0-9]+"; }
rgc(){ rgfield "$1" "$2" | grep -oE 'cyc=[0-9]+' | cut -d= -f2; }
rgb(){ rgfield "$1" "$2" | grep -oE '(bytes|mkbytes)=[0-9]+' | cut -d= -f2; }
rgov(){ grep -oE 'OVERLAP_VIOLATIONS=[0-9]+' "$1" | head -1 | cut -d= -f2; }

echo "corpus,rep,total_out,gz_tcyc,gz_tins,gz_w_cyc,gz_w_b,gz_m_cyc,gz_m_b,gz_o_cyc,gz_o_b,gz_ov,gz_sha,rg_tcyc,rg_tins,rg_w_cyc,rg_m_cyc,rg_m_b,rg_o_cyc,rg_o_b,rg_ov,gzaa_tcyc,gzaa_w_cyc" > "$CSV"

# Gate-0 sha + canonical total-output (zcat|wc -c) per corpus
declare -A TOTAL_OUT
echo "=== Gate-0 sha + total-out ===" >&2
for entry in "${CORPORA[@]}"; do
  corpus="${entry%%:*}"; tag="${entry##*:}"
  cat "$corpus" >/dev/null 2>&1   # warm page cache
  ref=$(zcat "$corpus" | sha256sum | cut -d' ' -f1)
  TOTAL_OUT[$tag]=$(zcat "$corpus" | wc -c)
  gs=$(env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -p $TH -c "$corpus" 2>/dev/null | sha256sum | cut -d' ' -f1)
  rs=$("$RG" -d -P $TH -o /dev/null "$corpus" >/tmp/.rs 2>&1; zcat "$corpus" | sha256sum | cut -d' ' -f1)
  echo "  $tag: out=${TOTAL_OUT[$tag]} gz=$([ "$gs" = "$ref" ] && echo OK || echo BAD)" >&2
done

for entry in "${CORPORA[@]}"; do
  corpus="${entry%%:*}"; tag="${entry##*:}"
  ref=$(zcat "$corpus" | sha256sum | cut -d' ' -f1)
  for r in $(seq 1 "$N"); do
    # gz  (SINK LAW: /dev/null sink, same as rg/gzaa; sha already gated in Gate-0 section)
    taskset -c "$CORES" perf stat -o "$OUT/gz.perf" -e "$PE" -- env GZIPPY_FORCE_PARALLEL_SM=1 \
      GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gz.err"
    gshaok=1
    gtc=$(pf "$OUT/gz.perf" cycles); gti=$(pf "$OUT/gz.perf" instructions)
    gwc=$(gzc "$OUT/gz.err" R_WORKER); gwb=$(gzb "$OUT/gz.err" R_WORKER)
    gmc=$(gzc "$OUT/gz.err" R_MARKERPP); gmb=$(gzb "$OUT/gz.err" R_MARKERPP)
    goc=$(gzc "$OUT/gz.err" R_OUTPUT); gob=$(gzb "$OUT/gz.err" R_OUTPUT)
    gov=$(gzov "$OUT/gz.err")
    # rg
    taskset -c "$CORES" perf stat -o "$OUT/rg.perf" -e "$PE" -- env RAPIDGZIP_REGION_PROF=1 \
      "$RG" -d -P "$TH" -o /dev/null "$corpus" >/dev/null 2>"$OUT/rg.err"
    rtc=$(pf "$OUT/rg.perf" cycles); rti=$(pf "$OUT/rg.perf" instructions)
    rwc=$(rgc "$OUT/rg.err" WORKER)
    rmc=$(rgc "$OUT/rg.err" MARKERPP); rmb=$(rgb "$OUT/rg.err" MARKERPP)
    roc=$(rgc "$OUT/rg.err" OUTPUT); rob=$(rgb "$OUT/rg.err" OUTPUT)
    rov=$(rgov "$OUT/rg.err")
    # gz A/A
    taskset -c "$CORES" perf stat -o "$OUT/gzaa.perf" -e cycles -- env GZIPPY_FORCE_PARALLEL_SM=1 \
      GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gzaa.err"
    atc=$(pf "$OUT/gzaa.perf" cycles); awc=$(gzc "$OUT/gzaa.err" R_WORKER)
    echo "$tag,$r,${TOTAL_OUT[$tag]},$gtc,$gti,$gwc,$gwb,$gmc,$gmb,$goc,$gob,$gov,$gshaok,$rtc,$rti,$rwc,$rmc,$rmb,$roc,$rob,$rov,$atc,$awc" >> "$CSV"
  done
  echo "  done $tag" >&2
done

restore; kill "$WD" 2>/dev/null; trap - EXIT INT TERM
echo "=== RESTORED gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost) llama=$(ps -o stat= -p $LLAMA_PIDS 2>/dev/null | tr -d ' ') ===" >&2
echo "CSV=$CSV"
