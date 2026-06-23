#!/usr/bin/env bash
# R_WORKER SUB-DECOMP capture: silesia(LOSS) + nasa(CONTROL) @ T4, N=9 interleaved
# (gz,rg,gzAA), per-SUB-region {cyc,calls} from the symmetric table-vs-decode partition.
# Box is FREE (no llama) — freeze gov=performance boost=0 (trap+watchdog restore). taskset
# 0-3, /dev/null both arms (SINK LAW). Gate-0: SUB_OVERLAP==0 both + non-inert + sha==zcat
# + conservation. Emits raw.csv for the fulcrum excess assembler.
set -u
GZ=/dev/shm/ar-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
OUT=/dev/shm/amd-subdecomp
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
N="${N:-9}"
CORES="${CORES:-0-3}"
TH=4
CORPORA=("/root/silesia.gz:silesia" "/root/nasa.gz:nasa")

unset GZIPPY_OVERLAP_WRITER RAPIDGZIP_AW_PROF RAPIDGZIP_WA_PROF

# ---- llama reappeared: pause the server + its bench driver for a clean frozen run.
#      GUARANTEED SIGCONT via trap + a detached <=600s watchdog. Do NOT restart/kill. ----
LLAMA_PIDS="$(pgrep -x llama-server) $(pgrep -f bench_he.py)"
echo "LLAMA pause targets: $LLAMA_PIDS" >&2

# ---- box save / trap / watchdog ----
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GF GV
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do GF+=("$f"); GV+=("$(cat "$f")"); done
restore(){
  for p in $LLAMA_PIDS; do kill -CONT "$p" 2>/dev/null; done
  for i in "${!GF[@]}"; do echo "${GV[$i]}" > "${GF[$i]}" 2>/dev/null; done
  echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
}
trap 'restore' EXIT INT TERM
# detached watchdog: <=600s then force CONT llama + restore gov/boost regardless
setsid sh -c "sleep 600; for p in $LLAMA_PIDS; do kill -CONT \$p 2>/dev/null; done; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand>\$f 2>/dev/null; done; echo 1>/sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WD=$!
# pause llama (server first usually, but order does not matter for SIGSTOP)
for p in $LLAMA_PIDS; do kill -STOP "$p" 2>/dev/null; done
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
sleep 1
echo "FROZEN gov=performance boost=0; llama STOP states=[$(for p in $LLAMA_PIDS; do printf '%s:%s ' $p $(ps -o stat= -p $p 2>/dev/null|tr -d ' '); done)]; watchdog=$WD cores=$CORES N=$N TH=$TH" >&2

PE="cycles,instructions"
pf(){ grep -E "[[:space:]]$2(:u)?\b" "$1" | grep -oE '[0-9,]+' | head -1 | tr -d ,; }
# gz extractors (subregion dump: "    R_TABLE  cyc= N hdr_calls= N ...", "    R_DECODE cyc= N calls= N", "  R_WORKER   cyc= N")
gzf(){ grep "$2" "$1" | grep "cyc=" | head -1 | grep -oE "$3= *[0-9]+" | head -1 | grep -oE '[0-9]+'; }
gzsubov(){ grep 'SUB_OVERLAP_VIOLATIONS=' "$1" | head -1 | grep -oE 'SUB_OVERLAP_VIOLATIONS=[0-9]+' | cut -d= -f2; }
# rg single line: "[RG-SUBREGION] R_TABLE cyc=A hdr_calls=B | R_DECODE cyc=C calls=D | ring_other cyc=E | WORKER cyc=F"
rgf(){ grep '\[RG-SUBREGION\] R_TABLE' "$1" | head -1; }
rgfield(){ rgf "$1" | grep -oE "$2 cyc=[0-9]+" | head -1 | grep -oE '[0-9]+'; }
rgsubov(){ grep '\[RG-SUBREGION\] SUB_OVERLAP_VIOLATIONS=' "$1" | head -1 | grep -oE 'SUB_OVERLAP_VIOLATIONS=[0-9]+' | cut -d= -f2; }

echo "corpus,rep,total_out,gz_tcyc,gz_w,gz_t,gz_tn,gz_d,gz_subov,rg_tcyc,rg_w,rg_t,rg_d,rg_subov,gzaa_w" > "$CSV"

# Gate-0 sha + total-out per corpus
declare -A TOTAL_OUT
echo "=== Gate-0 sha + total-out ===" >&2
for entry in "${CORPORA[@]}"; do
  corpus="${entry%%:*}"; tag="${entry##*:}"
  cat "$corpus" >/dev/null 2>&1
  ref=$(zcat "$corpus" | sha256sum | cut -d' ' -f1)
  TOTAL_OUT[$tag]=$(zcat "$corpus" | wc -c)
  gs=$(env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -p $TH -c "$corpus" 2>/dev/null | sha256sum | cut -d' ' -f1)
  rs=$("$RG" -d -P $TH -c "$corpus" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "  $tag: out=${TOTAL_OUT[$tag]} gz=$([ "$gs" = "$ref" ] && echo OK || echo BAD) rg=$([ "$rs" = "$ref" ] && echo OK || echo BAD)" >&2
done

for entry in "${CORPORA[@]}"; do
  corpus="${entry%%:*}"; tag="${entry##*:}"
  for r in $(seq 1 "$N"); do
    # gz
    taskset -c "$CORES" perf stat -o "$OUT/gz.perf" -e "$PE" -- env GZIPPY_FORCE_PARALLEL_SM=1 \
      GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gz.err"
    gtc=$(pf "$OUT/gz.perf" cycles)
    gw=$(gzf "$OUT/gz.err" "R_WORKER " cyc); gt=$(gzf "$OUT/gz.err" "R_TABLE " cyc)
    gtn=$(gzf "$OUT/gz.err" "R_TABLE " hdr_calls); gd=$(gzf "$OUT/gz.err" "R_DECODE " cyc)
    gov=$(gzsubov "$OUT/gz.err")
    # rg
    taskset -c "$CORES" perf stat -o "$OUT/rg.perf" -e "$PE" -- env RAPIDGZIP_REGION_PROF=1 \
      "$RG" -d -P "$TH" -o /dev/null "$corpus" >/dev/null 2>"$OUT/rg.err"
    rtc=$(pf "$OUT/rg.perf" cycles)
    rw=$(rgfield "$OUT/rg.err" WORKER); rt=$(rgfield "$OUT/rg.err" R_TABLE); rd=$(rgfield "$OUT/rg.err" R_DECODE)
    rov=$(rgsubov "$OUT/rg.err")
    # gz A/A
    taskset -c "$CORES" perf stat -o "$OUT/gzaa.perf" -e cycles -- env GZIPPY_FORCE_PARALLEL_SM=1 \
      GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$TH" -c "$corpus" >/dev/null 2>"$OUT/gzaa.err"
    aw=$(gzf "$OUT/gzaa.err" "R_WORKER " cyc)
    echo "$tag,$r,${TOTAL_OUT[$tag]},$gtc,$gw,$gt,$gtn,$gd,$gov,$rtc,$rw,$rt,$rd,$rov,$aw" >> "$CSV"
  done
  echo "  done $tag" >&2
done

restore; kill "$WD" 2>/dev/null; trap - EXIT INT TERM
echo "=== RESTORED gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost) llama=[$(for p in $LLAMA_PIDS; do printf '%s:%s ' $p $(ps -o stat= -p $p 2>/dev/null|tr -d ' '); done)] ===" >&2
echo "CSV=$CSV"
echo "--- last gz/rg subregion dumps ---" >&2
grep -E 'subregion-prof|RG-SUBREGION' "$OUT/gz.err" "$OUT/rg.err" 2>/dev/null | tail -8 >&2
