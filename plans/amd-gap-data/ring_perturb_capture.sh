#!/usr/bin/env bash
# RING / FOLD-DRAIN perturbation capture (Gate-2). silesia-T4 + monorepo-T2.
# Builds fulcrum `perturb` sweep dirs:
#   <cell>/{meta.txt,baseline.txt,baseline_recheck.txt,spin/t{10,20,30}.txt,
#           sleep/t{10,20,30}.txt,oracle_removed.txt}
# SLOW-KNOB = GZIPPY_RING_INJECT_NS (busy rdtsc-spin) + GZIPPY_SLOW_KIND=sleep
# control; REMOVAL-ORACLE = GZIPPY_FOLD_NODRAIN=1. /dev/null both arms (SINK LAW).
# Interleaved N reps. Box FROZEN (gov=performance boost=0) + llama paused SAFELY
# (SIGSTOP+watchdog+trap, SIGCONT after; never kill/restart). Frozen window <=8min.
set -u
GZ=/dev/shm/ring-target/release/gzippy
RG=/root/rg-build-src/build/src/tools/rapidgzip
[ -x "$RG" ] || RG=/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
OUT=/dev/shm/ring-perturb
rm -rf "$OUT"; mkdir -p "$OUT"
N="${N:-9}"
# cells: "corpus:tag:threads:cores"
CELLS=("/root/silesia.gz:silesia_T4:4:0-3" "/root/monorepo.gz:monorepo_T2:2:0-1")

now_ns(){ printf '%s\n' "$(( $(date +%s%N) ))"; }
# wall of a gz run in seconds (float), output to /dev/null
gzwall(){ # $1 extra-env-string (KEY=VAL ...), $2 corpus, $3 th, $4 cores
  local s e
  s=$(date +%s%N)
  taskset -c "$4" env GZIPPY_FORCE_PARALLEL_SM=1 $1 "$GZ" -d -p "$3" -c "$2" >/dev/null 2>/dev/null
  e=$(date +%s%N)
  awk -v a="$s" -v b="$e" 'BEGIN{printf "%.6f\n",(b-a)/1e9}'
}

# ---- llama pause (SAFE): trap + detached <=600s watchdog; SIGCONT after ----
LLAMA_PIDS="$(pgrep -x llama-server) $(pgrep -f bench_he.py)"
echo "LLAMA pause targets: $LLAMA_PIDS" >&2
BOOST_WAS=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 1)
declare -a GF GV
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do GF+=("$f"); GV+=("$(cat "$f")"); done
restore(){
  for p in $LLAMA_PIDS; do kill -CONT "$p" 2>/dev/null; done
  for i in "${!GF[@]}"; do echo "${GV[$i]}" > "${GF[$i]}" 2>/dev/null; done
  echo "$BOOST_WAS" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
}
trap 'restore' EXIT INT TERM
setsid sh -c "sleep 600; for p in $LLAMA_PIDS; do kill -CONT \$p 2>/dev/null; done; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ondemand>\$f 2>/dev/null; done; echo 1>/sys/devices/system/cpu/cpufreq/boost 2>/dev/null" >/dev/null 2>&1 &
WD=$!
for p in $LLAMA_PIDS; do kill -STOP "$p" 2>/dev/null; done
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null; done
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null
sleep 1
echo "FROZEN gov=performance boost=0; llama STOP=[$(for p in $LLAMA_PIDS; do printf '%s:%s ' $p $(ps -o stat= -p $p 2>/dev/null|tr -d ' '); done)] watchdog=$WD N=$N" >&2

for cell in "${CELLS[@]}"; do
  IFS=: read -r corpus tag th cores <<<"$cell"
  CD="$OUT/$tag"; mkdir -p "$CD/spin" "$CD/sleep"
  echo "=== cell $tag (corpus=$corpus th=$th cores=$cores) ===" >&2

  # Gate-0: baseline sha == zcat (knob OFF); rg self-test sha == zcat.
  ref=$(zcat "$corpus" | sha256sum | cut -d' ' -f1)
  gsha=$(taskset -c "$cores" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -p "$th" -c "$corpus" 2>/dev/null | sha256sum | cut -d' ' -f1)
  rsha=$("$RG" -d -P "$th" -c "$corpus" 2>/dev/null | sha256sum | cut -d' ' -f1)
  sha_ok=$([ "$gsha" = "$ref" ] && echo 1 || echo 0)
  echo "  sha gz=$([ "$gsha" = "$ref" ] && echo OK || echo BAD) rg=$([ "$rsha" = "$ref" ] && echo OK || echo BAD)" >&2

  # chunk count (R_WORKER calls) -> max chunks per thread (ceil)
  chunks=$(taskset -c "$cores" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_REGION_PROF=1 GZIPPY_VERBOSE=1 "$GZ" -d -p "$th" -c "$corpus" 2>&1 >/dev/null | grep 'R_WORKER ' | grep -oE 'calls= *[0-9]+' | grep -oE '[0-9]+' | head -1)
  maxcpt=$(( (chunks + th - 1) / th ))
  [ "$maxcpt" -lt 1 ] && maxcpt=1

  # baseline wall (best-of-5) to size injection
  bw=99
  for i in 1 2 3 4 5; do w=$(gzwall "" "$corpus" "$th" "$cores"); awk -v a="$w" -v b="$bw" 'BEGIN{exit !(a<b)}' && bw=$w; done
  # t30 injects ~15% of baseline wall on the critical thread; region_self_ms = 0.5*baseline_wall (injection reference)
  pcn_t30=$(awk -v bw="$bw" -v m="$maxcpt" 'BEGIN{printf "%d", 0.15*bw*1e9/m}')
  pcn_t20=$(( pcn_t30 * 2 / 3 ))
  pcn_t10=$(( pcn_t30 / 3 ))
  region_self_ms=$(awk -v bw="$bw" 'BEGIN{printf "%.3f", 0.5*bw*1000}')
  echo "  chunks=$chunks maxcpt=$maxcpt baseline_wall=${bw}s region_self_ms=$region_self_ms per_chunk_ns t10=$pcn_t10 t20=$pcn_t20 t30=$pcn_t30" >&2

  # non-inert probe: ring inject hits == chunks
  hits=$(taskset -c "$cores" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_RING_INJECT_NS=$pcn_t10 "$GZ" -d -p "$th" -c "$corpus" 2>&1 >/dev/null | grep 'RING inject' | grep -oE 'hits\(chunks\)=[0-9]+' | grep -oE '[0-9]+')
  echo "  non-inert RING hits=$hits (expect ~$chunks)" >&2

  : > "$CD/baseline.txt"; : > "$CD/baseline_recheck.txt"; : > "$CD/oracle_removed.txt"
  for p in 10 20 30; do : > "$CD/spin/t$p.txt"; : > "$CD/sleep/t$p.txt"; done

  # interleaved N reps: each rep runs every arm once (order fixed, drift hits all equally)
  for r in $(seq 1 "$N"); do
    gzwall "" "$corpus" "$th" "$cores" >> "$CD/baseline.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t10" "$corpus" "$th" "$cores" >> "$CD/spin/t10.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t20" "$corpus" "$th" "$cores" >> "$CD/spin/t20.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t30" "$corpus" "$th" "$cores" >> "$CD/spin/t30.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t10 GZIPPY_SLOW_KIND=sleep" "$corpus" "$th" "$cores" >> "$CD/sleep/t10.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t20 GZIPPY_SLOW_KIND=sleep" "$corpus" "$th" "$cores" >> "$CD/sleep/t20.txt"
    gzwall "GZIPPY_RING_INJECT_NS=$pcn_t30 GZIPPY_SLOW_KIND=sleep" "$corpus" "$th" "$cores" >> "$CD/sleep/t30.txt"
    gzwall "GZIPPY_FOLD_NODRAIN=1" "$corpus" "$th" "$cores" >> "$CD/oracle_removed.txt"
    gzwall "" "$corpus" "$th" "$cores" >> "$CD/baseline_recheck.txt"
  done

  {
    echo "region=ring_other(worker_residual)"
    echo "region_self_ms=$region_self_ms"
    echo "perturb_cmd=GZIPPY_RING_INJECT_NS spin/sleep + GZIPPY_FOLD_NODRAIN oracle (ring_other residual; does NOT isolate the fold-drain copy)"
    echo "sha_ok=$sha_ok"
    echo "cell_id=$tag"
    echo "freeze_state=gov=performance,boost=0,llama=STOP"
    echo "quiet_state=taskset $cores, /dev/null both arms, N=$N interleaved"
  } > "$CD/meta.txt"
  echo "  cell $tag done." >&2
done

restore; kill "$WD" 2>/dev/null; trap - EXIT INT TERM
echo "=== RESTORED gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) boost=$(cat /sys/devices/system/cpu/cpufreq/boost) llama=[$(for p in $LLAMA_PIDS; do printf '%s:%s ' $p $(ps -o stat= -p $p 2>/dev/null|tr -d ' '); done)] ===" >&2
echo "OUT=$OUT"
