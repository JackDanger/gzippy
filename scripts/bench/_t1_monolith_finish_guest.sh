#!/usr/bin/env bash
# _t1_monolith_finish_guest.sh — GATE the T1-MONOLITH-STREAMING native path vs
# igzip (the ISA-L C single-shot bar) and vs the thin-T1 baseline.
#
# Box is un-freezable (turbo on, governor powersave) → taskset -c PIN +
# interleaved randomized best-of-N (wall) + perf cyc/byte (frequency-invariant
# mechanism) + freq-stability gate. /dev/null both arms. sha==zcat checked.
#
# Arms (all gzippy at -p1 except igzip):
#   mono   : streaming-monolith native T1 default  (gzippy -d -c -p1)
#   mono2  : A/A self-test of mono
#   thin   : thin-T1 baseline (GZIPPY_NO_MONOLITH=1) — the BEFORE
#   igzip  : /usr/bin/igzip -d -c                    — the BAR
set -u
BIN=${BIN:-/dev/shm/t1mono-target/release/gzippy}
IGZIP=${IGZIP:-/usr/bin/igzip}
PIN=${PIN:-4}
WREPS=${WREPS:-11}      # wall best-of-N
PREPS=${PREPS:-7}       # perf samples per arm
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260622}
ART=${ART:-/dev/shm/t1mono-finish-art}
mkdir -p "$ART"; rm -f "$ART"/*.csv 2>/dev/null

echo "== T1-MONOLITH-FINISH gate (pin=cpu$PIN wreps=$WREPS preps=$PREPS) =="
echo "BIN sha=$(sha256sum "$BIN" | cut -c1-12)  HEAD=$(cd /mnt/internal/gz-head && git rev-parse --short HEAD)"
echo "load_start: $(cat /proc/loadavg)"
echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu$PIN/cpufreq/scaling_governor 2>/dev/null)"

ARMS="mono mono2 thin igzip"
run_arm() { # $1=arm $2=file  -> prints elapsed ns to stdout, writes nothing else
  local a=$1 f=$2 t0 t1
  t0=$(date +%s%N)
  case $a in
    mono|mono2) taskset -c "$PIN" "$BIN" -d -c -p 1 "$f" >/dev/null 2>/dev/null ;;
    thin)       env GZIPPY_NO_MONOLITH=1 taskset -c "$PIN" "$BIN" -d -c -p 1 "$f" >/dev/null 2>/dev/null ;;
    igzip)      taskset -c "$PIN" "$IGZIP" -d -c "$f" >/dev/null 2>/dev/null ;;
  esac
  t1=$(date +%s%N)
  echo $(( (t1 - t0) / 1000000 ))   # ms (integer)
}

perf_arm() { # $1=arm $2=file -> "cycles instructions minorfaults elapsed_s"
  local a=$1 f=$2 csv t0 t1; csv=$(mktemp)
  t0=$(date +%s%N)
  case $a in
    mono|mono2) taskset -c "$PIN" perf stat -x, -e cpu_core/cycles/,cpu_core/instructions/,minor-faults -- "$BIN" -d -c -p 1 "$f" >/dev/null 2>"$csv" ;;
    thin)       env GZIPPY_NO_MONOLITH=1 taskset -c "$PIN" perf stat -x, -e cpu_core/cycles/,cpu_core/instructions/,minor-faults -- "$BIN" -d -c -p 1 "$f" >/dev/null 2>"$csv" ;;
    igzip)      taskset -c "$PIN" perf stat -x, -e cpu_core/cycles/,cpu_core/instructions/,minor-faults -- "$IGZIP" -d -c "$f" >/dev/null 2>"$csv" ;;
  esac
  t1=$(date +%s%N)
  local cyc ins flt el
  cyc=$(awk -F, '/cpu_core\/cycles\//{print $1}' "$csv" | head -1)
  ins=$(awk -F, '/cpu_core\/instructions\//{print $1}' "$csv" | head -1)
  flt=$(awk -F, '/minor-faults/{print $1}' "$csv" | head -1)
  el=$(awk -v ns=$(( t1 - t0 )) 'BEGIN{printf "%.6f", ns/1e9}')
  rm -f "$csv"
  echo "${cyc:-0} ${ins:-0} ${flt:-0} ${el:-0}"
}

for corp in $CORPORA; do
  F=$GZDIR/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE"; continue; }
  REF=$(zcat "$F" | wc -c)
  echo "--- $corp  ref_bytes=$REF ---"
  # byte-exact sha==zcat (each arm once)
  rz=$(zcat "$F" | sha256sum | cut -c1-16)
  for a in mono thin igzip; do
    case $a in
      mono)  s=$(taskset -c "$PIN" "$BIN" -d -c -p1 "$F" 2>/dev/null | sha256sum | cut -c1-16) ;;
      thin)  s=$(env GZIPPY_NO_MONOLITH=1 taskset -c "$PIN" "$BIN" -d -c -p1 "$F" 2>/dev/null | sha256sum | cut -c1-16) ;;
      igzip) s=$(taskset -c "$PIN" "$IGZIP" -d -c "$F" 2>/dev/null | sha256sum | cut -c1-16) ;;
    esac
    [ "$s" = "$rz" ] && echo "  sha $a OK" || { echo "  sha $a MISMATCH ($s vs $rz) — ABORT"; exit 1; }
  done

  # ---- Phase A: interleaved best-of-N wall ----
  declare -A BEST WORST
  for a in $ARMS; do BEST[$a]=99999999; WORST[$a]=0; done
  for r in $(seq 1 "$WREPS"); do
    order=$(echo $ARMS | tr ' ' '\n' | shuf --random-source=<(yes "$SEED$corp$r"))
    for a in $order; do
      ms=$(run_arm "$a" "$F")
      awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
      awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
    done
  done
  for a in $ARMS; do
    sp=$(( ${WORST[$a]} - ${BEST[$a]} ))
    printf "  wall %-6s best=%6s ms  spread=%5s ms\n" "$a" "${BEST[$a]}" "$sp"
  done
  aa=$(( ${BEST[mono]} > ${BEST[mono2]} ? ${BEST[mono]} - ${BEST[mono2]} : ${BEST[mono2]} - ${BEST[mono]} ))
  printf "  A/A |mono-mono2| = %s ms\n" "$aa"
  printf "  RATIO mono/igzip = %s   thin/igzip = %s   mono/thin = %s\n" \
    "$(awk -v k=${BEST[mono]} -v c=${BEST[igzip]} 'BEGIN{printf "%.3f",k/c}')" \
    "$(awk -v k=${BEST[thin]} -v c=${BEST[igzip]} 'BEGIN{printf "%.3f",k/c}')" \
    "$(awk -v k=${BEST[mono]} -v c=${BEST[thin]} 'BEGIN{printf "%.3f",k/c}')"

  # ---- Phase B: perf cyc/byte + faults (mechanism + artifact) ----
  for a in mono thin igzip; do
    bc=99999999999; bi=0; bf=0; ge_min=999; ge_max=0
    for r in $(seq 1 "$PREPS"); do
      read cyc ins flt el <<<"$(perf_arm "$a" "$F")"
      echo "$corp,$a,$r,$cyc,$ins,$flt,$el,$REF" >> "$ART/samples.csv"
      # best (min) cycles as the representative cyc
      awk -v c="$cyc" -v b="$bc" 'BEGIN{exit !(c+0<b+0)}' && { bc=$cyc; bi=$ins; bf=$flt; }
      # GHz for freq-stability gate
      ghz=$(awk -v c="$cyc" -v e="$el" 'BEGIN{ if(e>0) printf "%.3f", c/e/1e9; else print 0}')
      awk -v g="$ghz" -v m="$ge_min" 'BEGIN{exit !(g+0<m+0)}' && ge_min=$ghz
      awk -v g="$ghz" -v m="$ge_max" 'BEGIN{exit !(g+0>m+0)}' && ge_max=$ghz
    done
    cpb=$(awk -v c="$bc" -v b="$REF" 'BEGIN{printf "%.3f", c/b}')
    ipb=$(awk -v i="$bi" -v b="$REF" 'BEGIN{printf "%.3f", i/b}')
    gspread=$(awk -v a="$ge_min" -v b="$ge_max" 'BEGIN{ if(b>0) printf "%.1f", (b-a)/b*100; else print 0}')
    printf "  perf %-6s cyc/B=%6s  ins/B=%6s  minorfaults=%9s  GHz[%s..%s] spread=%s%%\n" \
      "$a" "$cpb" "$ipb" "$bf" "$ge_min" "$ge_max" "$gspread"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "samples -> $ART/samples.csv"
echo "== DONE =="
