#!/usr/bin/env bash
# _marker_probe_guest.sh — MARKER-mode decode causal-perturbation probe (guest side).
#
# Measurement-only. Perturbs ONLY the u16 marker (CONTAINS_MARKERS=true) careful
# decode loop via GZIPPY_SLOW_MARKER_MODE (slow_knob.rs), proves it fired with the
# MARKER-specific GZIPPY_SLOW_MARKER_HITS counter, and runs a Gate-2 dose-response
# (spin + frequency-neutral sleep control) vs rapidgzip on a /dev/null sink.
set -u

SRC=${SRC:-/mnt/internal/gz-marker}
TARGET=${TARGET:-/dev/shm/gz-marker-target}
BIN=$TARGET/release/gzippy
RG=${RG:-/root/oracle_c/rapidgzip-native}
T=${T:-4}
MASK=${MASK:-0,2,4,6}
N=${N:-8}
ART=${ART:-/dev/shm/marker-art}
DO_BUILD=${DO_BUILD:-0}
mkdir -p "$ART"

say(){ echo "## $*"; }
fatal(){ echo "MARKER_FAIL=$1"; echo "MARKER_DONE"; exit "${2:-1}"; }

# ---- /dev/null sink integrity (HARD GATE: same sink both arms, never rm) ------
[ -c /dev/null ] || fatal "devnull-not-char-dev(before)" 2

# ---- build (optional) --------------------------------------------------------
if [ "$DO_BUILD" = 1 ]; then
  say "building gzippy-native (target-cpu=native) into $TARGET"
  cd "$SRC" || fatal "no-src:$SRC" 3
  RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR="$TARGET" \
    cargo build --release --no-default-features --features gzippy-native \
    2>&1 | grep -E 'Compiling gzippy|Finished|^error' | tail -8
fi
[ -x "$BIN" ] || fatal "no-binary:$BIN (run with DO_BUILD=1)" 3
echo "BIN_SHA=$(sha256sum "$BIN" | cut -c1-16)  head=$(cd "$SRC" && git rev-parse --short HEAD 2>/dev/null || echo NA)"

# ---- production-path assert ---------------------------------------------------
DBG=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p "$T" /root/silesia.gz 2>&1 >/dev/null | grep -m1 'path=' || true)
echo "ROUTING: ${DBG:-none}"
case "$DBG" in *ParallelSM*) ;; *) fatal "routing-not-ParallelSM:${DBG:-none}" 4;; esac

# ---- host freeze readback (informational; freeze is held from the mac) --------
echo "GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null||echo NA) NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null||echo NA) runnable=$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null||echo NA)"

# ============================================================================
# 1. MARKER-KNOB FIRED PROOF (instrument self-validation, GATE ZERO)
#    Run with both hit counters ON at mode=0 (count only, no perturbation) and
#    at mode=100. Marker hits must be > 0 and dose-independent (event count is a
#    property of the input, not the injection). Capture GZIPPY_VERBOSE for the
#    flip_to_clean / chunk context.
# ============================================================================
proof_corpus=/root/silesia.gz
for mode in 0 100; do
  LOG="$ART/hits_silesia_mode${mode}.log"
  GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 \
    GZIPPY_SLOW_HITS=1 GZIPPY_SLOW_MARKER_HITS=1 \
    GZIPPY_SLOW_MARKER_MODE=$mode \
    taskset -c "$MASK" "$BIN" -d -c -p "$T" "$proof_corpus" >/dev/null 2>"$LOG" || true
  CLEAN_H=$(grep -m1 'clean-loop inject hits' "$LOG" | grep -oE '[0-9]+' | tail -1)
  MARK_H=$(grep -m1 'marker-loop inject hits' "$LOG" | grep -oE '[0-9]+' | tail -1)
  echo "FIRED-PROOF mode=$mode : marker_hits=${MARK_H:-NA} clean_hits=${CLEAN_H:-NA}"
done
echo "--- GZIPPY_VERBOSE context (silesia) ---"
grep -iE 'chunk|flip|marker|window|isal_chunks' "$ART/hits_silesia_mode0.log" | head -25 || true
echo "----------------------------------------"

# Per-dose marker event-count elasticity (untimed; HITS counter ON). Documents
# whether the marker-mode WORK VOLUME itself changes with the injected slowdown
# (speculative-pipeline feedback). Captured separately so the timed runs below
# stay clean (no HITS atomic in the hot path).
echo "--- marker event-count elasticity (silesia, untimed) ---"
for mk in spin sleep; do for m in 0 50 100 200; do
  ek=""; [ "$mk" = sleep ] && ek="GZIPPY_SLOW_KIND=sleep"
  GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SLOW_MARKER_HITS=1 GZIPPY_SLOW_MARKER_MODE=$m $ek \
    taskset -c "$MASK" "$BIN" -d -c -p "$T" "$proof_corpus" >/dev/null 2>"$ART/el.log" || true
  mh=$(grep -m1 'marker-loop inject hits' "$ART/el.log" | grep -oE '[0-9]+' | tail -1)
  echo "elasticity kind=$mk mode=$m : marker_hits=${mh:-NA}"
done; done
echo "--------------------------------------------------------"

# ---- dose-0 SHA correctness gate (byte-exact vs gzip + rg) -------------------
sha_gate(){ # $1 corpus
  local c="$1" ref gz rgsha
  ref=$(gzip -dc "$c" | sha256sum | cut -d' ' -f1)
  GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p "$T" "$c" >"$ART/sha_gz.bin" 2>/dev/null || true
  gz=$(sha256sum "$ART/sha_gz.bin" | cut -d' ' -f1)
  "$RG" -d -c -f -P "$T" "$c" >"$ART/sha_rg.bin" 2>/dev/null || true
  rgsha=$(sha256sum "$ART/sha_rg.bin" | cut -d' ' -f1)
  rm -f "$ART/sha_gz.bin" "$ART/sha_rg.bin"
  if [ "$gz" = "$ref" ] && [ "$rgsha" = "$ref" ]; then
    echo "SHA-GATE $c : OK (gz==rg==gzip ref ${ref:0:16})"
  else
    fatal "sha-mismatch $c gz=${gz:0:16} rg=${rgsha:0:16} ref=${ref:0:16}" 5
  fi
}

# ============================================================================
# 2. DOSE-RESPONSE (Gate-2). Interleaved best-of-N on /dev/null sink.
#    Conditions per corpus: rg baseline; gz {0,50,100,200}% spin; gz {100,200}% sleep.
# ============================================================================
nowns(){ date +%s.%N; }
run_gz(){ # $1 corpus  $2 mode  $3 kind(spin|sleep)
  local c="$1" m="$2" k="$3" s e env=""
  [ "$m" -gt 0 ] && env="GZIPPY_SLOW_MARKER_MODE=$m"
  [ "$k" = sleep ] && env="$env GZIPPY_SLOW_KIND=sleep"
  s=$(nowns)
  env GZIPPY_FORCE_PARALLEL_SM=1 $env taskset -c "$MASK" "$BIN" -d -c -p "$T" "$c" >/dev/null 2>>"$ART/run.stderr"
  e=$(nowns); awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}'
}
run_rg(){ local c="$1" s e; s=$(nowns)
  "$RG" -d -c -f -P "$T" "$c" >/dev/null 2>>"$ART/run.stderr"
  e=$(nowns); awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}'; }

stats(){ tr ' ' '\n' <<<"$1" | grep -v '^$' | sort -n | awk '
  {v[NR]=$1} END{n=NR; if(n==0){print "NA NA NA";exit}
   min=v[1];max=v[n];med=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
   printf "%.4f %.4f %.0f", min, med, (min>0)?(max-min)/min*100:0}'; }

declare -A REG
CONDS="rg gz0spin gz50spin gz100spin gz200spin gz100sleep gz200sleep"

for corpus in /root/silesia.gz /root/squishy.gz; do
  name=$(basename "$corpus" .gz)
  say "===== CORPUS=$name (T=$T mask=$MASK N=$N) ====="
  sha_gate "$corpus"
  for c in $CONDS; do REG[$name.$c]=""; done
  # warmup round (i=0 discarded)
  for i in $(seq 0 "$N"); do
    [ -c /dev/null ] || fatal "devnull-vanished" 2
    t=$(run_rg "$corpus");         [ "$i" -gt 0 ] && REG[$name.rg]+=" $t"
    t=$(run_gz "$corpus" 0 spin);  [ "$i" -gt 0 ] && REG[$name.gz0spin]+=" $t"
    t=$(run_gz "$corpus" 50 spin); [ "$i" -gt 0 ] && REG[$name.gz50spin]+=" $t"
    t=$(run_gz "$corpus" 100 spin);[ "$i" -gt 0 ] && REG[$name.gz100spin]+=" $t"
    t=$(run_gz "$corpus" 200 spin);[ "$i" -gt 0 ] && REG[$name.gz200spin]+=" $t"
    t=$(run_gz "$corpus" 100 sleep);[ "$i" -gt 0 ] && REG[$name.gz100sleep]+=" $t"
    t=$(run_gz "$corpus" 200 sleep);[ "$i" -gt 0 ] && REG[$name.gz200sleep]+=" $t"
  done
  echo ""
  echo "==== DOSE-RESPONSE $name (min med spread%) ===="
  for c in $CONDS; do
    read mn md sp < <(stats "${REG[$name.$c]}")
    printf "%-12s min=%-8s med=%-8s spread=%s%%\n" "$c" "$mn" "$md" "$sp"
  done
  # rg ratio + per-dose delta vs gz0
  read rgmn _ _ < <(stats "${REG[$name.rg]}")
  read g0 _ _ < <(stats "${REG[$name.gz0spin]}")
  echo "rg/gz0 ratio = $(awk -v r="$rgmn" -v g="$g0" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')  (>1 means gz faster; gap=gz-rg)"
  for c in gz50spin gz100spin gz200spin gz100sleep gz200sleep; do
    read m _ _ < <(stats "${REG[$name.$c]}")
    echo "  Δ($c - gz0) = $(awk -v a="$m" -v b="$g0" 'BEGIN{printf "%+.4f s (%+.1f%%)", a-b, (b>0)?(a-b)/b*100:0}')"
  done
done

[ -c /dev/null ] || fatal "devnull-not-char-dev(after)" 2
echo "MARKER_DONE"
