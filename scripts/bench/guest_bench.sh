#!/usr/bin/env bash
# guest_bench.sh — interleaved gzippy-vs-rapidgzip wall benchmark.
#
# RUNS ON GUEST 199 (trainer). Invoked by host_lock_and_bench.sh over one ssh
# hop. The host has already locked frequency + frozen noisy neighbours and the
# gate has PASSED; this script just measures, with PROVENANCE-OR-NO-NUMBERS.
#
# Non-overridable correctness constants (NOT flags):
#   - iteration 0 is DROPPED (cache/JIT warmup).
#   - all output goes to /dev/null.
#   - EVERY run's stdout sha256 must match the gzip(1) reference decompression;
#     a single divergence makes the whole run UNTRUSTWORTHY.
#
# Args (env or flags):
#   THREADS="4 8 16"      thread counts to sweep (default "8")
#   N=9                   trials per cell (iter0 dropped => 9 kept; min 9)
#   --allow-dirty         stamp DIRTY loudly instead of aborting
#   --lever               causal-mode GZIPPY_SLOW_BOOTSTRAP sweep (opt-in)
#
# Emits a results table with min/median/sd/ratio per cell; sd>5% => cell FAILED;
# top-level RUN_TRUSTWORTHY=true only if provenance clean + all shas match + no
# cell sd>5%. GZIPPY_TIMELINE traces saved as artifacts (NOT auto-interpreted).
set -u

# ---- config -----------------------------------------------------------------
REPO="${REPO:-/root/gzippy}"
RAPIDGZIP="${RAPIDGZIP:-rapidgzip}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
THREADS="${THREADS:-8}"
N="${N:-9}"
SD_FAIL_PCT="${SD_FAIL_PCT:-5}"
ALLOW_DIRTY=0
LEVER=0
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts}"

for a in "$@"; do
  case "$a" in
    --allow-dirty) ALLOW_DIRTY=1;;
    --lever) LEVER=1;;
    T=*|THREADS=*) THREADS="$(echo "${a#*=}" | tr ',' ' ')";;
    N=*) N="${a#*=}";;
  esac
done
[ "$N" -ge 9 ] || N=9   # non-overridable floor

mkdir -p "$ARTDIR"
say() { echo "$@"; }

# ---- pin masks per thread count ---------------------------------------------
pin_mask() {
  case "$1" in
    4)  echo "0,2,4,6";;
    8)  echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *)  echo "";;
  esac
}
pin_label() { [ "$1" = 16 ] && echo "(T16 = SMT/oversubscribed P-cores)" || echo "(1 thread / P-core)"; }

# ---- PROVENANCE -------------------------------------------------------------
cd "$REPO" || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=no-repo $REPO"; exit 5; }

# Provenance dirtiness = gzippy's OWN tracked files + submodule POINTERS.
# Uncommitted edits INSIDE a vendor submodule (e.g. trace patches in
# vendor/rapidgzip) do not affect the gzippy binary, so --ignore-submodules=dirty
# excludes them while STILL catching a moved submodule pointer or any gzippy
# source change. The exact submodule commit is recorded below for full
# traceability. Untracked files DO count (they could be uncommitted source).
DIRTY_COUNT="$(git status --porcelain --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')"
SUBMOD_STATUS="$(git submodule status 2>/dev/null | awk '{printf "%s@%s ",$2,substr($1,1,12)}')"
PROV_DIRTY=0
if [ "$DIRTY_COUNT" != "0" ]; then
  if [ "$ALLOW_DIRTY" = 1 ]; then
    PROV_DIRTY=1
    say "## !!!!! WORKING TREE DIRTY ($DIRTY_COUNT files) — --allow-dirty: numbers are NOT publication-grade !!!!!"
  else
    echo "RUN_TRUSTWORTHY=false"
    echo "FAILURE=dirty-tree ($DIRTY_COUNT files); pass --allow-dirty to override (stamps DIRTY)"
    exit 6
  fi
fi

GZIPPY_SHA="$(git describe --always --dirty --tags 2>/dev/null || git rev-parse --short HEAD)"
GZIPPY_BRANCH="$(git branch --show-current 2>/dev/null)"
RG_VER="$($RAPIDGZIP --version 2>&1 | head -1)"
KERNEL="$(uname -r)"
CPU_MODEL="$(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ //')"
MICROCODE="$(awk -F: '/microcode/{print $2; exit}' /proc/cpuinfo | tr -d ' ')"

[ -f "$CORPUS" ] || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=corpus missing $CORPUS"; exit 7; }
CORPUS_SHA="$(sha256sum "$CORPUS" | cut -d' ' -f1)"

# ---- build gzippy (pure-rust-inflate, the sole production decode path) -------
say "## building gzippy --release --no-default-features --features pure-rust-inflate ..."
if ! cargo build --release --no-default-features --features pure-rust-inflate >"$ARTDIR/build.log" 2>&1; then
  echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=gzippy build failed (see artifacts/build.log)"
  tail -20 "$ARTDIR/build.log"
  exit 8
fi
GZIPPY="$REPO/target/release/gzippy"
[ -x "$GZIPPY" ] || { echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=gzippy binary not found at $GZIPPY"; exit 8; }

# ---- assert production routing ----------------------------------------------
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p 8 "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
say "## routing assert: $DBG"
case "$DBG" in
  *IsalParallelSM*) : ;;
  *) echo "RUN_TRUSTWORTHY=false"; echo "FAILURE=routing not IsalParallelSM: $DBG"; exit 9;;
esac

# ---- gzip(1) correctness reference ------------------------------------------
say "## computing gzip(1) reference sha256 ..."
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# prewarm corpus into page cache
cat "$CORPUS" >/dev/null 2>&1

# ---- provenance block (always printed) --------------------------------------
say "================ PROVENANCE ================"
say "gzippy:     $GZIPPY_SHA (branch $GZIPPY_BRANCH) dirty=$DIRTY_COUNT"
say "submodules: ${SUBMOD_STATUS:-<none>}"
say "rapidgzip:  $RG_VER"
say "corpus:     $CORPUS"
say "corpus_sha: $CORPUS_SHA"
say "ref_sha:    $REF_SHA  raw_bytes=$RAW_BYTES"
say "kernel:     $KERNEL"
say "cpu:        $CPU_MODEL  microcode=$MICROCODE"
say "trials/cell N=$N (iter0 dropped, output /dev/null, sha-verified each run)"
say "==========================================="

# ---- the interleaved bench engine -------------------------------------------
# returns via globals: arrays of per-trial seconds; sets DIVERGED on any sha miss
run_cmd_timed() { # run_cmd_timed <mask> <cmd...> ; echoes "secs sha"
  local mask="$1"; shift
  local out s e secs sha
  out="$(mktemp)"
  s=$(date +%s.%N)
  taskset -c "$mask" "$@" >"$out" 2>/dev/null
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  rm -f "$out"
  echo "$secs $sha"
}

stats() { # stats <space-separated-times> ; echoes "min median sd_pct"
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1; sum+=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1];
      mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      mean=sum/n; ss=0; for(i=1;i<=n;i++){d=v[i]-mean; ss+=d*d}
      sd=(n>1)?sqrt(ss/(n-1)):0;
      sdpct=(mean>0)?sd/mean*100:0;
      printf "%.4f %.4f %.1f", min, mid, sdpct;
    }'
}

DIVERGED=0
ANY_SD_FAIL=0
say ""
say "================ RESULTS ================="
printf "%-5s %-10s %-9s %-9s %-7s %-9s %-9s %-7s %s\n" \
  "T" "tool" "min(s)" "med(s)" "sd%" "MB/s" "ratio" "verdict" "pins"

declare -A REL

for T in $THREADS; do
  mask="$(pin_mask "$T")"
  if [ -z "$mask" ]; then say "## skip T=$T (no pin mask)"; continue; fi
  lbl="$(pin_label "$T")"

  GZ_T=""; RG_T=""
  # interleaved: each iteration runs gzippy then rapidgzip back-to-back so both
  # see identical per-trial contention. iter 0 dropped.
  for ((i=0;i<=N;i++)); do
    read gsec gsha < <(run_cmd_timed "$mask" \
        env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS")
    read rsec rsha < <(run_cmd_timed "$mask" \
        "$RAPIDGZIP" -d -c -f -P "$T" "$CORPUS")
    if [ "$i" -eq 0 ]; then continue; fi   # warmup drop
    GZ_T="$GZ_T $gsec"; RG_T="$RG_T $rsec"
    [ "$gsha" = "$REF_SHA" ] || { say "## !! CORRECTNESS DIVERGENCE gzippy T=$T trial=$i ($gsha != ref)"; DIVERGED=1; }
    [ "$rsha" = "$REF_SHA" ] || { say "## !! CORRECTNESS DIVERGENCE rapidgzip T=$T trial=$i ($rsha != ref)"; DIVERGED=1; }
  done

  read gmin gmed gsd < <(stats "$GZ_T")
  read rmin rmed rsd < <(stats "$RG_T")

  gmb=$(awk -v r="$RAW_BYTES" -v t="$gmin" 'BEGIN{print (t>0)?sprintf("%.0f",r/t/1e6):"0"}')
  rmb=$(awk -v r="$RAW_BYTES" -v t="$rmin" 'BEGIN{print (t>0)?sprintf("%.0f",r/t/1e6):"0"}')

  # ratio = rapidgzip_time / gzippy_time = gzippy_tput / rapidgzip_tput
  ratio=$(awk -v rg="$rmin" -v gz="$gmin" 'BEGIN{printf "%.3f", (gz>0)?rg/gz:0}')
  margin=$(awk -v a="$gsd" -v b="$rsd" 'BEGIN{m=(a>b)?a:b; print m/100}')
  verdict=$(awk -v r="$ratio" -v m="$margin" 'BEGIN{d=r-1; if(d>m)print "WIN"; else if(d<-m)print "LOSS"; else print "TIE"}')

  gflag=""; rflag=""
  awk -v s="$gsd" -v f="$SD_FAIL_PCT" 'BEGIN{exit !(s>f)}' && { gflag="FAILED"; ANY_SD_FAIL=1; }
  awk -v s="$rsd" -v f="$SD_FAIL_PCT" 'BEGIN{exit !(s>f)}' && { rflag="FAILED"; ANY_SD_FAIL=1; }

  printf "%-5s %-10s %-9s %-9s %-7s %-9s %-9s %-7s %s\n" \
    "$T" "gzippy" "$gmin" "$gmed" "$gsd${gflag:+!}" "$gmb" "$ratio" "$verdict" "$lbl"
  printf "%-5s %-10s %-9s %-9s %-7s %-9s %-9s %-7s %s\n" \
    "$T" "rapidgzip" "$rmin" "$rmed" "$rsd${rflag:+!}" "$rmb" "1.000" "ref" ""

  REL[$T]="$ratio:$verdict"

  # capture a timeline trace ARTIFACT (saved, not interpreted)
  GZIPPY_TIMELINE="$ARTDIR/timeline_T${T}.json" \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$mask" \
    "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ARTDIR/timeline.log" || true
done

# ---- optional causal lever (opt-in, NOT default) ----------------------------
if [ "$LEVER" = 1 ]; then
  say ""
  say "============== CAUSAL LEVER (GZIPPY_SLOW_BOOTSTRAP) =============="
  say "## wall-elasticity: spin-vs-sleep frequency-neutral control. T=8 mask."
  mask="$(pin_mask 8)"
  for mode in spin sleep; do
    sleepvar=""; [ "$mode" = sleep ] && sleepvar="GZIPPY_SLOW_BOOTSTRAP_SLEEP=1"
    for pct in 0 50 100; do
      env_mode="GZIPPY_SLOW_BOOTSTRAP=$pct $sleepvar"
      acc=""
      for ((i=0;i<=5;i++)); do
        read s _ < <(run_cmd_timed "$mask" \
          env GZIPPY_FORCE_PARALLEL_SM=1 $env_mode "$GZIPPY" -d -c -p 8 "$CORPUS")
        [ "$i" -eq 0 ] && continue
        acc="$acc $s"
      done
      read mn _ _ < <(stats "$acc")
      printf "  mode=%-5s slow=%-3s%% min=%ss\n" "$mode" "$pct" "$mn"
    done
  done
fi

# ---- top-level trust verdict ------------------------------------------------
TRUST=true
[ "$DIVERGED" = 0 ] || TRUST=false
[ "$ANY_SD_FAIL" = 0 ] || TRUST=false
[ "$PROV_DIRTY" = 0 ] || TRUST=false

say ""
say "================ VERDICT =================="
say "diverged=$DIVERGED  any_sd>${SD_FAIL_PCT}%=$ANY_SD_FAIL  dirty=$PROV_DIRTY"
say "RUN_TRUSTWORTHY=$TRUST"
say "artifacts: $ARTDIR (timelines NOT auto-interpreted; for manual fulcrum only)"
say "=========================================="
[ "$TRUST" = true ]
