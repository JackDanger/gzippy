#!/usr/bin/env bash
# guest_step0.sh — TIER-3 STEP-0 two discriminators (locked guest 199).
#
# (a) PARENT-CACHED-AT-STALL probe (GZIPPY_STALL_RESIDENCY_PROBE): at T8 on
#     silesia-large, classify whether the chunk CONTAINING each cold-get stall is
#     resident (CONTAINING_CACHED/IN_FLIGHT) or evicted (NOT_RESIDENT). Positive
#     control: GZIPPY_PREFETCH_CACHE_CAP {1=tiny, large} must move NOT_RESIDENT.
#     Byte-exact: every run sha-verified == gzip ref.
# (b) CONSUMER-BLOCK DECOMPOSE: capture clean-only trace_seed (placement-perfect
#     operating point) for the decode-wait/serial split (analyzed off-box by
#     consumer_block_decompose.py). Positive control: a SLOW-injected clean-only
#     trace — DECODE-WAIT must rise, SERIAL-BOOKKEEPING stay ~flat.
#
# Runs UNDER host_lock_and_bench.sh (freq pinned, watchdog armed). All config via
# args (the host invokes us over a bare ssh — no env passthrough).
# Args: BRANCH=reimplement-isa-l THREADS=8
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
THREADS="${THREADS:-8}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-step0}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
SEEDF="${SEEDF:-/dev/shm/gz_seed_windows_step0.bin}"

for a in "$@"; do
  case "$a" in
    BRANCH=*) BRANCH="${a#*=}";;
    THREADS=*) THREADS="$(echo "${a#*=}" | tr ',' ' ')";;
  esac
done
T="$(echo "$THREADS" | awk '{print $1}')"; [ -n "$T" ] || T=8

mkdir -p "$ARTDIR"
say() { echo "$@"; }
pin_mask() { case "$1" in 8) echo "0,2,4,6,8,10,12,14";; 4) echo "0,2,4,6";; 1) echo "0";; *) echo "0,2,4,6,8,10,12,14";; esac; }
mask="$(pin_mask "$T")"

# ---- sync + build -----------------------------------------------------------
cd "$REPO" || { echo "STEP0_FAILURE=no-repo"; echo "STEP0_GUEST_DONE"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true
DIRTY="$(git status --porcelain --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')"
[ "$DIRTY" = "0" ] || { echo "STEP0_FAILURE=dirty-tree ($DIRTY)"; echo "STEP0_GUEST_DONE"; exit 6; }

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
FEATURES="${GZIPPY_BUILD_FEATURES:-gzippy-native}"
say "## build gzippy ($FEATURES) @ $(git rev-parse --short HEAD) ..."
if ! cargo build --release --no-default-features --features "$FEATURES" >"$ARTDIR/build.log" 2>&1; then
  echo "STEP0_FAILURE=build"; tail -30 "$ARTDIR/build.log"; echo "STEP0_GUEST_DONE"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"
[ -f "$CORPUS" ] || { echo "STEP0_FAILURE=corpus"; echo "STEP0_GUEST_DONE"; exit 7; }
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
cat "$CORPUS" >/dev/null 2>&1

# routing assertion
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
case "$DBG" in *ParallelSM*) ;; *) echo "STEP0_FAILURE=routing $DBG"; echo "STEP0_GUEST_DONE"; exit 9;; esac

say "================ STEP-0 PROVENANCE ================"
say "branch=$BRANCH head=$(git rev-parse HEAD) T=$T"
say "corpus=$CORPUS ref_sha=$REF_SHA"
say "freq: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA) gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
say "=================================================="

drop_caches() { sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# ============================================================================
# DISCRIMINATOR (a) — parent-cached-at-stall (+ positive control + trace)
# ============================================================================
say ""; say "########## DISCRIMINATOR (a): PARENT-CACHED-AT-STALL ##########"

run_probe() { # <label> <extra-env...>   -> emits tally + sha
  local label="$1"; shift
  drop_caches
  local out; out="$(mktemp)"
  env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_STALL_RESIDENCY_PROBE=1 "$@" \
    taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$out" 2>"$ARTDIR/a_${label}.err" || true
  local sha; sha="$(sha_of "$out")"; rm -f "$out"
  local ok="MISMATCH"; [ "$sha" = "$REF_SHA" ] && ok="sha=OK"
  say "[a:$label] $ok"
  grep -m1 'STALL_RESIDENCY_PROBE:' "$ARTDIR/a_${label}.err" | sed "s/^/[a:$label] /"
}

# default caps (the verdict run) — also capture a trace for per-stall detail
drop_caches
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_STALL_RESIDENCY_PROBE=1 \
  GZIPPY_LOG_FILE="$ARTDIR/a_default_trace.log" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$ARTDIR/a_default.out" 2>"$ARTDIR/a_default.err" || true
A_DEF_SHA="$(sha_of "$ARTDIR/a_default.out")"; rm -f "$ARTDIR/a_default.out"
A_DEF_OK="MISMATCH"; [ "$A_DEF_SHA" = "$REF_SHA" ] && A_DEF_OK="sha=OK"
say "[a:default] $A_DEF_OK (trace -> a_default_trace.log)"
grep -m1 'STALL_RESIDENCY_PROBE:' "$ARTDIR/a_default.err" | sed 's/^/[a:default] /'
say "[a:default] per-stall classify (non-startup):"
grep 'classify' "$ARTDIR/a_default_trace.log" 2>/dev/null | grep -v '"decode_start":0,' | sed 's/^/[a:default]   /' | head -20

# positive control: tiny vs large cache cap
run_probe "cap1"   GZIPPY_PREFETCH_CACHE_CAP=1
run_probe "cap256" GZIPPY_PREFETCH_CACHE_CAP=256

# OFF==identity sanity (probe unset)
drop_caches
env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$ARTDIR/a_off.out" 2>/dev/null || true
A_OFF_SHA="$(sha_of "$ARTDIR/a_off.out")"; rm -f "$ARTDIR/a_off.out"
say "[a:off==identity] off_sha=$A_OFF_SHA default_sha=$A_DEF_SHA match=$([ "$A_OFF_SHA" = "$A_DEF_SHA" ] && echo YES || echo NO)"

# ============================================================================
# DISCRIMINATOR (b) — consumer-block decompose (clean-only trace + SLOW control)
# ============================================================================
say ""; say "########## DISCRIMINATOR (b): CONSUMER-BLOCK DECOMPOSE ##########"

# Capture aligned seed (p=1) for clean-only forcing.
drop_caches; rm -f "$SEEDF"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE="$SEEDF" \
  taskset -c "$(pin_mask 1)" "$GZIPPY" -d -c -p 1 "$CORPUS" >/dev/null 2>"$ARTDIR/b_cap.err" || true
CAP_WINS="$(grep -m1 'SEED_WINDOWS_CAPTURE wrote' "$ARTDIR/b_cap.err" | grep -o 'wrote [0-9]*' | grep -o '[0-9]*' || echo 0)"
say "[b] seed capture windows=$CAP_WINS"
[ "${CAP_WINS:-0}" -ge 1 ] || { echo "STEP0_FAILURE=capture-0-windows"; echo "STEP0_GUEST_DONE"; exit 10; }

cap_clean_trace() { # <label> <slow-env...>  -> clean-only trace at placement-perfect op-point
  local label="$1"; shift
  drop_caches
  local out; out="$(mktemp)"
  env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS="$SEEDF" \
    GZIPPY_LOG_FILE="$ARTDIR/b_${label}_trace.json" "$@" \
    taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$out" 2>"$ARTDIR/b_${label}.err" || true
  local sha; sha="$(sha_of "$out")"; rm -f "$out"
  local ok="MISMATCH"; [ "$sha" = "$REF_SHA" ] && ok="sha=OK"
  say "[b:$label] $ok trace -> b_${label}_trace.json"
}

# baseline clean-only (the decompose verdict input)
cap_clean_trace "clean"
# SLOW positive control: decode compute +100% (spin). DECODE-WAIT must rise.
cap_clean_trace "slow100" GZIPPY_SLOW_MODE=100 GZIPPY_SLOW_KIND=spin

say ""; say "STEP0_GUEST_DONE"
