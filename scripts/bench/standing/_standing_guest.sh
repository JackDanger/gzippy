#!/usr/bin/env bash
# _standing_guest.sh — guest-side half of the ONE-COMMAND ground-truth rig.
#
# Answers "where does gzippy-native stand vs the state of the art" deterministically
# on the Intel i7-13700T LXC. Builds gzippy-native (pure-Rust, C-FFI OFF the decode
# graph) at a pinned sha, GATE-0 self-validates, then runs an interleaved best-of-N
# matrix vs rapidgzip (T>1 SOTA) and igzip (T1 SOTA), and writes raw CSVs for
# standing_report.py.
#
# Invoked by standing.sh; all inputs arrive as env (see standing.sh remote_env()):
#   SHA CORPORA THREADS N GZ_SRC GZ_TARGET RG IGZIP PINBASE BUILD
#
# Discipline (CLAUDE.md Measurement PROTOCOL):
#   GATE-0  build sha==requested; build-flavor==parallel-sm+pure (FFI-off proof);
#           path=ParallelSM asserted per cell; every arm sha-verified == zcat;
#           same /dev/null sink for all arms; rg & igzip present.
#   GATE-1  interleaved best-of-N>=13 (analyzer); Delta<spread => TIE.
#   Self-test arms RG2/GZ2 (binary-vs-itself ~= 1.0) license trusting ratios on a
#           loaded shared box — if they fail the analyzer flags the cell untrusted.
# A run that fails a hard gate prints FAIL, writes FAIL to the .DONE file, exits != 0.
set -u

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/dev/shm/standing.log
DONE=/dev/shm/standing.DONE
OUT=/dev/shm/standing-art
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

SHA="${SHA:-HEAD}"
CORPORA="${CORPORA:-silesia monorepo nasa}"
THREADS="${THREADS:-1 2 4 8}"
N="${N:-13}"
GZ_SRC="${GZ_SRC:-/mnt/internal/gz-head}"
GZ_TARGET="${GZ_TARGET:-/dev/shm/standing-target}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
IGZIP="${IGZIP:-/usr/bin/igzip}"
PINBASE="${PINBASE:-0}"          # even P-cores: 0,2,4,6,8,10,12,14
BUILD="${BUILD:-1}"
CORPUS_DIR="${CORPUS_DIR:-/root}"

fail() { echo "STANDING_FAIL=$*"; echo "FAIL $*" > "$DONE"; exit 2; }

echo "== STANDING ground-truth rig =="
echo "host: $(uname -srm)  cores: $(nproc)"
echo "load_start: $(cat /proc/loadavg)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
echo "requested sha: $SHA  corpora: '$CORPORA'  threads: '$THREADS'  N: $N"

# ---------------------------------------------------------------------------
# BUILD gzippy-native at the requested sha (pure-rust-inflate => C-FFI OFF).
# ---------------------------------------------------------------------------
GZ="$GZ_TARGET/release/gzippy"
if [ "$BUILD" = 1 ]; then
  echo "--- BUILD (gzippy-native @ $SHA) ---"
  cd "$GZ_SRC" || fail "no GZ_SRC $GZ_SRC"
  git fetch origin kernel-converge-A --quiet 2>&1 | tail -2 || fail "git fetch failed"
  git reset --hard "$SHA" 2>&1 | tail -1 || fail "git reset $SHA failed"
  git submodule update --init --recursive 2>/dev/null || true   # FFI off => submodule optional
  BUILT_SHA="$(git rev-parse HEAD)"
  echo "checked-out sha: $BUILT_SHA"
  CARGO_TARGET_DIR="$GZ_TARGET" RUSTFLAGS="-C target-cpu=native" \
    timeout 600 cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -6 \
    || fail "cargo build failed"
else
  echo "--- BUILD skipped (BUILD=0) — reusing $GZ ---"
  cd "$GZ_SRC" 2>/dev/null && BUILT_SHA="$(git rev-parse HEAD)" || BUILT_SHA="?"
fi
[ -x "$GZ" ] || fail "gzippy binary missing: $GZ"

# ---------------------------------------------------------------------------
# GATE-0a: build identity + feature fingerprint (FFI-off proof) + sha pin.
# ---------------------------------------------------------------------------
echo "--- GATE0a: build identity ---"
SMALL=""
for c in $CORPORA; do [ -f "$CORPUS_DIR/$c.gz" ] && { SMALL="$CORPUS_DIR/$c.gz"; break; }; done
[ -n "$SMALL" ] || fail "no corpus present in $CORPUS_DIR for '$CORPORA'"
FLAVOR_LINE="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$SMALL" >/dev/null 2>/tmp/flav.err; grep -m1 'build-flavor=' /tmp/flav.err)"
FLAVOR="$(echo "$FLAVOR_LINE" | sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p')"
echo "build-flavor: '$FLAVOR'  (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "build-flavor='$FLAVOR' != parallel-sm+pure (FFI-off native build not confirmed)"
if [ "$SHA" != "HEAD" ] && [ "$BUILT_SHA" != "$SHA" ]; then
  case "$BUILT_SHA" in "$SHA"*) ;; *) fail "built sha $BUILT_SHA != requested $SHA";; esac
fi
echo "GZ=$GZ  sha=$(sha256sum "$GZ" | cut -c1-12)  built_sha=$BUILT_SHA"
echo "RG=$RG  $("$RG" --version 2>&1 | head -1 | cut -c1-60)"
[ -x "$RG" ] || fail "rapidgzip missing: $RG"
echo "IG=$IGZIP  $("$IGZIP" --version 2>&1 | head -1)"
[ -x "$IGZIP" ] || fail "igzip missing: $IGZIP"

# ---------------------------------------------------------------------------
# GATE-0b: per-corpus path=ParallelSM + sha-verify each arm == zcat ref.
# ---------------------------------------------------------------------------
echo "--- GATE0b: routing + correctness ---"
declare -A BYTES
for corp in $CORPORA; do
  F="$CORPUS_DIR/$corp.gz"
  [ -f "$F" ] || fail "corpus missing: $F"
  REF="$(zcat "$F" | sha256sum | cut -c1-16)"
  BYTES[$corp]="$(zcat "$F" | wc -c)"
  # production routing assertion (force the parallel SM engine at every T)
  PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" >/dev/null 2>/tmp/path.err; grep -m1 'path=' /tmp/path.err)"
  echo "$corp: $PATHL"
  # Accept either pure-Rust parallel production path: ParallelSM (the marker-pipeline
  # kernel under test) OR StoredParallel (the correct route for stored-dominated
  # corpora: storedheavy/storedmix/pure_stored). Both are C-FFI-off native paths;
  # the report notes which one each corpus took. Reject anything else (e.g. a libdeflate
  # fallback would mean the native build silently bailed).
  echo "$PATHL" | grep -qE 'path=(ParallelSM|StoredParallel)' || fail "$corp not routed to a pure-Rust parallel path: '$PATHL'"
  echo "$corp $(echo "$PATHL" | sed -n 's/.*path=\([A-Za-z]*\).*/\1/p')" >> "$OUT/paths.txt"
  GS="$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  RS="$(taskset -c "$PINBASE" "$RG" -d -c -P4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  IS="$(taskset -c "$PINBASE" "$IGZIP" -d -c "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  echo "  ref=$REF gz=$([ "$GS" = "$REF" ] && echo OK || echo BAD) rg=$([ "$RS" = "$REF" ] && echo OK || echo BAD) ig=$([ "$IS" = "$REF" ] && echo OK || echo BAD) bytes=${BYTES[$corp]}"
  [ "$GS" = "$REF" ] || fail "$corp gzippy sha mismatch"
  [ "$RS" = "$REF" ] || fail "$corp rapidgzip sha mismatch"
  [ "$IS" = "$REF" ] || fail "$corp igzip sha mismatch"
done
echo "GATE0 PASS"

# ---------------------------------------------------------------------------
# pin mask for a given thread count (even P-cores starting at PINBASE)
# ---------------------------------------------------------------------------
pin_mask() {  # $1=T -> echo "c0,c1,..."
  local t=$1 i=0 m=""
  while [ "$i" -lt "$t" ]; do
    [ -n "$m" ] && m="$m,"
    m="$m$((PINBASE + i*2))"
    i=$((i+1))
  done
  echo "$m"
}

# perf events: wall(duration_time, ns) + P-core cycles/instructions + task-clock(ms)
#  + cache refs/misses for an LLC-miss% report column (multiplexed; report-only).
EVENTS="duration_time,cpu_core/cycles/,cpu_core/instructions/,task-clock,cpu_core/cache-references/,cpu_core/cache-misses/"

run_one() {  # $1=arm $2=corp $3=T $4=rep
  local arm=$1 corp=$2 T=$3 r=$4 F="$CORPUS_DIR/$2.gz"
  local mask; mask="$(pin_mask "$T")"
  local csv="$OUT/$corp.T$T.$arm.$r.csv"
  case "$arm" in
    GZ|GZ2)
      taskset -c "$mask" perf stat -x, -e "$EVENTS" -- \
        env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$mask" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$csv" ;;
    RG|RG2)
      taskset -c "$mask" perf stat -x, -e "$EVENTS" -- \
        taskset -c "$mask" "$RG" -d -c -P"$T" "$F" >/dev/null 2>"$csv" ;;
    IG)
      # igzip is single-threaded; pin to the first core of the mask at every T.
      taskset -c "$PINBASE" perf stat -x, -e "$EVENTS" -- \
        taskset -c "$PINBASE" "$IGZIP" -d -c "$F" >/dev/null 2>"$csv" ;;
  esac
}

echo "--- MEASURE (interleaved GZ,GZ2,RG,RG2,IG per rep; N=$N) ---"
for corp in $CORPORA; do
  for T in $THREADS; do
    echo "  cell $corp T$T mask=$(pin_mask "$T") ..."
    for r in $(seq 1 "$N"); do
      run_one GZ  "$corp" "$T" "$r"
      run_one GZ2 "$corp" "$T" "$r"
      run_one RG  "$corp" "$T" "$r"
      run_one RG2 "$corp" "$T" "$r"
      run_one IG  "$corp" "$T" "$r"
    done
  done
done
echo "load_end: $(cat /proc/loadavg)"

# ---------------------------------------------------------------------------
# EFFCORES (Tool 2, opt-in EFFCORES=1): one extra GZIPPY_TIMELINE-instrumented
# decode per cell, reduced to the H-TAIL-vs-H-KERNEL discriminator. Runs AFTER
# the timed matrix so the trace instrument NEVER contaminates the Gate-1 walls.
# Also pulls rg --verbose "Blocks Total Fetched" as the non-inert chunk-count
# cross-check. Pure analysis; OFF by default (zero effect on the matrix).
# ---------------------------------------------------------------------------
if [ "${EFFCORES:-0}" = 1 ] && [ -f "$SELF_DIR/parallel_sm_tail_metric.py" ]; then
  echo "--- EFFCORES capture (post-matrix, instrumented; off the timed path) ---"
  : > "$OUT/effcores.txt"
  for corp in $CORPORA; do
    F="$CORPUS_DIR/$corp.gz"
    [ -f "$F" ] || continue
    for T in $THREADS; do
      [ "$T" -ge 2 ] || continue   # effcores is a T>=2 schedule metric
      mask="$(pin_mask "$T")"
      # rg's own chunk count for the cross-check
      rgv="$OUT/rgverbose.$corp.T$T.txt"
      taskset -c "$mask" "$RG" -d -c -P"$T" --verbose "$F" >/dev/null 2>"$rgv" || true
      nblk="$(grep -iE 'Total Fetched' "$rgv" | grep -oE '[0-9]+' | tail -1)"
      tj="$OUT/timeline.$corp.T$T.json"
      GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TIMELINE="$tj" taskset -c "$mask" \
        "$GZ" -d -c -p"$T" "$F" >/dev/null 2>>"$OUT/effcores.err" || true
      line="$(python3 "$SELF_DIR/parallel_sm_tail_metric.py" "$tj" \
                ${nblk:+--expected-chunks "$nblk"} --sink /dev/null \
                --label "$corp-T$T" 2>&1 | grep -E 'GATE-0|FORK LINE|effcores=' | tr '\n' ' ')"
      echo "$corp T$T (rg_blocks=${nblk:-?}): $line" | tee -a "$OUT/effcores.txt"
    done
  done
fi

# meta for the analyzer
{
  echo "sha $BUILT_SHA"
  echo "flavor $FLAVOR"
  echo "threads $THREADS"
  echo "n $N"
  echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  echo "gov $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
  echo "load_start $(cut -d' ' -f1 /proc/loadavg)"
  for corp in $CORPORA; do echo "bytes $corp ${BYTES[$corp]}"; done
} > "$OUT/meta.txt"

echo "=== ANALYZE ==="
python3 "$SELF_DIR/standing_report.py" "$OUT" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$DONE"
echo "=== STANDING_GUEST_DONE ($(date -u +%FT%TZ)) ==="
