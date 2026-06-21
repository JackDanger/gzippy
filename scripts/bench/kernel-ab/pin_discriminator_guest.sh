#!/usr/bin/env bash
# pin_discriminator_guest.sh — guest-side 3-arm PIN-DISCRIMINATOR (run FROZEN on
# guest 199, UNPINNED = production-style; NO external taskset so the OS/container
# decides placement — that is the whole question).
#
# Settles the fork: is the faithful fix for gzippy's silesia-T4 loss to DELETE the
# decode-worker pinning (let the OS schedule, like rapidgzip BlockFetcher.hpp:185)
# or a PORTABLE topology-aware distinct-physical pin?
#
# THREE byte-exact arms of gz-native (same binary, env-selected):
#   A = HEAD default     (with_pinning_for_capacity — SMT-packing pin; the bug)
#   B = GZIPPY_NO_PIN=1  (empty map, pin:None per worker — faithful to rg; candidate FIX)
#   C = GZIPPY_PHYS_PIN=1 (distinct-physical diagnostic)
# vs rapidgzip-native. silesia T2/T4/T8 + a 2nd corpus, interleaved best-of-N,
# /dev/null both arms, perf cyc/B + GHz. PLUS the NON-INERT /proc core-placement
# witness per arm (where the workers actually land).
#
# Inputs (env): SHA GZ_SRC GZ_TARGET RG CORPORA THREADS N CORPUS_DIR
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/dev/shm/pindisc.log
DONE=/dev/shm/pindisc.DONE
OUT=/dev/shm/pindisc-art
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

SHA="${SHA:-HEAD}"
GZ_SRC="${GZ_SRC:-/mnt/internal/gz-head}"
GZ_TARGET="${GZ_TARGET:-/dev/shm/pindisc-target}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORPORA="${CORPORA:-silesia monorepo}"
THREADS="${THREADS:-2 4 8}"
N="${N:-13}"
CORPUS_DIR="${CORPUS_DIR:-/root}"
BUILD="${BUILD:-1}"
GZ="$GZ_TARGET/release/gzippy"

fail() { echo "PINDISC_FAIL=$*"; echo "FAIL $*" > "$DONE"; exit 2; }

echo "== PIN-DISCRIMINATOR 3-arm (FROZEN, UNPINNED) =="
echo "host: $(uname -srm)  cores: $(nproc)"
echo "load_start: $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
echo "topology (sibling groups):"
for c in $(seq 0 15); do
  s=$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list 2>/dev/null || echo "?")
  echo "  cpu$c siblings=$s"
done | sort -u -t= -k2 | head -16

# ---------------------------------------------------------------------------
# BUILD gzippy-native at the requested sha (pure-rust-inflate => C-FFI OFF).
# ---------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
  echo "--- BUILD (gzippy-native @ $SHA) ---"
  cd "$GZ_SRC" || fail "no GZ_SRC $GZ_SRC"
  git fetch origin kernel-converge-A --quiet 2>&1 | tail -2 || fail "git fetch failed"
  git reset --hard "$SHA" 2>&1 | tail -1 || fail "git reset $SHA failed"
  git submodule update --init --recursive 2>/dev/null || true
  BUILT_SHA="$(git rev-parse HEAD)"
  echo "checked-out sha: $BUILT_SHA"
  CARGO_TARGET_DIR="$GZ_TARGET" RUSTFLAGS="-C target-cpu=native" \
    timeout 600 cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -6 \
    || fail "cargo build failed"
else
  echo "--- BUILD skipped ---"
  cd "$GZ_SRC" 2>/dev/null && BUILT_SHA="$(git rev-parse HEAD)" || BUILT_SHA="?"
fi
[ -x "$GZ" ] || fail "gzippy binary missing: $GZ"

# ---------------------------------------------------------------------------
# GATE-0a: build-flavor (FFI-off proof) + sha pin.
# ---------------------------------------------------------------------------
echo "--- GATE0a: build identity ---"
FIRST=""
for c in $CORPORA; do [ -f "$CORPUS_DIR/$c.gz" ] && { FIRST="$CORPUS_DIR/$c.gz"; break; }; done
[ -n "$FIRST" ] || fail "no corpus present in $CORPUS_DIR for '$CORPORA'"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$FIRST" >/dev/null 2>/tmp/flav.err || true
FLAVOR="$(sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p' /tmp/flav.err | head -1)"
echo "build-flavor: '$FLAVOR'  (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "build-flavor='$FLAVOR' != parallel-sm+pure"
if [ "$SHA" != "HEAD" ] && [ "$BUILT_SHA" != "$SHA" ]; then
  case "$BUILT_SHA" in "$SHA"*) ;; *) fail "built sha $BUILT_SHA != requested $SHA";; esac
fi
echo "GZ=$GZ  sha=$(sha256sum "$GZ" | cut -c1-16)  built_sha=$BUILT_SHA"
[ -x "$RG" ] || fail "rapidgzip missing: $RG"
echo "RG=$RG  $("$RG" --version 2>&1 | head -1 | cut -c1-60)"

# ---------------------------------------------------------------------------
# GATE-0b: per-corpus path=ParallelSM + sha-verify ALL THREE arms + rg == zcat.
# Pinning must NOT change output (byte-exact across A/B/C).
# ---------------------------------------------------------------------------
echo "--- GATE0b: routing + correctness (all 3 arms byte-exact) ---"
declare -A BYTES
for corp in $CORPORA; do
  F="$CORPUS_DIR/$corp.gz"
  [ -f "$F" ] || fail "corpus missing: $F"
  REF="$(zcat "$F" | sha256sum | cut -c1-16)"
  BYTES[$corp]="$(zcat "$F" | wc -c)"
  PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" >/dev/null 2>/tmp/path.err; grep -m1 'path=' /tmp/path.err)"
  echo "$corp: $PATHL"
  echo "$PATHL" | grep -q 'path=ParallelSM' || fail "$corp not routed to ParallelSM"
  SA="$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  SB="$(GZIPPY_NO_PIN=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  SC="$(GZIPPY_PHYS_PIN=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  SRG="$("$RG" -d -c -P4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  echo "  ref=$REF A=$SA B=$SB C=$SC rg=$SRG bytes=${BYTES[$corp]}"
  [ "$SA" = "$REF" ] || fail "$corp arm A sha mismatch"
  [ "$SB" = "$REF" ] || fail "$corp arm B sha mismatch"
  [ "$SC" = "$REF" ] || fail "$corp arm C sha mismatch"
  [ "$SRG" = "$REF" ] || fail "$corp rg sha mismatch"
done
echo "GATE0 PASS (all 3 arms + rg byte-exact == zcat)"

# ---------------------------------------------------------------------------
# NON-INERT CORE-PLACEMENT WITNESS — for EACH arm, prove where workers land.
# Run on silesia at T4 (the contested cell). UNPINNED (no taskset).
# ---------------------------------------------------------------------------
echo "--- CORE-PLACEMENT WITNESS (silesia T4, unpinned) ---"
PLACE="$SELF_DIR/pin_discriminator_placement.py"
[ -f "$PLACE" ] || PLACE="/root/pindisc/pin_discriminator_placement.py"
PF="$CORPUS_DIR/silesia.gz"
{
  python3 "$PLACE" "$GZ" "$PF" 4 A
  python3 "$PLACE" "$GZ" "$PF" 4 B GZIPPY_NO_PIN=1
  python3 "$PLACE" "$GZ" "$PF" 4 C GZIPPY_PHYS_PIN=1
} | tee "$OUT/placement.txt"

# ---------------------------------------------------------------------------
# MEASUREMENT — interleaved A,A2,B,B2,C,C2,RG,RG2 per rep. UNPINNED.
# /dev/null both arms. perf cyc/B + task-clock(GHz/effcores) + wall.
# ---------------------------------------------------------------------------
EVENTS="duration_time,cpu_core/cycles/,cpu_core/instructions/,task-clock,cpu_core/LLC-load-misses/"

run_gz() {  # $1=env $2=T $3=corp $4=outfile
  perf stat -x, -e "$EVENTS" -- \
    env $1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p"$2" "$CORPUS_DIR/$3.gz" >/dev/null 2>"$4"
}
run_rg() {  # $1=T $2=corp $3=outfile
  perf stat -x, -e "$EVENTS" -- \
    "$RG" -d -c -P"$1" "$CORPUS_DIR/$2.gz" >/dev/null 2>"$3"
}

# warmup (untimed)
for corp in $CORPORA; do
  run_gz "" 4 "$corp" "$OUT/warm.A.$corp"
  run_gz "GZIPPY_NO_PIN=1" 4 "$corp" "$OUT/warm.B.$corp"
  run_gz "GZIPPY_PHYS_PIN=1" 4 "$corp" "$OUT/warm.C.$corp"
  run_rg 4 "$corp" "$OUT/warm.rg.$corp"
done

echo "--- MEASURE interleaved (A,A2,B,B2,C,C2,RG,RG2; N=$N) ---"
: > "$OUT/meta.txt"
for corp in $CORPORA; do
  for T in $THREADS; do
    echo "  cell $corp T$T ..."
    echo "cell $corp $T" >> "$OUT/meta.txt"
    for r in $(seq 1 "$N"); do
      run_gz ""                 "$T" "$corp" "$OUT/$corp.T$T.A.$r.csv"
      run_gz ""                 "$T" "$corp" "$OUT/$corp.T$T.A2.$r.csv"
      run_gz "GZIPPY_NO_PIN=1"  "$T" "$corp" "$OUT/$corp.T$T.B.$r.csv"
      run_gz "GZIPPY_NO_PIN=1"  "$T" "$corp" "$OUT/$corp.T$T.B2.$r.csv"
      run_gz "GZIPPY_PHYS_PIN=1" "$T" "$corp" "$OUT/$corp.T$T.C.$r.csv"
      run_gz "GZIPPY_PHYS_PIN=1" "$T" "$corp" "$OUT/$corp.T$T.C2.$r.csv"
      run_rg "$T" "$corp" "$OUT/$corp.T$T.RG.$r.csv"
      run_rg "$T" "$corp" "$OUT/$corp.T$T.RG2.$r.csv"
    done
  done
done
echo "load_end: $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"

{
  echo "sha $BUILT_SHA"
  echo "flavor $FLAVOR"
  echo "n $N"
  for corp in $CORPORA; do echo "bytes $corp ${BYTES[$corp]}"; done
} >> "$OUT/meta.txt"

echo "=== ANALYZE ==="
REPORT="$SELF_DIR/pin_discriminator_report.py"
[ -f "$REPORT" ] || REPORT="/root/pindisc/pin_discriminator_report.py"
python3 "$REPORT" "$OUT" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$DONE"
echo "=== PINDISC_GUEST_DONE ($(date -u +%FT%TZ)) ==="
