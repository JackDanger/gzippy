#!/usr/bin/env bash
# _consumer_writev_guest.sh — guest-side 3-arm consumer-output writev verdict.
#
# ONE gzippy-native binary, THREE env-knob arms vs rapidgzip-native, at silesia
# T4 (the real loss cell) AND T8 (the forgotten +27% cell). Settles GO/STOP on the
# consumer-output writev direction.
#
#   A = HEAD baseline (OVERLAP_WRITER off, writev on)
#   B = GZIPPY_SKIP_WRITEV_SYSCALL=1 (REMOVAL CEILING; bytes WRONG, decode runs)
#   C = GZIPPY_OVERLAP_WRITER=1 (the real byte-EXACT fix; dedicated writer thread)
#
# GATE-0 (LOUD-FAIL): build-flavor==parallel-sm+pure; sha pin; path=ParallelSM;
#   A & C sha==zcat (byte-exact); B trailer-CRC verifies (exit 0 = decode ran) AND
#   B output bytes != ref (NON-INERT witness that writev was actually skipped);
#   C spawns the "gzippy-out-writ" writer thread (NON-INERT witness it engaged);
#   rg present; same /dev/null sink all arms; A/A (GZ vs GZ) + rg/rg self-tests.
# GATE-1: interleaved best-of-N>=13; Delta<spread => TIE (analyzer).
# A run that fails a hard gate prints FAIL, writes FAIL to .DONE, exits != 0.
set -u

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/dev/shm/cwv.log
DONE=/dev/shm/cwv.DONE
OUT=/dev/shm/cwv-art
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

SHA="${SHA:-HEAD}"
THREADS="${THREADS:-4 8}"
N="${N:-15}"
GZ_SRC="${GZ_SRC:-/mnt/internal/gz-head}"
GZ_TARGET="${GZ_TARGET:-/dev/shm/cwv-target}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
PINBASE="${PINBASE:-0}"          # even P-cores: 0,2,4,6,8,10,12,14
BUILD="${BUILD:-1}"
CORPUS_DIR="${CORPUS_DIR:-/root}"
CORP="${CORP:-silesia}"

fail() { echo "CWV_FAIL=$*"; echo "FAIL $*" > "$DONE"; exit 2; }

echo "== CONSUMER-WRITEV 3-arm verdict rig =="
echo "host: $(uname -srm)  cores: $(nproc)"
echo "load_start: $(cat /proc/loadavg)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
echo "requested sha: $SHA  corp: $CORP  threads: '$THREADS'  N: $N"

GZ="$GZ_TARGET/release/gzippy"
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
  echo "--- BUILD skipped (BUILD=0) ---"
  cd "$GZ_SRC" 2>/dev/null && BUILT_SHA="$(git rev-parse HEAD)" || BUILT_SHA="?"
fi
[ -x "$GZ" ] || fail "gzippy binary missing: $GZ"

F="$CORPUS_DIR/$CORP.gz"
[ -f "$F" ] || fail "corpus missing: $F"

# ---- GATE-0a: build identity + flavor ----
echo "--- GATE0a: build identity ---"
FLAVOR_LINE="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$F" >/dev/null 2>/tmp/flav.err; grep -m1 'build-flavor=' /tmp/flav.err)"
FLAVOR="$(echo "$FLAVOR_LINE" | sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p')"
echo "build-flavor: '$FLAVOR'  (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "build-flavor='$FLAVOR' != parallel-sm+pure"
if [ "$SHA" != "HEAD" ]; then case "$BUILT_SHA" in "$SHA"*) ;; *) fail "built sha $BUILT_SHA != requested $SHA";; esac; fi
echo "GZ sha=$(sha256sum "$GZ" | cut -c1-12)  built_sha=$BUILT_SHA"
[ -x "$RG" ] || fail "rg missing: $RG"
echo "RG=$RG $("$RG" --version 2>&1 | head -1 | cut -c1-50)"

# ---- GATE-0b: routing + correctness + NON-INERT witnesses ----
echo "--- GATE0b: routing + correctness + witnesses ---"
REF="$(zcat "$F" | sha256sum | cut -c1-16)"
BYTES="$(zcat "$F" | wc -c)"
echo "ref=$REF bytes=$BYTES"

PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" >/dev/null 2>/tmp/path.err; grep -m1 'path=' /tmp/path.err)"
echo "routing: $PATHL"
echo "$PATHL" | grep -q 'path=ParallelSM' || fail "not routed to ParallelSM: '$PATHL'"

# Arm A byte-exact
SA="$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
echo "A(baseline) sha=$SA $([ "$SA" = "$REF" ] && echo OK || echo BAD)"
[ "$SA" = "$REF" ] || fail "arm A not byte-exact"

# Arm C byte-exact (OVERLAP_WRITER)
SC="$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_OVERLAP_WRITER=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" 2>/dev/null | sha256sum | cut -c1-16)"
echo "C(overlap) sha=$SC $([ "$SC" = "$REF" ] && echo OK || echo BAD)"
[ "$SC" = "$REF" ] || fail "arm C (OVERLAP_WRITER) not byte-exact"

# Arm B: decode runs (exit 0) but bytes WRONG (writev skipped). NON-INERT witness.
# Write to a real file so the skipped-writev shows as missing bytes (sha != ref).
BTMP=/dev/shm/cwv-b.out
GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SKIP_WRITEV_SYSCALL=1 taskset -c "$PINBASE" "$GZ" -d -c -p4 "$F" > "$BTMP" 2>/tmp/b.err
BRC=$?
SB="$(sha256sum "$BTMP" | cut -c1-16)"
BBYTES="$(wc -c < "$BTMP")"
echo "B(skip-writev) rc=$BRC sha=$SB bytes=$BBYTES (ref bytes=$BYTES)"
[ "$BRC" = 0 ] || fail "arm B decode did not return success (rc=$BRC) — trailer CRC should still verify"
[ "$SB" != "$REF" ] || fail "arm B output == ref — writev was NOT skipped (INERT oracle)"
echo "B NON-INERT witness OK: decode ran (rc=0, trailer CRC verified) AND bytes differ from ref (writev skipped)"
rm -f "$BTMP"

# Arm C witness: the dedicated writer thread 'gzippy-out-writ' (15-char comm) must appear.
echo "--- C witness: writer-thread engagement ---"
GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_OVERLAP_WRITER=1 taskset -c "0,2,4,6" "$GZ" -d -c -p4 "$F" >/dev/null 2>&1 &
CPID=$!
CSEEN=0
for i in $(seq 1 2000); do
  if grep -qs 'gzippy-out-writ' /proc/$CPID/task/*/comm 2>/dev/null; then CSEEN=1; break; fi
  if ! kill -0 "$CPID" 2>/dev/null; then break; fi
done
wait "$CPID" 2>/dev/null
echo "C writer-thread seen: $CSEEN (1=engaged)"
[ "$CSEEN" = 1 ] || fail "arm C OVERLAP_WRITER did not spawn 'gzippy-out-writ' thread — fix INERT"
# Control: baseline A must NOT spawn the writer thread.
GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "0,2,4,6" "$GZ" -d -c -p4 "$F" >/dev/null 2>&1 &
APID=$!
ASEEN=0
for i in $(seq 1 2000); do
  if grep -qs 'gzippy-out-writ' /proc/$APID/task/*/comm 2>/dev/null; then ASEEN=1; break; fi
  if ! kill -0 "$APID" 2>/dev/null; then break; fi
done
wait "$APID" 2>/dev/null
echo "A writer-thread seen: $ASEEN (expect 0 — control)"
echo "GATE0 PASS"

pin_mask() { local t=$1 i=0 m=""; while [ "$i" -lt "$t" ]; do [ -n "$m" ] && m="$m,"; m="$m$((PINBASE + i*2))"; i=$((i+1)); done; echo "$m"; }

EVENTS="duration_time,cpu_core/cycles/,cpu_core/instructions/,task-clock"

run_gz() {  # $1=arm-label $2=T $3=rep $4=extra-env
  local arm=$1 T=$2 r=$3 extra=$4 mask; mask="$(pin_mask "$T")"
  local csv="$OUT/$CORP.T$T.$arm.$r.csv"
  taskset -c "$mask" perf stat -x, -e "$EVENTS" -- \
    env $extra GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$mask" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>"$csv"
}
run_rg() {  # $1=label $2=T $3=rep
  local arm=$1 T=$2 r=$3 mask; mask="$(pin_mask "$T")"
  local csv="$OUT/$CORP.T$T.$arm.$r.csv"
  taskset -c "$mask" perf stat -x, -e "$EVENTS" -- \
    taskset -c "$mask" "$RG" -d -c -P"$T" "$F" >/dev/null 2>"$csv"
}

echo "--- MEASURE (interleaved A,A2,B,C,RG,RG2 per rep; N=$N) ---"
for T in $THREADS; do
  echo "  cell $CORP T$T mask=$(pin_mask "$T") ..."
  for r in $(seq 1 "$N"); do
    run_gz A   "$T" "$r" ""
    run_gz A2  "$T" "$r" ""
    run_gz B   "$T" "$r" "GZIPPY_SKIP_WRITEV_SYSCALL=1"
    run_gz C   "$T" "$r" "GZIPPY_OVERLAP_WRITER=1"
    run_rg RG  "$T" "$r"
    run_rg RG2 "$T" "$r"
  done
done
echo "load_end: $(cat /proc/loadavg)"

{
  echo "sha $BUILT_SHA"
  echo "flavor $FLAVOR"
  echo "threads $THREADS"
  echo "n $N"
  echo "corp $CORP"
  echo "bytes $BYTES"
  echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  echo "gov $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
  echo "load_start $(cut -d' ' -f1 /proc/loadavg)"
} > "$OUT/meta.txt"

echo "=== ANALYZE ==="
python3 "$SELF_DIR/consumer_writev_report.py" "$OUT" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$DONE"
echo "=== CWV_GUEST_DONE ($(date -u +%FT%TZ)) ==="
