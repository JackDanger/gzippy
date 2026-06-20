#!/usr/bin/env bash
# _kernel_gate_guest.sh — guest-side half of the ONE-COMMAND kernel-change verdict.
#
# Builds BOTH gzippy-native binaries FRESH (FFI off, target-cpu=native, symbols
# kept so run_contig can be disassembled), GATE-0 self-validates, then runs a
# PAIRED INTERLEAVED A1,A2,B production-wall A/B and hands the perf CSVs to
# _kernel_gate_analyze.py for a per-corpus KEEP / TIE / REVERT verdict.
#
# Invoked by kernel_gate.sh; inputs arrive as env:
#   BASE CAND CORPORA THREADS N SRC TARGET RG IGZIP PINBASE FEATURES FLAVOR
#   CORPUS_DIR BRANCH
#
# GATE-0 (LOUD-FAIL -> no number):
#   (a) both builds: build-flavor == $FLAVOR (FFI-off native proof); built sha == requested
#   (b) NON-INERT .o-diff: run_contig disassembly sha A vs B MUST differ
#       (identical => "no kernel change, TIE by construction" — reported, not measured)
#   (c) sha == zcat for BOTH binaries on EVERY corpus (correct + non-inert)
#       + path == ParallelSM (production routing)
#   (d) same /dev/null sink both arms (by construction below)
#   (e) A/A self-test (A2-A1 ~ 0) + GHz spread + LLC-miss% reported by the analyzer
# GATE-1: paired Wilcoxon + bootstrap CI + Δ-vs-inter-run-spread (analyzer).
set -u

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/dev/shm/kgate.log
DONE=/dev/shm/kgate.DONE
OUT=/dev/shm/kgate-art
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1

BASE="${BASE:?BASE sha required}"
CAND="${CAND:?CAND sha required}"
CORPORA="${CORPORA:-silesia monorepo nasa}"
THREADS="${THREADS:-1}"
N="${N:-15}"
SRC="${SRC:-/mnt/internal/gz-head}"
TARGET="${TARGET:-/dev/shm/kgate-target}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
IGZIP="${IGZIP:-/usr/bin/igzip}"
PINBASE="${PINBASE:-4}"
FEATURES="${FEATURES:-gzippy-native}"
FLAVOR="${FLAVOR:-parallel-sm+pure}"
CORPUS_DIR="${CORPUS_DIR:-/root}"
BRANCH="${BRANCH:-kernel-converge-A}"

BIN_A=/dev/shm/kgate/A/gzippy
BIN_B=/dev/shm/kgate/B/gzippy
mkdir -p /dev/shm/kgate/A /dev/shm/kgate/B

fail() { echo "KGATE_FAIL=$*"; echo "FAIL $*" > "$DONE"; exit 2; }

echo "== KERNEL-GATE guest =="
echo "host: $(uname -srm)  cores: $(nproc)  load: $(cat /proc/loadavg)"
echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')  gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
echo "BASE=$BASE  CAND=$CAND  corpora='$CORPORA'  T=$THREADS  N=$N  pin=cpu$PINBASE"
echo "df /dev/shm: $(df -h /dev/shm | tail -1)"

# ---------------------------------------------------------------------------
# disassemble run_contig (production fn; #[inline(never)] => has a symbol when
# the binary is built STRIP=false) and hash it, address-normalised. This is the
# NON-INERT .o-diff gate's core (Gate-0b).
# ---------------------------------------------------------------------------
run_contig_sig() {  # $1=binary -> prints sha256 of the address-normalised run_contig disasm
  local bin=$1
  objdump -d -C --no-show-raw-insn "$bin" 2>/dev/null \
    | awk '/::run_contig>:/{f=1} f&&/^$/{f=0} f{print}' \
    | sed -E 's/^[[:space:]]*[0-9a-f]+:[[:space:]]*//; s/0x[0-9a-f]+/0xADDR/g; s/<[^>]*>//g' \
    | grep -vE '^[[:space:]]*$' \
    | sha256sum | cut -c1-16
}

# ---------------------------------------------------------------------------
# BUILD one sha into $TARGET (shared dir -> incremental), copy gzippy out.
# ---------------------------------------------------------------------------
build_one() {  # $1=sha $2=dest-binary
  local sha=$1 dest=$2
  cd "$SRC" || fail "no SRC $SRC"
  git fetch origin "$BRANCH" --quiet 2>&1 | tail -1 || fail "git fetch failed"
  git reset --hard "$sha" 2>&1 | tail -1 || fail "git reset $sha failed"
  git submodule update --init --recursive 2>/dev/null || true   # FFI off => optional
  local built; built="$(git rev-parse HEAD)"
  case "$built" in "$sha"*) ;; *) fail "built sha $built != requested $sha";; esac
  CARGO_TARGET_DIR="$TARGET" RUSTFLAGS="-C target-cpu=native" \
    CARGO_PROFILE_RELEASE_STRIP=false \
    timeout 600 cargo build --release --no-default-features --features "$FEATURES" 2>&1 | tail -4 \
    || fail "cargo build $sha failed"
  cp -f "$TARGET/release/gzippy" "$dest" || fail "copy gzippy ($sha) failed"
  echo "built $sha -> $dest  ($(du -h "$dest" | cut -f1))"
}

echo "--- BUILD baseline ($BASE) ---"
build_one "$BASE" "$BIN_A"
echo "--- BUILD candidate ($CAND) ---"
build_one "$CAND" "$BIN_B"

# ---------------------------------------------------------------------------
# GATE-0a: build-flavor (FFI-off proof) on BOTH binaries.
# ---------------------------------------------------------------------------
echo "--- GATE0a: build-flavor ---"
SMALL=""; for c in $CORPORA; do [ -f "$CORPUS_DIR/$c.gz" ] && { SMALL="$CORPUS_DIR/$c.gz"; break; }; done
[ -n "$SMALL" ] || fail "no corpus in $CORPUS_DIR for '$CORPORA'"
for tag in A B; do
  BIN=$([ "$tag" = A ] && echo "$BIN_A" || echo "$BIN_B")
  FL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c "$SMALL" >/dev/null 2>/tmp/kf.$tag; grep -m1 'build-flavor=' /tmp/kf.$tag | sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p')"
  echo "  arm$tag build-flavor='$FL'"
  [ "$FL" = "$FLAVOR" ] || fail "arm$tag build-flavor='$FL' != $FLAVOR (FFI-off native not confirmed)"
done

# ---------------------------------------------------------------------------
# GATE-0b: NON-INERT run_contig .o-diff.
# ---------------------------------------------------------------------------
echo "--- GATE0b: run_contig non-inert .o-diff ---"
SIG_A="$(run_contig_sig "$BIN_A")"; SIG_B="$(run_contig_sig "$BIN_B")"
echo "  run_contig sig: A=$SIG_A  B=$SIG_B"
[ -n "$SIG_A" ] && [ -n "$SIG_B" ] || fail "run_contig symbol not found (build not symboled? STRIP must be false)"
KERNEL_CHANGED=1
if [ "$SIG_A" = "$SIG_B" ]; then
  KERNEL_CHANGED=0
  echo "  run_contig IDENTICAL between $BASE and $CAND -> NO KERNEL CHANGE, TIE by construction."
else
  echo "  run_contig DIFFERS -> kernel change is real, proceeding to measure."
fi

# ---------------------------------------------------------------------------
# GATE-0c: per-corpus path=ParallelSM + sha==zcat for BOTH binaries.
# ---------------------------------------------------------------------------
echo "--- GATE0c: routing + correctness (both arms) ---"
declare -A BYTES
: > "$OUT/bytes.txt"
for corp in $CORPORA; do
  F="$CORPUS_DIR/$corp.gz"
  [ -f "$F" ] || fail "corpus missing: $F"
  REF="$(zcat "$F" | sha256sum | cut -c1-16)"
  BYTES[$corp]="$(zcat "$F" | wc -c)"
  echo "$corp ${BYTES[$corp]}" >> "$OUT/bytes.txt"
  PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$BIN_B" -d -c -p"$THREADS" "$F" >/dev/null 2>/tmp/kp; grep -m1 'path=' /tmp/kp)"
  echo "  $corp: $PATHL  bytes=${BYTES[$corp]}"
  echo "$PATHL" | grep -q 'path=ParallelSM' || fail "$corp not ParallelSM: '$PATHL'"
  AS="$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$BIN_A" -d -c -p"$THREADS" "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  BS="$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PINBASE" "$BIN_B" -d -c -p"$THREADS" "$F" 2>/dev/null | sha256sum | cut -c1-16)"
  echo "    ref=$REF A=$([ "$AS" = "$REF" ] && echo OK || echo BAD) B=$([ "$BS" = "$REF" ] && echo OK || echo BAD)"
  [ "$AS" = "$REF" ] || fail "$corp arm A sha mismatch"
  [ "$BS" = "$REF" ] || fail "$corp arm B sha mismatch"
done
echo "GATE0 PASS"

if [ "$KERNEL_CHANGED" = 0 ]; then
  # No kernel change — emit the TIE-by-construction verdict and stop (a measurement
  # would only confirm a wash). Still a fully self-validated, deterministic verdict.
  {
    echo "############ KERNEL-GATE VERDICT (no-op build) ############"
    echo "run_contig is byte-identical between BASE=$BASE and CAND=$CAND."
    echo "OVERALL VERDICT = TIE (by construction — no kernel change)"
  } | tee "$OUT/REPORT.txt"
  echo PASS > "$DONE"
  echo "=== KGATE_GUEST_DONE (no-op) ==="
  exit 0
fi

# ---------------------------------------------------------------------------
# pin mask for T threads (even P-cores from PINBASE)
# ---------------------------------------------------------------------------
pin_mask() { local t=$1 i=0 m=""; while [ "$i" -lt "$t" ]; do [ -n "$m" ] && m="$m,"; m="$m$((PINBASE + i*2))"; i=$((i+1)); done; echo "$m"; }

# wall(duration_time,ns) + cycles + instructions + task-clock(ms) + LLC refs/misses
EVENTS="duration_time,cpu_core/cycles/,cpu_core/instructions/,task-clock,cpu_core/cache-references/,cpu_core/cache-misses/"
MASK="$(pin_mask "$THREADS")"

run_one() {  # $1=arm(A1|A2|B) $2=corp $3=rep
  local arm=$1 corp=$2 r=$3 F="$CORPUS_DIR/$2.gz"
  local BIN; case "$arm" in A1|A2) BIN=$BIN_A;; B) BIN=$BIN_B;; esac
  taskset -c "$MASK" perf stat -x, -e "$EVENTS" -- \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p"$THREADS" "$F" \
    >/dev/null 2>"$OUT/$corp.$arm.$r.csv"
}

echo "--- MEASURE (interleaved A1,A2,B per rep; N=$N; T=$THREADS mask=$MASK) ---"
for corp in $CORPORA; do
  echo "  cell $corp ..."
  for r in $(seq 1 "$N"); do
    run_one A1 "$corp" "$r"
    run_one A2 "$corp" "$r"
    run_one B  "$corp" "$r"
  done
done
echo "load_end: $(cat /proc/loadavg)"

{
  echo "base $BASE"; echo "cand $CAND"; echo "threads $THREADS"; echo "n $N"
  echo "sig_a $SIG_A"; echo "sig_b $SIG_B"
  echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  echo "gov $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
} > "$OUT/meta.txt"

echo "=== ANALYZE ==="
python3 "$SELF_DIR/_kernel_gate_analyze.py" "$OUT" --threads "$THREADS" $CORPORA 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$DONE"
echo "=== KGATE_GUEST_DONE ($(date -u +%FT%TZ)) ==="
