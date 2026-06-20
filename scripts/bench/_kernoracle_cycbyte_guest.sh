#!/usr/bin/env bash
# NIGHT37 — kernel REMOVAL-ORACLE: production T1 cyc/byte A/B that BOUNDS the
# absolute on-wall share of gzippy's native CLEAN-decode engine (run_contig +
# its per-block table-build + clean glue) by swapping it for igzip's clean
# decode (the AVX2/BMI2 `_04` family via `isal_inflate`) on the SAME binary.
#
#   BASE   arm = GZIPPY_ISAL_ENGINE_ORACLE=0  -> native clean tail (run_contig)
#   ORACLE arm = GZIPPY_ISAL_ENGINE_ORACLE=1  -> igzip clean tail (_04 family)
#
# Built --no-default-features --features pure-rust-inflate,isal-compression so
# `isal_clean_tail` cfg is OFF (run_contig is the DEFAULT clean engine) yet ISA-L
# is linked (the engine-oracle can force it ON). This is the exact "measurement
# oracle on the native build" the GZIPPY_ISAL_ENGINE_ORACLE knob documents.
#
# Both arms decode the SAME deflate byte-identically (sha==zcat verified); only
# the clean-tail ENGINE differs, so the whole-program Δcyc/byte is attributable
# to the clean-decode engine. cyc/byte is a frequency-INDEPENDENT count, so an
# interleaved best-of-N/paired A/B on one isolated P-core is sound under box load
# (HARD SCOPE: T1 inner clean-decode share ONLY — NOT the T4/T8 wall).
#
# GATE-0 (printed FIRST; a run failing any is VOID):
#  (a) path=ParallelSM both arms (GZIPPY_DEBUG)
#  (b) NON-INERT: ORACLE arm isal_chunks>0 & isal_fallbacks==0 (engine truly _04);
#      BASE arm seeded_block+exact_block>0 & isal_chunks==0 (engine truly run_contig);
#      clean-byte fraction = UNIFIED_ROUTE_CLEAN_U8_BYTES(base)/ISIZE reported.
#  (c) SHA: both arms sha==zcat (byte-exact) on every corpus.
#  (d)/(e) GHz spread + LLC-miss% in the analyzer.
#
# Usage: BIN=/dev/shm/n37/release/gzippy PIN=4 REPS=15 CORPORA="silesia monorepo" \
#        bash _kernoracle_cycbyte_guest.sh
set -u

BIN=${BIN:-/dev/shm/n37/release/gzippy}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia monorepo"}
SHA_CORPORA=${SHA_CORPORA:-"silesia nasa monorepo"}
REF_GAP_SILESIA=${REF_GAP_SILESIA:-1.079}   # NIGHT36 igzip gap, silesia
REF_GAP_MONO=${REF_GAP_MONO:-1.204}         # NIGHT36 igzip gap, monorepo
OUT=/dev/shm/n37out
ANALYZE="$(dirname "$0")/_kernoracle_analyze.py"
rm -rf "$OUT"; mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }

echo "================ NIGHT37 KERNEL REMOVAL-ORACLE cyc/byte ================"
echo "BIN=$BIN  sha=$(sha256sum $BIN | cut -c1-16)  flavor=$($BIN --version 2>/dev/null || true)"
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA'  load:$(cat /proc/loadavg)"
echo "REF_GAP silesia=$REF_GAP_SILESIA monorepo=$REF_GAP_MONO (NIGHT36 vs igzip)"

run_arm() { # $1=oracle(0/1) $2=file ; extra env via $3 (verbose flags)
  local orc="$1" f="$2" extra="${3:-}"
  env $extra GZIPPY_ISAL_ENGINE_ORACLE=$orc GZIPPY_FORCE_PARALLEL_SM=1 \
    taskset -c "$PIN" "$BIN" -d -c -p1 "$f"
}

# ---------- GATE-0(a) routing + (b) non-inert counters ----------
FF=/root/$(echo $CORPORA | awk '{print $1}').gz
echo "--- GATE0(a)/(b) routing + engine-engagement (corpus=$FF) ---"
for orc in 0 1; do
  run_arm $orc "$FF" "GZIPPY_DEBUG=1 GZIPPY_VERBOSE=1" >/dev/null 2>"$OUT/g0.$orc"
  P=$(grep -oE "path=[A-Za-z]+" "$OUT/g0.$orc" | head -1)
  ISALC=$(grep -oE "isal_chunks=[0-9]+" "$OUT/g0.$orc" | head -1 | cut -d= -f2)
  ISALF=$(grep -oE "isal_fallbacks=[0-9]+" "$OUT/g0.$orc" | head -1 | cut -d= -f2)
  SEED=$(grep -oE "seeded_block=[0-9]+" "$OUT/g0.$orc" | head -1 | cut -d= -f2)
  EXACT=$(grep -oE "exact_block=[0-9]+" "$OUT/g0.$orc" | head -1 | cut -d= -f2)
  echo "    ORACLE=$orc  $P  isal_chunks=${ISALC:-?} isal_fallbacks=${ISALF:-?} seeded_block=${SEED:-?} exact_block=${EXACT:-?}"
done

# clean-byte fraction from BASE arm (UNIFIED clean u8 bytes / ISIZE)
ISIZE=$(zcat "$FF" | wc -c)
CLEANB=$(grep -oE "clean_u8_bytes=[0-9]+|UNIFIED_ROUTE_CLEAN_U8_BYTES=[0-9]+|clean_bytes=[0-9]+" "$OUT/g0.0" | head -1 | cut -d= -f2)
echo "    ISIZE=$ISIZE  base_clean_u8_bytes=${CLEANB:-NA}  (engine-swapped fraction)"

# ---------- GATE-0(c) sha==zcat both arms, all sha corpora ----------
echo "--- GATE0(c) byte-exactness (sha==zcat), both arms ---"
for c in $SHA_CORPORA; do
  F=/root/$c.gz; [ -f "$F" ] || { echo "    SKIP $c"; continue; }
  REF=$(zcat "$F" | sha256sum | cut -c1-16)
  S0=$(run_arm 0 "$F" 2>/dev/null | sha256sum | cut -c1-16)
  S1=$(run_arm 1 "$F" 2>/dev/null | sha256sum | cut -c1-16)
  echo "    $c ref=$REF base=$S0 oracle=$S1  $([ "$REF" = "$S0" ] && [ "$REF" = "$S1" ] && echo SHA_OK || echo SHA_MISMATCH)"
done
# T4/T8 byte-exact spot (oracle must engage cleanly at higher T too)
echo "--- T4/T8 byte-exact spot (silesia, both arms) ---"
for T in 4 8; do
  REF=$(zcat "$FF" | sha256sum | cut -c1-16)
  S0=$(env GZIPPY_ISAL_ENGINE_ORACLE=0 GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p$T "$FF" 2>/dev/null | sha256sum | cut -c1-16)
  S1=$(env GZIPPY_ISAL_ENGINE_ORACLE=1 GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p$T "$FF" 2>/dev/null | sha256sum | cut -c1-16)
  echo "    T$T ref=$REF base=$S0 oracle=$S1  $([ "$REF" = "$S0" ] && [ "$REF" = "$S1" ] && echo SHA_OK || echo SHA_MISMATCH)"
done

# ---------- MEASUREMENT: interleaved A1,A2(base self-test),B(oracle) ----------
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock"
echo "--- measuring (interleaved A1,A2,B per rep; A1/A2=BASE, B=ORACLE) ---"
: > "$OUT/bytes.txt"
# warm
run_arm 0 "$FF" >/dev/null 2>&1 || true
run_arm 1 "$FF" >/dev/null 2>&1 || true
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "  SKIP $corp"; continue; }
  echo "$corp $(zcat "$F" | wc -c)" >> "$OUT/bytes.txt"
  for r in $(seq 1 $REPS); do
    for arm in A1 A2 B; do
      case $arm in A1|A2) ORC=0;; B) ORC=1;; esac
      taskset -c "$PIN" perf stat -x, -e "$EVENTS" \
        -- env GZIPPY_ISAL_ENGINE_ORACLE=$ORC GZIPPY_FORCE_PARALLEL_SM=1 \
        taskset -c "$PIN" "$BIN" -d -c -p1 "$F" \
        >/dev/null 2>"$OUT/$corp.$arm.$r.csv"
    done
  done
done

echo "--- ANALYSIS ---"
python3 "$ANALYZE" "$OUT" "$REF_GAP_SILESIA" "$REF_GAP_MONO" $CORPORA
echo "DONE_NIGHT37_KERNORACLE"
