#!/usr/bin/env bash
# ZEN2 STEP-1 CEILING ORACLE measurer (3-arm interleave: GZ-base, GZ-ceiling, RG).
# Gate-0: flavor + baseline sha==zcat + ceiling NON-INERT (counter>0 AND sha differs)
#         + rg sha==zcat. Gate-1: interleaved best-of-N>=9, ratio vs A/A spread.
# GZ-ceiling output is WRONG-ON-PURPOSE (perturbation) -> NOT sha-checked, only proven
# non-inert. ratio_base = gz_base/rg (production), ratio_ceil = gz_ceil/rg (ceiling).
set -u
: "${GZ:?}"; : "${RG:?}"; : "${CORPUS_DIR:=/root}"; : "${OUT:=/dev/shm/zen-ceil-out}"
: "${N:=11}"; : "${CELLS:?}"   # CELLS = "silesia:4 silesia:2 squishy:4 ..." corpus:T
mkdir -p "$OUT"; LOG="$OUT/measure.log"; exec > "$LOG" 2>&1
rm -f "$OUT/DONE"
fail(){ echo "ZEN_FAIL=$*"; echo "FAIL $*" > "$OUT/DONE"; exit 2; }

echo "== ZEN ceiling measure $(date -u +%FT%TZ) =="
echo "host: $(uname -srm) cores=$(nproc)"
echo "load_start: $(cat /proc/loadavg)"
echo "gov: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
[ -x "$GZ" ] || fail "no gz $GZ"; [ -x "$RG" ] || fail "no rg $RG"

SIL="$CORPUS_DIR/silesia.gz"; [ -f "$SIL" ] || fail "no silesia"

# Gate-0a flavor
FLAVOR="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$SIL" >/dev/null 2>/tmp/flav.$$; grep -m1 'build-flavor=' /tmp/flav.$$ | sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p')"
echo "build-flavor: '$FLAVOR' (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "flavor='$FLAVOR'"
echo "gz_sha256: $(sha256sum "$GZ"|cut -c1-16)  rg: $("$RG" --version 2>&1|head -1)"

# Gate-4 path=ParallelSM
PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$SIL" >/dev/null 2>/tmp/p.$$; grep -m1 'path=' /tmp/p.$$)"
echo "silesia path: $PATHL"
echo "$PATHL" | grep -qE 'path=(ParallelSM|StoredParallel)' || fail "routing '$PATHL'"

# Gate-0b baseline (oracle OFF = production) sha==zcat + rg sha==zcat, per corpus in CELLS
SEEN=""
for cell in $CELLS; do
  corp="${cell%%:*}"; case " $SEEN " in *" $corp "*) continue;; esac; SEEN="$SEEN $corp"
  F="$CORPUS_DIR/$corp.gz"; [ -f "$F" ] || fail "missing $F"
  REF="$(zcat "$F"|sha256sum|cut -c1-16)"
  GB="$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" 2>/dev/null|sha256sum|cut -c1-16)"
  RS="$("$RG" -d -c -P4 "$F" 2>/dev/null|sha256sum|cut -c1-16)"
  echo "$corp: ref=$REF gz_base=$([ "$GB" = "$REF" ]&&echo OK||echo BAD) rg=$([ "$RS" = "$REF" ]&&echo OK||echo BAD)"
  [ "$GB" = "$REF" ] || fail "$corp gz_base sha (production must be correct)"
  [ "$RS" = "$REF" ] || fail "$corp rg sha"
  # Gate-0c ceiling NON-INERT: counter>0 AND output sha DIFFERS from baseline
  CHITS="$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING=1 GZIPPY_SLOW_HITS=1 "$GZ" -d -c -p4 "$F" 2>/tmp/ch.$$ >/tmp/co.$$; grep -m1 'marker-CEILING oracle hits' /tmp/ch.$$ | grep -oE '[0-9]+$')"
  GC="$(cat /tmp/co.$$|sha256sum|cut -c1-16)"
  echo "  ceiling: hits=${CHITS:-0} sha=$GC ($([ "$GC" != "$REF" ]&&echo DIFFERS-good||echo SAME-INERT))"
  [ "${CHITS:-0}" -gt 0 ] 2>/dev/null || fail "$corp ceiling INERT (hits=${CHITS:-0})"
  [ "$GC" != "$REF" ] || fail "$corp ceiling output == baseline (inert perturbation)"
done
echo "GATE0 PASS"

run_gz(){      perf stat -x, -e duration_time -- env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p"$1" "$2" >/dev/null 2>/tmp/pf.$$; grep ',ns,duration_time' /tmp/pf.$$|cut -d, -f1; }
run_gzceil(){  perf stat -x, -e duration_time -- env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING=1 "$GZ" -d -c -p"$1" "$2" >/dev/null 2>/tmp/pf.$$; grep ',ns,duration_time' /tmp/pf.$$|cut -d, -f1; }
run_rg(){      perf stat -x, -e duration_time -- "$RG" -d -c -P"$1" "$2" >/dev/null 2>/tmp/pf.$$; grep ',ns,duration_time' /tmp/pf.$$|cut -d, -f1; }

RAW="$OUT/raw.csv"; : > "$RAW"
echo "--- MEASURE interleaved (GZ,RG,GZc,GZ2,RG2) N=$N ---"
for cell in $CELLS; do
  corp="${cell%%:*}"; T="${cell##*:}"; F="$CORPUS_DIR/$corp.gz"
  echo "  cell $corp T$T ..."
  for r in $(seq 1 "$N"); do
    w=$(run_gz "$T" "$F");     echo "$corp,$T,GZ,$r,$w"   >> "$RAW"
    w=$(run_rg "$T" "$F");     echo "$corp,$T,RG,$r,$w"   >> "$RAW"
    w=$(run_gzceil "$T" "$F"); echo "$corp,$T,GZC,$r,$w"  >> "$RAW"
    w=$(run_gz "$T" "$F");     echo "$corp,$T,GZ2,$r,$w"  >> "$RAW"
    w=$(run_rg "$T" "$F");     echo "$corp,$T,RG2,$r,$w"  >> "$RAW"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "=== ANALYZE ==="
python3 "$OUT/report.py" "$RAW" 2>&1 | tee "$OUT/REPORT.txt"
echo PASS > "$OUT/DONE"
echo "=== ZEN_DONE $(date -u +%FT%TZ) ==="
