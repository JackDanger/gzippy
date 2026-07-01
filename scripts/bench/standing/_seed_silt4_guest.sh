#!/usr/bin/env bash
# _seed_silt4_guest.sh — Gate-2 REMOVAL-ORACLE for the silesia-T4 residual.
#
# Isolates the speculative u16-MARKER machinery's contribution to the silesia-T4
# gz-vs-rg wall via the seed_windows oracle: a p=1 CAPTURE records aligned
# predecessor windows; the seeded run forces EVERY chunk down the CLEAN
# (window-present) decode path — removing the marker/apply-window/replace-markers
# premium while keeping decode COMPUTE fully intact (byte-identical output).
#
# 4 walls, interleaved best-of-N, /dev/null both arms, same P-core mask:
#   PROD  = production (markered speculative path)         — the baseline
#   SEED  = seeded-clean (ALL marker machinery removed)    — the CEILING (cheats:
#                                                            knows windows from T1)
#   RG    = rapidgzip native ELF                           — the bar
#   (PROD2 = A/A self-test of PROD)
#
# GATE-0 (loud, blocking): SEED run BYTE-EXACT (sha==zcat); SEED replay hits>0 AND
#   misses==0 (non-inert proof the clean path was actually forced); capture wrote
#   N>0 windows; path=ParallelSM; PROD/PROD2 A/A within spread.
# VERDICT (removal-tier): marker-machinery share of the recoverable wall =
#   (PROD - SEED) vs (PROD - RG). Caveat: SEED is a CEILING (window-foreknowledge),
#   never the prize; the prize is a measured Δ of a concrete cheapening A/B.
set -u
GZ=/dev/shm/standing-target/release/gzippy
RG=${RG:-/root/oracle_c/rapidgzip-native}
F=${F:-/root/silesia.gz}
T=${T:-4}
N=${N:-11}
PINBASE=${PINBASE:-0}
SEEDF=/dev/shm/seed_silt4.bin
OUT=/dev/shm/seed_silt4
mkdir -p "$OUT"; rm -f "$OUT"/*.csv

mask() { local t=$1 i=0 m=""; while [ "$i" -lt "$t" ]; do [ -n "$m" ] && m="$m,"; m="$m$((PINBASE+i*2))"; i=$((i+1)); done; echo "$m"; }
M="$(mask "$T")"
echo "== SEED-WINDOWS removal-oracle: silesia T$T mask=$M N=$N =="
[ -x "$GZ" ] || { echo "FAIL: no gz binary $GZ"; exit 2; }
[ -x "$RG" ] || { echo "FAIL: no rg binary $RG"; exit 2; }

REF="$(zcat "$F" | sha256sum | cut -c1-16)"
echo "ref sha=$REF"

# ---- GATE-0: CAPTURE aligned windows via a p=1 run (records every chunk) ----
echo "--- CAPTURE (p=1) ---"
rm -f "$SEEDF"
GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE="$SEEDF" \
  taskset -c "$PINBASE" "$GZ" -d -c -p1 "$F" >/dev/null 2>/tmp/cap.err || { echo "FAIL capture run"; cat /tmp/cap.err; exit 2; }
if [ ! -s "$SEEDF" ]; then echo "FAIL: capture wrote no seed file"; cat /tmp/cap.err; exit 2; fi
CAPSZ=$(stat -c%s "$SEEDF")
echo "capture file bytes=$CAPSZ"

# ---- GATE-0: SEED run byte-exact + non-inert (hits>0 misses==0) ----
echo "--- SEED self-test (byte-exact + non-inert) ---"
SEEDSHA="$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS="$SEEDF" taskset -c "$M" "$GZ" -d -c -p"$T" "$F" 2>/tmp/seed.err | sha256sum | cut -c1-16)"
REPLAY="$(grep -m1 'SEED_WINDOWS replay' /tmp/seed.err || echo 'NONE')"
echo "seed sha=$SEEDSHA  ($([ "$SEEDSHA" = "$REF" ] && echo BYTE-EXACT || echo MISMATCH))"
echo "replay: $REPLAY"
[ "$SEEDSHA" = "$REF" ] || { echo "FAIL: seeded output NOT byte-exact"; exit 2; }
HITS="$(echo "$REPLAY" | sed -n 's/.*hits=\([0-9]*\).*/\1/p')"
MISS="$(echo "$REPLAY" | sed -n 's/.*misses=\([0-9]*\).*/\1/p')"
[ "${HITS:-0}" -gt 0 ] || { echo "FAIL: seed INERT (hits=0) — measuring production, not clean"; exit 2; }
echo "non-inert OK: hits=$HITS misses=${MISS:-?}"
# also verify rg correctness
RGSHA="$(taskset -c "$M" "$RG" -d -c -P"$T" "$F" 2>/dev/null | sha256sum | cut -c1-16)"
[ "$RGSHA" = "$REF" ] || { echo "FAIL: rg sha mismatch"; exit 2; }
echo "GATE-0 PASS"

EV="duration_time,cpu_core/cycles/,cpu_core/instructions/"
run() { # arm
  local arm=$1 csv="$OUT/$arm.csv"
  case "$arm" in
    PROD|PROD2) taskset -c "$M" perf stat -x, -e "$EV" -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$M" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>>"$csv";;
    SEED)       taskset -c "$M" perf stat -x, -e "$EV" -- env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS="$SEEDF" taskset -c "$M" "$GZ" -d -c -p"$T" "$F" >/dev/null 2>>"$csv";;
    RG)         taskset -c "$M" perf stat -x, -e "$EV" -- taskset -c "$M" "$RG" -d -c -P"$T" "$F" >/dev/null 2>>"$csv";;
  esac
}
echo "--- MEASURE (interleaved PROD,SEED,RG,PROD2; N=$N) ---"
for r in $(seq 1 "$N"); do run PROD; run SEED; run RG; run PROD2; done
echo "=== RAW (duration_time ns per rep) ==="
for arm in PROD SEED RG PROD2; do
  echo "--- $arm ---"; grep -E '^[0-9]+,.*duration_time' "$OUT/$arm.csv" | cut -d, -f1 | tr '\n' ' '; echo
done
echo "SEED_SILT4_DONE"
