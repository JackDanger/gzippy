#!/usr/bin/env bash
# step3_affinity_guest.sh — STEP-3 [Gate-2, byte-exact, zero decode-path] of the
# CONSUMER-CONFIRM plan (run on guest 199, FROZEN, UNPINNED = production-style).
#
# Tests P1's SMT-co-location hypothesis for the +17% UNPINNED cyc/B inflation
# directly: build-in affinity (GZIPPY_PHYS_PIN=1) pins gz's 4 workers + in-order
# consumer to DISTINCT PHYSICAL cores (one logical id per SMT sibling group),
# launched WITHOUT external taskset. Compares, UNPINNED:
#   gzbase  : gz default (GZIPPY_PHYS_PIN unset)   — reproduces the +19% loss
#   gzphys  : gz GZIPPY_PHYS_PIN=1                  — the affinity fix candidate
#   rg      : rapidgzip -P4                          — the parity target
# perf cyc/B + task-clock + GHz, N>=9 interleaved, /dev/null all arms, byte-exact,
# A/A self-tests, + a /proc affinity VERIFICATION (each gzphys thread on a
# distinct physical core).
#
# PASS: gzphys cyc/B falls toward ~9.6 AND gzphys/rg best <= ~1.05 (Delta>spread)
#       => SMT-co-location CONFIRMED + shippable byte-exact fix candidate.
# FLAT: gzphys ~= gzbase => SMT-co-location FALSIFIED (the +17% is something else).
#
# Usage:
#   GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
#   CORP=/root/silesia.gz N=11 bash step3_affinity_guest.sh
set -uo pipefail
GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORP="${CORP:-/root/silesia.gz}"
N="${N:-11}"
THREADS="${THREADS:-4}"
OUT="${OUT:-/dev/shm/step3_affinity}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYZE="$HERE/step3_affinity_analyze.py"
[ -f "$ANALYZE" ] || ANALYZE="/root/gzippy/scripts/bench/kernel-ab/step3_affinity_analyze.py"
[ -x "$GZ" ] || { echo "NO GZ $GZ"; exit 2; }
[ -x "$RG" ] || { echo "NO RG $RG"; exit 2; }
[ -f "$CORP" ] || { echo "NO CORP $CORP"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"

echo "============ STEP-3 INTERNAL PHYSICAL-CORE AFFINITY (UNPINNED silesia-T$THREADS) ============"
echo "GZ=$GZ  sha=$(sha256sum "$GZ"|cut -c1-16)"
echo "RG=$RG  sha=$(sha256sum "$RG"|cut -c1-16)"
echo "CORP=$CORP  N=$N"
echo "load: $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)  no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"

echo "--- GATE-0 FLAVOR/PATH ---"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p1 "$CORP" >/dev/null 2>"$OUT/dbg.txt" || true
grep -oiE 'flavor[=: ]+[a-z+-]+' "$OUT/dbg.txt" | head -1
PATHLINE=$(grep -oE 'path=[A-Za-z]+' "$OUT/dbg.txt" | head -1); echo "    $PATHLINE"

echo "--- GATE-0 BYTE-EXACT (both gz modes + rg, /dev/null) ---"
REF=$(zcat "$CORP" | sha256sum | cut -c1-16)
SB=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
SP=$(GZIPPY_PHYS_PIN=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
SR=$("$RG" -dc -P$THREADS "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
echo "    ref=$REF gzbase=$SB gzphys=$SP rg=$SR"
[ "$SB" = "$REF" ] && [ "$SP" = "$REF" ] && [ "$SR" = "$REF" ] || { echo "    BYTE MISMATCH — VOID"; exit 3; }
echo "    BYTE-EXACT PASS (gzbase==gzphys==rg==zcat)"
OUTBYTES=$(zcat "$CORP" | wc -c)
echo "    output_bytes=$OUTBYTES"

# ---- AFFINITY VERIFICATION: run gzphys, sample /proc tids' current CPU (field 39) ----
echo "--- AFFINITY VERIFICATION (gzphys: each thread on a distinct physical core) ---"
VERIFY_PY="$HERE/step3_verify_affinity.py"
[ -f "$VERIFY_PY" ] || VERIFY_PY="/root/gzippy/scripts/bench/kernel-ab/step3_verify_affinity.py"
python3 "$VERIFY_PY" "$GZ" "$CORP" "$THREADS" 2>&1 | sed 's/^/    /'

# ---- MEASUREMENT ----
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/LLC-load-misses/,task-clock,duration_time"
run_gz() {  # $1=env-extra $2=outfile
  perf stat -x, -e "$EVENTS" -- env $1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" >/dev/null 2>"$2"
}
run_rg() {  # $1=outfile
  perf stat -x, -e "$EVENTS" -- "$RG" -dc -P$THREADS "$CORP" >/dev/null 2>"$1"
}
# warmup
run_gz "" "$OUT/warm.base"; run_gz "GZIPPY_PHYS_PIN=1" "$OUT/warm.phys"; run_rg "$OUT/warm.rg"

echo "--- MEASURE interleaved gzbaseA,gzbaseB,gzphysA,gzphysB,rgA,rgB x N=$N ---"
for r in $(seq 1 "$N"); do
  run_gz "" "$OUT/gzbaseA.$r.csv"
  run_gz "" "$OUT/gzbaseB.$r.csv"
  run_gz "GZIPPY_PHYS_PIN=1" "$OUT/gzphysA.$r.csv"
  run_gz "GZIPPY_PHYS_PIN=1" "$OUT/gzphysB.$r.csv"
  run_rg "$OUT/rgA.$r.csv"
  run_rg "$OUT/rgB.$r.csv"
done
echo "load(post): $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"
echo "--- ANALYSIS ---"
python3 "$ANALYZE" "$OUT" "$OUTBYTES"
echo "DONE_STEP3_AFFINITY_GUEST"
