#!/usr/bin/env bash
# perthread_cpu_duty_guest.sh — STEP-1 [BLOCKING CONFIRM] driver (run on guest 199, FROZEN).
#
# Resolves the consumer-thread instrument contradiction with a CONSERVED per-thread
# OS-CPU duty-cycle measurement. GATE-0 flavor/path + byte-exact; clean perf
# task-clock reference (no poller) for the conservation gate; then the /proc
# per-tid duty rig (poller). If Sum(per-tid core-sec) reconciles to clean perf
# task-clock <=5%, the poller did not perturb and the duty numbers are honest.
#
# Usage:
#   GZ=/dev/shm/gzrg-target/release/gzippy CORP=/root/silesia.gz N=9 \
#   bash perthread_cpu_duty_guest.sh
set -uo pipefail
GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
CORP="${CORP:-/root/silesia.gz}"
N="${N:-9}"
THREADS="${THREADS:-4}"
OUT="${OUT:-/dev/shm/perthread_duty}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/perthread_cpu_duty.py"
[ -f "$PY" ] || PY="/root/gzippy/scripts/bench/kernel-ab/perthread_cpu_duty.py"
[ -x "$GZ" ] || { echo "NO GZ $GZ"; exit 2; }
[ -f "$CORP" ] || { echo "NO CORP $CORP"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"

echo "============ STEP-1 PER-THREAD OS-CPU DUTY (UNPINNED silesia-T$THREADS) ============"
echo "GZ=$GZ  sha=$(sha256sum "$GZ"|cut -c1-16)"
echo "CORP=$CORP  N=$N"
echo "load: $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)  no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"

echo "--- GATE-0 FLAVOR/PATH ---"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p1 "$CORP" >/dev/null 2>"$OUT/dbg.txt" || true
grep -oiE 'flavor[=: ]+[a-z+-]+' "$OUT/dbg.txt" | head -1
PATHLINE=$(grep -oE 'path=[A-Za-z]+' "$OUT/dbg.txt" | head -1)
echo "    $PATHLINE"
echo "$PATHLINE" | grep -q 'path=ParallelSM' && echo "    PATH PASS" || echo "    PATH WARN"

echo "--- GATE-0 BYTE-EXACT (/dev/null) ---"
REF=$(zcat "$CORP" | sha256sum | cut -c1-16)
SG=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
echo "    ref=$REF gz=$SG"
[ "$SG" = "$REF" ] || { echo "    BYTE MISMATCH — VOID"; exit 3; }
echo "    BYTE-EXACT PASS"

# ---- clean perf task-clock reference (NO poller), N runs, median ----
echo "--- CLEAN perf task-clock reference (no poller, N=$N) ---"
: > "$OUT/taskclock.txt"
# warmup
env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" >/dev/null 2>/dev/null
for r in $(seq 1 "$N"); do
  perf stat -x, -e task-clock -- env GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p$THREADS "$CORP" \
    >/dev/null 2>"$OUT/tc.$r.csv"
  # task-clock is reported in ms in the first field
  ms=$(awk -F, '/task-clock/{print $1; exit}' "$OUT/tc.$r.csv")
  echo "$ms" >> "$OUT/taskclock.txt"
done
MED_TC_MS=$(sort -n "$OUT/taskclock.txt" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
MED_TC_S=$(awk "BEGIN{printf \"%.4f\", $MED_TC_MS/1000.0}")
echo "    median task-clock = $MED_TC_MS ms = $MED_TC_S s"

# ---- per-tid duty rig (poller) ----
echo "--- PER-TID DUTY RIG ---"
python3 "$PY" --gz "$GZ" --corpus "$CORP" --threads "$THREADS" --n "$N" \
  --sample-ms 1.0 --perf-taskclock-sec "$MED_TC_S" --tol 0.05 \
  --json "$OUT/duty.json"
echo "load(post): $(cat /proc/loadavg)  procs_running=$(awk '/procs_running/{print $2}' /proc/stat)"
echo "DONE_PERTHREAD_DUTY_GUEST"
