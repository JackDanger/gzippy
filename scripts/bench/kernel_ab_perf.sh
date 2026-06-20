#!/usr/bin/env bash
# NIGHT31 — paired interleaved perf for the isolated kernel-vs-kernel A/B.
# Runs ARM A and ARM B alternately, N reps each, pinned to one core, capturing
# perf-stat cycles+instructions around the WHOLE process (the kernel loop is
# >>99% of process work at reps=4290 — one block parse vs 4290 block decodes).
# Reports per-run cyc/B + instr/B + IPC; the caller computes paired CI.
set -euo pipefail

BIN=/dev/shm/kab-target/release/examples/kernel_ab
PIN=${PIN:-4}
REPS=${REPS:-4290}        # ~256 MiB total
N=${N:-13}                # >= 11
OUT=${OUT:-/dev/shm/kab_perf.csv}

[ -x "$BIN" ] || { echo "no binary $BIN"; exit 2; }

echo "arm,run,cycles,instructions,bytes,cyc_per_byte,instr_per_byte,ipc" > "$OUT"

run_one() {
  local arm="$1" run="$2"
  # perf stat: cycles + instructions for the whole process. The example prints
  # bytes on stdout; perf writes counts to stderr (parsed below).
  local perflog bytes cyc ins
  perflog=$(mktemp)
  bytes=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" \
      perf stat -x, -e cycles,instructions \
      "$BIN" --arm "$arm" --reps "$REPS" 2>"$perflog" \
      | grep '^ARM=' | sed -E 's/.*bytes=([0-9]+).*/\1/')
  cyc=$(grep -E ',cycles' "$perflog" | head -1 | cut -d, -f1)
  ins=$(grep -E ',instructions' "$perflog" | head -1 | cut -d, -f1)
  rm -f "$perflog"
  # Strip any thousands separators perf might emit.
  cyc=${cyc//[^0-9]/}; ins=${ins//[^0-9]/}
  local cpb ipb ipc
  cpb=$(awk "BEGIN{printf \"%.4f\", $cyc/$bytes}")
  ipb=$(awk "BEGIN{printf \"%.4f\", $ins/$bytes}")
  ipc=$(awk "BEGIN{printf \"%.4f\", $ins/$cyc}")
  echo "$arm,$run,$cyc,$ins,$bytes,$cpb,$ipb,$ipc" | tee -a "$OUT"
}

# Warm both once (page-in, freq ramp) — not recorded.
GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm a --reps 200 >/dev/null 2>&1 || true
GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm b --reps 200 >/dev/null 2>&1 || true

for r in $(seq 1 "$N"); do
  # Randomize arm order each pair to decorrelate.
  if [ $((RANDOM % 2)) -eq 0 ]; then
    run_one a "$r"; run_one b "$r"
  else
    run_one b "$r"; run_one a "$r"
  fi
done

echo "=== SUMMARY (median + IQR) ==="
python3 - "$OUT" <<'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
def col(arm,k): return sorted(float(r[k]) for r in rows if r['arm']==arm)
def med(x): return st.median(x)
def q(x,p):
    i=max(0,min(len(x)-1,int(round(p*(len(x)-1))))); return x[i]
for arm in ('a','b'):
    cpb=col(arm,'cyc_per_byte'); ipb=col(arm,'instr_per_byte'); ipc=col(arm,'ipc')
    print(f"ARM {arm}: n={len(cpb)} cyc/B med={med(cpb):.4f} [{q(cpb,.25):.4f},{q(cpb,.75):.4f}]  "
          f"instr/B med={med(ipb):.4f}  IPC med={med(ipc):.4f}")
a=col('a','cyc_per_byte'); b=col('b','cyc_per_byte')
print(f"GAP cyc/B (A-B) median = {med(a)-med(b):+.4f}")
ai=col('a','instr_per_byte'); bi=col('b','instr_per_byte')
print(f"GAP instr/B (A-B) median = {med(ai)-med(bi):+.4f}")
print(f"IPC A med={med(col('a','ipc')):.4f}  IPC B med={med(col('b','ipc')):.4f}  dIPC={med(col('a','ipc'))-med(col('b','ipc')):+.4f}")
PY
