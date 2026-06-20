#!/usr/bin/env bash
# NIGHT32 — isolated STATELESS-kernel A/B (the STRONG deliverable).
# Three arms, paired + interleaved, pinned to one P-core, perf-stat per run:
#   a   = gz run_contig            (resumable, D-1 anchor SAVED every iter)
#   as  = gz run_contig_stateless  (D-1 anchor SHED — GZIPPY_STATELESS_KERNEL=1, --arm a)
#   b   = igzip decode_huffman_code_block_stateless_04
# Question: how much of the +3.40 instr/B (+0.89 cyc/B) ARM A residual over _04
# does shedding the resumable contract close.
#
# GATE-0: the harness self-tests byte-exact (ARM A dies on mismatch) AND prints
# stateless_entries(warm) > 0 for the 'as' arm (non-inert). This script ALSO
# runs an A-vs-A self-test (same resumable kernel both pseudo-arms) and reports
# the ratio — if it is not ~1.0 the box is too loaded to trust any Delta.
set -euo pipefail

BIN=${BIN:-/dev/shm/kab-target/release/examples/kernel_ab}
PIN=${PIN:-4}
REPS=${REPS:-4290}        # ~256 MiB total
N=${N:-13}                # >= 13 (Gate-1)
OUT=${OUT:-/dev/shm/kab_stateless_perf.csv}

[ -x "$BIN" ] || { echo "no binary $BIN"; exit 2; }

echo "arm,run,cycles,instructions,bytes,cyc_per_byte,instr_per_byte,ipc" > "$OUT"

# arm label -> (env, binary --arm)
run_one() {
  local arm="$1" run="$2"
  local env_stateless="" binarm="a"
  case "$arm" in
    a)  env_stateless="";                          binarm="a" ;;
    a2) env_stateless="";                          binarm="a" ;;  # self-test twin (resumable)
    as) env_stateless="GZIPPY_STATELESS_KERNEL=1"; binarm="a" ;;
    b)  env_stateless="";                          binarm="b" ;;
    *)  echo "bad arm $arm"; exit 2 ;;
  esac
  local perflog bytes cyc ins
  perflog=$(mktemp)
  bytes=$(env $env_stateless GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" \
      perf stat -x, -e cycles,instructions \
      "$BIN" --arm "$binarm" --reps "$REPS" 2>"$perflog" \
      | grep '^ARM=' | sed -E 's/.*bytes=([0-9]+).*/\1/')
  cyc=$(grep -E 'cpu_core/cycles/' "$perflog" | head -1 | cut -d, -f1)
  ins=$(grep -E 'cpu_core/instructions/' "$perflog" | head -1 | cut -d, -f1)
  if [ -z "$cyc" ]; then
    cyc=$(grep -E ',cycles,|,cycles$|cycles,' "$perflog" | head -1 | cut -d, -f1)
    ins=$(grep -E ',instructions,|,instructions$|instructions,' "$perflog" | head -1 | cut -d, -f1)
  fi
  rm -f "$perflog"
  cyc=${cyc//[^0-9]/}; ins=${ins//[^0-9]/}
  local cpb ipb ipc
  cpb=$(awk "BEGIN{printf \"%.4f\", $cyc/$bytes}")
  ipb=$(awk "BEGIN{printf \"%.4f\", $ins/$bytes}")
  ipc=$(awk "BEGIN{printf \"%.4f\", $ins/$cyc}")
  echo "$arm,$run,$cyc,$ins,$bytes,$cpb,$ipb,$ipc" | tee -a "$OUT"
}

# Warm each path once (page-in, freq ramp) — not recorded. Also surfaces the
# Gate-0 byte-exact + non-inert lines on stderr.
GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm a --reps 200 >/dev/null 2>&1 || true
GZIPPY_STATELESS_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm a --reps 200 >/dev/null 2>&1 || true
GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm b --reps 200 >/dev/null 2>&1 || true

# GATE-0 NON-INERT + BYTE-EXACT (loud): a single full warm of the stateless arm
# with stderr shown — must say byte-exact OK and stateless_entries(warm) > 0.
echo "=== GATE-0 stateless non-inert + byte-exact (stderr) ==="
GZIPPY_STATELESS_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$BIN" --arm a --reps 1 2>&1 >/dev/null | grep -E 'Gate-0|stateless|mismatch|INERT' || true

for r in $(seq 1 "$N"); do
  # Randomize the order of the three measured arms each round to decorrelate.
  arms=(a as b)
  # Fisher-Yates-ish shuffle via sort -R
  shuffled=$(printf '%s\n' "${arms[@]}" | sort -R)
  for arm in $shuffled; do run_one "$arm" "$r"; done
done

# A-vs-A self-test: run the resumable kernel as two pseudo-arms a few times.
echo "=== A-vs-A SELF-TEST (resumable both; ratio must be ~1.0) ==="
SELFOUT=/dev/shm/kab_selftest.csv
echo "arm,run,cycles,instructions,bytes,cyc_per_byte,instr_per_byte,ipc" > "$SELFOUT"
OUT="$SELFOUT"
for r in $(seq 1 7); do
  if [ $((RANDOM % 2)) -eq 0 ]; then run_one a "$r"; run_one a2 "$r"
  else run_one a2 "$r"; run_one a "$r"; fi
done
OUT=${OUT:-/dev/shm/kab_stateless_perf.csv}

echo "=== SUMMARY ==="
python3 /dev/shm/kab_stateless_analyze.py /dev/shm/kab_stateless_perf.csv "$SELFOUT"
