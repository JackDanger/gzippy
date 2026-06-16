#!/usr/bin/env bash
# _topdown_guest.sh — capture Intel TopdownL1 (+ near-taken branch density) for
# base(TRIPLE) vs SINGLE on a corpus at T1, P-core pinned. Reconciles the
# frontend-vs-backend mechanism question for the multisym-degree sweep.
#
# env: BASE_BIN AFTER_BIN CORPUS MASK N
set -u
BASE_BIN="${BASE_BIN:?}"; AFTER_BIN="${AFTER_BIN:?}"; CORPUS="${CORPUS:?}"
MASK="${MASK:-0}"; N="${N:-7}"
SINK=/dev/shm/.td_sink.bin

run_td() { # <label> <bin>
  local label="$1" bin="$2" pstat
  pstat="$(mktemp /tmp/.td_XXXXXX)"
  # TopdownL1 metric group on the P-core PMU; add branches + near-taken.
  perf stat -M TopdownL1 \
    -e cpu_core/branches/,cpu_core/branch-misses/,cpu_core/instructions/ \
    -o "$pstat" -- taskset -c "$MASK" "$bin" -d -c -p 1 "$CORPUS" >"$SINK" 2>/dev/null
  echo "===TOPDOWN $label==="
  grep -E 'tma_|retiring|bad.spec|frontend|backend|Retiring|Bad|Frontend|Backend|insn per|branches|branch-misses|instructions' "$pstat" || cat "$pstat"
  rm -f "$pstat"
}

echo "## corpus=$CORPUS mask=$MASK N=$N"
# warmup
taskset -c "$MASK" "$BASE_BIN" -d -c -p 1 "$CORPUS" >"$SINK" 2>/dev/null
for ((i=0;i<N;i++)); do run_td BASE "$BASE_BIN"; run_td AFTER "$AFTER_BIN"; done
rm -f "$SINK"
echo TOPDOWN_DONE
