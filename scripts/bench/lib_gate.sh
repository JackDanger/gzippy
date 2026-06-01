#!/usr/bin/env bash
# lib_gate.sh — frequency-stability gate.
#
# RUNS ON HOST. Sourced by host_lock_and_bench.sh.
#
# The gate proves the cores are actually running at a fixed, locked frequency
# BEFORE any timing rows are emitted. It does this with the aperf/mperf
# invariant: under a steady all-core load, the ratio Δaperf/Δmperf is the
# achieved-vs-base frequency multiplier. With turbo off + governor=performance +
# min==max, every P-core should report the SAME ratio (~1.0 relative to base
# clock) and the derived MHz should sit within tolerance of the platform base.
#
# A gate FAIL is fatal: the caller restores the box and emits ONLY a failed-gate
# record (never timing rows). Untrustworthy frequency state must never be
# silently turned into "results".
#
# References:
#   MSR 0xE8 = IA32_APERF, 0xE7 = IA32_MPERF (per-logical-cpu, count actual vs
#   reference TSC-rate cycles). MSR 0xCE = MSR_PLATFORM_INFO; bits 15:8 are the
#   max non-turbo ratio (×100 MHz) = the base clock.

GATE_RATIO_LO="${GATE_RATIO_LO:-0.985}"
GATE_RATIO_HI="${GATE_RATIO_HI:-1.015}"
GATE_MHZ_TOL_PCT="${GATE_MHZ_TOL_PCT:-1.5}"
GATE_SAMPLE_SECS="${GATE_SAMPLE_SECS:-1}"
GATE_PCORES="${GATE_PCORES:-0 2 4 6 8 10 12 14}"

# base_mhz_from_msr — decode MSR_PLATFORM_INFO bits 15:8 → base MHz.
base_mhz_from_msr() {
  local raw ratio
  raw="$(rdmsr -p 0 0xCE 2>/dev/null)" || return 1
  [ -n "$raw" ] || return 1
  # raw is hex with no 0x prefix. ratio = bits 15:8.
  ratio=$(( ( 0x${raw} >> 8 ) & 0xff ))
  echo $(( ratio * 100 ))
}

# _rd_msr_dec <cpu> <msr-hex> — read MSR as a decimal integer.
_rd_msr_dec() {
  local v
  v="$(rdmsr -p "$1" -d "$2" 2>/dev/null)" || return 1
  [ -n "$v" ] || return 1
  echo "$v"
}

# run_gate — load P-cores, sample aperf/mperf delta, validate ratio + MHz.
# Echoes a human-readable report to stdout. Returns 0 = PASS, nonzero = FAIL.
# On return, sets global GATE_REPORT (multi-line) and GATE_STATUS (PASS|FAIL).
run_gate() {
  local base_mhz
  base_mhz="$(base_mhz_from_msr)" || { GATE_STATUS=FAIL; GATE_REPORT="gate: cannot read base MHz (MSR 0xCE)"; echo "$GATE_REPORT"; return 1; }

  # Pin a steady all-core load onto the P-cores for the sampling window.
  # stress-ng --cpu 8 with a taskset mask covering the P-core logical ids.
  local mask
  mask="$(echo "$GATE_PCORES" | tr ' ' ',')"
  taskset -c "$mask" stress-ng --cpu 8 --timeout "$(( GATE_SAMPLE_SECS + 3 ))s" >/dev/null 2>&1 &
  local stress_pid=$!

  # Let the load ramp before first sample.
  sleep 1

  # First snapshot.
  local -A a0 m0 a1 m1
  local cpu
  for cpu in $GATE_PCORES; do
    a0[$cpu]="$(_rd_msr_dec "$cpu" 0xE8)"
    m0[$cpu]="$(_rd_msr_dec "$cpu" 0xE7)"
  done
  sleep "$GATE_SAMPLE_SECS"
  for cpu in $GATE_PCORES; do
    a1[$cpu]="$(_rd_msr_dec "$cpu" 0xE8)"
    m1[$cpu]="$(_rd_msr_dec "$cpu" 0xE7)"
  done

  # Stop the load.
  kill "$stress_pid" 2>/dev/null
  wait "$stress_pid" 2>/dev/null

  local status=PASS
  local report="gate: base=${base_mhz}MHz tol_ratio=[${GATE_RATIO_LO},${GATE_RATIO_HI}] tol_mhz=±${GATE_MHZ_TOL_PCT}% sample=${GATE_SAMPLE_SECS}s"
  report+=$'\n'

  for cpu in $GATE_PCORES; do
    local da dm
    if [ -z "${a0[$cpu]}" ] || [ -z "${a1[$cpu]}" ] || [ -z "${m0[$cpu]}" ] || [ -z "${m1[$cpu]}" ]; then
      report+="  cpu${cpu}: MSR read failed -> FAIL"$'\n'
      status=FAIL
      continue
    fi
    da=$(( a1[$cpu] - a0[$cpu] ))
    dm=$(( m1[$cpu] - m0[$cpu] ))
    if [ "$dm" -le 0 ]; then
      report+="  cpu${cpu}: Δmperf<=0 (idle/wrap) -> FAIL"$'\n'
      status=FAIL
      continue
    fi
    # ratio = Δaperf/Δmperf ; achieved MHz = base * ratio.
    local line
    line="$(awk -v da="$da" -v dm="$dm" -v base="$base_mhz" \
                -v lo="$GATE_RATIO_LO" -v hi="$GATE_RATIO_HI" -v tol="$GATE_MHZ_TOL_PCT" \
      'BEGIN{
         r=da/dm; mhz=base*r;
         dpct=(mhz-base)/base*100; if(dpct<0)dpct=-dpct;
         ok=(r>=lo && r<=hi && dpct<=tol)?"OK":"FAIL";
         printf "  cpu%-2s ratio=%.4f mhz=%.0f (%+.2f%% vs base) -> %s\n", "X", r, mhz, (mhz-base)/base*100, ok;
       }')"
    line="${line/cpuX /cpu${cpu} }"
    report+="$line"$'\n'
    case "$line" in *FAIL*) status=FAIL;; esac
  done

  GATE_STATUS="$status"
  GATE_REPORT="$report"
  printf '%s' "$report"
  [ "$status" = PASS ]
}
