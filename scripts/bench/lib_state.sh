#!/usr/bin/env bash
# lib_state.sh — atomic state read/write for the clean-bench host lock.
#
# RUNS ON HOST. Sourced by host_lock_and_bench.sh and gzippy_bench_restore.sh.
#
# The state file records every host setting that the lock MUTATES, captured
# BEFORE the first mutation, so restore is exact. Restore falls back to these
# baked-in safe defaults if the state file is missing/partial (e.g. the script
# was kill -9'd between arming the watchdog and writing state — which cannot
# happen because we write state BEFORE arming, but defaults are the belt to the
# watchdog's suspenders).
#
# State-file format: one `KEY=VALUE` per line. Multi-value keys (per-cpu
# governor/min/max, frozen-guest list) are space-joined on a single line.
#
# All settings the lock touches are VOLATILE (sysfs / MSR / cgroup.freeze), so a
# host reboot is itself a full restore. Nothing here writes anything persistent.

# ---- baked-in safe defaults (the i7-13700T "neurotic" baseline) -------------
# Source: probed live 2026-05-31. no_turbo=0, governor=performance,
# scaling_min=800000, scaling_max=4800000, MSR 0x620=0x82b, THP=madvise.
STATE_DEFAULT_NO_TURBO=0
STATE_DEFAULT_GOVERNOR=performance
STATE_DEFAULT_MIN_FREQ=800000
STATE_DEFAULT_MAX_FREQ=4800000
STATE_DEFAULT_UNCORE_0x620=0x82b
STATE_DEFAULT_THP=madvise

# write_state <statefile> — capture current host state atomically (tmp+rename).
# Captures into KEY=VALUE lines. Idempotent: re-writes the whole file.
write_state() {
  local sf="$1" tmp
  tmp="$(mktemp "${sf}.tmp.XXXXXX")" || return 1

  {
    echo "STATE_VERSION=1"
    echo "CAPTURED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # no_turbo (single global knob on intel_pstate)
    echo "NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"

    # THP
    # /sys exposes "always [madvise] never" — extract the bracketed token.
    local thp
    thp="$(sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    echo "THP=${thp}"

    # MSR 0x620 uncore (read on cpu 0; value is package-scoped)
    echo "UNCORE_0x620=0x$(rdmsr -p 0 0x620 2>/dev/null)"

    # Per-cpu governor / min / max. One token per online cpu, in cpu-index order.
    local govs="" mins="" maxs="" c
    for d in /sys/devices/system/cpu/cpu[0-9]*; do
      [ -d "$d/cpufreq" ] || continue
      c="${d##*/cpu}"
      govs="$govs ${c}:$(cat "$d/cpufreq/scaling_governor" 2>/dev/null)"
      mins="$mins ${c}:$(cat "$d/cpufreq/scaling_min_freq" 2>/dev/null)"
      maxs="$maxs ${c}:$(cat "$d/cpufreq/scaling_max_freq" 2>/dev/null)"
    done
    echo "GOVERNORS=${govs# }"
    echo "MIN_FREQS=${mins# }"
    echo "MAX_FREQS=${maxs# }"
  } >"$tmp" || { rm -f "$tmp"; return 1; }

  # fsync the tmp file then atomically rename into place.
  sync "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$sf" || { rm -f "$tmp"; return 1; }
  sync 2>/dev/null || true
  return 0
}

# state_get <statefile> <KEY> — echo VALUE, or empty if absent.
state_get() {
  local sf="$1" key="$2"
  [ -f "$sf" ] || return 1
  sed -n "s/^${key}=//p" "$sf" | head -1
}
