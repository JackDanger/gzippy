#!/usr/bin/env bash
# gzippy_bench_restore.sh — IDEMPOTENT restore of the host to baseline.
#
# RUNS ON HOST. Two callers:
#   1. host_lock_and_bench.sh trap / normal teardown (fast path).
#   2. the systemd-run watchdog timer (gzippy-bench-restore) — the PRIMARY
#      guarantee. If the laptop ssh dies / the host script is kill -9'd, the
#      watchdog still fires and restores the box.
#
# Reads the state file written before any mutation; falls back to baked-in safe
# defaults (lib_state.sh) for any missing/unparsable field. Every restore step
# is idempotent — running twice is a no-op the second time. Verifies readbacks
# and logs mismatches; exit status reflects whether the box is at baseline.
#
# Usage: gzippy_bench_restore.sh <statefile>
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_state.sh
. "$HERE/lib_state.sh"

SF="${1:-/run/gzippy-bench/state}"
LOG="${RESTORE_LOG:-/run/gzippy-bench/restore.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

log() { echo "[restore $(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

log "begin restore from statefile=$SF (exists=$([ -f "$SF" ] && echo yes || echo no))"

# ---- resolve target values (state file, else safe defaults) -----------------
get() { # get KEY DEFAULT
  local v; v="$(state_get "$SF" "$1" 2>/dev/null)"
  if [ -n "$v" ]; then echo "$v"; else echo "$2"; fi
}

T_NO_TURBO="$(get NO_TURBO "$STATE_DEFAULT_NO_TURBO")"
T_THP="$(get THP "$STATE_DEFAULT_THP")"
T_UNCORE="$(get UNCORE_0x620 "$STATE_DEFAULT_UNCORE_0x620")"
GOVERNORS="$(state_get "$SF" GOVERNORS 2>/dev/null)"
MIN_FREQS="$(state_get "$SF" MIN_FREQS 2>/dev/null)"
MAX_FREQS="$(state_get "$SF" MAX_FREQS 2>/dev/null)"
FROZEN="$(state_get "$SF" FROZEN_GUESTS 2>/dev/null)"

# ---- 1. THAW frozen guests (do this FIRST — most important for service) -----
# cgroup v2 freeze path for LXC. Idempotent: writing 0 to an already-thawed
# cgroup is a no-op.
thaw_guest() {
  local id="$1" f
  for f in \
    "/sys/fs/cgroup/lxc/$id/cgroup.freeze" \
    "/sys/fs/cgroup/lxc.payload.$id/cgroup.freeze"; do
    if [ -w "$f" ]; then echo 0 >"$f" 2>/dev/null && log "thawed guest $id ($f)"; fi
  done
}
if [ -n "$FROZEN" ]; then
  for id in $FROZEN; do thaw_guest "$id"; done
else
  log "no frozen-guest list in state; scanning for any frozen lxc cgroups"
  for f in /sys/fs/cgroup/lxc/*/cgroup.freeze /sys/fs/cgroup/lxc.payload.*/cgroup.freeze; do
    [ -e "$f" ] || continue
    if [ "$(cat "$f" 2>/dev/null)" = "1" ]; then echo 0 >"$f" 2>/dev/null && log "thawed (scan) $f"; fi
  done
fi

# ---- 2. no_turbo ------------------------------------------------------------
if [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  echo "$T_NO_TURBO" >/sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null \
    && log "no_turbo <- $T_NO_TURBO"
fi

# ---- 3. uncore MSR 0x620 (package-scoped; -a writes all) --------------------
if [ -n "$T_UNCORE" ]; then
  wrmsr -a 0x620 "$T_UNCORE" 2>/dev/null && log "uncore 0x620 <- $T_UNCORE"
fi

# ---- 4. per-cpu governor / min / max ----------------------------------------
restore_percpu() { # restore_percpu "<id:val ...>" <cpufreq-file>
  local pairs="$1" file="$2" pair cpu val
  [ -n "$pairs" ] || return 0
  for pair in $pairs; do
    cpu="${pair%%:*}"; val="${pair#*:}"
    [ -n "$cpu" ] && [ -n "$val" ] || continue
    local p="/sys/devices/system/cpu/cpu${cpu}/cpufreq/${file}"
    [ -w "$p" ] && echo "$val" >"$p" 2>/dev/null
  done
}
# Order matters: set governor first, then min, then max (avoids min>max rejects).
restore_percpu "$GOVERNORS" scaling_governor
restore_percpu "$MIN_FREQS" scaling_min_freq
restore_percpu "$MAX_FREQS" scaling_max_freq
[ -n "$GOVERNORS" ] && log "per-cpu governor/min/max restored from state" \
  || log "no per-cpu cpufreq in state; left as-is (defaults are volatile, reboot clears)"

# ---- 5. THP -----------------------------------------------------------------
if [ -n "$T_THP" ] && [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
  echo "$T_THP" >/sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null \
    && log "THP <- $T_THP"
fi

# ---- 6. release the C-state floor holder ------------------------------------
# host_lock_and_bench.sh writes the holder pid into the statefile dir.
PIDF="$(dirname "$SF")/cstate_holder.pid"
if [ -f "$PIDF" ]; then
  hp="$(cat "$PIDF" 2>/dev/null)"
  if [ -n "$hp" ] && kill -0 "$hp" 2>/dev/null; then
    kill "$hp" 2>/dev/null && log "released C-state holder pid=$hp"
  fi
  rm -f "$PIDF" 2>/dev/null
fi

# ---- 7. verify readbacks ----------------------------------------------------
rc=0
nt="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
[ "$nt" = "$T_NO_TURBO" ] || { log "MISMATCH no_turbo=$nt want=$T_NO_TURBO"; rc=1; }
un="0x$(rdmsr -p 0 0x620 2>/dev/null)"
[ "$un" = "$T_UNCORE" ] || { log "MISMATCH uncore=$un want=$T_UNCORE"; rc=1; }
# any guest still frozen?
for f in /sys/fs/cgroup/lxc/*/cgroup.freeze /sys/fs/cgroup/lxc.payload.*/cgroup.freeze; do
  [ -e "$f" ] || continue
  if [ "$(cat "$f" 2>/dev/null)" = "1" ]; then log "MISMATCH still-frozen $f"; rc=1; fi
done

if [ "$rc" = 0 ]; then log "RESTORE VERIFIED: no_turbo=$nt uncore=$un guests-thawed"; else log "RESTORE INCOMPLETE (rc=$rc)"; fi
exit "$rc"
