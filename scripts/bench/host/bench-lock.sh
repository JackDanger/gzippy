#!/usr/bin/env bash
# bench-lock.sh — THE single host-side (neurotic) freeze lifecycle, DECOUPLED.
#
# This is the consolidation of the four overlapping host freeze scripts
# (/root/clock_freeze.sh, /root/host_freeze_watchdog.sh, /root/freeze_wrapper.sh,
# /root/freeze_profile.sh) into ONE standalone acquire/release/status lock that
# the repo measurement spine (scripts/bench/parity.sh, oracle.sh) brackets its
# runs with — instead of each driver freezing by hand (and each missing a
# different noisy neighbor).
#
# DEPLOY: lives on neurotic at /root/bench-lock.sh. Edit there (it is host-side
# and privileged); this repo copy is the source-of-truth mirror — keep them in
# sync. It REUSES the proven plumbing already on the host:
#     /root/gzippy-bench/lib_state.sh            (atomic baseline capture)
#     /root/gzippy-bench/gzippy_bench_restore.sh (idempotent restore)
#     /root/gzippy-bench/lib_gate.sh             (aperf/mperf freq-stability gate)
# and the systemd-run watchdog pattern from host_lock_and_bench.sh.
#
# THE ROOT FIX (why this exists): the old /root/clock_freeze.sh froze ONLY LXC
# 105 (plex) + 111 (frigate). Plex was caught, but transmission(170), sabnzbd(166),
# llama(211), immich-ml(212) and the kvm VM ran FREE — loadavg bounced 1.6↔11 and
# every absolute number was a load artifact. bench-lock uses an ALLOWLIST: it
# freezes EVERY running LXC except the few needed for access (the bench guest +
# DNS + VPN/route + proxy), so ALL noisy neighbors are paused for the window.
#
#   acquire [TTL]   capture baseline -> arm watchdog -> freeze noisy LXCs ->
#                   no_turbo/uncore/governor lock -> SETTLE -> VERIFY QUIET
#                   (loadavg readback). Prints `BENCH_LOCK=quiet` only if the box
#                   is genuinely quiet; `BENCH_LOCK=loaded loadavg=..` otherwise.
#   release         restore baseline (thaw ALL, no_turbo=0, freqs back) + disarm.
#   status          print freeze set + loadavg + watchdog + freq state.
#   verify          re-read loadavg only (cheap quiet re-check between trials).
#
# Restore is GUARANTEED three ways: (1) release, (2) the systemd watchdog timer
# fires after TTL even if the caller dies, (3) every mutation is volatile sysfs/
# MSR/cgroup so a reboot is a full restore. Plex is NEVER left paused.
set -u

BENCH_DIR="${BENCH_DIR:-/root/gzippy-bench}"
RUNDIR="${RUNDIR:-/run/gzippy-bench}"
SF="$RUNDIR/state"
WD_UNIT="gzippy-bench-restore"
WATCHDOG_TTL_DEFAULT=2700          # 45 min — generous upper bound for a measure turn

# Access-keep allowlist: the bench guest + infra we must not freeze to keep the
# double-hop + name resolution + routing alive. EVERYTHING else running gets
# frozen (the allowlist-not-blocklist design — a new noisy LXC is frozen by
# default, never silently free).
KEEP_ALLOWLIST="${KEEP_ALLOWLIST:-199 152 153 115 116 109}"  # bench-target + DNS + VPN/route + proxy
SELF_ACCESS_REQUIRED="${SELF_ACCESS_REQUIRED:-199 152 153 115 116}"

# CONNECTIVITY-CRITICAL GUARD (defense-in-depth, 2026-06-22).
# The allowlist above is BY VMID and therefore FRAGILE: a renamed CT, a new
# WireGuard/VPN/DNS CT with a different id, or a caller that overrides
# KEEP_ALLOWLIST could silently push a connectivity-critical guest into the
# freeze set — pausing WireGuard cuts the VPN to the bench boxes (and possibly
# the operator's link to the host itself). So, INDEPENDENT of the allowlist, we
# NEVER freeze any running CT whose hostname/tags/description identify it as
# WireGuard/VPN/DNS/proxy/gateway. This guard is unconditional and cannot be
# disabled by an env override of KEEP_ALLOWLIST. If a CT's identity cannot be
# read (pct config fails), we treat it as critical and skip it — fail SAFE
# toward connectivity (a non-quiet bench is recoverable; a severed VPN is not).
CRITICAL_REGEX="${CRITICAL_REGEX:-wireguard|wg|vpn|dns|pihole|gateway|router|proxy}"

# is_connectivity_critical <id> — 0 if the CT must never be frozen.
is_connectivity_critical() {
  local id="$1" cfg name
  cfg="$(pct config "$id" 2>/dev/null)"
  # Identity unreadable -> fail safe (treat as critical, never freeze).
  [ -n "$cfg" ] || return 0
  name="$(printf '%s\n' "$cfg" | awk -F': ' '/^hostname:/{print $2}')"
  printf '%s\n' "$name" | grep -qiE "$CRITICAL_REGEX" && return 0
  # also match tags:/description: lines (e.g. tags: dns / tags: vpn)
  printf '%s\n' "$cfg" | grep -iE '^(tags|description):' | grep -qiE "$CRITICAL_REGEX" && return 0
  return 1
}

# Quiet threshold. We gate on INSTANTANEOUS runnable-process count, NOT the 1-min
# loadavg: loadavg is a ~60s exponential average that still carries the pre-freeze
# neighbor load for a full minute AFTER the neighbors are paused (measured: a box
# with procs_running=1 and <2% CPU still shows loadavg ~2.3 right after freezing a
# busy transmission). So loadavg is a FALSE not-quiet signal. procs_running
# (/proc/stat) is the live count of runnable tasks — on a genuinely quiet frozen
# box it sits at 1-2 (just the sampler itself). We sample it over a short window
# and require the average (minus our own 1 sampler) to be at/under the threshold.
QUIET_MAX_RUNNABLE="${QUIET_MAX_RUNNABLE:-2.0}"   # avg runnable tasks incl. our sampler
QUIET_SAMPLES="${QUIET_SAMPLES:-4}"               # 1Hz samples to average
QUIET_MAX_LOADAVG="${QUIET_MAX_LOADAVG:-2.0}"     # reported for context only (lagging)
SETTLE_SECS="${SETTLE_SECS:-2}"                   # brief settle before sampling

[ -d "$BENCH_DIR" ] || { echo "BENCH_LOCK_FAIL=no-bench-dir:$BENCH_DIR (proven plumbing missing)"; exit 5; }
# shellcheck source=/dev/null
. "$BENCH_DIR/lib_state.sh" 2>/dev/null || { echo "BENCH_LOCK_FAIL=no-lib_state"; exit 5; }

mkdir -p "$RUNDIR"
LOG="$RUNDIR/bench-lock.log"
log() { echo "[bench-lock $(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

in_list() { local x="$1"; shift; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }

loadavg1() { cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA; }

# Compute freeze set = running LXCs - KEEP_ALLOWLIST - connectivity-critical CTs.
# The connectivity-critical guard is UNCONDITIONAL: even if a WireGuard/VPN/DNS
# CT is somehow absent from KEEP_ALLOWLIST (renamed id, env override), it is
# still excluded from FREEZE and recorded in PROTECTED for the status readout.
compute_freeze() {
  local running id
  mapfile -t running < <(pct list 2>/dev/null | awk 'NR>1 && $2=="running"{print $1}')
  [ "${#running[@]}" -gt 0 ] || { echo "BENCH_LOCK_FAIL=no-running-lxcs (pct list failed?)" >&2; return 1; }
  RUNNING_LXC=("${running[@]}")
  FREEZE=()
  PROTECTED=()
  for id in "${running[@]}"; do
    if in_list "$id" $KEEP_ALLOWLIST; then continue; fi
    if is_connectivity_critical "$id"; then PROTECTED+=("$id"); continue; fi
    FREEZE+=("$id")
  done
}

# procs_running — instantaneous count of runnable tasks (field 2 of the
# procs_running line in /proc/stat). On a quiet box this is ~1 (the reader).
procs_running() { awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA; }

# Verify quiet using the INSTANTANEOUS runnable count averaged over a short window
# (NOT the lagging 1-min loadavg). Returns 0 quiet / 1 loaded. Loadavg is printed
# for context only.
verify_quiet() {
  local n sum=0 cnt=0 r avg l
  for ((n=0; n<QUIET_SAMPLES; n++)); do
    r="$(procs_running)"
    [ "$r" = NA ] && { echo "BENCH_LOCK=unknown procs_running=NA"; return 0; }
    sum=$((sum + r)); cnt=$((cnt + 1))
    [ "$n" -lt $((QUIET_SAMPLES - 1)) ] && sleep 1
  done
  avg="$(awk -v s="$sum" -v c="$cnt" 'BEGIN{printf "%.2f", (c>0)?s/c:0}')"
  l="$(loadavg1)"
  local hot
  hot="$(awk -v a="$avg" -v m="$QUIET_MAX_RUNNABLE" 'BEGIN{print (a+0>m+0)?1:0}')"
  if [ "$hot" = 1 ]; then
    echo "BENCH_LOCK=loaded runnable_avg=$avg (>$QUIET_MAX_RUNNABLE) loadavg1=$l — a neighbor escaped the freeze; do NOT bank an absolute number"
    return 1
  fi
  echo "BENCH_LOCK=quiet runnable_avg=$avg (<=$QUIET_MAX_RUNNABLE) loadavg1=$l(lagging-EMA,context-only)"
  return 0
}

case "${1:-status}" in
  acquire)
    TTL="${2:-$WATCHDOG_TTL_DEFAULT}"

    # 1. self-access guard: the keep guests we depend on MUST be running.
    compute_freeze || exit 5
    for need in $SELF_ACCESS_REQUIRED; do
      in_list "$need" "${RUNNING_LXC[@]}" || { log "ABORT self-access guard: required guest $need not running"; echo "BENCH_LOCK_FAIL=self-access:$need-not-running"; exit 3; }
    done
    log "running LXCs: ${RUNNING_LXC[*]}"
    log "freeze set (running - keep - critical): ${FREEZE[*]:-<none>}"
    log "connectivity-protected (never frozen): ${PROTECTED[*]:-<none>}"
    log "VMs left running: $(qm list 2>/dev/null | awk 'NR>1 && $3=="running"{printf "%s ",$1}')"

    # 2. write baseline state BEFORE any mutation (atomic), record freeze list.
    write_state "$SF" || { echo "BENCH_LOCK_FAIL=state-write"; exit 5; }
    printf 'FROZEN_GUESTS=%s\n' "${FREEZE[*]:-}" >>"$SF"
    log "baseline captured: no_turbo=$(state_get "$SF" NO_TURBO) uncore=$(state_get "$SF" UNCORE_0x620)"

    # 3. ARM watchdog BEFORE first mutation (primary restore guarantee).
    systemctl stop  "${WD_UNIT}.timer"   2>/dev/null || true
    systemctl reset-failed "${WD_UNIT}.service" 2>/dev/null || true
    if systemd-run --on-active="$TTL" --unit="$WD_UNIT" \
         bash "$BENCH_DIR/gzippy_bench_restore.sh" "$SF" >/dev/null 2>&1; then
      log "watchdog ARMED: unit=$WD_UNIT TTL=${TTL}s"
    else
      echo "BENCH_LOCK_FAIL=watchdog-arm (refusing to mutate without a restore guarantee)"; exit 5
    fi

    # 4. APPLY LOCK (all volatile).
    # 4a. freeze noisy LXCs (THE fix — pauses plex+transmission+sabnzbd+llama+immich…)
    for id in "${FREEZE[@]:-}"; do
      [ -n "$id" ] || continue
      f="/sys/fs/cgroup/lxc/$id/cgroup.freeze"
      [ -w "$f" ] && echo 1 >"$f" 2>/dev/null && log "froze guest $id"
    done
    # 4b. no_turbo=1 (neutralize busy-vs-sleep turbo deflation)
    echo 1 >/sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null && log "no_turbo <- 1"
    # 4c. uncore lock (fixed ratio, package-scoped)
    wrmsr -a 0x620 0x1e1e 2>/dev/null && log "uncore 0x620 <- 0x1e1e"
    # 4d. governor=performance + min==max pin (turbo off => max is base)
    for d in /sys/devices/system/cpu/cpu[0-9]*; do
      [ -d "$d/cpufreq" ] || continue
      [ -w "$d/cpufreq/scaling_governor" ] && echo performance >"$d/cpufreq/scaling_governor" 2>/dev/null
      mx="$(cat "$d/cpufreq/scaling_max_freq" 2>/dev/null)"
      [ -n "$mx" ] && [ -w "$d/cpufreq/scaling_min_freq" ] && echo "$mx" >"$d/cpufreq/scaling_min_freq" 2>/dev/null
    done
    log "governor=performance, min=max pinned"

    # 5. SETTLE then VERIFY QUIET. The freeze makes quiet actually achievable.
    #    The runnable count can momentarily blip from periodic host daemons
    #    (pvestatd polls every few seconds), so we re-check ONCE after a short
    #    wait before declaring loaded — a single transient blip must not flag a
    #    box that is otherwise quiet.
    sleep "$SETTLE_SECS"
    echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) frozen=[${FREEZE[*]:-}] ttl=${TTL}s wd=$(systemctl is-active ${WD_UNIT}.timer 2>/dev/null)"
    if verify_quiet; then
      qrc=0
    else
      sleep 3
      verify_quiet; qrc=$?    # re-check; the second line is authoritative
    fi
    # acquire SUCCEEDS even if not-yet-quiet (caller decides whether to wait/retry
    # via `verify`), but the line above tells the truth so a loaded number is never
    # silently banked.
    exit 0
    ;;

  release)
    bash "$BENCH_DIR/gzippy_bench_restore.sh" "$SF" 2>&1 | tail -3 || log "restore returned nonzero"
    systemctl stop  "${WD_UNIT}.timer"   2>/dev/null || true
    systemctl reset-failed "${WD_UNIT}.service" 2>/dev/null || true
    # final readback so the caller sees the box is actually restored.
    echo "BENCH_LOCK=released no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) frozen_now=[$(for f in /sys/fs/cgroup/lxc/*/cgroup.freeze; do [ "$(cat "$f" 2>/dev/null)" = 1 ] && echo -n "${f%/cgroup.freeze}" | sed 's#.*/##' | tr '\n' ' '; done)] wd=$(systemctl is-active ${WD_UNIT}.timer 2>/dev/null || echo inactive)"
    ;;

  verify)
    verify_quiet
    ;;

  status)
    compute_freeze 2>/dev/null || true
    echo "loadavg=$(cat /proc/loadavg 2>/dev/null)"
    echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) cpu0_cur=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"
    echo "frozen_now=[$(for f in /sys/fs/cgroup/lxc/*/cgroup.freeze; do [ "$(cat "$f" 2>/dev/null)" = 1 ] && echo -n "$(basename $(dirname $f)) "; done)]"
    echo "watchdog=$(systemctl is-active ${WD_UNIT}.timer 2>/dev/null || echo inactive)"
    echo "keep_allowlist=[$KEEP_ALLOWLIST] would_freeze=[${FREEZE[*]:-}]"
    echo "connectivity_protected=[${PROTECTED[*]:-}] (never frozen; matched /$CRITICAL_REGEX/ on hostname/tags/description)"
    ;;

  *)
    echo "usage: bench-lock.sh {acquire [TTL] | release | status | verify}" >&2
    exit 2
    ;;
esac
