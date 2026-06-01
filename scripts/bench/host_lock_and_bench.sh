#!/usr/bin/env bash
# host_lock_and_bench.sh — the single trap-protected lock lifecycle owner.
#
# RUNS ON HOST (neurotic), invoked from the laptop by clean_bench.sh:
#   ssh neurotic 'bash /root/gzippy-bench/host_lock_and_bench.sh ARGS'
#
# Responsibilities, in order:
#   1. resolve RUNNING guests dynamically (pct list).
#   2. compute freeze = running_LXCs - KEEP_ALLOWLIST (allowlist, not blocklist).
#      ASSERT keep guests are running, else ABORT (self-access guard).
#   3. write the state file ATOMICALLY before any mutation.
#   4. ARM the systemd-run watchdog BEFORE the first mutation (primary restore
#      guarantee — fires even if laptop ssh dies / this script is kill -9'd).
#   5. trap restore on EXIT/INT/TERM (fast-path nicety).
#   6. APPLY the lock (all volatile): freeze LXCs, no_turbo=1, uncore lock,
#      C-state floor, governor/min/max.
#   7. GATE: aperf/mperf stability; FAIL => restore + failed-gate record only.
#   8. run guest_bench.sh on 199 (one hop).
#   9. restore + disarm watchdog + verify readbacks.
#
# VMs (qm list, e.g. 102 ha) are LEFT RUNNING — we do not cgroup.freeze VMs.
# This is documented and accepted: the gate verifies the resulting core
# stability regardless, and freezing only the noisy LXCs is the simple, safe
# baseline. A VM-induced gate fail aborts cleanly (no bad numbers emitted).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/lib_state.sh"
. "$HERE/lib_gate.sh"

# ---- config -----------------------------------------------------------------
KEEP_ALLOWLIST="199 152 153 115 116 109"   # bench-target + DNS + VPN/route + lab-proxy
SELF_ACCESS_REQUIRED="199 152 153 115 116"  # must be running or we abort
WATCHDOG_TTL="${WATCHDOG_TTL:-2700}"        # seconds; 2-3x worst-case bench wall
RUNDIR="${RUNDIR:-/run/gzippy-bench}"
SF="$RUNDIR/state"
WD_UNIT="gzippy-bench-restore"

GUEST_IP="${GUEST_IP:-10.30.0.199}"
GUEST_SCRIPT_DIR="${GUEST_SCRIPT_DIR:-/root/gzippy-bench}"

# pass-through args go to guest_bench.sh
GUEST_ARGS="$*"

mkdir -p "$RUNDIR"
RUNLOG="$RUNDIR/host.log"
log() { echo "[host $(date -u +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

abort() { log "ABORT: $*"; exit 3; }

# ---- restore plumbing -------------------------------------------------------
ARMED=0
restore_now() {
  # idempotent; safe to call from trap and from normal teardown.
  bash "$HERE/gzippy_bench_restore.sh" "$SF" || log "restore returned nonzero"
  if [ "$ARMED" = 1 ]; then
    systemctl stop "${WD_UNIT}.timer" 2>/dev/null || true
    systemctl reset-failed "${WD_UNIT}.service" 2>/dev/null || true
    ARMED=0
    log "watchdog disarmed"
  fi
}
trap 'rc=$?; log "trap fired (rc=$rc)"; restore_now; exit $rc' EXIT INT TERM

# ---- 1+2. resolve guests, compute freeze set, self-access guard -------------
mapfile -t RUNNING_LXC < <(pct list 2>/dev/null | awk 'NR>1 && $2=="running"{print $1}')
[ "${#RUNNING_LXC[@]}" -gt 0 ] || abort "no running LXCs resolved (pct list failed?)"
log "running LXCs: ${RUNNING_LXC[*]}"

in_list() { local x="$1"; shift; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }

# self-access guard: required keep guests must be running.
for need in $SELF_ACCESS_REQUIRED; do
  in_list "$need" "${RUNNING_LXC[@]}" || abort "self-access guard: required guest $need not running"
done

FREEZE=()
for id in "${RUNNING_LXC[@]}"; do
  in_list "$id" $KEEP_ALLOWLIST || FREEZE+=("$id")
done
log "freeze set (running - keep): ${FREEZE[*]:-<none>}"
log "VMs left running: $(qm list 2>/dev/null | awk 'NR>1 && $3=="running"{printf "%s ",$1}')"

# ---- 3. write state file BEFORE any mutation --------------------------------
write_state "$SF" || abort "could not write state file $SF"
# append the frozen-guest list (write_state doesn't know it)
printf 'FROZEN_GUESTS=%s\n' "${FREEZE[*]:-}" >>"$SF"
log "state written: $SF"
log "baseline captured: no_turbo=$(state_get "$SF" NO_TURBO) uncore=$(state_get "$SF" UNCORE_0x620) thp=$(state_get "$SF" THP)"

# ---- 4. ARM WATCHDOG before first mutation ----------------------------------
systemctl stop "${WD_UNIT}.timer" 2>/dev/null || true
systemctl reset-failed "${WD_UNIT}.service" 2>/dev/null || true
if systemd-run --on-active="${WATCHDOG_TTL}" --unit="$WD_UNIT" \
     bash "$HERE/gzippy_bench_restore.sh" "$SF" >/dev/null 2>&1; then
  ARMED=1
  log "watchdog ARMED: unit=$WD_UNIT TTL=${WATCHDOG_TTL}s (fires if this script dies)"
else
  abort "could not arm watchdog — refusing to mutate the box without a restore guarantee"
fi

# ---- 6. APPLY LOCK (all volatile) -------------------------------------------
# 6a. freeze noisy LXCs
for id in "${FREEZE[@]:-}"; do
  [ -n "$id" ] || continue
  f="/sys/fs/cgroup/lxc/$id/cgroup.freeze"
  [ -w "$f" ] && echo 1 >"$f" 2>/dev/null && log "froze guest $id"
done

# 6b. no_turbo=1
echo 1 >/sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null && log "no_turbo <- 1"

# 6c. uncore lock 0x1e1e (fixed ratio) — package-scoped, -a all cpus
wrmsr -a 0x620 0x1e1e 2>/dev/null && log "uncore 0x620 <- 0x1e1e"

# 6d. C-state floor: compile + launch the holder, record its pid
HOLDER_BIN="$RUNDIR/cstate_hold"
if cc -O2 -o "$HOLDER_BIN" "$HERE/cstate_hold.c" 2>>"$RUNLOG"; then
  "$HOLDER_BIN" >/dev/null 2>>"$RUNLOG" &
  # $! is the holder pid; restore kills it to release the /dev/cpu_dma_latency fd.
  echo "$!" >"$RUNDIR/cstate_holder.pid"
  log "C-state floor held (pid $!)"
else
  log "WARN: could not compile C-state holder; proceeding without C-state floor"
fi

# 6e. governor + min==max pin (lock every cpufreq cpu to performance, max=max)
for d in /sys/devices/system/cpu/cpu[0-9]*; do
  [ -d "$d/cpufreq" ] || continue
  [ -w "$d/cpufreq/scaling_governor" ] && echo performance >"$d/cpufreq/scaling_governor" 2>/dev/null
  mx="$(cat "$d/cpufreq/scaling_max_freq" 2>/dev/null)"
  [ -n "$mx" ] && [ -w "$d/cpufreq/scaling_min_freq" ] && echo "$mx" >"$d/cpufreq/scaling_min_freq" 2>/dev/null
done
log "governor=performance, min=max pinned (turbo off => max is base)"

# small settle
sleep 1

# ---- 7. GATE ----------------------------------------------------------------
log "running frequency-stability gate..."
if run_gate >>"$RUNLOG" 2>&1; then
  log "GATE PASS"
else
  log "GATE FAIL — emitting failed-gate record, NO timing rows"
  {
    echo "RUN_TRUSTWORTHY=false"
    echo "FAILURE=gate"
    echo "GATE_STATUS=$GATE_STATUS"
    echo "$GATE_REPORT"
  } | tee "$RUNDIR/result.txt"
  exit 4   # trap restores + disarms
fi
# record the gate proof for the results bundle
{ echo "GATE_STATUS=$GATE_STATUS"; printf '%s' "$GATE_REPORT"; } >"$RUNDIR/gate.txt"

# ---- 8. run guest bench (one hop) -------------------------------------------
# TESTSLEEP: self-test hook — keep the box LOCKED for N seconds instead of
# running the guest hop, so the watchdog-kill test can exercise the real
# apply+gate+watchdog path. Never set in production.
if [ -n "${TESTSLEEP:-}" ]; then
  log "TESTSLEEP=$TESTSLEEP: holding lock (self-test mode), no guest bench"
  sleep "$TESTSLEEP"
  log "TESTSLEEP elapsed; trap will restore"
  exit 0
fi

log "launching guest bench on $GUEST_IP ..."
GUEST_CMD="bash $GUEST_SCRIPT_DIR/guest_bench.sh $GUEST_ARGS"
# accept-new: the host may not yet have the guest key cached on a fresh boot.
if ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
     "root@$GUEST_IP" "$GUEST_CMD" 2>&1 | tee "$RUNDIR/result.txt"; then
  log "guest bench completed"
else
  log "guest bench returned nonzero (see result.txt)"
fi

# ---- 9. teardown via trap (restore + disarm + verify) -----------------------
log "host script done; trap will restore + disarm + verify"
exit 0
