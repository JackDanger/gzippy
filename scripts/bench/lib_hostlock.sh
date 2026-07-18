# lib_hostlock.sh — laptop-side bracket around the host (neurotic) freeze.
#
# Sourced by parity.sh and oracle.sh. Brackets a measurement with the standalone
# host-side /root/bench-lock.sh (deployed from scripts/bench/host/bench-lock.sh):
#   acquire -> freezes ALL noisy LXCs (plex/transmission/sabnzbd/llama/immich…),
#              no_turbo=1, governor pin, SETTLE, then verifies QUIET (loadavg).
#   release -> thaws everything + disarms the watchdog (ALWAYS, via trap).
#
# WHY: the spine used to do its own -J double-hop and freeze NOTHING, so a number
# was taken on a Plex-loaded box (loadavg 1.6↔11). bench-lock pauses the
# neighbors so the box is genuinely quiet; this helper makes the spine USE it and
# guarantees the box is always restored even if the measure aborts.
#
# Requires (from guest.env): $SSH_JUMP (ssh to the host), $JUMP.
# Knobs: HOSTLOCK_TTL (watchdog seconds, default 1800), QUIET_MAX_LOADAVG.

HOSTLOCK_TTL="${HOSTLOCK_TTL:-1800}"
HOSTLOCK_BIN="${HOSTLOCK_BIN:-/root/bench-lock.sh}"
_HOSTLOCK_HELD=0

# Push the current repo copy of bench-lock.sh to the host so host and repo never
# drift (the source-of-truth mirror rule). Best-effort: a host without write here
# still uses whatever /root/bench-lock.sh it has.
hostlock_sync() {
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [ -f "$here/host/bench-lock.sh" ] || return 0
  timeout 40 scp -o ConnectTimeout=15 "$here/host/bench-lock.sh" "$JUMP:$HOSTLOCK_BIN" >/dev/null 2>&1 \
    && timeout 15 $SSH_JUMP "chmod +x $HOSTLOCK_BIN" >/dev/null 2>&1 || true
}

# hostlock_acquire — freeze the box, arm the watchdog, verify quiet. Echoes the
# bench-lock line. Sets _HOSTLOCK_HELD=1 and arms the EXIT trap so the box is
# ALWAYS released. Returns 0 if quiet, 1 if the freeze could not make it quiet
# (caller decides; the line tells the truth either way).
hostlock_acquire() {
  hostlock_sync
  echo "=== host freeze: acquire (ttl=${HOSTLOCK_TTL}s) via $JUMP ==="
  local out
  out="$(timeout 90 $SSH_JUMP "bash $HOSTLOCK_BIN acquire $HOSTLOCK_TTL" 2>&1)"
  echo "$out"
  case "$out" in *BENCH_LOCK_FAIL=*) echo "## host freeze FAILED to acquire — see above"; return 2;; esac
  _HOSTLOCK_HELD=1
  trap 'hostlock_release' EXIT INT TERM
  case "$out" in *BENCH_LOCK=quiet*) return 0;; *) return 1;; esac
}

# hostlock_verify — cheap quiet re-check (between trials / before banking).
hostlock_verify() {
  timeout 30 $SSH_JUMP "bash $HOSTLOCK_BIN verify" 2>&1
}

# hostlock_release — idempotent thaw+disarm. Safe from a trap.
hostlock_release() {
  [ "$_HOSTLOCK_HELD" = 1 ] || return 0
  _HOSTLOCK_HELD=0
  trap - EXIT INT TERM
  echo "=== host freeze: release (thaw + disarm watchdog) ==="
  timeout 60 $SSH_JUMP "bash $HOSTLOCK_BIN release" 2>&1 | tail -3 || \
    echo "## WARN: release hop failed — the systemd watchdog will force-thaw within TTL"
}
