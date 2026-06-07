#!/bin/sh
# Singleton-leader lock for gzippy orchestration (macOS has no flock).
# Prevents duplicate-leader collisions via an atomic mkdir lock that
# records the holder pid. Cloned from scripts/cargo-lock.sh's
# mkdir-mutex + stale-pid-reclaim pattern.
#
# Usage:
#   scripts/leader-lock.sh acquire   # exits 0 if acquired, 1 if a LIVE leader holds it
#   scripts/leader-lock.sh release   # releases if held by this caller's parent shell
#   scripts/leader-lock.sh status    # prints holder pid (if any)
#
# The lock is NON-BLOCKING: if a live pid holds it, acquire EXITS immediately
# so the duplicate leader can bail out. Stale locks (owner pid gone) are
# reclaimed automatically.
LOCKDIR="/tmp/gzippy-leader.lock.d"

case "${1:-}" in
  acquire)
    if ! mkdir "$LOCKDIR" 2>/dev/null; then
      # Lock exists — check if the owner is still alive.
      if [ -f "$LOCKDIR/pid" ]; then
        owner=$(cat "$LOCKDIR/pid" 2>/dev/null)
        if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
          echo "leader-lock: HELD by live pid $owner — refusing to start duplicate leader" >&2
          exit 1
        fi
        echo "leader-lock: reclaiming stale lock (owner ${owner:-unknown} gone)" >&2
        rm -rf "$LOCKDIR"
        mkdir "$LOCKDIR" 2>/dev/null || { echo "leader-lock: lost reclaim race" >&2; exit 1; }
      else
        # Lock dir with no pid file — treat as stale.
        rm -rf "$LOCKDIR"
        mkdir "$LOCKDIR" 2>/dev/null || { echo "leader-lock: lost reclaim race" >&2; exit 1; }
      fi
    fi
    # Record the holder. Caller passes its pid as $2, else uses this script's ppid.
    holder="${2:-$PPID}"
    echo "$holder" > "$LOCKDIR/pid"
    echo "leader-lock: ACQUIRED by pid $holder" >&2
    exit 0
    ;;
  release)
    rm -rf "$LOCKDIR"
    echo "leader-lock: released" >&2
    exit 0
    ;;
  status)
    if [ -f "$LOCKDIR/pid" ]; then
      owner=$(cat "$LOCKDIR/pid" 2>/dev/null)
      if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
        echo "leader-lock: HELD by live pid $owner"
      else
        echo "leader-lock: stale (owner ${owner:-unknown} gone)"
      fi
    else
      echo "leader-lock: free"
    fi
    exit 0
    ;;
  *)
    echo "usage: $0 {acquire [pid]|release|status}" >&2
    exit 2
    ;;
esac
