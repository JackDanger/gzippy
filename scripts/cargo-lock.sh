#!/bin/sh
# Portable global build/test mutex for gzippy (macOS has no flock).
# Serializes ALL cargo/rustc/test invocations system-wide via an atomic
# mkdir lock, so concurrent builds/tests are STRUCTURALLY IMPOSSIBLE.
#
# Usage:
#   scripts/cargo-lock.sh cargo build ...
#   scripts/cargo-lock.sh sh -c 'cargo build && cargo test ...'
#
# Honors a max wait (default 3600s). Cleans up the lock on exit/signal.
LOCKDIR="/tmp/gzippy-cargo.lock.d"
MAX_WAIT="${GZIPPY_LOCK_WAIT:-3600}"
waited=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "cargo-lock: timed out after ${MAX_WAIT}s waiting for $LOCKDIR" >&2
    exit 75
  fi
  # If the lock is stale (owning pid gone), reclaim it.
  if [ -f "$LOCKDIR/pid" ]; then
    owner=$(cat "$LOCKDIR/pid" 2>/dev/null)
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
      echo "cargo-lock: reclaiming stale lock (owner $owner gone)" >&2
      rm -rf "$LOCKDIR"
      continue
    fi
  fi
  sleep 2
  waited=$((waited + 2))
done
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
"$@"
status=$?
exit $status
