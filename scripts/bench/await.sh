#!/usr/bin/env bash
# await.sh — a SELF-CLEANING synchronous waiter (P4 of OWNER-HELP-SUGGESTIONS).
#
# Replaces the two orphan-generating patterns the owner re-pays every turn:
#   - the keep-alive sentinel  `nohup sleep 86400 >/dev/null 2>&1 &`
#   - the advisor-wait spin     `until [ -s plans/X.md ]; do sleep 3; done`
#     and `until … || ! pgrep -f "claude -p"; do sleep 3; done`
# Both leave detached `sleep`/`claude -p` children when the turn ends.
#
# await.sh traps EXIT/INT/TERM and kills its OWN child processes (pkill -P $$), so
# it can NEVER orphan a sleep. It polls in the FOREGROUND and returns when the
# wait condition is met — no background, no yield, no sentinel.
#
# Wait conditions (pick one):
#   <file>            wait until the file exists and is non-empty (-s)
#   --pid <PID>       wait until the process exits
#   --pgrep <PAT>     wait until NO process matches the pgrep -f pattern
#
# Usage:
#   scripts/bench/await.sh plans/X-advisor-verdict.md            # wait for verdict file
#   scripts/bench/await.sh --pid 12345                           # wait for a pid
#   scripts/bench/await.sh --pgrep 'claude -p --model opus'      # wait for advisor to exit
#   scripts/bench/await.sh plans/X.md --timeout 600 --interval 3
set -euo pipefail

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}"; }

# Self-cleaning: on ANY exit path, reap our own children so no sleep can survive.
# We BACKGROUND each sleep and `wait` on it; a signal interrupts `wait` immediately
# (a foreground `sleep` would defer the trap until it returned), so cleanup fires
# at once and no detached sleep is ever orphaned.
cleanup() { [ -n "${nap:-}" ] && kill "$nap" 2>/dev/null; pkill -P $$ 2>/dev/null || true; }
on_signal() { cleanup; exit 130; }
trap cleanup EXIT
trap on_signal INT TERM

MODE=""; TARGET=""; TIMEOUT=900; INTERVAL=3
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --pid) MODE=pid; TARGET="$2"; shift;;
    --pgrep) MODE=pgrep; TARGET="$2"; shift;;
    --timeout) TIMEOUT="$2"; shift;;
    --interval) INTERVAL="$2"; shift;;
    -*) echo "await.sh: unknown flag '$1'" >&2; usage; exit 2;;
    *) MODE="${MODE:-file}"; TARGET="$1";;
  esac
  shift
done
[ -n "$MODE" ] && [ -n "$TARGET" ] || { echo "await.sh: need a wait target" >&2; usage; exit 2; }

met() {
  case "$MODE" in
    file)  [ -s "$TARGET" ];;
    pid)   ! kill -0 "$TARGET" 2>/dev/null;;
    pgrep) ! pgrep -f "$TARGET" >/dev/null 2>&1;;
  esac
}

waited=0
while ! met; do
  if [ "$waited" -ge "$TIMEOUT" ]; then
    echo "await.sh: TIMEOUT after ${TIMEOUT}s waiting on [$MODE] $TARGET" >&2
    exit 124
  fi
  # Background the sleep and `wait` on it so a TERM/INT interrupts immediately and
  # the trap reaps it — a foreground sleep would defer the signal until it returned.
  sleep "$INTERVAL" & nap=$!
  wait "$nap" 2>/dev/null || true
  nap=""
  waited=$((waited + INTERVAL))
done

echo "await.sh: condition met after ${waited}s — [$MODE] $TARGET"
