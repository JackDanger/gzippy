#!/usr/bin/env bash
# orphan-check.sh — the ONE canonical orphan sweep (P4 of OWNER-HELP-SUGGESTIONS).
#
# Replaces the re-typed 5-line ritual at the top/bottom of nearly every turn
#   (pgrep -fl "claude -p"; pgrep -fl "sleep [0-9]"; … ; ssh … 'pgrep -fl gzippy …')
# with one command that sweeps BOTH the local box and the guest (via the pinned
# double-hop) and prints LOCAL_CLEAN / GUEST_CLEAN or the offenders.
#
# By default it only REPORTS. Pass --kill to actually terminate local stray
# `claude -p`, detached `sleep [0-9]+`, and (with --kill) leftover guest
# gzippy/rapidgzip measurement processes. It NEVER kills this script's own
# children or the parent shell.
#
# Usage:
#   scripts/bench/orphan-check.sh           # report local + guest
#   scripts/bench/orphan-check.sh --kill    # report AND kill local strays (+guest perf procs)
#   scripts/bench/orphan-check.sh --local   # local only (skip the guest hop)
#   scripts/bench/orphan-check.sh --help
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,22p' "${BASH_SOURCE[0]}"; }

KILL=0; LOCAL_ONLY=0
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0;;
    --kill) KILL=1;;
    --local) LOCAL_ONLY=1;;
    *) echo "orphan-check.sh: unknown arg '$a'" >&2; usage; exit 2;;
  esac
done

SELF=$$

# ---- LOCAL sweep ------------------------------------------------------------
echo "=== LOCAL orphan sweep ==="
# A "stray" is, conservatively, ONLY this campaign's self-inflicted orphan
# signatures — NOT every sleep on the box (a broad `sleep [0-9]` would flag/kill
# legitimate sleeps belonging to other agents or the supervisor).
#   1. `claude -p` advisor processes left running;
#   2. the keep-alive sentinel `sleep 86400` (any long nohup sentinel);
#   3. a `sleep` that has been REPARENTED to init (ppid==1) — a true orphan.
local_offenders=0
flag_one() { # <pid> <descr>
  local pid="$1" descr="$2"
  [ "$pid" = "$SELF" ] && return
  local ppid; ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$ppid" = "$SELF" ] && return     # our own child — not an orphan
  echo "  LOCAL stray: $pid  $descr"
  local_offenders=$((local_offenders + 1))
  if [ "$KILL" = 1 ]; then
    kill "$pid" 2>/dev/null && echo "    -> killed $pid" || echo "    -> could not kill $pid"
  fi
}

# 1+2: explicit campaign signatures (claude -p, the long-sentinel sleep).
for pat in 'claude -p' 'sleep 86400' 'nohup sleep'; do
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    flag_one "${line%% *}" "${line#* }"
  done < <(pgrep -fl "$pat" 2>/dev/null || true)
done

# 3: sleeps reparented to init (ppid==1) — genuinely orphaned, regardless of duration.
while IFS= read -r pid; do
  [ -z "$pid" ] && continue
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ "$ppid" = "1" ]; then
    flag_one "$pid" "$(ps -o command= -p "$pid" 2>/dev/null) [reparented to init]"
  fi
done < <(pgrep -x sleep 2>/dev/null || true)

if [ "$local_offenders" -eq 0 ]; then echo "LOCAL_CLEAN"; else echo "LOCAL_OFFENDERS=$local_offenders"; fi

# ---- GUEST sweep ------------------------------------------------------------
if [ "$LOCAL_ONLY" = 1 ]; then exit 0; fi

if [ ! -f "$HERE/guest.env" ]; then
  echo "## orphan-check: guest.env absent — skipping guest sweep" >&2
  exit 0
fi
# shellcheck source=/dev/null
. "$HERE/guest.env"

echo "=== GUEST orphan sweep (via $SSH_GUEST) ==="
GUEST_KILL=""
[ "$KILL" = 1 ] && GUEST_KILL='for p in $(pgrep -f "gzippy -d|rapidgzip|measure.sh|fulcrum_total_capture" 2>/dev/null); do kill "$p" 2>/dev/null && echo "  -> killed guest $p"; done'
REMOTE='
  self=$$
  found=0
  for pat in "gzippy -d" "rapidgzip" "measure.sh" "fulcrum_total_capture" "sleep 86400"; do
    # -a prints the cmdline; drop our OWN sweep shell + any pgrep/sed/bash wrapper
    # whose cmdline merely CONTAINS the pattern (the self-detection race that
    # flagged the sweep'\''s own ssh subshells as offenders). Keep only lines whose
    # process is the actual binary, not a shell mentioning it.
    out="$(pgrep -af "$pat" 2>/dev/null | awk -v me="$self" '\''$1!=me && $0 !~ /pgrep|orphan-check|GUEST stray|[ =]for pat/'\'' || true)"
    [ -n "$out" ] && { echo "$out" | sed "s/^/  GUEST stray: /"; found=1; }
  done
  '"$GUEST_KILL"'
  [ "$found" = 0 ] && echo GUEST_CLEAN || echo GUEST_OFFENDERS
'
if ! timeout 60 $SSH_GUEST "$REMOTE"; then
  echo "## orphan-check: guest hop FAILED — could not sweep the guest (check connectivity)." >&2
  exit 1
fi
