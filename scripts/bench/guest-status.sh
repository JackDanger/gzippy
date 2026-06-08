#!/usr/bin/env bash
# guest-status.sh — the ONE guest discovery probe (P1 of OWNER-HELP-SUGGESTIONS).
#
# Replaces the bespoke per-turn one-liner
#   (hostname; nproc; perf_event_paranoid; ls …; which rapidgzip; ls silesia …)
# that appeared in dozens of transcripts, each variant slightly different.
#
# Prints, in ONE double-hop, the facts a measurement turn needs BEFORE trusting a
# number: host identity, core count, frequency-lock state (governor/no_turbo),
# the CANONICAL binary's mtime+sha (stale-binary trap), corpus presence+size, and
# the rapidgzip version. All paths come from guest.env so this can never disagree
# with parity.sh about WHERE.
#
# Usage:
#   scripts/bench/guest-status.sh           # probe the guest
#   scripts/bench/guest-status.sh --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

# shellcheck source=/dev/null
. "$HERE/guest.env"

# The remote probe is a SINGLE self-contained shell snippet (no multi-line python).
# Every command is fail-soft (|| echo NA) so one missing tool never aborts the whole
# report — the point is a complete picture, not the first failure.
REMOTE_PROBE='
  echo "host=$(hostname 2>/dev/null || echo NA)"
  echo "nproc=$(nproc 2>/dev/null || echo NA)"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
  echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
  echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo NA)"
  echo "mem_avail_mb=$(free -m 2>/dev/null | awk "/Mem:/{print \$7}" || echo NA)"
  echo "guest_src='"$GUEST_SRC"'"
  if [ -d '"$GUEST_SRC"'/.git ]; then
    echo "guest_src_head=$(git -C '"$GUEST_SRC"' rev-parse --short HEAD 2>/dev/null || echo NA)"
    echo "guest_src_branch=$(git -C '"$GUEST_SRC"' rev-parse --abbrev-ref HEAD 2>/dev/null || echo NA)"
    echo "guest_src_dirty=$(git -C '"$GUEST_SRC"' status --porcelain 2>/dev/null | wc -l | tr -d " ")"
  else
    echo "guest_src_head=ABSENT (no checkout at '"$GUEST_SRC"')"
  fi
  if [ -x '"$GZIPPY_BIN"' ]; then
    echo "gzippy_bin='"$GZIPPY_BIN"'"
    echo "gzippy_mtime=$(date -r '"$GZIPPY_BIN"' "+%Y-%m-%d %H:%M:%S" 2>/dev/null || stat -c %y '"$GZIPPY_BIN"' 2>/dev/null || echo NA)"
    echo "gzippy_sha=$(sha256sum '"$GZIPPY_BIN"' 2>/dev/null | cut -c1-16 || echo NA)"
  else
    echo "gzippy_bin=ABSENT ('"$GZIPPY_BIN"')"
  fi
  if [ -f '"$CORPUS"' ]; then
    echo "corpus='"$CORPUS"'"
    echo "corpus_bytes=$(wc -c < '"$CORPUS"' 2>/dev/null || echo NA)"
    echo "corpus_gz_sha=$(sha256sum '"$CORPUS"' 2>/dev/null | cut -c1-16 || echo NA)"
  else
    echo "corpus=ABSENT ('"$CORPUS"')"
  fi
  echo "rapidgzip=$(command -v '"$RG"' 2>/dev/null || echo NOT-IN-PATH)"
  echo "rapidgzip_version=$('"$RG"' --version 2>&1 | head -1 || echo NA)"
  echo "rg_trace_bin=$([ -x '"$RG_TRACE"' ] && echo PRESENT || echo absent)"
'

echo "=== guest-status (via: $SSH_GUEST) ==="
# shellcheck disable=SC2086
if ! timeout 60 $SSH_GUEST "$REMOTE_PROBE"; then
  rc=$?
  echo "guest-status: probe FAILED (rc=$rc) — connectivity or auth problem on the $JUMP -> $GUEST hop." >&2
  echo "guest-status: diagnose the hop before any measurement (see plans/GUEST.md)." >&2
  exit "$rc"
fi
echo "=== end guest-status ==="
