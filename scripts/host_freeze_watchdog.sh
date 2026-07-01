#!/usr/bin/env bash
# HOST-side (neurotic) freeze of noisy-neighbor guests with a SELF-RESTORING
# watchdog that thaws even if the controlling ssh session dies. Idempotent.
#   start <secs>  -> freeze 111+105, arm a detached watchdog that force-thaws
#                    after <secs> no matter what (default 900s)
#   stop          -> thaw 111+105, kill watchdog
#   status        -> print freeze state + watchdog pid
set -u
F111=/sys/fs/cgroup/lxc/111/cgroup.freeze
F105=/sys/fs/cgroup/lxc/105/cgroup.freeze
WDPID=/run/gzippy_freeze_wd.pid

thaw(){ echo 0 > "$F111" 2>/dev/null; echo 0 > "$F105" 2>/dev/null; }

case "${1:-status}" in
  start)
    secs="${2:-900}"
    # kill any prior watchdog
    [ -f "$WDPID" ] && kill "$(cat $WDPID)" 2>/dev/null
    echo 1 > "$F111"; echo 1 > "$F105"
    # detached watchdog: force-thaw after secs even if parent ssh dies
    setsid bash -c "sleep $secs; echo 0 > $F111; echo 0 > $F105; rm -f $WDPID" </dev/null >/dev/null 2>&1 &
    echo $! > "$WDPID"
    echo "frozen: 111=$(cat $F111) 105=$(cat $F105) watchdog=$(cat $WDPID) ttl=${secs}s"
    ;;
  stop)
    [ -f "$WDPID" ] && { kill "$(cat $WDPID)" 2>/dev/null; rm -f "$WDPID"; }
    thaw
    echo "thawed: 111=$(cat $F111) 105=$(cat $F105) watchdog=killed"
    ;;
  status)
    echo "111=$(cat $F111 2>/dev/null) 105=$(cat $F105 2>/dev/null) watchdog_pid=$(cat $WDPID 2>/dev/null || echo none)"
    ;;
esac
