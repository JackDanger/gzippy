#!/usr/bin/env bash
# HOST-side wrapper: freeze camera+plex containers, run an inner bench inside
# lxc/199, always thaw (trap + watchdog backup). Usage: freeze_wrapper.sh <inner-cmd...>
set -u
F111=/sys/fs/cgroup/lxc/111/cgroup.freeze
F105=/sys/fs/cgroup/lxc/105/cgroup.freeze
thaw(){ echo 0 > "$F111" 2>/dev/null; echo 0 > "$F105" 2>/dev/null; }
trap thaw EXIT INT TERM HUP
# watchdog backup: force-thaw after 600s no matter what
setsid bash -c "sleep 600; echo 0 > $F111; echo 0 > $F105" </dev/null >/dev/null 2>&1 &
WD=$!
echo 1 > "$F111"; echo 1 > "$F105"; sleep 2
echo "frozen: 111=$(cat "$F111") 105=$(cat "$F105")"
echo "------------------------------------------------------------"
"$@"
echo "------------------------------------------------------------"
thaw
kill "$WD" 2>/dev/null
echo "thawed: 111=$(cat "$F111") 105=$(cat "$F105")"
