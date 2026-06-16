#!/bin/bash
# Freeze preamble for the Neurotic baseline.
echo "=== governor set ==="
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 && echo "governor=performance set" || echo "governor write FAILED (container?)"
echo "=== no_turbo set ==="
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null && echo "no_turbo=1 set" || echo "no_turbo write FAILED"
echo "=== kill stale neighbors (our prior find/aw_bench) ==="
pkill -9 -f aw_bench 2>/dev/null && echo "killed aw_bench" || echo "no aw_bench"
# only kill find procs in D state that are ours
ps -eo pid,stat,comm | awk '$2 ~ /D/ && $3=="find"{print $1}' | while read p; do kill -9 "$p" 2>/dev/null && echo "killed D-state find $p"; done
echo "=== drop caches ==="
sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null && echo "caches dropped" || echo "drop_caches FAILED"
echo "=== READBACK ==="
echo -n "governor[0,2,4]: "; for c in 0 2 4; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null; done | tr '\n' ' '; echo
echo -n "no_turbo: "; cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null; echo
echo -n "nproc: "; nproc
echo -n "procs_running: "; awk '/procs_running/{print $2}' /proc/stat
echo -n "loadavg: "; cat /proc/loadavg
echo "=== top CPU consumers ==="
ps -eo pid,stat,pcpu,comm --sort=-pcpu | head -8
echo "=== online cpus ==="
cat /sys/devices/system/cpu/online 2>/dev/null
