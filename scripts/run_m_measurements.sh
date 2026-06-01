#!/usr/bin/env bash
# Combined guest runner (RUN ON GUEST 199, invoked by host_lock_and_bench.sh
# AFTER the host froze guests + locked frequency + the gate PASSED).
# Runs, in order: provenance witness, M2 (/dev/null 3-knob factorial + stats),
# M1 (tmpfs write-vs-wait). Emits RUN_TRUSTWORTHY from provenance.
set -u
S=/root/gzippy/scripts
N="${N:-9}"
K="${K:-5000}"

echo "######## PROVENANCE ########"
bash "$S/m_provenance.sh"

echo
echo "######## M2 — /dev/null 3-knob factorial (decode x consumer x marker-resolve) ########"
N="$N" K="$K" bash "$S/m2_factorial_devnull.sh" | tee /tmp/m2.out
echo "-------- M2 STATS --------"
python3 "$S/m2_stats.py" /tmp/m2.out

echo
echo "######## M1 — tmpfs write-time vs next-chunk-wait (T8) ########"
N="${M1_N:-11}" bash "$S/m1_write_vs_wait.sh"
