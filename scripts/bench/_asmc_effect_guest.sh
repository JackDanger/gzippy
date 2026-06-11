#!/bin/bash
# Rung (c) effect-counter verification (needs GZIPPY_VERBOSE for the dump).
set -u
echo "-- T1 silesia ON --"
GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0 /root/bin-asmc-native -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T1 silesia OFF (kill-switch) --"
GZIPPY_ASM_KERNEL=0 GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0 /root/bin-asmc-native -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T8 silesia ON --"
GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asmc-native -d -c -p8 /root/silesia.gz 2>&1 >/dev/null | grep "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T8 model ON --"
GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asmc-native -d -c -p8 /root/model.gz 2>&1 >/dev/null | grep "asm-kernel" || echo "NO-ASM-LINE"
