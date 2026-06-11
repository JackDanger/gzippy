#!/bin/bash
# Rung (c) sha grid + effect verification (asma pattern, c-stage binaries).
# Arms: on (bin-asmc-native), off (bin-asmc-native + kill), base (bin-asmc-base).
set -u
for C in silesia model; do
  for T in 1 8; do
    MASK=0; [ "$T" = 8 ] && MASK=0,2,4,6,8,10,12,14
    for arm in on off base; do
      case $arm in
        on)   BIN=/root/bin-asmc-native; ENVV="" ;;
        off)  BIN=/root/bin-asmc-native; ENVV="GZIPPY_ASM_KERNEL=0" ;;
        base) BIN=/root/bin-asmc-base;   ENVV="" ;;
      esac
      S=$(env $ENVV GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $MASK $BIN -d -c -p$T /root/$C.gz 2>/dev/null | sha256sum | cut -c1-16)
      echo "SHA,$C,$T,$arm,$S"
    done
  done
done
echo "== effect verification =="
echo "-- T1 silesia ON --"
GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0 /root/bin-asmc-native -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T1 silesia OFF (kill-switch) --"
GZIPPY_ASM_KERNEL=0 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0 /root/bin-asmc-native -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T8 silesia ON --"
GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asmc-native -d -c -p8 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
echo "-- T8 model ON --"
GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asmc-native -d -c -p8 /root/model.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
