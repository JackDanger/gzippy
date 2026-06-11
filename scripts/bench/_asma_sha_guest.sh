#!/bin/bash
set -u
for C in silesia model; do
  for T in 1 8; do
    MASK=0; [ "$T" = 8 ] && MASK=0,2,4,6,8,10,12,14
    for arm in on off base; do
      case $arm in
        on)   BIN=/root/bin-asma-native; ENVV="" ;;
        off)  BIN=/root/bin-asma-native; ENVV="GZIPPY_ASM_KERNEL=0" ;;
        base) BIN=/root/bin-asma-base;   ENVV="" ;;
      esac
      S=$(env $ENVV GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $MASK $BIN -d -c -p$T /root/$C.gz 2>/dev/null | sha256sum | cut -c1-16)
      echo "SHA,$C,$T,$arm,$S"
    done
  done
done
echo "== effect verification (T8 silesia) =="
GZIPPY_ASM_STATS=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asma-native -d -c -p8 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
GZIPPY_ASM_KERNEL=0 GZIPPY_ASM_STATS=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6,8,10,12,14 /root/bin-asma-native -d -c -p8 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
GZIPPY_ASM_STATS=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0 /root/bin-asma-native -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep -E "asm-kernel" || echo "NO-ASM-LINE"
