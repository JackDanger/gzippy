#!/bin/bash
# Rung (c) final sha grid: 5 corpora x T{1,8} x {on, off, base}.
set -u
FAIL=0
for C in silesia model bignasa storedheavy storedmix; do
  for T in 1 8; do
    MASK=0; [ "$T" = 8 ] && MASK=0,2,4,6,8,10,12,14
    REF=""
    for arm in on off base; do
      case $arm in
        on)   BIN=/root/bin-asmc-native; ENVV="" ;;
        off)  BIN=/root/bin-asmc-native; ENVV="GZIPPY_ASM_KERNEL=0" ;;
        base) BIN=/root/bin-asmc-base;   ENVV="" ;;
      esac
      S=$(env $ENVV GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $MASK $BIN -d -c -p$T /root/$C.gz 2>/dev/null | sha256sum | cut -c1-16)
      echo "SHA,$C,$T,$arm,$S"
      if [ -z "$REF" ]; then REF=$S; elif [ "$S" != "$REF" ]; then echo "MISMATCH,$C,$T,$arm"; FAIL=1; fi
    done
  done
done
[ "$FAIL" = 0 ] && echo "GRID-ALL-MATCH" || echo "GRID-FAIL"
