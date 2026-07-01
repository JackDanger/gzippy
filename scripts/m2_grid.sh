#!/bin/bash
# M2 byte-exact grid: {silesia,model,bignasa} x T{1,2,4,8,16} x {native,isal}
# sha256-verified against expected pins. Exit non-zero on ANY mismatch.
set -u
EXP_SILESIA="${EXP_SILESIA:?}"
EXP_MODEL="${EXP_MODEL:?}"
EXP_BIGNASA="${EXP_BIGNASA:?}"
fail=0
for bin in /root/bin-m2-native /root/bin-m2-isal; do
  for arc in silesia model bignasa; do
    case "$arc" in
      silesia) exp="$EXP_SILESIA" ;;
      model)   exp="$EXP_MODEL" ;;
      bignasa) exp="$EXP_BIGNASA" ;;
    esac
    for t in 1 2 4 8 16; do
      got=$(GZIPPY_FORCE_PARALLEL_SM=1 timeout 300 "$bin" -d -c -p "$t" "/root/$arc.gz" | sha256sum | cut -d' ' -f1)
      if [ "$got" = "$exp" ]; then
        echo "OK   $(basename "$bin") $arc T$t $got"
      else
        echo "FAIL $(basename "$bin") $arc T$t got=$got exp=$exp"
        fail=1
      fi
    done
  done
done
exit $fail
