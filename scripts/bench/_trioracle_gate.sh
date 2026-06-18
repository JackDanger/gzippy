#!/usr/bin/env bash
# Byte-exact tri-oracle gate for the asm_kernel dist-preload change.
# Cross-checks the reference plaintext across gzip+igzip+libdeflate+pigz, then
# verifies BOTH gzippy flavors (native, isal) at T1/T4/T8 produce sha-identical
# output on 5 corpora. Any mismatch => exit 1 (REVERT trigger).
set -u
NATIVE=/root/gz-new-native
ISAL=/root/gz-new-isal
declare -A CORP=(
  [silesia]=/root/silesia.gz
  [nasa]=/root/nasa.gz
  [monorepo]=/root/monorepo.gz
  [squishy]=/root/squishy.gz
  [large]=/root/bignasa.gz
)
fail=0
printf "%-10s %-12s %-12s %-12s %-12s\n" CORPUS ORACLES native-T148 isal-T148 VERDICT
for name in silesia nasa monorepo squishy large; do
  f=${CORP[$name]}
  ref=$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)
  ig=$(igzip -dc "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  ld=$(libdeflate-gunzip -c "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  pg=$(pigz -dc "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  oracles_ok="OK"
  [ "$ref" = "$ig" ] && [ "$ref" = "$ld" ] && [ "$ref" = "$pg" ] || oracles_ok="ORACLE-DIVERGE"
  nat_ok="OK"; isa_ok="OK"
  for T in 1 4 8; do
    s=$(GZIPPY_FORCE_PARALLEL_SM=1 "$NATIVE" -d -c -p "$T" "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
    [ "$s" = "$ref" ] || { nat_ok="FAIL@T$T($s)"; fail=1; }
    s=$(GZIPPY_FORCE_PARALLEL_SM=1 "$ISAL" -d -c -p "$T" "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
    [ "$s" = "$ref" ] || { isa_ok="FAIL@T$T($s)"; fail=1; }
  done
  [ "$oracles_ok" != "OK" ] && fail=1
  v=PASS; { [ "$nat_ok" != OK ] || [ "$isa_ok" != OK ] || [ "$oracles_ok" != OK ]; } && v=FAIL
  printf "%-10s %-12s %-12s %-12s %-12s\n" "$name" "$oracles_ok" "$nat_ok" "$isa_ok" "$v"
done
echo "---"
[ $fail -eq 0 ] && echo "TRIORACLE_GATE: PASS (all corpora, both flavors, T1/T4/T8 sha-identical)" || echo "TRIORACLE_GATE: FAIL"
exit $fail
