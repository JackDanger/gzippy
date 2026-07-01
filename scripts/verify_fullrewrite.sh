#!/usr/bin/env bash
# Byte-exact tri-oracle + asm-engagement self-validation for the
# igzip-full-rewrite MOVDQU back-ref copy. Run on neurotic.
set -u
GZ=/root/gz-fullrewrite/target/release/gzippy
CORPORA="silesia nasa monorepo squishy"
THREADS="1 4 8"
fail=0

echo "=== GATE 2: instrument self-validation (KERN_ENTRIES>0, copy exercised) ==="
# T4 silesia: window-absent chunks fire run_contig; ASM_STATS dumps counters.
GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  $GZ -d -c -p4 /root/silesia.gz >/dev/null 2>/tmp/asmstats.txt
grep -iE "KERN_ENTRIES|KERN_ASM_BYTES|run_contig|kernel" /tmp/asmstats.txt | head -10
echo "(if no KERN line above, engagement not proven on this run)"

echo
echo "=== GATE 1: tri-oracle byte-exact (gzippy vs gzip/igzip/libdeflate) ==="
for c in $CORPORA; do
  src=/root/$c.gz
  [ -f "$src" ] || { echo "MISSING $src"; continue; }
  # reference shas (three independent oracles)
  ref_gzip=$(gzip -dc "$src" 2>/dev/null | sha256sum | cut -d' ' -f1)
  ref_igzip=$(igzip -dc "$src" 2>/dev/null | sha256sum | cut -d' ' -f1)
  ref_ld=$(libdeflate-gunzip -c "$src" 2>/dev/null | sha256sum | cut -d' ' -f1)
  oracle_ok="OK"
  [ "$ref_gzip" = "$ref_igzip" ] && [ "$ref_gzip" = "$ref_ld" ] || oracle_ok="ORACLE-DISAGREE($ref_gzip/$ref_igzip/$ref_ld)"
  for t in $THREADS; do
    got=$(GZIPPY_FORCE_PARALLEL_SM=1 $GZ -d -c -p$t "$src" 2>/dev/null | sha256sum | cut -d' ' -f1)
    if [ "$got" = "$ref_gzip" ]; then
      echo "PASS  $c T$t  sha=${got:0:12}  [$oracle_ok]"
    else
      echo "FAIL  $c T$t  got=${got:0:12} want=${ref_gzip:0:12}  [$oracle_ok]"
      fail=1
    fi
  done
done

echo
if [ "$fail" = 0 ]; then echo "ALL BYTE-EXACT PASS"; else echo "BYTE-EXACT FAILURES PRESENT"; fi
exit $fail
