#!/usr/bin/env bash
# D1: byte-exact sha==zcat grid (silesia/monorepo/nasa x T1/T4/T8 x {asmoff,asmon})
# + kill-switch byte-exact (GZIPPY_FLAT_CLEAN=0 routes asmoff to engine B)
# + engine-A non-inert proof (FLAT_CONTIG calls>0 on asmoff, ==0 with kill-switch / on asmon)
set -uo pipefail
O=/dev/shm/ixv
CORPORA="silesia monorepo nasa"
FAIL=0
echo "== reference shas (zcat) =="
declare -A REF
for c in $CORPORA; do
  REF[$c]=$(zcat /root/$c.gz | sha256sum | cut -d' ' -f1)
  echo "  $c -> ${REF[$c]}"
done

echo "== byte-exact grid =="
for bin in gz-asmoff gz-asmon; do
  for c in $CORPORA; do
    for t in 1 4 8; do
      s=$(GZIPPY_FORCE_PARALLEL_SM=1 "$O/$bin" -d -c -p$t /root/$c.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
      if [ "$s" = "${REF[$c]}" ]; then verdict=OK; else verdict="MISMATCH"; FAIL=1; fi
      echo "  $bin $c T$t : $verdict"
    done
  done
done

echo "== kill-switch (GZIPPY_FLAT_CLEAN=0 -> engine B on asmoff) byte-exact =="
for c in $CORPORA; do
  s=$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/$c.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
  if [ "$s" = "${REF[$c]}" ]; then verdict=OK; else verdict="MISMATCH"; FAIL=1; fi
  echo "  gz-asmoff[FLAT_CLEAN=0] $c T1 : $verdict"
done

echo "== Gate-0c non-inert: FLAT_CONTIG calls per config (silesia T1) =="
echo -n "  asmoff default (engine A ON): "; GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig || echo "(none)"
echo -n "  asmoff FLAT_CLEAN=0 (engine B): "; GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig || echo "(none — engine B, expected)"
echo -n "  asmon (run_contig, no engine A compiled): "; GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmon" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig || echo "(none — cfg'd out, expected)"

echo "D1_FAIL=$FAIL"
echo "IXV_D1_DONE"
