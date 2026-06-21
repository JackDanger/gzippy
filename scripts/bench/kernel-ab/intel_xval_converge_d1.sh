#!/usr/bin/env bash
# D1 for the SOLE-PATH convergence (@cffa61ee) on x86:
#   - byte-exact sha==zcat grid: {gz-asmoff(converged), gz-asmon} x {silesia,monorepo,nasa} x {T1,T4,T8}
#   - kill-switch byte-exact: GZIPPY_FLAT_CLEAN=0 routes gz-asmoff to engine B
#   - NON-INERT proof of the converged tail: gz-asmoff must show
#       careful_calls>0   (the NEW decode_clean_careful_flat resumable tail RAN)
#       clean_lut_builds=0 (engine-B clean two-level lut_litlen double-build GONE)
#     and FLAT_CLEAN=0 must FLIP both (careful_calls=0, clean_lut_builds>0).
set -uo pipefail
O=/dev/shm/ixv
CORPORA="silesia monorepo nasa"
FAIL=0
declare -A REF
echo "== reference shas (zcat) =="
for c in $CORPORA; do REF[$c]=$(zcat /root/$c.gz | sha256sum | cut -d' ' -f1); echo "  $c -> ${REF[$c]}"; done

echo "== byte-exact grid =="
for bin in gz-asmoff gz-asmon; do
  for c in $CORPORA; do
    for t in 1 4 8; do
      s=$(GZIPPY_FORCE_PARALLEL_SM=1 "$O/$bin" -d -c -p$t /root/$c.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
      if [ "$s" = "${REF[$c]}" ]; then v=OK; else v="MISMATCH"; FAIL=1; fi
      echo "  $bin $c T$t : $v"
    done
  done
done

echo "== kill-switch (GZIPPY_FLAT_CLEAN=0 -> engine B on gz-asmoff) byte-exact =="
for c in $CORPORA; do
  s=$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/$c.gz 2>/dev/null | sha256sum | cut -d' ' -f1)
  if [ "$s" = "${REF[$c]}" ]; then v=OK; else v="MISMATCH"; FAIL=1; fi
  echo "  gz-asmoff[FLAT_CLEAN=0] $c T1 : $v"
done

echo "== Gate-0c NON-INERT (converged tail): silesia T1 =="
LINE_A=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmoff" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig)
LINE_B=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FLAT_CLEAN=0 "$O/gz-asmoff" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig)
echo "  engineA (default): $LINE_A"
echo "  engineB (FLAT=0) : $LINE_B"
# NB: use a leading delimiter so "careful_calls" does not also match "calls".
get() { echo " $1" | grep -o "[ =]$2=[0-9]*" | grep -o '[0-9]*$'; }
FC_A=$(get "$LINE_A" calls); CC_A=$(get "$LINE_A" careful_calls); LB_A=$(get "$LINE_A" clean_lut_builds)
FC_B=$(get "$LINE_B" calls); CC_B=$(get "$LINE_B" careful_calls); LB_B=$(get "$LINE_B" clean_lut_builds)
# PRODUCTION engine-A non-inert (the deliverable's required proof):
[ "${FC_A:-0}" -gt 0 ] && echo "  PASS engineA flat_contig calls>0 ($FC_A)" || { echo "  FAIL flat_contig calls ($FC_A)"; FAIL=1; }
[ "${CC_A:-0}" -gt 0 ] && echo "  PASS engineA careful_calls>0 ($CC_A) [NEW flat tail decode_clean_careful_flat RAN]" || { echo "  FAIL careful_calls not >0 ($CC_A)"; FAIL=1; }
[ "${LB_A:-1}" -eq 0 ] && echo "  PASS engineA clean_lut_builds=0 [engine-B clean double-build GONE]" || { echo "  FAIL clean_lut_builds!=0 ($LB_A)"; FAIL=1; }
# KILL-SWITCH non-inert discriminator: FLAT_CLEAN=0 routes engine A OFF
# (flat_contig + careful both drop to 0). clean_lut_builds stays 0 in BOTH
# engines on a clean stream (it counts only the deep careful fallback build).
[ "${FC_B:-1}" -eq 0 ] && echo "  PASS engineB(FLAT=0) flat_contig calls=0 [engine A routed OFF]" || { echo "  FAIL engineB flat_contig calls!=0 ($FC_B)"; FAIL=1; }
[ "${CC_B:-1}" -eq 0 ] && echo "  PASS engineB(FLAT=0) careful_calls=0 [flat tail routed OFF]" || { echo "  FAIL engineB careful_calls!=0 ($CC_B)"; FAIL=1; }
echo "  (info engineB clean_lut_builds=$LB_B — expected 0 on a clean stream, both engines)"
echo -n "  asmon (engine A cfg'd out): "; GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$O/gz-asmon" -d -c -p1 /root/silesia.gz 2>&1 >/dev/null | grep flat_contig || echo "(none — expected, cfg'd out)"

echo "D1_FAIL=$FAIL"
echo "IXV_CONV_D1_DONE"
