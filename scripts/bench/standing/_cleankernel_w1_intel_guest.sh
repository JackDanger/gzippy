#!/usr/bin/env bash
# W1-on-Intel: where does rg's ISA-L beat gz on silesia clean decode — the
# COPY or the Huffman inner-kernel? gz clean-path causal slow-inject (decode vs
# store) on x86 + NODECODE removal + rg --verbose phase split. Wall time
# best-of-N (perf blocked in LXC). /dev/null sink, sha-gated.
set -uo pipefail
GZ="${GZ:-/dev/shm/cleankernel-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
RGW="${RGW:-/usr/local/bin/rapidgzip}"   # wheel build has --verbose
CORP="${CORP:-/root/silesia.gz}"
N="${N:-11}"
REF="$(zcat "$CORP" | sha256sum | cut -d' ' -f1)"

timeit(){ local s e; s=$(date +%s.%N); "$@" >/dev/null 2>/dev/null; e=$(date +%s.%N); awk "BEGIN{print ($e-$s)*1000}"; }
bestof(){ local n=$1; shift; local m="" v; for i in $(seq 1 "$n"); do v=$(timeit "$@"); m=$(awk "BEGIN{print (\"$m\"==\"\"||$v<$m)?$v:$m}"); done; echo "$m"; }

echo "== uptime =="; uptime
# record pass for NODECODE
CAP=/dev/shm/ck_sil_p1.cap
GZIPPY_ORACLE_RECORD=$CAP "$GZ" -dc -p1 "$CORP" >/dev/null 2>/dev/null
SND="$(GZIPPY_ORACLE_NODECODE=$CAP "$GZ" -dc -p1 "$CORP" 2>/dev/null | sha256sum | cut -d' ' -f1)"
echo "nodecode sha==ref: $([ "$SND" = "$REF" ] && echo YES || echo NO)"
# byte-transparency of slow knobs
for k in GZIPPY_SLOW_DECODE GZIPPY_SLOW_STORE; do
  S="$(env $k=100 "$GZ" -dc -p1 "$CORP" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  echo "$k=100 sha==ref: $([ "$S" = "$REF" ] && echo YES || echo NO)"
done

echo "== gz -p1 silesia clean-path wall (best-of-$N ms) =="
B=$(bestof "$N" "$GZ" -dc -p1 "$CORP"); echo "  base        : $B"
ND=$(GZIPPY_ORACLE_NODECODE=$CAP bestof "$N" "$GZ" -dc -p1 "$CORP"); echo "  nodecode    : $ND  (removes Huffman decode+bitread)"
SD=$(GZIPPY_SLOW_DECODE=100 bestof "$N" "$GZ" -dc -p1 "$CORP"); echo "  slowdec100  : $SD"
SS=$(GZIPPY_SLOW_STORE=100 bestof "$N" "$GZ" -dc -p1 "$CORP"); echo "  slowsto100  : $SS"
python3 - "$B" "$ND" "$SD" "$SS" <<'PY'
import sys
b,nd,sd,ss=map(float,sys.argv[1:5])
print(f"  [x86] NODECODE decode-wall share = {(b-nd)/b*100:.1f}%  (copy/store/rest = {nd/b*100:.1f}%)")
print(f"  [x86] causal slow-inject: SLOW_DECODE +{sd-b:.0f}ms  SLOW_STORE +{ss-b:.0f}ms")
print(f"  [x86] store/decode WALL-criticality ratio = {(ss-b)/(sd-b):.2f}  (>1 => COPY/STORE more wall-critical)")
PY

echo "== rg --verbose phase split (silesia P4) =="
$RGW -dc -P4 --verbose "$CORP" >/dev/null 2>/tmp/rgv.txt || $RG -dc -P4 --verbose "$CORP" >/dev/null 2>/tmp/rgv.txt || true
grep -iE 'Decompress|decode|copy|window|marker|checksum|alloc|block.find|Replac' /tmp/rgv.txt | head -40
