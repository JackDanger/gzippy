#!/usr/bin/env bash
# _cleankernel_silt4_guest.sh — W3 reproducibility of silesia-T4 gz/rg on Intel.
# Interleaved best-of-N, /dev/null both arms, sha==zcat gate, rg-vs-rg A/A self-test.
# Trust the gz/rg ratio ONLY if the A/A self-test spread is <= TOL (box quiet enough).
set -uo pipefail
GZ="${GZ:-/dev/shm/cleankernel-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORP="${CORP:-/root/silesia.gz}"
T="${T:-4}"
N="${N:-15}"
TOL="${TOL:-5.0}"   # percent: A/A ratio must be within TOL of 1.0 AND spread<=TOL

command -v "$GZ" >/dev/null 2>&1 || [ -x "$GZ" ] || { echo "NO GZ $GZ"; exit 2; }
[ -x "$RG" ] || { echo "NO RG $RG"; exit 2; }

echo "== uptime =="; uptime
echo "== build-flavor/path =="
GZIPPY_DEBUG=1 "$GZ" -dc -p"$T" "$CORP" >/dev/null 2>/tmp/dbg.txt || true
grep -i 'flavor\|path=' /tmp/dbg.txt || true

REF="$(zcat "$CORP" | sha256sum | cut -d' ' -f1)"
echo "ref sha: ${REF:0:16}"
SG="$("$GZ" -dc -p"$T" "$CORP" 2>/dev/null | sha256sum | cut -d' ' -f1)"
SR="$("$RG" -dc -P"$T" "$CORP" 2>/dev/null | sha256sum | cut -d' ' -f1)"
echo "gz sha==ref: $([ "$SG" = "$REF" ] && echo YES || echo NO)   rg sha==ref: $([ "$SR" = "$REF" ] && echo YES || echo NO)"
[ "$SG" = "$REF" ] || { echo "GZ BYTE MISMATCH"; exit 3; }
[ "$SR" = "$REF" ] || { echo "RG BYTE MISMATCH"; exit 3; }

# nanosecond wall timer, stdout->/dev/null both arms
t(){ local s e; s=$(date +%s.%N); "$@" -dc >/dev/null 2>/dev/null; : ; }
timeit(){ # $@ = full command (already includes -dc target); returns ms
  local s e
  s=$(python3 -c 'import time;print(time.perf_counter())')
  "$@" >/dev/null 2>/dev/null
  e=$(python3 -c 'import time;print(time.perf_counter())')
  python3 -c "print(($e-$s)*1000)"
}

declare -a GZT RGT RGB   # gz, rg, rg-B (A/A)
# warmup
"$GZ" -dc -p"$T" "$CORP" >/dev/null 2>/dev/null
"$RG" -dc -P"$T" "$CORP" >/dev/null 2>/dev/null
for i in $(seq 1 "$N"); do
  GZT+=("$(timeit "$GZ" -dc -p"$T" "$CORP")")
  RGT+=("$(timeit "$RG" -dc -P"$T" "$CORP")")
  RGB+=("$(timeit "$RG" -dc -P"$T" "$CORP")")
done

python3 - "$TOL" <<'PY' "${GZT[@]}" "|" "${RGT[@]}" "|" "${RGB[@]}"
import sys, statistics
tol=float(sys.argv[1])
rest=sys.argv[2:]
g=rest[:rest.index("|")]; rest=rest[rest.index("|")+1:]
r=rest[:rest.index("|")]; b=rest[rest.index("|")+1:]
g=[float(x) for x in g]; r=[float(x) for x in r]; b=[float(x) for x in b]
def stat(v):
    return min(v), statistics.median(v), (max(v)-min(v))/min(v)*100
gm,gmd,gsp=stat(g); rm,rmd,rsp=stat(r); bm,bmd,bsp=stat(b)
print(f"N={len(g)}")
print(f"gz  best={gm:.1f}ms med={gmd:.1f} spr={gsp:.1f}%")
print(f"rgA best={rm:.1f}ms med={rmd:.1f} spr={rsp:.1f}%")
print(f"rgB best={bm:.1f}ms med={bmd:.1f} spr={bsp:.1f}%")
aa=rm/bm
print(f"A/A rg(best)/rg(best) = {aa:.4f}  (|AA-1|={abs(aa-1)*100:.2f}%, tol={tol}%)")
aa_med=rmd/bmd
print(f"A/A rg(med)/rg(med)   = {aa_med:.4f}")
trust = abs(aa-1)*100<=tol and rsp<=tol and bsp<=tol and gsp<=tol
print(f"GATE-0 A/A trust: {'PASS' if trust else 'FAIL'} (need |AA-1|<=tol AND all spreads<=tol)")
print(f"==> gz/rg (best) = {gm/rm:.4f}   gz/rg (med) = {gmd/rmd:.4f}")
print(f"    gz {'SLOWER' if gm>rm else 'FASTER'} than rg by {abs(gm/rm-1)*100:.1f}% (best-of-N)")
print(f"    [{'TRUSTED' if trust else 'UNTRUSTED — box too loaded, reproducibility OWED'}]")
PY
