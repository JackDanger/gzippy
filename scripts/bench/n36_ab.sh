#!/usr/bin/env bash
# NIGHT36 — production-wall conv-vs-base A/B for the run_contig `ret=1` hoist.
# Arms interleaved per rep on the SAME pinned P-core, SAME /dev/null sink:
#   A1,A2 = BASE (NIGHT35 native, no hoist) + self-test
#   B     = CONV (NIGHT35 + ret=1 hoist)
#   IG,IG2= igzip bar + self-test (for the gap-to-igzip context)
# medΔ (B-A1) NEGATIVE => conv FASTER than base (the change wins). cyc/B is
# frequency-invariant; GHz + LLC reported by the analyzer.
set -u
BASE=${BASE:-/dev/shm/n36base/release/gzippy}
CONV=${CONV:-/dev/shm/n36/release/gzippy}
IG=${IG:-/usr/bin/igzip}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia monorepo"}
OUT=${OUT:-/dev/shm/n36ab}
ANALYZE=${ANALYZE:-/root/distpreload-harness/_distpreload_paired_analyze.py}
LOG=${OUT}.log
DONE=${OUT}.DONE
rm -f "$DONE"; rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$LOG" 2>&1
EVENTS="cpu_core/instructions/,cpu_core/cycles/,task-clock,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/"

echo "== NIGHT36 conv-vs-base production A/B =="
echo "BASE sha=$(sha256sum $BASE|cut -c1-12)  CONV sha=$(sha256sum $CONV|cut -c1-12)"
echo "igzip: $($IG --version 2>&1|head -1)  pin=cpu$PIN reps=$REPS corpora='$CORPORA'"
echo "load_start=$(cat /proc/loadavg)"

# ---- GATE-0 (a)(b)(c)(f): sha==zcat all arms + routing + KERN consumer fired ----
GATE_FAIL=0
: > "$OUT/bytes.txt"
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "NO FILE $F"; GATE_FAIL=1; continue; }
  REF=$(zcat "$F"|sha256sum|cut -c1-16)
  BY=$(zcat "$F"|wc -c)
  echo "$corp $BY" >> "$OUT/bytes.txt"
  IGS=$(taskset -c $PIN $IG -d -c "$F" 2>/dev/null|sha256sum|cut -c1-16)
  BAS=$(taskset -c $PIN $BASE -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  CON=$(taskset -c $PIN $CONV -d -c -p1 "$F" 2>/dev/null|sha256sum|cut -c1-16)
  echo "GATE0 $corp ref=$REF ig=$([ "$IGS" = "$REF" ]&&echo OK||{ echo BAD;GATE_FAIL=1;}) base=$([ "$BAS" = "$REF" ]&&echo OK||{ echo BAD;GATE_FAIL=1;}) conv=$([ "$CON" = "$REF" ]&&echo OK||{ echo BAD;GATE_FAIL=1;})"
done
# routing + KERN consumer fired on conv
P=$(GZIPPY_DEBUG=1 taskset -c $PIN $CONV -d -c -p1 /root/silesia.gz 2>&1 >/dev/null|grep -o "path=[A-Za-z]*"|head -1)
KE=$(GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 taskset -c $PIN $CONV -d -c -p1 /root/nasa.gz 2>&1 >/dev/null|grep -o "entries=[0-9]*"|head -1)
echo "GATE4 routing $P  consumer $KE"
[ "$P" = "path=ParallelSM" ] || GATE_FAIL=1
case "$KE" in entries=0|"") GATE_FAIL=1;; esac
if [ "$GATE_FAIL" != 0 ]; then echo "GATE FAIL"; echo FAIL>"$DONE"; exit 2; fi
echo "GATE0/4 PASS"

run_gz(){ local BIN=$1 corp=$2 arm=$3 r=$4 F=/root/$2.gz CSV="$OUT/$2.$3.$4.csv"
  taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN "$BIN" -d -c -p1 "$F" >/dev/null 2>"$CSV"; }
run_ig(){ local corp=$1 arm=$2 r=$3 F=/root/$1.gz CSV="$OUT/$1.$2.$3.csv"
  taskset -c $PIN perf stat -x, -e "$EVENTS" -- taskset -c $PIN $IG -d -c "$F" >/dev/null 2>"$CSV"; }

echo "--- MEASURE interleaved (A1=base,A2=base,B=conv,IG,IG2) ---"
for corp in $CORPORA; do
  for r in $(seq 1 $REPS); do
    run_gz $BASE $corp A1 $r
    run_gz $BASE $corp A2 $r
    run_gz $CONV $corp B  $r
    run_ig $corp IG  $r
    run_ig $corp IG2 $r
  done
done
echo "load_end=$(cat /proc/loadavg)"

echo "######## ANALYSIS 1: CONV - BASE (the change verdict) ########"
python3 "$ANALYZE" "$OUT" --tag conv_minus_base $CORPORA

# derived dir: CONV - IGZIP (gap closed by conv)
D2="$OUT"_cvig; rm -rf "$D2"; mkdir -p "$D2"; cp "$OUT/bytes.txt" "$D2/"
for corp in $CORPORA; do for r in $(seq 1 $REPS); do
  cp "$OUT/$corp.IG.$r.csv"  "$D2/$corp.A1.$r.csv"
  cp "$OUT/$corp.IG2.$r.csv" "$D2/$corp.A2.$r.csv"
  cp "$OUT/$corp.B.$r.csv"   "$D2/$corp.B.$r.csv"
done; done
echo "######## ANALYSIS 2: CONV - IGZIP (gap to igzip) ########"
python3 "$ANALYZE" "$D2" --tag conv_minus_igzip $CORPORA

# derived dir: BASE - IGZIP (prior gap)
D3="$OUT"_bvig; rm -rf "$D3"; mkdir -p "$D3"; cp "$OUT/bytes.txt" "$D3/"
for corp in $CORPORA; do for r in $(seq 1 $REPS); do
  cp "$OUT/$corp.IG.$r.csv"  "$D3/$corp.A1.$r.csv"
  cp "$OUT/$corp.IG2.$r.csv" "$D3/$corp.A2.$r.csv"
  cp "$OUT/$corp.A1.$r.csv"  "$D3/$corp.B.$r.csv"
done; done
echo "######## ANALYSIS 3: BASE - IGZIP (prior gap) ########"
python3 "$ANALYZE" "$D3" --tag base_minus_igzip $CORPORA

echo PASS > "$DONE"
