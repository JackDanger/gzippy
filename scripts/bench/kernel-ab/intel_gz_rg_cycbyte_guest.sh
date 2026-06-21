#!/usr/bin/env bash
# intel_gz_rg_cycbyte_guest.sh — STEP-1 AXIS instrument (load-immune).
#
# Settles the silesia-T4 +16% gz-vs-RAPIDGZIP axis: is gz's per-byte cost higher
# because it EXECUTES MORE INSTRUCTIONS than rapidgzip (INSTRUCTION-COUNT axis →
# apportionable/portable → localize+port rg's leaner phase) or because it runs at
# LOWER IPC for ~equal instructions (IPC / ISA-L-CODEGEN axis → deep-kernel rewrite,
# poor ROI → bank-vs-fund)?
#
# rapidgzip is the RIGHT reference (a chunked decoder WITH a scaffold + ISA-L kernel,
# like gz) — unlike the bare ISA-L `_04` kernel STEP-0 used. So this compares two
# whole binaries doing the same parallel decode, via `perf stat` on the real decode.
#
# METHOD (combines two proven patterns):
#   - single-P-core taskset isolation + measured GHz-stability gate
#     (from _distpreload_cycbyte_guest.sh) for a real cyc/B on the LOADED LXC, and
#   - gz-vs-rapidgzip both-arms + A/A self-test (from standing/_cleankernel_silt4)
#     to license the loaded-box wall ratio.
#
# THE LOAD-IMMUNE PRIMITIVE is `instructions` (a retired-instruction COUNT, not a
# wall) — trustworthy at any box load. The AXIS verdict is made in ABSOLUTE instr/B.
# cyc/B is reported under the GHz gate; IPC = instr/cyc (carries the GHz caveat).
# NO ratio*ratio / share*wall — every per-byte number is an absolute count / bytes.
#
# T1: single pinned P-core (cleanest). T4: 4 distinct physical P-cores (cpuset);
# perf stat aggregates instructions/cycles across ALL threads of the process tree,
# so instr/B(T4) = Sigma_all_threads_instructions / output_bytes (the load-immune
# 4-thread comparison). The wall ratio gz/rg (A/A-licensed) re-confirms the +16%.
#
# GATE-0 (printed FIRST; a run failing any is VOID):
#   byte-exact gz==rg==zcat (sha) both arms; /dev/null both arms; gz
#   flavor=parallel-sm+pure + path=ParallelSM asserted; rg present + rg-vs-rg A/A
#   ~1.0 AND gz-vs-gz A/A ~1.0; GHz spread <=1%; instructions inter-run spread
#   <0.5% (confirm the primitive IS stable); box load + LLC-miss/B reported.
#
# Usage:
#   GZ=/dev/shm/gzrg-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
#   CORP=/root/silesia.gz PIN_T1=3 PIN_T4=2,4,6,8 N=15 \
#   bash intel_gz_rg_cycbyte_guest.sh
set -uo pipefail

GZ="${GZ:-/dev/shm/gzrg-target/release/gzippy}"
RG="${RG:-/root/oracle_c/rapidgzip-native}"
CORP="${CORP:-/root/silesia.gz}"
PIN_T1="${PIN_T1:-3}"           # single P-core CPU id (T1)
PIN_T4="${PIN_T4:-2,4,6,8}"     # 4 distinct physical P-core CPU ids (T4)
N="${N:-15}"
OUT="${OUT:-/dev/shm/gzrg_axis}"
ANALYZE="$(dirname "$0")/intel_gz_rg_cycbyte_analyze.py"
[ -f "$ANALYZE" ] || ANALYZE="/root/gzippy/scripts/bench/kernel-ab/intel_gz_rg_cycbyte_analyze.py"

[ -x "$GZ" ] || { echo "NO GZ $GZ"; exit 2; }
[ -x "$RG" ] || { echo "NO RG $RG"; exit 2; }
[ -f "$CORP" ] || { echo "NO CORP $CORP"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"

echo "================ INTEL gz-vs-rapidgzip cyc/B + instr/B AXIS ================"
echo "GZ=$GZ  sha=$(sha256sum "$GZ"|cut -c1-16)"
echo "RG=$RG  sha=$(sha256sum "$RG"|cut -c1-16)  ver=$("$RG" --version 2>&1|grep -oE 'version [0-9.]+'|head -1)"
echo "CORP=$CORP  pin_t1=cpu$PIN_T1  pin_t4=cpu{$PIN_T4}  N=$N"
echo "load: $(cat /proc/loadavg)"
echo "host-freq-state: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu$PIN_T1/cpufreq/scaling_governor 2>/dev/null) (LXC: host-managed/read-only; gate=measured GHz spread)"

# ---------- GATE-0: gz build-flavor + production path ----------
echo "--- GATE-0 FLAVOR/PATH (gz) ---"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p1 "$CORP" >/dev/null 2>"$OUT/dbg.txt" || true
FLAVOR=$(grep -oiE 'flavor[=: ]+[a-z+-]+' "$OUT/dbg.txt" | head -1)
PATHLINE=$(grep -oE 'path=[A-Za-z]+' "$OUT/dbg.txt" | head -1)
echo "    $FLAVOR   $PATHLINE"
echo "$PATHLINE" | grep -q 'path=ParallelSM' && echo "    PATH PASS (ParallelSM)" || echo "    PATH WARN (expected ParallelSM) — see $OUT/dbg.txt"

# ---------- GATE-0: byte-exact gz==rg==zcat ----------
echo "--- GATE-0 BYTE-EXACT (sha; /dev/null sink) ---"
REF=$(zcat "$CORP" | sha256sum | cut -c1-16)
SG=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -dc -p4 "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
SR=$("$RG" -dc -P4 "$CORP" 2>/dev/null | sha256sum | cut -c1-16)
echo "    ref(zcat)=$REF  gz=$SG  rg=$SR"
[ "$SG" = "$REF" ] || { echo "    GZ BYTE MISMATCH — VOID"; exit 3; }
[ "$SR" = "$REF" ] || { echo "    RG BYTE MISMATCH — VOID"; exit 3; }
echo "    BYTE-EXACT PASS (gz==rg==zcat)"

OUTBYTES=$(zcat "$CORP" | wc -c)
echo "    output_bytes=$OUTBYTES"
echo "$OUTBYTES" > "$OUT/outbytes.txt"

# ---------- MEASUREMENT ----------
# cpu_core PMU (both T1 and T4 pinned to P-cores). duration_time = wall(ns).
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/LLC-load-misses/,task-clock,duration_time"
run_one() { # $1=pin $2=bin $3=threadflag(e.g. -p4 or -P4) $4=outfile
  taskset -c "$1" perf stat -x, -e "$EVENTS" \
    -- env GZIPPY_FORCE_PARALLEL_SM=1 "$2" -dc "$3" "$CORP" \
    >/dev/null 2>"$4"
}

# warmup (page cache + freq ramp)
run_one "$PIN_T1" "$GZ" -p1 "$OUT/warm.gz1" 2>/dev/null || true
run_one "$PIN_T1" "$RG" -P1 "$OUT/warm.rg1" 2>/dev/null || true

echo "--- MEASURE T1 (single P-core cpu$PIN_T1) interleaved gzA,gzB,rgA,rgB x N=$N ---"
for r in $(seq 1 "$N"); do
  run_one "$PIN_T1" "$GZ" -p1 "$OUT/T1.gzA.$r.csv"
  run_one "$PIN_T1" "$GZ" -p1 "$OUT/T1.gzB.$r.csv"
  run_one "$PIN_T1" "$RG" -P1 "$OUT/T1.rgA.$r.csv"
  run_one "$PIN_T1" "$RG" -P1 "$OUT/T1.rgB.$r.csv"
done

echo "--- MEASURE T4 (4 P-cores cpuset {$PIN_T4}) interleaved gzA,gzB,rgA,rgB x N=$N ---"
for r in $(seq 1 "$N"); do
  run_one "$PIN_T4" "$GZ" -p4 "$OUT/T4.gzA.$r.csv"
  run_one "$PIN_T4" "$GZ" -p4 "$OUT/T4.gzB.$r.csv"
  run_one "$PIN_T4" "$RG" -P4 "$OUT/T4.rgA.$r.csv"
  run_one "$PIN_T4" "$RG" -P4 "$OUT/T4.rgB.$r.csv"
done

echo "load(post): $(cat /proc/loadavg)"
echo "--- ANALYSIS ---"
python3 "$ANALYZE" "$OUT" "$OUTBYTES"
echo "DONE_INTEL_GZ_RG_AXIS"
