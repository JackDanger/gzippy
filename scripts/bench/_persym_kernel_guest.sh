#!/bin/bash
# _persym_kernel_guest.sh — per-SYMBOL inner-Huffman kernel comparison instrument.
#
# Compares gznative run_contig vs igzip decode_huffman_code_block_stateless_04 vs
# rapidgzip Block::read on a literal-heavy input. The deficit is PER-SYMBOL, so
# every count is normalized by the stream's symbol count (literals+backrefs, a
# CONSTANT identical across all three tools for a given .gz — obtained once from
# rapidgzip --analyze).
#
# Reports per tool: instr/sym, cyc/sym, IPC, branch-miss/sym, TopdownL1
# (frontend/backend/bad-spec/retiring), and the kernel SELF-fraction (perf report)
# so the whole-program perf-stat can be attributed to the kernel.
#
# GATE-0 SELF-VALIDATION (printed; a run that fails these is VOID):
#   (a) gz path == ParallelSM (GZIPPY_DEBUG)
#   (b) gz run_contig ENGAGED: KERN_ENTRIES>0 and KERN_ASM_BYTES within 20% of the
#       decompressed size (the kernel actually processed the bulk of the bytes)
#   (c) sha(gz -d) == sha(reference)  [Rule 4]
#   (d) kernel self-fraction printed for all 3 (isolation proof)
#   (e) load average printed (cyc/IPC/topdown are cyc-based; instr/branches exact)
#
# Usage: _persym_kernel_guest.sh <corpus> <total_symbols> [pin_core] [reps]
set -u
CORP="${1:?corpus name e.g. silesia}"
SYM="${2:?total symbol count from rapidgzip --analyze}"
PIN="${3:-2}"
REPS="${4:-7}"

GZ_PROD=/root/gzippy/target/release/gzippy   # stripped production (perf stat)
GZ_SYM=/root/gz-native-sym                    # debuginfo, identical flags (annotate/report)
IGZIP=/usr/bin/igzip
RG=/root/oracle_c/rapidgzip-native
F=/root/$CORP.gz
OUT=/tmp/persym/$CORP
mkdir -p "$OUT"

echo "================ PER-SYMBOL KERNEL COMPARE: $CORP ================"
echo "file=$F symbols=$SYM pin_core=$PIN reps=$REPS  load:$(cat /proc/loadavg)"
echo "gz=$GZ_PROD ($(sha256sum $GZ_PROD|cut -c1-12)) sym=$GZ_SYM"

# ---- GATE-0(a)+(b): path + kernel-engaged self-validation -------------------
echo "--- GATE0: gz path + run_contig engaged ---"
GZIPPY_DEBUG=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
  taskset -c "$PIN" "$GZ_PROD" -d -c -p1 "$F" >/dev/null 2>/tmp/persym/g0.$CORP
grep -E "path=" /tmp/persym/g0.$CORP | head -1
ASMLINE=$(grep "asm-kernel:c" /tmp/persym/g0.$CORP | tail -1)
echo "$ASMLINE"
RAW=$(zcat "$F" | wc -c)
ASMB=$(echo "$ASMLINE" | sed -n 's/.*asm_bytes=\([0-9]*\).*/\1/p')
ENTR=$(echo "$ASMLINE" | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
echo "  raw_bytes=$RAW kernel_asm_bytes=${ASMB:-0} entries=${ENTR:-0} frac=$(awk -v a=${ASMB:-0} -v r=$RAW 'BEGIN{printf "%.3f", r?a/r:0}')"

# ---- GATE-0(c): sha ---------------------------------------------------------
REFSHA=$(zcat "$F" | sha256sum | cut -c1-16)
GZSHA=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ_PROD" -d -c -p1 "$F" 2>/dev/null | sha256sum | cut -c1-16)
echo "  sha ref=$REFSHA gz=$GZSHA  $([ "$REFSHA" = "$GZSHA" ] && echo SHA_OK || echo SHA_MISMATCH)"

# ---- per-tool runner -------------------------------------------------------
run_tool() {  # $1=tag  $2...=command
  local tag="$1"; shift
  # exact counts (load-robust): instructions, cycles, branches, branch-misses
  taskset -c "$PIN" perf stat -r "$REPS" \
    -e instructions,cycles,branches,branch-misses \
    -- taskset -c "$PIN" bash -c "$* >/dev/null 2>&1" 2>"$OUT/$tag.stat"
  # TopdownL1 (cyc-based; load-sensitive — load printed above)
  taskset -c "$PIN" perf stat -r "$REPS" -M TopdownL1 \
    -- taskset -c "$PIN" bash -c "$* >/dev/null 2>&1" 2>"$OUT/$tag.td"
}

GZCMD="GZIPPY_FORCE_PARALLEL_SM=1 $GZ_PROD -d -c -p1 $F"
IGCMD="$IGZIP -d -c $F"
RGCMD="$RG -P 1 -d -c $F"

echo "--- running perf stat (instr/cyc/branches + TopdownL1), $REPS reps each ---"
run_tool gznative "$GZCMD"
run_tool igzip    "$IGCMD"
run_tool rg       "$RGCMD"

# ---- kernel SELF-fraction (perf report self %) -----------------------------
echo "--- kernel self-fraction (perf report) ---"
kfrac() {  # $1=tag $2=cmd $3=symbol-regex
  local tag="$1" cmd="$2" re="$3"
  taskset -c "$PIN" perf record -F 2999 -o "$OUT/$tag.rec" \
    -- taskset -c "$PIN" bash -c "$cmd >/dev/null 2>&1" >/dev/null 2>&1
  local pct
  pct=$(perf report -i "$OUT/$tag.rec" --stdio --no-children 2>/dev/null \
        | grep -vE '^#|^$' | grep -E "$re" | head -1 | awk '{print $1}')
  echo "  $tag kernel self = ${pct:-NA}  ($re)"
}
kfrac gznative "$GZCMD" "run_contig"
kfrac igzip    "$IGCMD" "decode_huffman_code_block_stateless"
kfrac rg       "$RGCMD" "Block.*read|readInternal|inflate"

# ---- SUMMARY (per-symbol normalized) ---------------------------------------
echo "======================= SUMMARY (per symbol, /$SYM) ======================="
printf "%-10s %12s %12s %8s %12s\n" tool instr/sym cyc/sym IPC bmiss/sym
for tag in gznative igzip rg; do
  read I C B M < <(awk '
    /cpu_core\/instructions\// {gsub(/,/,"",$1); ins=$1}
    /cpu_core\/cycles\// {gsub(/,/,"",$1); cyc=$1}
    /cpu_core\/branches\// {gsub(/,/,"",$1); br=$1}
    /cpu_core\/branch-misses\// {gsub(/,/,"",$1); bm=$1}
    END{print ins, cyc, br, bm}' "$OUT/$tag.stat")
  awk -v t=$tag -v i=$I -v c=$C -v m=$M -v s=$SYM 'BEGIN{
    printf "%-10s %12.3f %12.3f %8.3f %12.4f\n", t, i/s, c/s, c?i/c:0, m/s}'
done
echo "--- TopdownL1 (% slots) ---"
printf "%-10s %10s %10s %10s %10s\n" tool frontend backend bad-spec retiring
for tag in gznative igzip rg; do
  FE=$(grep 'tma_frontend_bound' "$OUT/$tag.td" | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  BE=$(grep 'tma_backend_bound'  "$OUT/$tag.td" | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  BS=$(grep 'tma_bad_speculation' "$OUT/$tag.td" | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  RT=$(grep 'tma_retiring'        "$OUT/$tag.td" | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  printf "%-10s %10s %10s %10s %10s\n" "$tag" "${FE:-NA}" "${BE:-NA}" "${BS:-NA}" "${RT:-NA}"
done
echo "raw perf outputs in $OUT/*.stat *.td"
echo "DONE_PERSYM_$CORP"
