#!/usr/bin/env bash
# T1 RECOVERABILITY oracle (cursor-directed, CLAUDE Gate-2 "confirm the wall moves
# before a work-stretch"): GZIPPY_KERNEL_INJECT adds byte-transparent dummy instrs
# per run_contig iteration. Sweep {0,1,2,4} and measure d(cyc/B)/d(instr/B). If
# cyc/B rises PROPORTIONALLY with injected instr/B, the run_contig instruction
# surplus is on the RETIRING critical path => the ~2 instr/B gap is RECOVERABLE.
# Flat cyc/B => slack (floor). Byte-transparent: sha==zcat at EVERY level (Gate-0).
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
GZDIR=${GZDIR:-/root}
PIN=${PIN:-4}
N=${N:-11}
CORPORA=${CORPORA:-"silesia nasa"}
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then
  CYC=cpu_core/cycles/; INS=cpu_core/instructions/
else CYC=cycles; INS=instructions; fi
EV="$CYC,$INS"
perfrun() { perf stat -x, -e "$EV" taskset -c "$PIN" "$@" >/dev/null 2>/tmp/ps.txt
  awk -F, '/cycles/{c=$1} /instructions/{i=$1} END{print c, i}' /tmp/ps.txt; }
echo "corpus,inject,rep,rawbytes,cyc,instr"
for c in $CORPORA; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || { echo "# MISSING $f" 1>&2; continue; }
  rb=$(zcat "$f" 2>/dev/null | wc -c)
  sZ=$(zcat "$f" | sha256sum | cut -d' ' -f1)
  # Gate-0: byte-transparent sha at every inject level
  for inj in 0 1 2 4; do
    s=$(GZIPPY_KERNEL_INJECT=$inj GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
    echo "# SHA[$c inj=$inj]=$([ "$s" = "$sZ" ] && echo OK || echo BAD)" 1>&2
  done
  for rep in $(seq 1 "$N"); do
    for inj in 0 1 2 4; do
      set -- $(perfrun env GZIPPY_KERNEL_INJECT=$inj GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f")
      echo "$c,$inj,$rep,$rb,$1,$2"
    done
  done
done
echo "# done" 1>&2
