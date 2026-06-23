#!/usr/bin/env bash
# T1 stateless-kernel A/B (re-gate NIGHT32 at HEAD now that T1 is INSTRUCTION-BOUND):
#   BASE = run_contig (resumable anchor)   CAND = GZIPPY_STATELESS_KERNEL=1 (anchor
#   deleted, -0.607 instr/B, byte-exact on valid T1 clean blocks). Paired-interleaved,
#   perf stat cyc+instr, /dev/null, single-core. Gate-0: sha==zcat BOTH arms (byte-
#   exact) + a non-zero CAND instr/B drop = the stateless path actually ran (non-inert).
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
GZDIR=${GZDIR:-/root}
PIN=${PIN:-4}
N=${N:-13}
CORPORA=${CORPORA:-"silesia nasa"}
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then
  CYC=cpu_core/cycles/; INS=cpu_core/instructions/
else CYC=cycles; INS=instructions; fi
EV="$CYC,$INS"
perfrun() { perf stat -x, -e "$EV" taskset -c "$PIN" "$@" >/dev/null 2>/tmp/ps.txt
  awk -F, '/cycles/{c=$1} /instructions/{i=$1} END{print c, i}' /tmp/ps.txt; }
echo "corpus,arm,rep,rawbytes,cyc,instr"
for c in $CORPORA; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || { echo "# MISSING $f" 1>&2; continue; }
  rb=$(zcat "$f" 2>/dev/null | wc -c)
  sB=$(GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  sC=$(GZIPPY_STATELESS_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  sZ=$(zcat "$f" | sha256sum | cut -d' ' -f1)
  echo "# SHA[$c] BASE=$([ "$sB" = "$sZ" ] && echo OK || echo BAD) CAND=$([ "$sC" = "$sZ" ] && echo OK || echo BAD) rawbytes=$rb" 1>&2
  for rep in $(seq 1 "$N"); do
    set -- $(perfrun env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f");                        echo "$c,BASE,$rep,$rb,$1,$2"
    set -- $(perfrun env GZIPPY_STATELESS_KERNEL=1 GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f"); echo "$c,CAND,$rep,$rb,$1,$2"
  done
done
echo "# done" 1>&2
