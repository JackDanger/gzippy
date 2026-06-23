#!/usr/bin/env bash
# 2-binary T1 A/B gate: BASE_BIN vs CAND_BIN, perf stat cyc+instr, paired-
# interleaved, single-core, /dev/null. Computes paired Δ cyc/B + instr/B per
# corpus + a spread-gated verdict. Gate-0: both arms sha==zcat. For gating the
# heroic T1 instruction-cut asm changes (cyc/B is the verdict, instr/B the proof).
set -u
BASE_BIN=${BASE_BIN:?BASE_BIN required}
CAND_BIN=${CAND_BIN:?CAND_BIN required}
GZDIR=${GZDIR:-/root}
PIN=${PIN:-4}
N=${N:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then
  CYC=cpu_core/cycles/; INS=cpu_core/instructions/
else CYC=cycles; INS=instructions; fi
EV="$CYC,$INS"
perfrun() { perf stat -x, -e "$EV" taskset -c "$PIN" "$@" >/dev/null 2>/tmp/ps.txt
  awk -F, '/cycles/{c=$1} /instructions/{i=$1} END{print c, i}' /tmp/ps.txt; }
echo "corpus,arm,rep,rawbytes,cyc,instr"
for c in $CORPORA; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || { echo "# MISSING $f" 1>&2; continue; }
  rb=$(zcat "$f" 2>/dev/null | wc -c); sZ=$(zcat "$f" | sha256sum | cut -d' ' -f1)
  sB=$(GZIPPY_FORCE_PARALLEL_SM=1 "$BASE_BIN" -d -c -p 1 "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  sC=$(GZIPPY_FORCE_PARALLEL_SM=1 "$CAND_BIN" -d -c -p 1 "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "# SHA[$c] BASE=$([ "$sB" = "$sZ" ] && echo OK || echo BAD) CAND=$([ "$sC" = "$sZ" ] && echo OK || echo BAD)" 1>&2
  for rep in $(seq 1 "$N"); do
    set -- $(perfrun env GZIPPY_FORCE_PARALLEL_SM=1 "$BASE_BIN" -d -c -p 1 "$f"); echo "$c,BASE,$rep,$rb,$1,$2"
    set -- $(perfrun env GZIPPY_FORCE_PARALLEL_SM=1 "$CAND_BIN" -d -c -p 1 "$f"); echo "$c,CAND,$rep,$rb,$1,$2"
  done
done
echo "# done" 1>&2
