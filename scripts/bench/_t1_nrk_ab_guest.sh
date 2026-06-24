#!/usr/bin/env bash
# T1 NRK A/B (same binary, env toggle): BASE=run_contig (resumable), CAND=NRK.
# perf stat cyc+instr, paired-interleaved, /dev/null, single-core. Gate-0: both
# arms sha==zcat; CAND instr/B drop = non-inert proof NRK ran. cyc/B = the verdict.
set -u
B=${BIN:-/dev/shm/nrk-target/release/gzippy}
GZDIR=${GZDIR:-/root}; PIN=${PIN:-4}; N=${N:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then CYC=cpu_core/cycles/; INS=cpu_core/instructions/; else CYC=cycles; INS=instructions; fi
pr(){ perf stat -x, -e "$CYC,$INS" taskset -c "$PIN" "$@" >/dev/null 2>/tmp/ps.txt; awk -F, '/cycles/{c=$1}/instructions/{i=$1}END{print c,i}' /tmp/ps.txt; }
echo "corpus,arm,rep,rawbytes,cyc,instr"
for c in $CORPORA; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || continue
  rb=$(zcat "$f" 2>/dev/null|wc -c); sZ=$(zcat "$f"|sha256sum|cut -d' ' -f1)
  sB=$(GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p1 "$f" 2>/dev/null|sha256sum|cut -d' ' -f1)
  sC=$(GZIPPY_T1_NRK=1 GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p1 "$f" 2>/dev/null|sha256sum|cut -d' ' -f1)
  echo "# SHA[$c] BASE=$([ "$sB" = "$sZ" ]&&echo OK||echo BAD) CAND=$([ "$sC" = "$sZ" ]&&echo OK||echo BAD)" 1>&2
  for rep in $(seq 1 "$N"); do
    set -- $(pr env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p1 "$f"); echo "$c,BASE,$rep,$rb,$1,$2"
    set -- $(pr env GZIPPY_T1_NRK=1 GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p1 "$f"); echo "$c,CAND,$rep,$rb,$1,$2"
  done
done
echo "# done" 1>&2
