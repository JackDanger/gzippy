#!/usr/bin/env bash
# T1-vs-igzip KERNEL LOCATE (re-establish the gap + mechanism at HEAD before any
# heroic asm). perf stat cyc/instr/IPC/branch-miss for gz-native -p1 vs igzip,
# single-core pinned, N reps. Computes cyc/B, instr/B, IPC, brmiss/kB. Arch-aware
# PMU (Intel hybrid cpu_core/ vs AMD plain). /dev/null, sha-gated.
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
GZDIR=${GZDIR:-/root}
PIN=${PIN:-4}
N=${N:-9}
CORPORA=${CORPORA:-"silesia nasa"}

# arch-aware perf events
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then
  CYC=cpu_core/cycles/; INS=cpu_core/instructions/; BR=cpu_core/branches/; BRM=cpu_core/branch-misses/
  echo "# PMU: Intel hybrid cpu_core/"
else
  CYC=cycles; INS=instructions; BR=branches; BRM=branch-misses
  echo "# PMU: plain core counters (AMD)"
fi
EV="$CYC,$INS,$BR,$BRM"

rawbytes() { zcat "$1" 2>/dev/null | wc -c; }
# one perf run -> "cyc instr br brmiss" (sum), /dev/null
perfrun() { # $1=cmd... ; prints "cyc instr br brmiss"
  perf stat -x, -e "$EV" taskset -c "$PIN" "$@" >/dev/null 2>/tmp/ps.txt
  awk -F, 'BEGIN{c=i=b=m=0}
    /cycles/{c=$1} /instructions/{i=$1} /branches/{b=$1} /branch-misses/{m=$1}
    END{print c, i, b, m}' /tmp/ps.txt
}

echo "corpus,tool,rep,rawbytes,cyc,instr,branches,brmiss"
for c in $CORPORA; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || { echo "# MISSING $f" 1>&2; continue; }
  rb=$(rawbytes "$f")
  # sha gate
  s1=$("$B" -d -c "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  s2=$(zcat "$f" | sha256sum | cut -d' ' -f1)
  echo "# SHA[$c]=$([ "$s1" = "$s2" ] && echo OK || echo BAD) rawbytes=$rb" 1>&2
  for rep in $(seq 1 "$N"); do
    set -- $(perfrun env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p 1 "$f"); echo "$c,GZ,$rep,$rb,$1,$2,$3,$4"
    set -- $(perfrun igzip -d -c "$f");                                    echo "$c,IGZIP,$rep,$rb,$1,$2,$3,$4"
  done
done
echo "# done" 1>&2
