#!/usr/bin/env bash
# T1 cross-tool matrix (cursor-directed): gz-native vs igzip / libdeflate / pigz /
# rapidgzip / gunzip at T1, single-core pinned, interleaved, /dev/null all arms.
# North-star checks at T1: beat pigz (FALSIFIER: gz/pigz must be <1), beat rg,
# beat libdeflate+igzip (the gated ISA-L floor). sha-verified for gz.
set -u
B=${BIN:-/dev/shm/tri-target/release/gzippy}
RG=${RG:-/usr/local/bin/rapidgzip}
GZDIR=${GZDIR:-/root}
N=${N:-13}
PIN=${PIN:-4}
TS(){ date +%s.%N; }
el(){ awk "BEGIN{printf \"%.6f\", $1-$2}"; }
# each arm: timed wall, /dev/null, single core
gz(){   local a b; a=$(TS); GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$PIN" "$B" -d -c -p 1 "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }
ig(){   local a b; a=$(TS); taskset -c "$PIN" igzip -d -c "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }
ld(){   local a b; a=$(TS); taskset -c "$PIN" libdeflate-gunzip -c "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }
pg(){   local a b; a=$(TS); taskset -c "$PIN" pigz -d -c -p 1 "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }
rg(){   local a b; a=$(TS); taskset -c "$PIN" "$RG" -d -c -P 1 "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }
gu(){   local a b; a=$(TS); taskset -c "$PIN" gunzip -c "$1" >/dev/null 2>/dev/null; b=$(TS); el "$b" "$a"; }

# Gate-0: gz sha==zcat per corpus
for c in silesia monorepo nasa; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || { echo "# MISSING $f" 1>&2; continue; }
  s1=$("$B" -d -c "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  s2=$(zcat "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "# SHA[$c]=$([ "$s1" = "$s2" ] && echo OK || echo MISMATCH) sz=$(stat -c%s "$f")" 1>&2
done

echo "corpus,arm,rep,wall_s"
for c in silesia monorepo nasa; do
  f="$GZDIR/$c.gz"; [ -f "$f" ] || continue
  for rep in $(seq 1 "$N"); do
    echo "$c,GZ,$rep,$(gz "$f")"
    echo "$c,IGZIP,$rep,$(ig "$f")"
    echo "$c,LIBDEFLATE,$rep,$(ld "$f")"
    echo "$c,PIGZ,$rep,$(pg "$f")"
    echo "$c,RG,$rep,$(rg "$f")"
    echo "$c,GUNZIP,$rep,$(gu "$f")"
  done
done
echo "# done $(date -u +%FT%TZ)" 1>&2
