#!/usr/bin/env bash
# Cycle/IPC before-after for the SegmentedU8 memcpy-append fix (segmented_buffer.rs ONLY).
# Interleaved N>=12: each trial runs base, after, rg back-to-back (same window).
# perf cpu_core cycles+instructions, /dev/null sink (char-dev guarded), sha-gated.
set -u
BASE=/dev/shm/gz-base-fresh/release/gzippy
AFTER=/dev/shm/gz-after-fresh/release/gzippy
RG=/root/oracle_c/rapidgzip-native
N="${1:-14}"
T=4
CORES="0,2,4,6"
PIN="taskset -c $CORES"
EV="cpu_core/cycles/u,cpu_core/instructions/u"
OUT=/dev/shm/memcpy_cyc.csv
PS=/dev/shm/_p.stat

[ -c /dev/null ] || { echo "SINK_FAIL: /dev/null not a char device (before)"; exit 9; }
for b in "$BASE" "$AFTER" "$RG"; do [ -x "$b" ] || { echo "MISSING_BIN: $b"; exit 8; }; done

echo "=== GATE4: path assertion (base & after) ==="
for tag in base after; do
  bin=$BASE; [ "$tag" = after ] && bin=$AFTER
  p=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 $bin -d -c -p 4 /root/monorepo.gz 2>&1 >/dev/null | grep -o 'path=[A-Za-z]*' | head -1)
  echo "  $tag $p"
done

echo "=== sha-gate (base, after vs gzip ref) silesia+monorepo+nasa, T1+T4 ==="
sha_ok=1
for c in silesia monorepo nasa; do
  f=/root/$c.gz
  ref=$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)
  for tag in base after; do
    bin=$BASE; [ "$tag" = after ] && bin=$AFTER
    for t in 1 4; do
      s=$(GZIPPY_FORCE_PARALLEL_SM=1 $bin -d -c -p "$t" "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)
      if [ "$s" = "$ref" ]; then ok=PASS; else ok=FAIL; sha_ok=0; fi
      echo "  $c $tag T$t $ok"
    done
  done
done
[ "$sha_ok" = 1 ] || { echo "SHA_GATE_FAILED — aborting"; exit 7; }

run() { # tag bin corpus  -> append cycles,instr row
  local tag="$1" bin="$2" f="$3" extra="$4"
  perf stat -x, -e "$EV" -o "$PS" -- env GZIPPY_FORCE_PARALLEL_SM=1 $PIN $bin $extra -p "$T" "$f" >/dev/null 2>/dev/null
  local cyc ins
  cyc=$(awk -F, '$3 ~ /cycles/{gsub(/[^0-9]/,"",$1);print $1}' "$PS")
  ins=$(awk -F, '$3 ~ /instructions/{gsub(/[^0-9]/,"",$1);print $1}' "$PS")
  echo "$f_label,$trial,$tag,$cyc,$ins" >> "$OUT"
}
run_rg() {
  local f="$1"
  perf stat -x, -e "$EV" -o "$PS" -- $PIN $RG -d -c -P "$T" "$f" >/dev/null 2>/dev/null
  local cyc ins
  cyc=$(awk -F, '$3 ~ /cycles/{gsub(/[^0-9]/,"",$1);print $1}' "$PS")
  ins=$(awk -F, '$3 ~ /instructions/{gsub(/[^0-9]/,"",$1);print $1}' "$PS")
  echo "$f_label,$trial,rg,$cyc,$ins" >> "$OUT"
}

echo "corpus,trial,arm,cycles,instructions" > "$OUT"
for c in silesia monorepo; do
  f=/root/$c.gz
  f_label=$c
  for trial in $(seq 1 "$N"); do
    run base  "$BASE"  "$f" "-d -c"
    run after "$AFTER" "$f" "-d -c"
    run_rg "$f"
  done
  echo "  done $c"
done

[ -c /dev/null ] || { echo "SINK_FAIL: /dev/null not a char device (after)"; exit 9; }
echo "CSV_WRITTEN=$OUT"
wc -l "$OUT"
echo "MEMCPY_CYC_DONE"
