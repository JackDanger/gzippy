#!/usr/bin/env bash
# PHASE 2/3 — instr-per-byte (B) + topdown memory/core-bound (C), gz vs rg.
# Frozen box. Pinned P-cores. cpu_core PMU. /dev/null sink. sha-spotcheck done.
set -u
B=/dev/shm/gz-b22-target/release/gzippy
RG=/root/oracle_c/rapidgzip-native
N="${N:-13}"
OUT=/dev/shm/redecode_perf.csv
TDOUT=/dev/shm/redecode_topdown.csv
WOUT=/dev/shm/redecode_wall.csv
test -c /dev/null || { echo "FAIL devnull-not-char"; exit 2; }
: > "$OUT"; : > "$TDOUT"; : > "$WOUT"
echo "corpus,T,tool,rep,instr,cycles" >> "$OUT"
echo "corpus,T,tool,mem_bound_pct,core_bound_pct" >> "$TDOUT"
echo "corpus,T,tool,rep,wall_s" >> "$WOUT"

mask_for() { case "$1" in 1) echo 0;; 4) echo 0,2,4,6;; esac; }
cmd_for() { # tool corpus T
  if [ "$1" = gz ]; then echo "env GZIPPY_FORCE_PARALLEL_SM=1 $B -d -c -p $3 $2";
  else echo "$RG -d -c -P $3 $2"; fi
}

# ---- Pass A: instructions + cycles, interleaved gz<->rg ----
for spec in "silesia /root/silesia.gz" "nasa /root/nasa.gz"; do
  set -- $spec; name="$1"; path="$2"
  for t in 1 4; do
    mask=$(mask_for "$t")
    for rep in $(seq 1 "$N"); do
      for tool in gz rg; do
        c=$(cmd_for "$tool" "$path" "$t")
        v=$(taskset -c "$mask" perf stat -x, -e cpu_core/instructions/,cpu_core/cycles/ -- $c >/dev/null 2>/tmp/ps.txt; \
            awk -F, '/instructions/{i=$1} /cycles/{cy=$1} END{print i" "cy}' /tmp/ps.txt)
        echo "$name,$t,$tool,$rep,${v% *},${v#* }" >> "$OUT"
      done
    done
  done
done

# ---- Pass B: wall, interleaved (no perf overhead) ----
for spec in "silesia /root/silesia.gz" "nasa /root/nasa.gz"; do
  set -- $spec; name="$1"; path="$2"
  for t in 1 4; do
    mask=$(mask_for "$t")
    for rep in $(seq 1 9); do
      for tool in gz rg; do
        c=$(cmd_for "$tool" "$path" "$t")
        s=$(date +%s.%N)
        taskset -c "$mask" $c >/dev/null 2>/dev/null
        e=$(date +%s.%N)
        echo "$name,$t,$tool,$rep,$(awk "BEGIN{print $e-$s}")" >> "$WOUT"
      done
    done
  done
done

# ---- Pass C: topdown L2 memory/core bound (median of 5) ----
for spec in "silesia /root/silesia.gz" "nasa /root/nasa.gz"; do
  set -- $spec; name="$1"; path="$2"
  for t in 1 4; do
    mask=$(mask_for "$t")
    for tool in gz rg; do
      c=$(cmd_for "$tool" "$path" "$t")
      for rep in 1 2 3 4 5; do
        taskset -c "$mask" perf stat -M tma_memory_bound,tma_core_bound -x, -- $c >/dev/null 2>/tmp/td.txt
        mb=$(awk -F, '/tma_memory_bound/{print $(NF-1)}' /tmp/td.txt | tail -1)
        cb=$(awk -F, '/tma_core_bound/{print $(NF-1)}' /tmp/td.txt | tail -1)
        echo "$name,$t,$tool,$mb,$cb" >> "$TDOUT"
      done
    done
  done
done

test -c /dev/null && echo "devnull_char_ok_after"
echo "DONE; OUT=$OUT TDOUT=$TDOUT WOUT=$WOUT"
wc -l "$OUT" "$TDOUT" "$WOUT"
