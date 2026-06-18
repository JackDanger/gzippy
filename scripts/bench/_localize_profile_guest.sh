#!/bin/bash
# Localize the ~2.1 cyc/B per-iteration machinery gap: per-region cyc/B for
# gzippy run_contig vs igzip decode_huffman_code_block_stateless_04.
# Records perf data + total cycles for both tools on silesia+nasa, pinned cpu4.
# Self-validating: prints kernel self%, sha match, sample counts, GHz.
set -e
CORE=4
OUT=/tmp/loc_prof
rm -rf "$OUT"; mkdir -p "$OUT"
LOOPS=${LOOPS:-10}
GZBIN=/tmp/symtarget/release/gzippy
SYM_GZ="gzippy::decompress::parallel::asm_kernel::imp::run_contig"
SYM_IG="decode_huffman_code_block_stateless_04"

# raise sample rate for this run; restore at end
OLD_RATE=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
echo 5000 > /proc/sys/kernel/perf_event_max_sample_rate || true
trap 'echo "$OLD_RATE" > /proc/sys/kernel/perf_event_max_sample_rate 2>/dev/null || true' EXIT

for corpus in silesia nasa; do
  IN=/root/$corpus.gz
  BYTES=$(zcat "$IN" | wc -c)
  echo "=== CORPUS=$corpus BYTES=$BYTES ==="

  # sha self-validation (both tools == zcat)
  REF=$(zcat "$IN" | sha256sum | cut -d' ' -f1)
  GZSHA=$(GZIPPY_FORCE_PARALLEL_SM=1 $GZBIN -d -c -p1 "$IN" 2>/dev/null | sha256sum | cut -d' ' -f1)
  IGSHA=$(igzip -d -c "$IN" 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "SHA ref=$REF"
  echo "SHA gz =$GZSHA $([ "$GZSHA" = "$REF" ] && echo MATCH || echo MISMATCH)"
  echo "SHA ig =$IGSHA $([ "$IGSHA" = "$REF" ] && echo MATCH || echo MISMATCH)"

  # ---- gzippy: total cycles + record ----
  taskset -c $CORE perf stat -x, -r 3 -e cpu_core/cycles/,cpu_core/instructions/ -- \
    bash -c "GZIPPY_FORCE_PARALLEL_SM=1 $GZBIN -d -c -p1 $IN >/dev/null" 2> "$OUT/gz_$corpus.stat"
  taskset -c $CORE perf record -F5000 -e cpu_core/cycles/ -o "$OUT/gz_$corpus.data" -- \
    bash -c "for i in \$(seq 1 $LOOPS); do GZIPPY_FORCE_PARALLEL_SM=1 $GZBIN -d -c -p1 $IN >/dev/null; done" 2>/dev/null
  perf report -i "$OUT/gz_$corpus.data" --stdio -n 2>/dev/null | grep -vE "^#|^$" | head -6 > "$OUT/gz_$corpus.report"
  perf annotate -i "$OUT/gz_$corpus.data" --stdio -s "$SYM_GZ" 2>/dev/null > "$OUT/gz_$corpus.annot"

  # ---- igzip: total cycles + record ----
  taskset -c $CORE perf stat -x, -r 3 -e cpu_core/cycles/,cpu_core/instructions/ -- \
    bash -c "igzip -d -c $IN >/dev/null" 2> "$OUT/ig_$corpus.stat"
  taskset -c $CORE perf record -F5000 -e cpu_core/cycles/ -o "$OUT/ig_$corpus.data" -- \
    bash -c "for i in \$(seq 1 $LOOPS); do igzip -d -c $IN >/dev/null; done" 2>/dev/null
  perf report -i "$OUT/ig_$corpus.data" --stdio -n 2>/dev/null | grep -vE "^#|^$" | head -6 > "$OUT/ig_$corpus.report"
  perf annotate -i "$OUT/ig_$corpus.data" --stdio -s "$SYM_IG" 2>/dev/null > "$OUT/ig_$corpus.annot"

  echo "--- gz report ---"; cat "$OUT/gz_$corpus.report"
  echo "--- ig report ---"; cat "$OUT/ig_$corpus.report"
done
echo "ANNOT FILES:"; ls -la "$OUT"/*.annot
echo "DONE"
