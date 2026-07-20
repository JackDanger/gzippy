#!/bin/bash
# Solvency frozen frontier runs: 3 corpora x {T8,T16}. no-orphan via fulcrum freeze run.
set -u
FR=/root/fulcrum-fr/target/release/fulcrum
GZ=/root/gzippy-inc6/target/release/gzippy
CD=/root/frontier-corpora
declare -A SHA=(
  [sil]=41ab303ad745caf086f0fd824d4b824a8eaf926187b41164db140d7e42a9fe28
  [text]=49cf128ffd5c5316975719d5ec4d4a66c6e054a361af483701fcd24564b5d8c8
  [bin]=29fad544cee0fc0e4cd1d1aaafda1253d15cb832355a6564a01b203e90c9570b
)
run_one() {
  local corpus="$1" threads="$2" pin="$3"
  local out=/root/frontier_solvency_${corpus}_T${threads}.json
  echo "############ RUN corpus=$corpus T=$threads pin=$pin ############"
  local S=$SECONDS
  $FR freeze run --procs 'llama-swap,llama-server' --ttl-s 1500 -- \
    $FR frontier \
      --ours gzippy --ours-cmd "$GZ -{level} -c -p {threads} {corpus}" \
      --ours-levels 1-9 \
      --rival "pigz=pigz -{level} -c -p {threads} {corpus}=1-9" \
      --rival "igzip=igzip -{level} -c {corpus}=0-3" \
      --rival "libdeflate=libdeflate-gzip -{level} -c {corpus}=1-9" \
      --corpus $CD/$corpus --threads $threads \
      --roundtrip-cmd "gzip -dc" --input-sha ${SHA[$corpus]} \
      --n 9 --coarse-reps 5 --size-reps 2 --sink /dev/null \
      --pin "$pin" --box solvency --out $out 2>&1 | \
      grep -E "FRONTIER=|FREEZE=|RESTORE=|curve=|OPEN|VOID|dropped|LEVEL-VOID|wrote" | tail -40
  echo "ELAPSED_SEC=$((SECONDS-S)) for $corpus T$threads"
  echo "LLAMA_STAT_AFTER: $(ps -o stat= -C llama-server | tr '\n' ' ')"
  echo
}
for c in sil text bin; do
  run_one "$c" 8 "0-7"
  run_one "$c" 16 "0-15"
done
echo "ALL_SOLVENCY_RUNS_DONE"
