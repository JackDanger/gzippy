#!/usr/bin/env bash
# Side-by-side machine-code comparison of two symbols in a binary — built to
# diff OUR inflate hot loop against libdeflate's (both are statically linked
# into examples/inner_bench, so they live in the same binary and were compiled
# for the same target-cpu). Answers "where do we emit more instructions per
# symbol than libdeflate" at the actual-machine-code level, not just counts.
#
# Usage (on the bench host, x86_64):
#   cargo build --release --no-default-features --features pure-rust-inflate --example inner_bench
#   bash scripts/asm_compare.sh target/release/examples/inner_bench \
#        decode_huffman_libdeflate_style deflate_decompress_bmi2
#
# Default symbols compare our consume_first hot loop vs libdeflate's BMI2 path.
set -u
BIN="${1:?binary path}"
A="${2:-decode_huffman_libdeflate_style}"   # ours
B="${3:-deflate_decompress_bmi2}"           # libdeflate (BMI2 target variant)

DIS=$(mktemp)
objdump -d -C --no-show-raw-insn "$BIN" > "$DIS" 2>/dev/null

extract() { awk -v s="$1" 'index($0,s)&&/>:$/{f=1;next} f&&/^[0-9a-f]+ <.*>:$/{exit} f' "$DIS"; }
count()   { grep -cE '^[[:space:]]+[0-9a-f]+:' ; }
opmix()   { grep -oE ':[[:space:]]+[a-z][a-z0-9]+' | awk '{print $NF}' | sort | uniq -c | sort -rn; }

fa=$(mktemp); fb=$(mktemp)
extract "$A" > "$fa"; extract "$B" > "$fb"
na=$(count < "$fa"); nb=$(count < "$fb")

echo "=================================================================="
printf "  %-40s %d static instrs\n" "$A" "$na"
printf "  %-40s %d static instrs\n" "$B" "$nb"
echo "=================================================================="
echo "OPCODE                  OURS   LIBDEFLATE   ratio"
# union of top opcodes
join -a1 -a2 -e0 -o '0,1.2,2.2' \
  <(opmix < "$fa" | awk '{print $2" "$1}' | sort -k1,1) \
  <(opmix < "$fb" | awk '{print $2" "$1}' | sort -k1,1) 2>/dev/null \
  | sort -k2 -rn | awk '$2+$3>0{r=($3>0)?$2/$3:99; printf "%-18s %8d %10d   %5.2fx\n",$1,$2,$3,r}' | head -24

rm -f "$DIS" "$fa" "$fb"
