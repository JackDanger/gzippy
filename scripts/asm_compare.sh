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

# --- STRUCTURAL TELLS: the instructions that explain a 2-6x COMPUTE gap ---
# (added for the bootstrap-vs-clean decoder analysis: the slow deflate_block
#  marker loop vs the fast clean loop. The diff here IS the lever.)
echo
echo "=== STRUCTURAL TELLS (usual 2-6x culprits — diff A vs B) ==="
tells() { local f="$1"; printf "u16store(movw)=%s ring-mask(and 0xffff..)=%s idiv=%s shifts(modulo?)=%s branches=%s vector(xmm/ymm)=%s BMI2(pext/bzhi)=%s scalar-load(movz)=%s" \
  "$(grep -cE 'mov[[:space:]]+word|[[:space:]]movw' "$f")" "$(grep -cE '\band[[:space:]].*0x(ffff|1ffff|3ffff|7fff)' "$f")" \
  "$(grep -cE '[[:space:]]idiv' "$f")" "$(grep -cE '[[:space:]](shl|shr|sar)[[:space:]]' "$f")" \
  "$(grep -cE ':[[:space:]]+j[a-z]' "$f")" "$(grep -cE 'xmm|ymm' "$f")" \
  "$(grep -cE 'pext|pdep|bzhi|shlx|shrx' "$f")" "$(grep -cE '[[:space:]]movz' "$f")"; }
printf "  A %-30s %s\n" "$A" "$(tells "$fa")"
printf "  B %-30s %s\n" "$B" "$(tells "$fb")"
echo "  => the per-store ring-mask 'and' is a SERIAL dependency on every write (latency chain);"
echo "     movw is 2x store bandwidth; missing xmm/BMI2 in A vs B = missed vectorization. Fix the diff."

# --- deeper microarch (run these on the box; llvm-mca/perf give port-pressure + critical-path) ---
MCA=$(command -v llvm-mca 2>/dev/null || command -v llvm-mca-19 2>/dev/null || command -v llvm-mca-16 2>/dev/null || true)
echo
if [ -n "${MCA:-}" ]; then
  echo "=== llvm-mca port-pressure + critical-path (re-emit Intel asm for it) ==="
  echo "  RUSTFLAGS='-C target-cpu=native' cargo asm --release --intel <fn> | $MCA -mcpu=native | grep -iE 'Cycles|uOps|IPC|RThroughput|Bottleneck|critical'"
else
  echo "=== llvm-mca MISSING (the precise port/critical-path tool): apt-get install -y llvm ==="
fi
echo "=== live per-instruction cycles (the stalling instruction IS the lever): ==="
echo "  RUSTFLAGS='-C target-cpu=native -C debuginfo=1 -C strip=none' cargo build --release --no-default-features --features pure-rust-inflate --example inner_bench"
echo "  perf record -e cycles:u -- target/release/examples/inner_bench bootstrap; perf annotate --stdio $A"

rm -f "$DIS" "$fa" "$fb"
