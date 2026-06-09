#!/usr/bin/env bash
# disasm_proof.sh — STEP 0 owed ISA-L Level-2 machine-code confirmation.
#
# Proves gzippy-isal links the AVX2 nasm igzip inflate kernel
# (decode_huffman_code_block_stateless_04) and compares it to rapidgzip's.
#
# The release gzippy binary is STRIPPED, so we cannot slice it by symbol. Strategy:
#   1. The ISA-L _04.o / _01.o objects in the cargo build dir RETAIN symbols — disasm
#      them and count AVX2/BMI2/SSE in the kernel body (proves it IS the AVX2 nasm
#      kernel, not the scalar C *_base fallback).
#   2. Prove LINKAGE despite stripping: extract a distinctive run of machine-code
#      bytes from the _04.o kernel and confirm the same bytes are present in the
#      stripped gzippy binary's .text.
#   3. rapidgzip's extension .so is NOT stripped of these statics — disasm its
#      stateless_04 the same way and histogram it.
#
# Read-only (objdump/grep); does NOT execute either binary. Safe on a live box.
set -uo pipefail

GZ_BIN="${GZ_BIN:-/root/gzippy-bench/target/release/gzippy}"
ISAL_O04="${ISAL_O04:-/root/gzippy-bench/target/release/build/isal-sys-0060809a15c21a4d/out/isa-l/igzip/igzip_decode_block_stateless_04.o}"
ISAL_O01="${ISAL_O01:-/root/gzippy-bench/target/release/build/isal-sys-0060809a15c21a4d/out/isa-l/igzip/igzip_decode_block_stateless_01.o}"
RG_SO="${RG_SO:-/usr/local/lib/python3.13/dist-packages/rapidgzip.cpython-313-x86_64-linux-gnu.so}"
OBJDUMP="${OBJDUMP:-objdump}"

AVX2_RE='\b(vp[a-z]+|vmov[a-z]*|vbroadcast|vperm|vpbroadcast|vinsert|vextract|vpand|vpor|vpxor|vpshuf|vpcmp|vpmovmsk|vpgather)\b'
BMI2_RE='\b(pext|pdep|bzhi|shlx|shrx|sarx|rorx|mulx|andn|blsr|blsi|tzcnt|lzcnt|bextr)\b'
SSE_RE='\b(movdqa|movdqu|movaps|movups|pshufb|pand|por|pxor|pcmpeq|pmovmskb|punpck)\b'

# disasm one symbol from a file that HAS symbols; histogram AVX2/BMI2/SSE.
histo_sym() { # <file> <symregex>
  local f="$1" pat="$2"
  "$OBJDUMP" -d --no-show-raw-insn "$f" 2>/dev/null \
   | awk -v pat="$pat" '
       /^[0-9a-f]+ <.*>:/ { name=$0; sub(/.*</,"",name); sub(/>:.*/,"",name);
                            inblk = (name ~ pat)?1:0; if(inblk) total=total; next }
       inblk && /^[[:space:]]*[0-9a-f]+:/ { sub(/^[^\t]*\t/,""); print }
     ' \
   | awk -v a="$AVX2_RE" -v b="$BMI2_RE" -v s="$SSE_RE" '
       { total++; if($0 ~ a) avx++; if($0 ~ b) bmi++; if($0 ~ s) sse++ }
       END { printf "total=%d avx2=%d bmi2=%d sse=%d\n", total+0, avx+0, bmi+0, sse+0 }'
}

echo "############ A. gzippy ISA-L kernel objects (symbols retained) ############"
echo "## _04.o symbols present:"; "$OBJDUMP" -t "$ISAL_O04" 2>/dev/null | grep -iE 'stateless|decode_huffman' | awk '{print "   "$NF}' | sort -u | head
echo "## decode_huffman_code_block_stateless_04 histogram (AVX2 kernel):"
echo -n "   "; histo_sym "$ISAL_O04" 'decode_huffman_code_block_stateless_04'
echo "## decode_huffman_code_block_stateless_01 histogram (SSE kernel):"
echo -n "   "; histo_sym "$ISAL_O01" 'decode_huffman_code_block_stateless_01'

echo ""
echo "############ B. LINKAGE proof — kernel bytes present in stripped gzippy ############"
# Grab a 24-byte run of raw machine code from deep inside the _04 kernel (skip the
# prologue), as a hex needle, then search the stripped binary for it.
NEEDLE="$("$OBJDUMP" -d "$ISAL_O04" 2>/dev/null \
  | awk '/<decode_huffman_code_block_stateless_04>:/{f=1;next} f&&/^[[:space:]]*[0-9a-f]+:/{n++; if(n>=40 && n<=51){ # bytes from a stable mid-body region
        line=$0; sub(/^[[:space:]]*[0-9a-f]+:[[:space:]]*/,"",line); sub(/[[:space:]]+[a-z].*/,"",line); printf "%s ", line } }
      f&&/^$/{exit}' | tr -d ' \t')"
echo "## needle (mid-kernel _04 bytes, hex): ${NEEDLE:0:48}..."
if [ -n "$NEEDLE" ]; then
  # binary-grep the needle in the gzippy .text
  if xxd -p "$GZ_BIN" 2>/dev/null | tr -d '\n' | grep -q "$NEEDLE"; then
    echo "## RESULT: AVX2 _04 kernel bytes FOUND in stripped gzippy binary => LINKED."
  else
    echo "## RESULT: needle NOT found verbatim (PIE/reloc may have rewritten an operand) — trying a shorter run"
    SHORT="${NEEDLE:0:32}"
    if xxd -p "$GZ_BIN" 2>/dev/null | tr -d '\n' | grep -q "$SHORT"; then
      echo "## RESULT(short): _04 kernel bytes FOUND in gzippy => LINKED."
    else
      echo "## RESULT(short): still not found — fall back to .a membership proof below."
    fi
  fi
fi
echo "## archive membership (libisal.a contains the _04 object the linker consumed):"
AR="$(dirname "$ISAL_O04")"; AR="$(cd "$AR/../.." && pwd)/.libs/libisal.a"
[ -f "$AR" ] && ar t "$AR" 2>/dev/null | grep -iE 'stateless_0[14]' | sed 's/^/   /' || echo "   (libisal.a not at expected path)"

echo ""
echo "############ C. rapidgzip .so ISA-L kernel ############"
echo "## rapidgzip stateless symbols present:"
nm "$RG_SO" 2>/dev/null | grep -iE 'stateless|decode_huffman' | awk '{print "   "$NF}' | sort -u | head
echo "## rapidgzip decode_huffman_code_block_stateless_04 histogram:"
echo -n "   "; histo_sym "$RG_SO" 'decode_huffman_code_block_stateless_04'
echo "## rapidgzip decode_huffman_code_block_stateless_01 histogram:"
echo -n "   "; histo_sym "$RG_SO" 'decode_huffman_code_block_stateless_01'

echo ""
echo "DISASM_PROOF_DONE"
