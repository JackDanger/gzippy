#!/usr/bin/env python3
"""Generate two extreme corpora to bound the backref-COPY vs Huffman-DECODE
split of the aarch64 clean-decode-core instr/B (DELIVERABLE 1).

- literal-heavy (random bytes): ~1 symbol per output byte, ~0 backref copy.
  instr/B here is dominated by Huffman-decode + bitreader (per-SYMBOL cost).
- backref-heavy (long repeats): ~1 symbol per ~258 output bytes, ~all bytes
  produced by the backref word-copy. instr/B here exposes the per-COPY-BYTE
  cost (Huffman/bitreader amortized ~258x).

Both are gzipped with the system gzip (-6) so the DEFLATE block/table
structure is realistic. Sizes kept modest (instr/B is a ratio, size-neutral).
"""
import os
import random
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
N = 32 * 1024 * 1024  # 32 MiB uncompressed each

random.seed(0xC0FFEE)

# literal-heavy THROUGH HUFFMAN: random bytes over a small (16-symbol)
# alphabet. Skewed entropy → gzip emits DYNAMIC-Huffman blocks (NOT stored),
# but random order → almost no backrefs > chance, so ~every output byte is a
# literal SYMBOL decoded through the Huffman/bitreader path.
lit = os.path.join(OUT, "decomp_literal.bin")
mask4 = bytes(b & 0x0F for b in range(256))  # 256-entry translate table → alphabet 16
with open(lit, "wb") as f:
    f.write(random.randbytes(N).translate(mask4))

# backref-heavy: a tiny seed repeated → long matches, few literals
rep = os.path.join(OUT, "decomp_backref.bin")
seed = random.randbytes(64)  # 64-byte motif, repeats fill 32 MiB
with open(rep, "wb") as f:
    reps = N // len(seed)
    f.write(seed * reps)
    f.write(seed[: N - reps * len(seed)])

for src in (lit, rep):
    dst = src + ".gz"
    with open(dst, "wb") as o:
        subprocess.run(["gzip", "-6", "-c", src], stdout=o, check=True)
    print(f"{dst}  uncompressed={N}  compressed={os.path.getsize(dst)}")
    os.remove(src)
