#!/usr/bin/env python3
# Emit a gzip member whose deflate dynamic block OVER-SUBSCRIBES the litlen
# Huffman tree (257 symbols all at code-length 1). A correct decoder must
# return an error (InvalidCodeLengths / over-subscription), never crash/hang.
# This exercises the upstream Kraft screen (set_and_expand_lit_len_huffcode)
# which NIGHT39 preserves — the per-symbol guard it removes was downstream of it.
import sys, struct

class BW:
    def __init__(self): self.bits=[]
    def w(self, val, n):       # write n bits, LSB first (deflate order)
        for i in range(n): self.bits.append((val>>i)&1)
    def bytes(self):
        while len(self.bits)%8: self.bits.append(0)
        out=bytearray()
        for i in range(0,len(self.bits),8):
            b=0
            for j in range(8): b|=self.bits[i+j]<<j
            out.append(b)
        return bytes(out)

bw=BW()
bw.w(1,1)            # BFINAL=1
bw.w(2,2)            # BTYPE=10 dynamic
bw.w(257-257,5)      # HLIT  -> 257 litlen codes
bw.w(1-1,5)          # HDIST -> 1 dist code
bw.w(19-4,4)         # HCLEN -> 19 code-length codes
order=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
# Only CL symbol "1" gets length 1 (its canonical 1-bit code is "0").
for sym in order:
    bw.w(1 if sym==1 else 0, 3)
# Emit 258 (=257 litlen + 1 dist) code lengths, each = CL symbol "1" -> bit 0.
for _ in range(258):
    bw.w(0,1)        # 1-bit Huffman code "0" == CL symbol 1 == "code length 1"

deflate=bw.bytes()
hdr=b"\x1f\x8b\x08\x00"+struct.pack("<I",0)+b"\x00\x03"  # gzip header
trailer=struct.pack("<II",0,0)                            # bogus CRC+ISIZE (decode fails first)
open(sys.argv[1],"wb").write(hdr+deflate+trailer)
print(f"wrote {sys.argv[1]} ({len(hdr)+len(deflate)+len(trailer)} bytes, over-subscribed litlen tree)")
