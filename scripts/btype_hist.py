#!/usr/bin/env python3
"""Walk raw deflate blocks from a gzip file, histogram BTYPE for the first N blocks."""
import sys, struct

path = sys.argv[1]
maxblocks = int(sys.argv[2]) if len(sys.argv) > 2 else 200
d = open(path, "rb").read()

# parse gzip header
assert d[0] == 0x1F and d[1] == 0x8B
flg = d[3]
i = 10
if flg & 4:  # FEXTRA
    xlen = struct.unpack("<H", d[i : i + 2])[0]
    i += 2 + xlen
if flg & 8:  # FNAME
    while d[i] != 0:
        i += 1
    i += 1
if flg & 16:  # FCOMMENT
    while d[i] != 0:
        i += 1
    i += 1
if flg & 2:  # FHCRC
    i += 2

# bit reader over raw deflate
bitpos = i * 8
def getbit():
    global bitpos
    byte = d[bitpos >> 3]
    b = (byte >> (bitpos & 7)) & 1
    bitpos += 1
    return b
def getbits(n):
    v = 0
    for k in range(n):
        v |= getbit() << k
    return v

hist = {0: 0, 1: 0, 2: 0}
nblocks = 0
# Only stored blocks are easy to skip exactly; for fixed/dynamic we can't fully
# parse here, so we just read the first block's header of each only while stored.
# Limited walk: read BFINAL+BTYPE, and if stored skip its payload; else stop
# (we only need the type mix of the leading run, enough to confirm fixed != dynamic).
while nblocks < maxblocks and (bitpos >> 3) < len(d) - 8:
    bfinal = getbit()
    btype = getbits(2)
    if btype == 3:
        break
    hist[btype] += 1
    nblocks += 1
    if btype == 0:
        # align to byte, read LEN
        if bitpos & 7:
            bitpos += 8 - (bitpos & 7)
        ln = struct.unpack("<H", d[bitpos >> 3 : (bitpos >> 3) + 2])[0]
        bitpos += 32  # LEN + NLEN
        bitpos += ln * 8
    else:
        # cannot cheaply skip a coded block here; stop the walk
        break
    if bfinal:
        break

print(f"first-blocks BTYPE hist (stored-walkable prefix): stored={hist[0]} fixed={hist[1]} dynamic={hist[2]} (walked {nblocks})")
print(f"FIRST block: bfinal/btype from byte0=0x{d[i]:02x} -> btype={(d[i]>>1)&3}")
