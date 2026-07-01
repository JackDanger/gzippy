#!/usr/bin/env python3
"""Generate a btype01/stored-heavy payload (mirrors routing.rs make_btype01_heavy_data),
write raw to stdout. Pipe through `gzip -1` for fixed-Huffman-heavy gzip."""
import sys

size = int(sys.argv[1]) if len(sys.argv) > 1 else 40 * 1024 * 1024
phrases = [b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "]
out = bytearray()
rng = 0xB0BD1EC0DE
MASK = (1 << 64) - 1
while len(out) < size:
    rng = (rng * 6364136223846793005 + 1) & MASK
    if (rng >> 32) % 100 < 70:
        out.append((rng >> 16) & 0xFF)
    else:
        p = phrases[rng % len(phrases)]
        out += p[: min(len(p), size - len(out))]
del out[size:]
sys.stdout.buffer.write(bytes(out))
