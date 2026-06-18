#!/usr/bin/env python3
# Gate-2 perturbation: inject K serially-DEPENDENT, value-NEUTRAL ops
# (rol/ror pairs on {bitbuf}) right after each refill's `or {bitbuf}, {t2}`,
# extending the bitbuf-merge -> preload/consume recurrence. If cyc/byte rises
# proportionally, the refill/bit-mgmt chain is on the critical recurrence.
import sys
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 4   # K = 2*PAIRS dependent ops
path = sys.argv[1]
src = open(path).read()
anchor = '                "or {bitbuf}, {t2}",\n'
inj = ''.join('                "rol {bitbuf}, 1",\n                "ror {bitbuf}, 1",\n' for _ in range(PAIRS))
n = src.count(anchor)
assert n >= 1, f"anchor not found in {path}"
src2 = src.replace(anchor, anchor + inj)
open(path, 'w').write(src2)
print(f"patched {n} refill site(s); injected {2*PAIRS} dep ops per site (PAIRS={PAIRS})")
