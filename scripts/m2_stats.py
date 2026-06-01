#!/usr/bin/env python3
# Parse "mode/cell: t1 t2 ..." lines from m2_factorial_devnull.sh output and
# compute per-cell median/sd, the three MAIN EFFECTS (decode/consumer/resolve),
# all PAIRWISE interactions, and the 3-way interaction, separately for spin and
# sleep. Reads stdin or a file arg.
import sys, statistics as st

cells = {}  # (mode,cell) -> [times]
for line in (open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin):
    line = line.strip()
    if "/" in line and ":" in line and line[:4] in ("spin", "slee"):
        head, rest = line.split(":", 1)
        mode, cell = head.split("/")
        vals = [float(x) for x in rest.split()]
        if vals:
            cells[(mode, cell)] = vals

def med(m, c):
    return st.median(cells[(m, c)])

def sd(m, c):
    v = cells[(m, c)]
    return st.pstdev(v) if len(v) > 1 else 0.0

# 8 cells: 000 B00 0C0 00R BC0 B0R 0CR BCR
order = ["000", "B00", "0C0", "00R", "BC0", "B0R", "0CR", "BCR"]
for mode in ("spin", "sleep"):
    if (mode, "000") not in cells:
        continue
    print(f"\n===== MODE={mode} (/dev/null sink) =====")
    print(f"{'cell':<5}{'median_s':>10}{'sd_s':>9}{'sd%':>7}")
    base = med(mode, "000")
    pooled = []
    for c in order:
        if (mode, c) not in cells:
            continue
        m, s = med(mode, c), sd(mode, c)
        pooled.append(s)
        print(f"{c:<5}{m:>10.4f}{s:>9.4f}{(100*s/m if m else 0):>6.1f}%")
    noise = st.mean(pooled) if pooled else 0
    print(f"pooled_per_cell_sd(noise_band) = {1000*noise:.1f} ms")

    def d(c):  # delta vs baseline in ms
        return 1000 * (med(mode, c) - base)

    # MAIN EFFECTS (single-knob vs baseline)
    print("-- MAIN EFFECTS (vs 000 baseline) --")
    print(f"  decode   (B00-000): {d('B00'):+8.1f} ms  ({100*d('B00')/(1000*base):+5.1f}%)")
    print(f"  consumer (0C0-000): {d('0C0'):+8.1f} ms  ({100*d('0C0')/(1000*base):+5.1f}%)")
    print(f"  resolve  (00R-000): {d('00R'):+8.1f} ms  ({100*d('00R')/(1000*base):+5.1f}%)")

    # PAIRWISE INTERACTIONS: I(XY) = (XY) - (X0) - (0Y) + (00)
    # using ms-deltas: int = d(XY) - d(X) - d(Y)
    def inter2(both, a, b):
        return d(both) - d(a) - d(b)
    print("-- PAIRWISE INTERACTIONS  I = coupled - sum(independent)  (|I| < noise => INDEPENDENT) --")
    print(f"  decode x consumer (BC0): {inter2('BC0','B00','0C0'):+8.1f} ms")
    print(f"  decode x resolve  (B0R): {inter2('B0R','B00','00R'):+8.1f} ms")
    print(f"  consumer x resolve(0CR): {inter2('0CR','0C0','00R'):+8.1f} ms")

    # 3-WAY: I3 = d(BCR) - d(B00)-d(0C0)-d(00R)
    #              + (pairwise corrections cancel in the delta-form sum) ->
    # exact 3-way contrast on raw cell medians:
    # I3 = BCR - BC0 - B0R - 0CR + B00 + 0C0 + 00R - 000
    def M(c):
        return med(mode, c)
    i3 = 1000*(M('BCR') - M('BC0') - M('B0R') - M('0CR') + M('B00') + M('0C0') + M('00R') - M('000'))
    print(f"-- 3-WAY INTERACTION (decode x consumer x resolve): {i3:+.1f} ms --")
