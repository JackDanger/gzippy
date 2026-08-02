import sys
rows = [l.split() for l in open('l1sizes.txt').read().strip().split('\n')[1:]]
rivals = {'gzip': 3, 'pigz': 4, 'libdeflate': 5}
print(f"{'rival':<11} {'fail now':>9} {'fail after':>11} {'closed':>7} {'OPENED':>7}")
tot_now = tot_after = tot_closed = tot_open = 0
opened, closed = [], []
for name, idx in rivals.items():
    fn = fa = cl = op = 0
    for r in rows:
        f, ours, ldx = r[0], int(r[1]), int(r[2])
        riv = int(r[idx])
        now_fail, after_fail = ours > riv, ldx > riv
        fn += now_fail; fa += after_fail
        if now_fail and not after_fail: cl += 1; closed.append((name, f))
        if after_fail and not now_fail: op += 1; opened.append((name, f, ours, ldx, riv))
    print(f"{name:<11} {fn:>9} {fa:>11} {cl:>7} {op:>7}")
    tot_now += fn; tot_after += fa; tot_closed += cl; tot_open += op
print(f"{'TOTAL':<11} {tot_now:>9} {tot_after:>11} {tot_closed:>7} {tot_open:>7}")
print()
print("CELLS THAT WOULD OPEN (clause 3 is ABSOLUTE — any one of these blocks it):")
for name, f, ours, ldx, riv in opened:
    print(f"  {name:<11} {f:<22} ours={ours} -> ldx={ldx}  rival={riv}  excess={ldx-riv} ({100*(ldx-riv)/riv:.3f}%)")
print()
print("CELLS THAT WOULD CLOSE:")
for name, f in closed:
    print(f"  {name:<11} {f}")
