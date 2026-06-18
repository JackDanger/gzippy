#!/usr/bin/env python3
# persym_corr.py — correlate perf-script sample IPs to objdump instructions for one
# function, bypassing the perf-annotate "no samples" bug on giant inline-asm symbols.
#
# Robust to PIE/ASLR: takes a list of (ip) for a SINGLE symbol (pre-filtered by the
# caller's `grep <symbol>`), anchors offset = ip - min(ip) (the symbol's first
# sampled instruction; the iteration-top is always hot so min ~= fn start), and joins
# with objdump of the function at its nm address. RECORD WITH setarch -R so every
# process shares one base — otherwise mixed ASLR bases corrupt the histogram.
#
# Usage: persym_corr.py <ip_file> <binary> <fn_addr_hex> <fn_size_hex> <label> [topN]
import sys, subprocess, collections

ipf, binary, fn_addr, fn_size, label = sys.argv[1:6]
topN = int(sys.argv[6]) if len(sys.argv) > 6 else 30
fa = int(fn_addr, 16); fs = int(fn_size, 16)

ips = []
for line in open(ipf):
    s = line.strip().split()
    if not s:
        continue
    try:
        ips.append(int(s[0], 16))
    except ValueError:
        continue
if not ips:
    print("no samples"); sys.exit(0)
mn = min(ips); mx = max(ips)
span = mx - mn
total = len(ips)
hist = collections.Counter(ip - mn for ip in ips)

# objdump the function by nm address, keyed by offset from fn start
od = subprocess.run(
    ["objdump", "-d", "-M", "intel",
     f"--start-address={fa}", f"--stop-address={fa+fs+16}", binary],
    capture_output=True, text=True).stdout
insn = {}
for ln in od.splitlines():
    ln = ln.strip()
    if ":" not in ln:
        continue
    head = ln.split(":", 1)[0]
    try:
        a = int(head, 16)
    except ValueError:
        continue
    rest = ln.split(":", 1)[1].strip()
    parts = rest.split("\t")
    insn[a - fa] = parts[-1].split("#")[0].strip() if parts else rest

print(f"== {label}: {total} samples, observed span 0x{span:x} (nm fn size 0x{fs:x}) ==")
if span > fs + 0x40:
    print(f"  !! WARN observed span 0x{span:x} > nm size 0x{fs:x}: min-anchor may be off / multi-base")
for offv, c in hist.most_common(topN):
    pct = 100.0 * c / total
    print(f"  {pct:5.2f}%  +0x{offv:<4x} {insn.get(offv,'(?)')}" )
