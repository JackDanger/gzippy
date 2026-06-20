#!/usr/bin/env python3
import csv, sys, statistics as st
from collections import defaultdict

# Uncompressed byte counts (guest corpora). Override if corpora change.
SIZE = {"silesia":211968000, "monorepo":50915328, "nasa":205242368}
CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ixv_perf.csv"
I = defaultdict(list); C = defaultdict(list)
with open(CSV) as f:
    for r in csv.DictReader(f):
        k=(r["test"],r["corpus"],r["arm"])
        I[k].append(int(r["instructions"])); C[k].append(int(r["cycles"]))

def md_sp(v):
    m=st.median(v); return m,(max(v)-min(v))/m*100

def block(test, aArm, bArm, aLbl, bLbl, winlabel):
    print("="*94); print(test); print("="*94)
    for c in ["silesia","monorepo","nasa"]:
        sz=SIZE[c]
        Ai,Asi=md_sp(I[(test[0],c,aArm)]); Bi,Bsi=md_sp(I[(test[0],c,bArm)])
        A2i,_=md_sp(I[(test[0],c,"A2_self")])
        Ac,_=md_sp(C[(test[0],c,aArm)]); Bc,_=md_sp(C[(test[0],c,bArm)])
        ri=Bi/Ai; rc=Bc/Ac; selft=A2i/Ai
        eff=(ri-1)*100; sp=Asi+Bsi
        if abs(eff)<=sp: v="TIE (Δ<=spread)"
        elif eff>0: v=winlabel
        else: v="B WINS"
        print(f"[{c}] out={sz/1e6:.0f}MB")
        print(f"  A {aLbl:22s} instr/B={Ai/sz:7.3f} cyc/B={Ac/sz:6.3f} (spread {Asi:.2f}%)")
        print(f"  B {bLbl:22s} instr/B={Bi/sz:7.3f} cyc/B={Bc/sz:6.3f} (spread {Bsi:.2f}%)")
        print(f"  >> instr B/A={ri:.4f}x  Δ={eff:+.2f}% vs spread {sp:.2f}% -> {v}")
        print(f"  >> cyc   B/A={rc:.4f}x  [NOISY: unfrozen LXC, no_turbo=0/powersave — context only]   selftest A2/A={selft:.5f}")
    print()

block(("D2","engine A (flat,asm-off) vs engine B (two-level) — GZIPPY_FLAT_CLEAN kill-switch, SAME binary"),
      "A_engineA","B_engineB","engineA","engineB","engineA WINS")
block(("D3","engine A (flat pure-Rust, asm-OFF) vs run_contig (x86 BMI2 ASM, asm-ON)  [PIVOTAL SOLE-PATH]"),
      "A_engineA_asmoff","B_runcontig_asmon","engineA(asmoff)","run_contig(asmon)","engineA WINS (asm RETIRABLE on instr)")
