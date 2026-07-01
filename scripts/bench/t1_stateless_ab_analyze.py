#!/usr/bin/env python3
"""T1 stateless A/B: paired Δ cyc/B + instr/B (CAND=stateless vs BASE=run_contig).
CAND instr/B drop = non-inert proof the stateless path ran. cyc/B drop = the win."""
import sys, csv, statistics as st
from collections import defaultdict
d=defaultdict(lambda:defaultdict(lambda:defaultdict(list))); rb={}
lines=[l for l in open(sys.argv[1]) if not l.lstrip().startswith('#')]
for r in csv.DictReader(lines):
    if not r.get('corpus'): continue
    c=r['corpus']; rb[c]=float(r['rawbytes'])
    d[c][r['arm']]['cyc'].append(float(r['cyc'])); d[c][r['arm']]['instr'].append(float(r['instr']))
def med(x): return st.median(x)
for c in d:
    b,a=d[c]['BASE'],d[c]['CAND']
    bc,ac=med(b['cyc'])/rb[c],med(a['cyc'])/rb[c]
    bi,ai=med(b['instr'])/rb[c],med(a['instr'])/rb[c]
    spread=(max(b['cyc'])-min(b['cyc']))/med(b['cyc'])*100
    dcyc=(ac-bc)/bc*100; dins=(ai-bi)/bi*100
    v="WIN" if dcyc<-spread else ("REGRESS" if dcyc>spread else "TIE(<spread)")
    print(f"{c}: cyc/B {bc:.3f}->{ac:.3f} ({dcyc:+.2f}%)  instr/B {bi:.3f}->{ai:.3f} ({dins:+.2f}%)  spread={spread:.2f}%  => {v}")
    print(f"     non-inert: CAND instr/B {'DROPPED' if ai<bi-1e-6 else 'UNCHANGED (INERT? stateless path may not have run)'}")
