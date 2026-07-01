#!/usr/bin/env python3
"""2-binary T1 A/B: paired Δ cyc/B + instr/B (CAND vs BASE), spread-gated verdict.
WIN if Δcyc/B median < -spread (CAND faster & significant); REGRESS if > +spread."""
import sys, csv, statistics as st
from collections import defaultdict
d=defaultdict(lambda:defaultdict(lambda:defaultdict(list))); rb={}
lines=[l for l in open(sys.argv[1]) if not l.lstrip().startswith('#')]
for r in csv.DictReader(lines):
    if not r.get('corpus'): continue
    c=r['corpus']; rb[c]=float(r['rawbytes'])
    d[c][r['arm']]['cyc'].append(float(r['cyc'])); d[c][r['arm']]['instr'].append(float(r['instr']))
def med(x): return st.median(x)
print(f"{'corpus':9} {'BASE c/B':>9} {'CAND c/B':>9} {'Δcyc%(med)':>10} {'pairedΔ% [lo,hi]':>20} {'Δinstr%':>8} {'spread%':>8}  verdict")
for c in d:
    b,a=d[c]['BASE'],d[c]['CAND']
    bc,ac=med(b['cyc'])/rb[c], med(a['cyc'])/rb[c]
    bi,ai=med(b['instr'])/rb[c], med(a['instr'])/rb[c]
    spread=(max(b['cyc'])-min(b['cyc']))/med(b['cyc'])*100
    dc=(ac-bc)/bc*100; di=(ai-bi)/bi*100
    # paired Δ% per rep (interleaved BASE_i,CAND_i): the proper significance test
    n=min(len(b['cyc']),len(a['cyc']))
    pd=sorted([(a['cyc'][i]-b['cyc'][i])/b['cyc'][i]*100 for i in range(n)])
    pmed=med(pd); plo,phi=pd[0],pd[-1]
    # significant if ALL paired deltas same sign (CI excludes 0 at the extremes)
    sig = (phi<0) or (plo>0)
    v=("WIN(paired)" if (sig and pmed<0) else ("REGRESS(paired)" if (sig and pmed>0)
       else ("WIN" if dc<-spread else ("REGRESS" if dc>spread else "TIE"))))
    print(f"{c:9} {bc:9.3f} {ac:9.3f} {dc:+10.2f} {('['+format(plo,'+.2f')+','+format(phi,'+.2f')+']'):>20} {di:+8.2f} {spread:8.2f}  {v}")
