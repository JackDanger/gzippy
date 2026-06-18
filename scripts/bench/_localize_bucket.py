#!/usr/bin/env python3
# Bucket perf-annotate self% into per-region cyc/B for gzippy run_contig and
# igzip decode_huffman kernel. Regions: a=bit-mgmt(consume+refill), b=copy,
# c=classify/decode/table/EOB, d=loop-overhead(guards/store/advance/spill).
# Self-validates: per-region % sums to ~100; prints leftover.
import re, sys

# ordered (start_addr, region); region applies until next start. END sentinel.
GZ = [
 (0xc3f60,'d'),(0xc3fcf,'d'),(0xc3fea,'c'),(0xc401c,'c'),(0xc4026,'a'),
 (0xc402e,'a'),(0xc404e,'c'),(0xc405c,'d'),(0xc4067,'a'),(0xc409a,'c'),
 (0xc40f8,'c'),(0xc4117,'c'),(0xc412d,'d'),(0xc4139,'a'),(0xc4141,'c'),
 (0xc414d,'d'),(0xc4153,'c'),(0xc415d,'d'),(0xc4169,'c'),(0xc4171,'d'),
 (0xc4173,'c'),(0xc4182,'c'),(0xc419b,'a'),(0xc41ad,'c'),(0xc41dc,'b'),
 (0xc41ec,'a'),(0xc4212,'c'),(0xc4220,'b'),(0xc427f,'d'),(0xc4287,'b'),
 (0xc4366,'d'),(0xc43d7,'END'),
]
IG = [
 (0x38be0,'d'),(0x38c61,'a'),(0x38c7c,'c'),(0x38c8a,'d'),(0x38c9c,'c'),
 (0x38d0e,'a'),(0x38d16,'d'),(0x38d1d,'c'),(0x38d32,'a'),(0x38d3d,'c'),
 (0x38d51,'c'),(0x38d56,'a'),(0x38d69,'c'),(0x38d7a,'c'),(0x38d87,'d'),
 (0x38d8c,'c'),(0x38dde,'a'),(0x38de6,'c'),(0x38df9,'c'),(0x38e06,'c'),
 (0x38e13,'a'),(0x38e16,'c'),(0x38e19,'b'),(0x38e2c,'b'),(0x38e79,'d'),
 (0x38ed4,'c'),(0x39193,'END'),
]
def region_of(addr, table):
    r=None
    for start,reg in table:
        if reg=='END':
            return r if addr<start else None
        if addr>=start: r=reg
        else: break
    return r

line_re=re.compile(r'^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):')
def bucket(path, table):
    acc={'a':0.0,'b':0.0,'c':0.0,'d':0.0}; tot=0.0; leftover=0.0
    for line in open(path):
        m=line_re.match(line)
        if not m: continue
        pct=float(m.group(1)); addr=int(m.group(2),16)
        tot+=pct
        reg=region_of(addr,table)
        if reg in acc: acc[reg]+=pct
        else: leftover+=pct
    return acc,tot,leftover

NAMES={'a':'bit-mgmt(consume+refill)','b':'backref-copy','c':'classify/decode/table/EOB','d':'loop-overhead(guard/store/spill)'}
# kernel cyc/B (kernel_self% * total_cyc / bytes)
KCB={ 'gz_silesia':4.889,'ig_silesia':3.981,'gz_nasa':1.843,'ig_nasa':1.390 }
TABLES={'gz':GZ,'ig':IG}
for corpus in ['silesia','nasa']:
    print(f"\n===== {corpus.upper()} — per-region cyc/B (kernel-to-kernel) =====")
    print(f"{'region':40s} {'gz %':>7s} {'ig %':>7s} {'gz c/B':>8s} {'ig c/B':>8s} {'gz-ig':>8s}")
    gza,gzt,gzl=bucket(f"/tmp/loc_prof/gz_{corpus}.annot",GZ)
    iga,igt,igl=bucket(f"/tmp/loc_prof/ig_{corpus}.annot",IG)
    gzk=KCB[f'gz_{corpus}']; igk=KCB[f'ig_{corpus}']
    for reg in ['a','b','c','d']:
        gzp=gza[reg]/gzt*100; igp=iga[reg]/igt*100
        gzc=gzp/100*gzk; igc=igp/100*igk
        print(f"{NAMES[reg]:40s} {gzp:7.1f} {igp:7.1f} {gzc:8.3f} {igc:8.3f} {gzc-igc:+8.3f}")
    print(f"{'KERNEL TOTAL':40s} {'':7s} {'':7s} {gzk:8.3f} {igk:8.3f} {gzk-igk:+8.3f}")
    print(f"  [self-validate] gz samples-sum%={gzt:.1f} leftover%={gzl:.2f} | ig sum%={igt:.1f} leftover%={igl:.2f}")
