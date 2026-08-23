import json,gzip,sys,subprocess,bisect,collections,re
prof,binp=sys.argv[1],sys.argv[2]
raw=open(prof,'rb').read()
if raw[:2]==b'\x1f\x8b': raw=gzip.decompress(raw)
d=json.loads(raw)
counts=collections.Counter()
for t in d["threads"]:
    st,ft,fnt,strs=t["stackTable"],t["frameTable"],t["funcTable"],t["stringArray"]
    for s in (t["samples"].get("stack") or []):
        if s is None: continue
        counts[strs[fnt["name"][ft["func"][st["frame"][s]]]]]+=1
out=subprocess.run(["nm","-n","--defined-only",binp],capture_output=True,text=True).stdout
syms=[]
for l in out.splitlines():
    p=l.split(None,2)
    if len(p)==3 and p[1].lower()=="t":
        try: syms.append((int(p[0],16),p[2]))
        except ValueError: pass
syms.sort(); addrs=[a for a,_ in syms]
def dem(n):
    n=re.sub(r'17h[0-9a-f]{16}E?$','',n.lstrip('_')); n=re.sub(r'^ZN','',n)
    parts=re.findall(r'(\d+)([A-Za-z0-9_$\.]+)',n)
    return "::".join(p[1] for p in parts) if parts else n
agg=collections.Counter(); tot=sum(counts.values()) or 1
for name,c in counts.items():
    try: va=0x100000000+int(name,16)
    except ValueError: agg[name]+=c; continue
    i=bisect.bisect_right(addrs,va)-1
    agg[dem(syms[i][1]) if i>=0 else "?"]+=c
print(f"self time ({tot} samples)")
for n,c in agg.most_common(10):
    print(f"  {100*c/tot:5.1f}%  {n[:96]}")
