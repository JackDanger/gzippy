import csv, collections, statistics
RAW={'silesia':211968000,'squishy':400391411}
rows=collections.defaultdict(lambda:collections.defaultdict(list))
with open('/tmp/ab_raw.csv') as f:
    for line in f:
        c,T,arm,r,w,cy,ins=line.strip().split(',')
        rows[(c,int(T))][arm].append((int(w),int(cy),int(ins)))
print(f"{'cell':<16}{'arm':<7}{'min_cyc/B':>11}{'med_wall_ms':>13}{'cyc_spread%':>12}{'wall_spread%':>13}")
verdicts=[]
for cell in sorted(rows):
    c,T=cell; raw=RAW[c]
    stat={}
    for arm in ('base','branch'):
        ws=[x[0] for x in rows[cell][arm]]
        cys=[x[1] for x in rows[cell][arm]]
        mincy=min(cys); cyB=mincy/raw
        medw=statistics.median(ws)/1e6
        cyspread=(max(cys)-min(cys))/min(cys)*100
        wspread=(max(ws)-min(ws))/min(ws)*100
        stat[arm]=(mincy,cyB,medw,cyspread,wspread)
        print(f"{c+' T'+str(T):<16}{arm:<7}{cyB:>11.3f}{medw:>13.1f}{cyspread:>12.2f}{wspread:>13.2f}")
    # delta on min cycles/byte (load-robust primary)
    bcy=stat['base'][1]; rcy=stat['branch'][1]
    dpct=(rcy-bcy)/bcy*100          # negative = branchless faster
    spread=max(stat['base'][3],stat['branch'][3])  # cyc inter-run spread
    if abs(dpct)<=spread: v='TIE'
    elif dpct<0: v='WIN(branchless)'
    else: v='REGRESSION'
    verdicts.append((f"{c} T{T}",dpct,spread,v))
    print(f"  -> Δcyc/B={dpct:+.2f}%  spread=±{spread:.2f}%  VERDICT={v}\n")
print("==== VERDICT TABLE (Δ = branchless vs base, cyc/byte) ====")
for n,d,s,v in verdicts:
    print(f"{n:<14} Δ={d:+6.2f}%  spread=±{s:.2f}%  {v}")
