#!/bin/bash
# (1) High-N reconfirm of the decisive silesia T1 cell (cyc/byte).
# (2) Branch/IPC/bad-spec diagnostics at T8 (the branchless change's target metrics).
set -u
BASE=/root/bin-ab-base
BRANCH=/root/bin-ab-branchless
declare -A RAW=( [silesia]=211968000 [squishy]=400391411 )

echo "===== (1) silesia T1 high-N reconfirm (cyc/byte, interleaved) ====="
NB=""; NR=""
for r in $(seq 1 21); do
  taskset -c 2 perf stat -x, -e cycles -- taskset -c 2 $BASE   -d -c -p1 /root/silesia.gz >/dev/null 2>/tmp/a
  cb=$(awk -F, '/cpu_core\/cycles\//{print $1}' /tmp/a); NB="$NB $cb"
  taskset -c 2 perf stat -x, -e cycles -- taskset -c 2 $BRANCH -d -c -p1 /root/silesia.gz >/dev/null 2>/tmp/a
  cr=$(awk -F, '/cpu_core\/cycles\//{print $1}' /tmp/a); NR="$NR $cr"
done
echo "base cyc:$NB"
echo "branch cyc:$NR"
echo "$NB|$NR|${RAW[silesia]}" | awk -F'|' '{
  n=split($1,b," "); split($2,r," "); raw=$3;
  bmin=1e18;rmin=1e18;bmax=0;rmax=0;
  for(i=1;i<=n;i++){if(b[i]<bmin)bmin=b[i];if(b[i]>bmax)bmax=b[i];if(r[i]<rmin)rmin=r[i];if(r[i]>rmax)rmax=r[i]}
  bcyB=bmin/raw; rcyB=rmin/raw; d=(rcyB-bcyB)/bcyB*100;
  bsp=(bmax-bmin)/bmin*100; rsp=(rmax-rmin)/rmin*100; sp=(bsp>rsp?bsp:rsp);
  printf "  base min cyc/B=%.3f (spread %.2f%%)  branch min cyc/B=%.3f (spread %.2f%%)\n",bcyB,bsp,rcyB,rsp;
  v=(d<0?(-d<=sp?"TIE":"WIN"):(d<=sp?"TIE":"REGRESSION"));
  printf "  Δ=%+.2f%%  spread=±%.2f%%  VERDICT=%s\n",d,sp,v;
}'

echo; echo "===== (2) T8 diagnostics: IPC / bad-spec / branch-miss-per-byte (r=9 each) ====="
diag() { # $1=bin $2=corpus $3=tag
  local bin="$1" c="$2" tag="$3" pin=0,2,4,6,8,10,12,14 raw=${RAW[$2]}
  taskset -c $pin perf stat -r9 -x, -e instructions,cycles,branches,branch-misses \
    -- taskset -c $pin "$bin" -d -c -p8 /root/$c.gz >/dev/null 2>/tmp/s
  taskset -c $pin perf stat -r9 -M TopdownL1 \
    -- taskset -c $pin "$bin" -d -c -p8 /root/$c.gz >/dev/null 2>/tmp/td
  local I C B M
  I=$(awk -F, '/cpu_core\/instructions\//{print $1}' /tmp/s)
  C=$(awk -F, '/cpu_core\/cycles\//{print $1}' /tmp/s)
  B=$(awk -F, '/cpu_core\/branches\//{print $1}' /tmp/s)
  M=$(awk -F, '/cpu_core\/branch-misses\//{print $1}' /tmp/s)
  BS=$(grep tma_bad_speculation /tmp/td | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  RT=$(grep tma_retiring /tmp/td | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %')
  awk -v t="$tag" -v i=$I -v c=$C -v b=$B -v m=$M -v raw=$raw -v bs="$BS" -v rt="$RT" 'BEGIN{
    printf "  %-22s IPC=%.3f  bmiss/KB=%.3f  bmiss%%ofbr=%.3f  badspec=%s%%  retiring=%s%%\n",
      t, c?i/c:0, m/raw*1000, b?m/b*100:0, bs, rt }'
}
for c in silesia squishy; do
  diag "$BASE"   "$c" "$c base"
  diag "$BRANCH" "$c" "$c branch"
done
echo DONE
