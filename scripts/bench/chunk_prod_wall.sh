#!/usr/bin/env bash
# chunk_prod_wall.sh — T1 wall (/dev/null + tmpfs) for old vs new prod binary + igzip,
# and T>1 (T4/T8) wall+RSS IDENTITY check (new keeps 4 MiB at T>1 => must == old).
set -u
OLD=${OLD:-/root/bin/gzippy-t1prod}
NEW=${NEW:-/root/bin/gzippy-chunkt1}
IGZIP=${IGZIP:-/usr/bin/igzip}
PIN=${PIN:-4}
N=${N:-9}
M=${M:-5}
CORPORA=${CORPORA:-"nasa silesia"}
TMPF=/dev/shm/sink.out
echo "==== PROD WALL  old=$(sha256sum $OLD|cut -c1-12) new=$(sha256sum $NEW|cut -c1-12)  bestof=$N pin=cpu$PIN  $(date) ===="
echo "load:$(cat /proc/loadavg)"

bw() { # $1 bin $2 corpus $3 T $4 sink(null|tmpfs)
  local bin=$1 F=/root/$2.gz T=$3 sink=$4 best=99 i s e w
  for i in $(seq 1 $N); do
    s=$(date +%s.%N)
    if [ "$sink" = null ]; then taskset -c $PIN env GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p$T "$F" >/dev/null 2>&1
    else taskset -c $PIN env GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p$T "$F" >"$TMPF" 2>/dev/null; fi
    e=$(date +%s.%N); w=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.3f",b-a}')
    awk -v w=$w -v b=$best 'BEGIN{exit !(w<b)}' && best=$w
  done; echo "$best"; }
ig() { local F=/root/$1.gz sink=$2 best=99 i s e w
  for i in $(seq 1 $N); do s=$(date +%s.%N)
    if [ "$sink" = null ]; then taskset -c $PIN "$IGZIP" -d -c "$F" >/dev/null 2>&1; else taskset -c $PIN "$IGZIP" -d -c "$F" >"$TMPF" 2>/dev/null; fi
    e=$(date +%s.%N); w=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.3f",b-a}'); awk -v w=$w -v b=$best 'BEGIN{exit !(w<b)}' && best=$w; done; echo "$best"; }
rss() { local bin=$1 F=/root/$2.gz T=$3 best=99999999 i r
  for i in $(seq 1 $M); do r=$(GZIPPY_FORCE_PARALLEL_SM=1 /usr/bin/time -v "$bin" -d -c -p$T "$F" 2>&1 >/dev/null|awk '/Maximum resident/{print $NF}'); [ -n "$r" ] && awk -v r=$r -v b=$best 'BEGIN{exit !(r<b)}' && best=$r; done; echo "$best"; }

echo "--- T1 wall ---"
printf "%-10s %-8s %8s %8s\n" corpus arm null tmpfs
for c in $CORPORA; do
  o_n=$(bw "$OLD" $c 1 null); o_t=$(bw "$OLD" $c 1 tmpfs)
  n_n=$(bw "$NEW" $c 1 null); n_t=$(bw "$NEW" $c 1 tmpfs)
  i_n=$(ig $c null); i_t=$(ig $c tmpfs)
  printf "%-10s %-8s %8s %8s\n" "$c" old "$o_n" "$o_t"
  printf "%-10s %-8s %8s %8s\n" "$c" new "$n_n" "$n_t"
  printf "%-10s %-8s %8s %8s\n" "$c" igzip "$i_n" "$i_t"
  echo "   new-vs-igzip: null=$(awk -v g=$n_n -v i=$i_n 'BEGIN{printf "+%.0f%%",100*(g-i)/i}') tmpfs=$(awk -v g=$n_t -v i=$i_t 'BEGIN{printf "+%.0f%%",100*(g-i)/i}')  new-vs-old: null=$(awk -v n=$n_n -v o=$o_n 'BEGIN{printf "%+.1f%%",100*(n-o)/o}')"
done

echo "--- T>1 IDENTITY (T4/T8 wall+RSS old vs new; expect ~equal) ---"
printf "%-10s %-3s %-6s %8s %12s\n" corpus T arm wall rss_kB
for c in $CORPORA; do for T in 4 8; do
  printf "%-10s %-3s %-6s %8s %12s\n" "$c" "$T" old "$(bw $OLD $c $T null)" "$(rss $OLD $c $T)"
  printf "%-10s %-3s %-6s %8s %12s\n" "$c" "$T" new "$(bw $NEW $c $T null)" "$(rss $NEW $c $T)"
done; done
echo "DONE_PROD_WALL"
