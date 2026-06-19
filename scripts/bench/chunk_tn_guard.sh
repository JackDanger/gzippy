#!/usr/bin/env bash
# chunk_tn_guard.sh — T>1 NO-REGRESSION guard for GZIPPY_CHUNK_KIB=$BKIB.
# best-of-N wall (/dev/null) + min-of-M maxRSS + byte-exact sha, T in $THREADS.
# A=default chunk, B=GZIPPY_CHUNK_KIB=$BKIB. SAME binary, env-only diff.
set -u
GZIPPY=${GZIPPY:-/root/bin/gzippy-t1prod}
BKIB=${BKIB:-1024}
N=${N:-9}
M=${M:-5}
CORPORA=${CORPORA:-"nasa silesia"}
THREADS=${THREADS:-"4 8"}
ENVB="GZIPPY_CHUNK_KIB=$BKIB"
echo "==== T>1 GUARD A=default B=$ENVB  bestof=$N rss=min$M  $(date) ===="
echo "GZIPPY=$GZIPPY sha=$(sha256sum "$GZIPPY"|cut -c1-12) load:$(cat /proc/loadavg)"

best_wall() { # $1=corpus $2=T $3=envprefix
  local F=/root/$1.gz T=$2 envp=$3 best=99 i s e w
  for i in $(seq 1 $N); do
    s=$(date +%s.%N)
    env $envp GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p$T "$F" >/dev/null 2>&1
    e=$(date +%s.%N)
    w=$(awk -v a=$s -v b=$e 'BEGIN{printf "%.3f", b-a}')
    awk -v w=$w -v b=$best 'BEGIN{exit !(w<b)}' && best=$w
  done
  echo "$best"
}
min_rss() { # $1 corpus $2 T $3 envp
  local F=/root/$1.gz T=$2 envp=$3 best=99999999 i r
  for i in $(seq 1 $M); do
    r=$(env $envp GZIPPY_FORCE_PARALLEL_SM=1 /usr/bin/time -v "$GZIPPY" -d -c -p$T "$F" 2>&1 >/dev/null | awk '/Maximum resident/{print $NF}')
    [ -n "$r" ] && awk -v r=$r -v b=$best 'BEGIN{exit !(r<b)}' && best=$r
  done
  echo "$best"
}

printf "%-10s %-3s %-8s %10s %12s %-10s\n" corpus T arm wall rss_kB sha
for corp in $CORPORA; do
  REF=$(zcat /root/$corp.gz|sha256sum|cut -c1-12)
  for T in $THREADS; do
    for arm in default B; do
      [ "$arm" = default ] && envp="" || envp="$ENVB"
      sha=$(env $envp GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p$T /root/$corp.gz 2>/dev/null|sha256sum|cut -c1-12)
      ok=$([ "$sha" = "$REF" ] && echo OK || echo MISMATCH)
      w=$(best_wall $corp $T "$envp")
      rss=$(min_rss $corp $T "$envp")
      printf "%-10s %-3s %-8s %10s %12s %-10s\n" "$corp" "$T" "$arm" "$w" "$rss" "$ok"
    done
  done
done
echo "DONE_CHUNK_TN_GUARD"
