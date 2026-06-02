#!/usr/bin/env bash
# guest_ceiling_bench.sh — FOOTPRINT CEILING ORACLE whole-system A/B.
# ONE binary (target/release/gzippy on origin/feat/footprint-ceiling-oracle).
# Arms: BASELINE (knob OFF, sha-verified) vs CEILING (GZIPPY_FOOTPRINT_CEILING=1,
# sha-EXEMPT) vs rapidgzip. wall + maxRSS + minflt (file & pipe), interleaved N,
# then IPC perf pass. Plus the T1-T16 thread-sweep (gz/rg wall + IPC).
set -u
cd /root/gzippy || { echo RUN_TRUSTWORTHY=false; echo FAILURE=no-repo; exit 5; }
B=target/release/gzippy
C="${CORPUS:-benchmark_data/silesia-large.gz}"
N="${N:-11}"
RG="${RG:-rapidgzip}"
[ -x "$B" ] || { echo RUN_TRUSTWORTHY=false; echo FAILURE=no-bin; exit 8; }
REF_SHA="$(gzip -dc "$C" | sha256sum | cut -d' ' -f1)"
RAW="$(gzip -dc "$C" | wc -c)"
cat "$C" >/dev/null 2>&1
echo "## route: $(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 $B -d -c -p 8 "$C" 2>&1 >/dev/null | grep -m1 path=)"
echo "## isal_sym=$(nm -D $B 2>/dev/null | grep -c isal_inflate) rg=$($RG --version 2>&1|head -1) raw=$RAW N=$N"
echo "## REF_SHA=$REF_SHA"

pin(){ case "$1" in 1) echo 0;; 2) echo 0,2;; 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;; 16) echo 0-15;; *) echo "";; esac; }
median(){ echo "$1"|tr ' ' '\n'|grep -v '^$'|sort -n|awk '{v[NR]=$1}END{n=NR;if(n==0){print 0;exit}print (n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2}'; }
maxof(){ echo "$1"|tr ' ' '\n'|grep -v '^$'|sort -n|tail -1; }
minof(){ echo "$1"|tr ' ' '\n'|grep -v '^$'|sort -n|head -1; }

# timed <mask> <sink> <cmd...> -> "secs sha rss_kb minflt"
timed(){ local mask="$1" sink="$2"; shift 2; local tf; tf="$(mktemp)"; local s e sha
  if [ "$sink" = file ]; then local d; d="$(mktemp)"; s=$(date +%s.%N); taskset -c "$mask" /usr/bin/time -v "$@" >"$d" 2>"$tf"; e=$(date +%s.%N); sha=$(sha256sum "$d"|cut -d' ' -f1); rm -f "$d"
  else local o; o="$(mktemp)"; s=$(date +%s.%N); taskset -c "$mask" /usr/bin/time -v "$@" 2>"$tf" | sha256sum >"$o"; e=$(date +%s.%N); sha=$(cut -d' ' -f1 "$o"); rm -f "$o"; fi
  local secs rss mf; secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}'); rss=$(grep "Maximum resident" "$tf"|grep -oE "[0-9]+"|head -1); mf=$(grep "Minor" "$tf"|grep -oE "[0-9]+"|head -1); rm -f "$tf"; echo "$secs ${sha:0:12} ${rss:-0} ${mf:-0}"; }

# cmd_for <arm> <T> -> echoes the full argv (binary + args, env handled by caller)
# (functions can't be exec'd by /usr/bin/time, so we build the argv inline.)

DIV=0
echo ""; echo "===== CEILING A/B: WALL(med s) maxRSS(MB) minflt(max) — file & pipe, T8 & T16 ====="
printf "%-4s %-5s %-12s %-9s %-10s %-10s %-7s %s\n" T sink arm med_s rss_MB minflt MB/s rg_ratio
for sink in file pipe; do for T in 8 16; do
  m="$(pin "$T")"; declare -A W R F; for a in base ceil rg; do W[$a]=""; R[$a]=""; F[$a]=""; done
  for ((i=0;i<=N;i++)); do
    read bs bsh br bm < <(timed "$m" "$sink" env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p "$T" "$C")
    read cs csh cr cm < <(timed "$m" "$sink" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FOOTPRINT_CEILING=1 "$B" -d -c -p "$T" "$C")
    read rs rsh rr rm < <(timed "$m" "$sink" "$RG" -d -c -f -P "$T" "$C")
    [ "$i" -eq 0 ] && continue
    W[base]+=" $bs"; R[base]+=" $br"; F[base]+=" $bm"
    W[ceil]+=" $cs"; R[ceil]+=" $cr"; F[ceil]+=" $cm"
    W[rg]+=" $rs"; R[rg]+=" $rr"; F[rg]+=" $rm"
    if [ "$sink" = file ]; then
      [ "${bsh}" = "${REF_SHA:0:12}" ] || { echo "!!base DIV T$T i$i $bsh"; DIV=1; }
      [ "${rsh}" = "${REF_SHA:0:12}" ] || { echo "!!rg DIV T$T i$i $rsh"; DIV=1; }
      # ceil is sha-EXEMPT (correctness-breaking) — record only
    fi
  done
  rgm="$(median "${W[rg]}")"
  for a in base ceil rg; do
    md="$(median "${W[$a]}")"; rmb=$(awk -v k="$(maxof "${R[$a]}")" 'BEGIN{printf "%.0f",k/1024}'); mf="$(maxof "${F[$a]}")"
    mb=$(awk -v r="$RAW" -v t="$md" 'BEGIN{print(t>0)?sprintf("%.0f",r/t/1e6):"0"}')
    ra=$(awk -v x="$md" -v y="$rgm" 'BEGIN{printf "%.3f",(y>0)?x/y:0}')
    nm=$([ "$a" = base ]&&echo gz-base||([ "$a" = ceil ]&&echo gz-CEIL||echo rapidgzip))
    printf "%-4s %-5s %-12s %-9s %-10s %-10s %-7s %s\n" "$T" "$sink" "$nm" "$md" "$rmb" "$mf" "$mb" "$ra"
  done
done; done

echo ""; echo "===== IPC (instr/cycle) n=5 file sink, T8 & T16 ====="
EV=instructions,cycles
ipc(){ local mask="$1"; shift; local tf;tf="$(mktemp)"; taskset -c "$mask" perf stat -x, -e "$EV" -- "$@" >/dev/null 2>"$tf"; awk -F, '/instructions/{i=$1}/cycles/{c=$1}END{printf "%.3f",(c>0)?i/c:0}' "$tf"; rm -f "$tf"; }
printf "%-4s %-12s %s\n" T arm "IPC(med5)"
for T in 8 16; do m="$(pin "$T")"
  for a in base ceil rg; do v=""
    for ((i=0;i<5;i++)); do
      case "$a" in base) x=$(ipc "$m" env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p "$T" "$C");;
        ceil) x=$(ipc "$m" env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_FOOTPRINT_CEILING=1 "$B" -d -c -p "$T" "$C");;
        rg) x=$(ipc "$m" "$RG" -d -c -f -P "$T" "$C");; esac
      v+=" $x"; done
    nm=$([ "$a" = base ]&&echo gz-base||([ "$a" = ceil ]&&echo gz-CEIL||echo rapidgzip))
    printf "%-4s %-12s %s\n" "$T" "$nm" "$(median "$v")"
  done
done

echo ""; echo "===== THREAD-SWEEP T1-T16: gz-base vs rapidgzip (wall ratio + IPC), file sink, N>=7 ====="
printf "%-4s %-10s %-10s %-9s %-8s %-8s %s\n" T gz_med_s rg_med_s gz_ratio gz_IPC rg_IPC onset
NS=7
for T in 1 2 4 8 16; do m="$(pin "$T")"; gw=""; rw=""
  for ((i=0;i<=NS;i++)); do
    read bs bsh br bm < <(timed "$m" file env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p "$T" "$C")
    read rs rsh rr rm < <(timed "$m" file "$RG" -d -c -f -P "$T" "$C")
    [ "$i" -eq 0 ] && continue
    gw+=" $bs"; rw+=" $rs"
    [ "$bsh" = "${REF_SHA:0:12}" ] || { echo "!!sweep base DIV T$T i$i"; DIV=1; }
  done
  gm="$(median "$gw")"; rm2="$(median "$rw")"; ra=$(awk -v x="$gm" -v y="$rm2" 'BEGIN{printf "%.3f",(y>0)?x/y:0}')
  gi=""; ri=""
  for ((i=0;i<5;i++)); do gi+=" $(ipc "$m" env GZIPPY_FORCE_PARALLEL_SM=1 "$B" -d -c -p "$T" "$C")"; ri+=" $(ipc "$m" "$RG" -d -c -f -P "$T" "$C")"; done
  printf "%-4s %-10s %-10s %-9s %-8s %-8s %s\n" "$T" "$gm" "$rm2" "$ra" "$(median "$gi")" "$(median "$ri")" ""
done

echo ""; echo "diverged(base/rg only)=$DIV"
echo "RUN_TRUSTWORTHY=$([ "$DIV" = 0 ]&&echo true||echo false)"
