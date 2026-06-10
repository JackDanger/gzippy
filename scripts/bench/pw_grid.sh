#!/bin/sh
# pw_grid.sh — UNFROZEN grid x3 for the per-worker retention lever (task step 4).
# Runs ON THE GUEST. OFF (GZIPPY_PW_RETAIN=0) vs ON (default) for /root/bin-pw.
# Cells: {model,bignasa,silesia}xT8 + silesiaxT4. /usr/bin/time -v for wall+RSS.
# One sha-verified run per (arm,corpus). Plus pgfault delta via /proc/vmstat on model T8.
set -u
BIN=/root/bin-pw
OUT=/dev/shm/pw_grid
mkdir -p "$OUT"

cell() { # $1=corpus $2=threads $3=arm(off|on)
  f=/root/$1.gz
  if [ "$3" = off ]; then export GZIPPY_PW_RETAIN=0; else unset GZIPPY_PW_RETAIN; fi
  for i in 1 2 3; do
    /usr/bin/time -v "$BIN" -d -c -p "$2" "$f" >/dev/null 2>"$OUT/$1-T$2-$3-r$i.time"
  done
  unset GZIPPY_PW_RETAIN
}

shaver() { # $1=corpus $2=threads $3=arm $4=expected_prefix
  f=/root/$1.gz
  if [ "$3" = off ]; then export GZIPPY_PW_RETAIN=0; else unset GZIPPY_PW_RETAIN; fi
  got=$("$BIN" -d -c -p "$2" "$f" 2>/dev/null | sha256sum | cut -c1-16)
  unset GZIPPY_PW_RETAIN
  if [ "$got" = "$4" ]; then echo "SHA-OK $1 T$2 $3 $got"; else echo "SHA-MISMATCH $1 T$2 $3 got=$got want=$4"; fi
}

pgf() { awk '/pgfault/ {print $2}' /proc/vmstat; }

# Interleave OFF/ON per cell to share thermal/cache state.
for c in "model 8" "bignasa 8" "silesia 8" "silesia 4"; do
  set -- $c
  cell "$1" "$2" off
  cell "$1" "$2" on
done

# sha verification: one run per (arm,corpus-cell)
shaver model 8 off 80521b40281d6ce7
shaver model 8 on  80521b40281d6ce7
shaver bignasa 8 off 255c34ef2e0fdefe
shaver bignasa 8 on  255c34ef2e0fdefe
shaver silesia 8 off 028bd002c89c9a90
shaver silesia 8 on  028bd002c89c9a90
shaver silesia 4 off 028bd002c89c9a90
shaver silesia 4 on  028bd002c89c9a90

# pgfault delta, model T8: process-level minflt/majflt already in time -v;
# also capture system pgfault around one run per arm.
for arm in off on; do
  if [ "$arm" = off ]; then export GZIPPY_PW_RETAIN=0; else unset GZIPPY_PW_RETAIN; fi
  b=$(pgf)
  "$BIN" -d -c -p 8 /root/model.gz >/dev/null 2>&1
  a=$(pgf)
  unset GZIPPY_PW_RETAIN
  echo "vmstat-pgfault model T8 $arm $((a - b))"
done

# summarize
echo "== SUMMARY (corpus threads arm: wall_s maxrss_kb minflt majflt) =="
for t in "$OUT"/*.time; do
  n=$(basename "$t" .time)
  w=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "$t")
  r=$(awk -F': ' '/Maximum resident/ {print $2}' "$t")
  mn=$(awk -F': ' '/Minor .*page faults/ {print $2}' "$t")
  mj=$(awk -F': ' '/Major .*page faults/ {print $2}' "$t")
  echo "$n: wall=$w rss=$r minflt=$mn majflt=$mj"
done
