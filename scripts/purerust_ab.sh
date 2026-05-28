#!/usr/bin/env bash
# Rigorous interleaved A/B for the pure-rust decoder vs isal vs rapidgzip.
#
# Per plans/rust-rapidgzip.md §4: n>=20, per-iter MEDIAN, interleaved
# (alternate the three tools every trial so thermal/clock drift cancels),
# trial-1 reported separately, load<4 gate. Wall pass first; a short
# perf-stat pass (counters are load-independent) reports cycles/IPC/faults.
#
# Usage:
#   bash scripts/purerust_ab.sh <fixture.gz> <raw_bytes> [trials] [threads]
set -euo pipefail

FIXTURE="${1:?fixture.gz}"
RAW="${2:?raw bytes}"
TRIALS="${3:-20}"
THREADS="${4:-$(nproc)}"
BIN=/tmp/bench-bin
RG=/root/gzippy/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip

# ── load gate ────────────────────────────────────────────────────────────
LOAD=$(awk '{print int($1)}' /proc/loadavg)
echo "load(1m)=$(cut -d' ' -f1 /proc/loadavg)  threads=$THREADS  trials=$TRIALS  raw=${RAW}B"
if [ "$LOAD" -ge 4 ]; then
  echo "WARN: load >= 4 — wall numbers are NOT trustworthy. Counters still valid."
fi

declare -A CMD
CMD[purerust]="$BIN/gzippy-purerust -d -c -p $THREADS $FIXTURE"
CMD[isal]="$BIN/gzippy-isal -d -c -p $THREADS $FIXTURE"
CMD[rapidgzip]="$RG -d -P $THREADS -c $FIXTURE"
ORDER=(purerust isal rapidgzip)

# ── correctness: all three must agree byte-for-byte ───────────────────────
H=$(${CMD[purerust]} 2>/dev/null | sha256sum | cut -d' ' -f1)
for t in isal rapidgzip; do
  h=$(${CMD[$t]} 2>/dev/null | sha256sum | cut -d' ' -f1)
  [ "$h" = "$H" ] || { echo "FAIL: $t hash $h != purerust $H"; exit 2; }
done
echo "correctness OK (sha256=$H)"

# ── wall pass (interleaved) ───────────────────────────────────────────────
declare -A TIMES
for t in "${ORDER[@]}"; do TIMES[$t]=""; done
for ((i=1; i<=TRIALS; i++)); do
  for t in "${ORDER[@]}"; do
    s=$(date +%s.%N); ${CMD[$t]} >/dev/null 2>&1; e=$(date +%s.%N)
    el=$(awk "BEGIN{printf \"%.4f\", $e-$s}")
    TIMES[$t]="${TIMES[$t]} $el"
  done
done

echo
echo "tool        trial1_MBps   median_MBps(t2..N)   min   max"
for t in "${ORDER[@]}"; do
  echo "${TIMES[$t]}" | RAW=$RAW T=$t awk '
    { for(i=1;i<=NF;i++) a[i]=$i; n=NF }
    END{
      raw=ENVIRON["RAW"]+0; tool=ENVIRON["T"];
      t1=raw/a[1]/1e6;
      m=n-1; for(i=2;i<=n;i++) r[i-1]=a[i];
      for(i=1;i<=m;i++)for(j=i+1;j<=m;j++)if(r[j]<r[i]){x=r[i];r[i]=r[j];r[j]=x}
      med=(m%2)?r[(m+1)/2]:(r[m/2]+r[m/2+1])/2;
      printf "%-10s  %9.0f     %12.0f      %4.0f  %4.0f\n",
        tool, t1, raw/med/1e6, raw/r[m]/1e6, raw/r[1]/1e6
    }'
done
