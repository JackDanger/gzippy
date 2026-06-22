#!/usr/bin/env bash
# _t1_monolith_falsifier.sh — gated T1-MONOLITH falsifier measurement.
#
# Arms (all decode-only timed, /dev/null sink, interleaved best-of-N, pin cpu$PIN):
#   igzip      IBIN igzip            ISA-L monolith one-shot WITH CRC = the x86 bar
#   prodN      NBIN prod             gzippy-NATIVE T1-monolith  = THE SHIP TARGET (falsifier)
#   prodN2     NBIN prod             A/A self-test (native monolith vs itself)
#   prodNthin  NBIN prod NO_MONOLITH gzippy-NATIVE legacy thin-T1 = the BEFORE baseline
#   prodI      IBIN prod             gzippy-isal T1-monolith (kernel ref, ~ igzip)
#
# Derived per cell:
#   FALSIFIER  (prodN    - igzip)/igzip    [native monolith vs igzip; <=1.10 = CONFIRMED]
#   BEFORE     (prodNthin- igzip)/igzip    [native legacy thin-T1 vs igzip = starting point]
#   KERNEL     (prodN    - prodI)/prodI    [native vs isal under the monolith driver]
#   isal_mono  (prodI    - igzip)/igzip    [isal monolith vs igzip; should be ~0]
#
# Gate-0: every arm bytes==zcat (REF); A/A |prodN-prodN2| << inter-arm Δ.
set -u
NBIN=${NBIN:-/dev/shm/tn/release/examples/streaming_thin}
IBIN=${IBIN:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260622}

echo "== T1-MONOLITH FALSIFIER (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "nbin sha=$(sha256sum "$NBIN" | cut -c1-12)  ibin sha=$(sha256sum "$IBIN" | cut -c1-12)"

ARMS="igzip prodN prodN2 prodNthin prodI"
declare -A BIN MODE ENV
# T1-monolith is OPT-IN (GZIPPY_MONOLITH=1); default `prod` = legacy thin-T1.
BIN[igzip]=$IBIN;     MODE[igzip]=igzip; ENV[igzip]=""
BIN[prodN]=$NBIN;     MODE[prodN]=prod;  ENV[prodN]="GZIPPY_MONOLITH=1"
BIN[prodN2]=$NBIN;    MODE[prodN2]=prod; ENV[prodN2]="GZIPPY_MONOLITH=1"
BIN[prodNthin]=$NBIN; MODE[prodNthin]=prod; ENV[prodNthin]=""
BIN[prodI]=$IBIN;     MODE[prodI]=prod;  ENV[prodI]="GZIPPY_MONOLITH=1"

echo "-- non-inert routing proofs (GZIPPY_DEBUG, first corpus) --"
F0=$GZDIR/$(echo $CORPORA | awk '{print $1}').gz
for a in prodN prodNthin prodI; do
  dbg=$(env ${ENV[$a]} GZIPPY_DEBUG=1 taskset -c $PIN "${BIN[$a]}" "${MODE[$a]}" "$F0" 2>&1 >/dev/null | grep -iE 'MONOLITH|thin-T1' | head -1)
  printf "  %-10s : %s\n" "$a" "${dbg:-<none>}"
done

for corp in $CORPORA; do
  F=$GZDIR/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE ($F)"; continue; }
  REF=$(zcat "$F" | wc -c)
  echo "--- $corp  ref_bytes=$REF ---"
  declare -A BEST BYTES
  for a in $ARMS; do BEST[$a]=999999; BYTES[$a]=0; done
  for r in $(seq 1 "$REPS"); do
    order=$(echo $ARMS | tr ' ' '\n' | shuf --random-source=<(yes "$SEED$r"))
    for a in $order; do
      line=$(env ${ENV[$a]} taskset -c $PIN "${BIN[$a]}" "${MODE[$a]}" "$F" 2>/dev/null | grep '^RESULT')
      ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
      by=$(echo "$line" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
      [ -z "$ms" ] && { echo "  ARM $a FAILED rep $r"; continue; }
      BYTES[$a]=$by
      awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
    done
  done
  # Gate-0 bytes==zcat
  g0="PASS"
  for a in $ARMS; do [ "${BYTES[$a]}" = "$REF" ] || g0="FAIL($a=${BYTES[$a]})"; done
  ig=${BEST[igzip]}; pn=${BEST[prodN]}; pn2=${BEST[prodN2]}; pt=${BEST[prodNthin]}; pi=${BEST[prodI]}
  fals=$(awk -v p=$pn -v i=$ig 'BEGIN{printf "%.4f", p/i}')
  bef=$(awk -v p=$pt -v i=$ig 'BEGIN{printf "%.4f", p/i}')
  ker=$(awk -v p=$pn -v i=$pi 'BEGIN{printf "%+.2f%%", (p-i)/i*100}')
  imo=$(awk -v p=$pi -v i=$ig 'BEGIN{printf "%.4f", p/i}')
  aa=$(awk -v a=$pn -v b=$pn2 'BEGIN{printf "%.2f", (a>b)?(a-b):(b-a)}')
  echo "  igzip=${ig}ms prodN=${pn}ms prodNthin=${pt}ms prodI=${pi}ms  (A/A |prodN-prodN2|=${aa}ms)"
  echo "  FALSIFIER prodN/igzip=${fals}   BEFORE prodNthin/igzip=${bef}   KERNEL(nat-isal)=${ker}   isal_mono/igzip=${imo}   Gate0=$g0"
done
echo "load_end: $(cat /proc/loadavg)"
