#!/usr/bin/env bash
# _t1_cache_residency_guest.sh — T1-CACHE-RESIDENCY lever discrimination.
#
# Measures the chunked STREAMING thin-T1 production path (streaming_thin `prod`
# mode = decompress_parallel(&[u8],…,1)) against the igzip ISA-L monolith bar,
# perturbing the cache-residency levers via byte-transparent env oracles, to find
# WHICH mechanism pays BEFORE baking it into the production default.
#
# Two binaries (set BIN_N native ship target, BIN_I isal for the igzip bar):
#   BIN_I = gzippy-isal streaming_thin   (igzip mode = ISA-L monolith bar)
#   BIN_N = gzippy-native streaming_thin (prod mode  = pure-Rust ship target)
#
# Arms (all `prod` on BIN_N except the bar; the bar `igzip` runs on BIN_I):
#   igzip      ISA-L monolith bar (BIN_I)                        = the x86 bar
#   prod       native thin-T1, 1 MiB chunk, no recycle (current default)
#   prod2      A/A self-test of prod
#   chunk2m    prod + GZIPPY_CHUNK_KIB=2048           (lever a: 2 MiB chunk)
#   manpool    prod + GZIPPY_MANUAL_BUFFER_POOL=1     (lever b1: recycle, ratio reserve)
#   respool    prod + GZIPPY_RESIDENT_OUTPUT_POOL=1   (lever b1+b2: recycle + 64MiB fixed)
#   c2m_man    prod + CHUNK_KIB=2048 + MANUAL_BUFFER_POOL=1  (a+b1 combined)
#
# Derived per cell:
#   native baseline (prod − igzip)/igzip
#   lever a         (prod − chunk2m)/prod    [+ = 2MiB chunk faster]
#   lever b1        (prod − manpool)/prod
#   lever b1+b2     (prod − respool)/prod
#   a+b1            (prod − c2m_man)/prod
#   best native/igzip = min(arm)/igzip across all arms
#
# /dev/null both arms, per-rep RANDOMIZED order, best-of-N (min ms), pin cpu$PIN.
set -u
BIN_N=${BIN_N:-/dev/shm/tn/release/examples/streaming_thin}
BIN_I=${BIN_I:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260622}

echo "== T1-CACHE-RESIDENCY discrimination (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "BIN_N sha=$(sha256sum "$BIN_N" | cut -c1-12)  BIN_I sha=$(sha256sum "$BIN_I" | cut -c1-12)"

ARMS="igzip prod prod2 chunk2m manpool respool c2m_man"
declare -A BINOF MODE ENV
BINOF[igzip]=$BIN_I;  MODE[igzip]=igzip;  ENV[igzip]=""
BINOF[prod]=$BIN_N;   MODE[prod]=prod;    ENV[prod]=""
BINOF[prod2]=$BIN_N;  MODE[prod2]=prod;   ENV[prod2]=""
BINOF[chunk2m]=$BIN_N; MODE[chunk2m]=prod; ENV[chunk2m]="GZIPPY_CHUNK_KIB=2048"
BINOF[manpool]=$BIN_N; MODE[manpool]=prod; ENV[manpool]="GZIPPY_MANUAL_BUFFER_POOL=1"
BINOF[respool]=$BIN_N; MODE[respool]=prod; ENV[respool]="GZIPPY_RESIDENT_OUTPUT_POOL=1"
BINOF[c2m_man]=$BIN_N; MODE[c2m_man]=prod; ENV[c2m_man]="GZIPPY_CHUNK_KIB=2048 GZIPPY_MANUAL_BUFFER_POOL=1"

# Non-inert proof: one GZIPPY_DEBUG line per perturbed arm.
echo "-- non-inert proofs (GZIPPY_DEBUG one-shot, first corpus) --"
F0=$GZDIR/$(echo $CORPORA | awk '{print $1}').gz
for a in prod chunk2m manpool respool c2m_man; do
  dbg=$(env ${ENV[$a]} GZIPPY_DEBUG=1 taskset -c $PIN "${BINOF[$a]}" "${MODE[$a]}" "$F0" 2>&1 >/dev/null | grep -i 'thin-T1\|stride\|resident' | head -1)
  printf "  %-9s : %s\n" "$a" "${dbg:-<no debug line>}"
done

for corp in $CORPORA; do
  F=$GZDIR/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE ($F)"; continue; }
  REF=$(zcat "$F" | wc -c)
  echo "--- $corp  ref_bytes=$REF ---"
  declare -A BEST WORST BYTES
  for a in $ARMS; do BEST[$a]=999999; WORST[$a]=0; BYTES[$a]=0; done
  for r in $(seq 1 "$REPS"); do
    order=$(echo $ARMS | tr ' ' '\n' | shuf --random-source=<(yes "$SEED$r"))
    for a in $order; do
      line=$(env ${ENV[$a]} taskset -c $PIN "${BINOF[$a]}" "${MODE[$a]}" "$F" 2>/dev/null | grep '^RESULT')
      ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
      by=$(echo "$line" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
      [ -z "$ms" ] && { echo "  ARM $a FAILED rep $r"; continue; }
      BYTES[$a]=$by
      awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
      awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
    done
  done
  GATE=PASS
  for a in $ARMS; do
    sp=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
    bok=OK; [ "${BYTES[$a]}" = "$REF" ] || { bok="BAD(${BYTES[$a]})"; GATE=FAIL; }
    printf "  %-9s best=%9s ms  spread=%7s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$sp" "$bok"
  done
  aa=$(awk -v a="${BEST[prod]}" -v b="${BEST[prod2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  printf "  A/A |prod-prod2|=%s ms   GATE0(bytes)=%s\n" "$aa" "$GATE"
  for pair in "baseline:prod:igzip" "leverA_2m:chunk2m:igzip" "leverB1_man:manpool:igzip" "leverB12_res:respool:igzip" "leverAB:c2m_man:igzip"; do
    nm=${pair%%:*}; rest=${pair#*:}; arm=${rest%%:*}; bar=${rest#*:}
    printf "  %-13s %s/igzip = %s   (Δ=%s ms)\n" "$nm" "$arm" \
      "$(awk -v k="${BEST[$arm]}" -v c="${BEST[$bar]}" 'BEGIN{printf "%.3f",k/c}')" \
      "$(awk -v k="${BEST[$arm]}" -v c="${BEST[$bar]}" 'BEGIN{printf "%.3f",k-c}')"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
