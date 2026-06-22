#!/usr/bin/env bash
# _prod_path_locate_guest.sh — PROD-PATH LOCATE: measure the REAL production T1
# decode path (`decompress_parallel(&[u8], …, 1)` via streaming_thin `prod` mode)
# vs the igzip monolith bar, then removal-oracle each Layer-A per-chunk candidate
# IN THE REAL PATH. The prior SCAFFOLD-LOCATE used the streaming_thin `cheap`
# PROXY which (a) ADDED a phantom input .to_vec copy and (b) OMITTED the real
# production driver. This races the actual production driver instead.
#
# All `prod*` arms run the SAME gzippy-isal binary: each T1 chunk's clean tail
# decodes through REAL ISA-L `_04` (isal_clean_tail cfg) — kernel IDENTICAL to
# igzip — so the prod−igzip delta IS the production driver/scaffold overhead.
#
#   igzip       ISA-L monolith one-shot, WITH CRC = the x86 bar
#   prod        real production T1 (decompress_parallel,1), CRC ON  = the product
#   prod2       A/A self-test (prod vs prod)
#   prodcrcoff  prod + GZIPPY_ORACLE_CRC_OFF=1   → CRC second-touch removed (#5)
#   prodbig     prod + GZIPPY_CHUNK_KIB=huge     → per-chunk count collapsed
#                  (bounds the per-chunk family #1 alloc/reserve + #2 window-roll
#                   + #3 isal lifecycle TOGETHER; stride clamps to ~file/3 at T1)
#   prodpool    prod + GZIPPY_RESIDENT_OUTPUT_POOL=1 → reserve pinned 64MiB (#1)
#
# Derived per cell:
#   REAL SCAFFOLD     (prod    − igzip)/igzip   [both WITH CRC — fair]
#   CRC share         (prod    − prodcrcoff)/prod
#   SCAFFOLD−CRC      (prodcrcoff − igzip)/igzip
#   per-chunk family  (prod    − prodbig)/prod   [+ = real per-chunk cost]
#   reserve(#1)       (prod    − prodpool)/prod
#
# /dev/null both arms, per-rep RANDOMIZED order, best-of-N (min ms), pin cpu$PIN.
# Gate-0: every arm bytes==zcat; A/A |prod-prod2| << inter-arm Δ.
set -u
BIN=${BIN:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260621}
BIGKIB=${BIGKIB:-9999999}

echo "== PROD-PATH LOCATE oracle (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "bin sha=$(sha256sum "$BIN" | cut -c1-12)"

ARMS="igzip prod prod2 prodcrcoff prodbig prodpool"
declare -A MODE ENV
MODE[igzip]=igzip;      ENV[igzip]=""
MODE[prod]=prod;        ENV[prod]=""
MODE[prod2]=prod;       ENV[prod2]=""
MODE[prodcrcoff]=prod;  ENV[prodcrcoff]="GZIPPY_ORACLE_CRC_OFF=1"
MODE[prodbig]=prod;     ENV[prodbig]="GZIPPY_CHUNK_KIB=$BIGKIB"
MODE[prodpool]=prod;    ENV[prodpool]="GZIPPY_RESIDENT_OUTPUT_POOL=1"

# Non-inert proof: print one GZIPPY_DEBUG line per perturbed arm so the env knob's
# effect is visible (stride= for prodbig; thin-T1 fires for all prod*).
echo "-- non-inert proofs (GZIPPY_DEBUG one-shot, first corpus) --"
F0=$GZDIR/$(echo $CORPORA | awk '{print $1}').gz
for a in prod prodcrcoff prodbig prodpool; do
  dbg=$(env ${ENV[$a]} GZIPPY_DEBUG=1 taskset -c $PIN "$BIN" "${MODE[$a]}" "$F0" 2>&1 >/dev/null | grep -i 'thin-T1\|stride\|resident\|crc' | head -2 | tr '\n' ' ')
  printf "  %-11s : %s\n" "$a" "${dbg:-<no debug line>}"
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
      line=$(env ${ENV[$a]} taskset -c $PIN "$BIN" "${MODE[$a]}" "$F" 2>/dev/null | grep '^RESULT')
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
    printf "  %-11s best=%9s ms  spread=%7s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$sp" "$bok"
  done
  aa=$(awk -v a="${BEST[prod]}" -v b="${BEST[prod2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  printf "  A/A |prod-prod2|=%s ms\n" "$aa"
  echo "  GATE0(bytes)=$GATE"
  printf "  REAL SCAFFOLD  (prod-igzip)/igzip      = %s%%  (Δ=%s ms)\n" \
    "$(awk -v k="${BEST[prod]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.2f",(k-c)/c*100}')" \
    "$(awk -v k="${BEST[prod]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",k-c}')"
  printf "  CRC share      (prod-prodcrcoff)/prod  = %s%%  (Δ=%s ms)\n" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodcrcoff]}" 'BEGIN{printf "%.2f",(k-v)/k*100}')" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodcrcoff]}" 'BEGIN{printf "%.3f",k-v}')"
  printf "  SCAFFOLD-CRC   (prodcrcoff-igzip)/igzip= %s%%\n" \
    "$(awk -v k="${BEST[prodcrcoff]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.2f",(k-c)/c*100}')"
  printf "  per-chunk fam  (prod-prodbig)/prod     = %s%%  (Δ=%s ms)\n" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodbig]}" 'BEGIN{printf "%.2f",(k-v)/k*100}')" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodbig]}" 'BEGIN{printf "%.3f",k-v}')"
  printf "  reserve #1     (prod-prodpool)/prod    = %s%%  (Δ=%s ms)\n" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodpool]}" 'BEGIN{printf "%.2f",(k-v)/k*100}')" \
    "$(awk -v k="${BEST[prod]}" -v v="${BEST[prodpool]}" 'BEGIN{printf "%.3f",k-v}')"
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
