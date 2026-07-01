#!/usr/bin/env bash
# _thin_oracle_local.sh — route-A removal-oracle interleaved race.
# Arms (all decode-only internal timing, /dev/null sink, same example binary):
#   prod  = full parallel-SM T=1 (production scaffold)
#   gzippy= thin streaming kernel (route-A: bookkeeping shed)
#   libdeflate / zlibng / igzip(if present) = comparators
# Per-rep RANDOMIZED arm order; best-of-N (min ms) per arm; reports spread.
set -u
BIN=${BIN:-./target/release/examples/streaming_thin}
F=${1:?usage: _thin_oracle_local.sh <file.gz> [reps]}
REPS=${2:-9}
HAVE_IGZIP=${HAVE_IGZIP:-0}
ARMS="prod gzippy libdeflate zlibng"
[ "$HAVE_IGZIP" = 1 ] && ARMS="$ARMS igzip"

echo "== thin removal-oracle race  file=$F reps=$REPS arms='$ARMS' =="
# self-test: byte counts must agree across arms (non-inert + correctness)
declare -A BEST WORST BYTES
for a in $ARMS; do BEST[$a]=999999; WORST[$a]=0; done

for r in $(seq 1 "$REPS"); do
  order=$(echo $ARMS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for a in $order; do
    line=$("$BIN" "$a" "$F" 2>/dev/null | grep '^RESULT')
    ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
    by=$(echo "$line" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
    [ -z "$ms" ] && { echo "ARM $a FAILED rep $r"; continue; }
    BYTES[$a]=$by
    awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
    awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
  done
done

echo "--- best-of-$REPS (ms), spread = worst-best ---"
ref_bytes=""
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-11s best=%8s ms  spread=%6s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$spread" "${BYTES[$a]}"
  [ -z "$ref_bytes" ] && ref_bytes="${BYTES[$a]}"
  [ "${BYTES[$a]}" = "$ref_bytes" ] || echo "    !! BYTE COUNT MISMATCH vs $ref_bytes (NON-INERT/CORRECTNESS FAIL)"
done

echo "--- ratios (gzippy_thin / comparator), >1 = gzippy slower ---"
gz=${BEST[gzippy]}
for a in $ARMS; do
  [ "$a" = gzippy ] && continue
  printf "  thin / %-11s = %s\n" "$a" "$(awk -v g="$gz" -v c="${BEST[$a]}" 'BEGIN{printf "%.3f", g/c}')"
done
echo "--- prod / comparators (full scaffold) ---"
pr=${BEST[prod]}
for a in $ARMS; do
  [ "$a" = prod ] && continue
  printf "  prod / %-11s = %s\n" "$a" "$(awk -v p="$pr" -v c="${BEST[$a]}" 'BEGIN{printf "%.3f", p/c}')"
done
echo "--- route-A capture: (prod-thin)/prod = scaffold share removable by route A ---"
awk -v p="$pr" -v t="$gz" 'BEGIN{printf "  (prod-thin)/prod = %.1f%%\n", (p-t)/p*100}'
