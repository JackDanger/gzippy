#!/usr/bin/env bash
# _scaffold_locate_guest.sh — MEASUREMENT 3: LOCATE the gz-contig-driver SCAFFOLD
# cost by sub-region (removal-oracle). ALL arms run the SAME isal binary; the
# inner decode (igzip read_header + _04/_base kernel) is held CONSTANT across
# every arm. The `cheap` baseline = gz contig driver wrapping that kernel. Each
# `cheap_*` arm REMOVES one driver sub-region; (cheap - cheap_X)/cheap BOUNDS
# that sub-region's share of the driver wall. `igzip` = the ISA-L monolith bar;
# CLEAN SCAFFOLD = (cheap - igzip)/igzip.
#
#   cheap        baseline gz contig driver (== igzipbarecheap)
#   cheap_nosink remove per-flush output sink writes
#   cheap_nozero uninit contig buffer (remove the cap zeroing)
#   cheap_noin   borrow input (remove the input .to_vec memcpy)
#   cheap_big    whole-output uninit buffer (window memmove + per-batch flush
#                never fire; single final flush)
#   igzip        ISA-L monolith (WITH CRC) — the x86 bar
# A/A self-test: cheap2 (cheap vs cheap).
# /dev/null both arms, per-rep RANDOMIZED order, best-of-N (min ms), pin cpu$PIN.
# Gate-0: every arm bytes==zcat; A/A spread << inter-arm Δ.
set -u
BIN=${BIN:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260621}

echo "== scaffold-LOCATE oracle (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "bin sha=$(sha256sum "$BIN" | cut -c1-12)"

ARMS="cheap cheap2 cheap_nosink cheap_nozero cheap_noin cheap_big igzip"
declare -A MARG
MARG[cheap]=igzipbarecheap; MARG[cheap2]=igzipbarecheap
MARG[cheap_nosink]=cheap_nosink; MARG[cheap_nozero]=cheap_nozero
MARG[cheap_noin]=cheap_noin; MARG[cheap_big]=cheap_big
MARG[igzip]=igzip

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
      line=$(taskset -c $PIN "$BIN" "${MARG[$a]}" "$F" 2>/dev/null | grep '^RESULT')
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
    printf "  %-13s best=%9s ms  spread=%7s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$sp" "$bok"
  done
  aa=$(awk -v a="${BEST[cheap]}" -v b="${BEST[cheap2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  printf "  A/A self-test |cheap-cheap2|=%s ms\n" "$aa"
  echo "  GATE0(bytes)=$GATE"
  printf "  CLEAN SCAFFOLD (cheap-igzip)/igzip = %s%%\n" \
    "$(awk -v k="${BEST[cheap]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.2f",(k-c)/c*100}')"
  # Per-sub-region share of the gz driver wall: (cheap - cheap_X)/cheap.
  # Positive => removing X SPEEDS UP (X is a real driver cost). Negative =>
  # removing X SLOWS DOWN (X is net-beneficial; do NOT remove).
  for x in cheap_nosink cheap_nozero cheap_noin cheap_big; do
    printf "  REGION %-13s (cheap-%s)/cheap = %s%%  (Δ=%s ms)\n" "$x" "$x" \
      "$(awk -v k="${BEST[cheap]}" -v v="${BEST[$x]}" 'BEGIN{printf "%.2f",(k-v)/k*100}')" \
      "$(awk -v k="${BEST[cheap]}" -v v="${BEST[$x]}" 'BEGIN{printf "%.3f",k-v}')"
  done
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
