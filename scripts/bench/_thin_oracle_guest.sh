#!/usr/bin/env bash
# _thin_oracle_guest.sh — route-A removal-oracle interleaved race on neurotic.
# native binary: prod / gzippy(thin) / libdeflate / zlibng   (pure-Rust kernel)
# isal binary:   igzip                                       (the x86 bar)
# All arms: decode-only INTERNAL timing, /dev/null sink, pinned to one P-core,
# per-rep RANDOMIZED arm order (incl. an A/A self-test arm), best-of-N (min ms).
# Gate-0: every arm's byte count == zcat ref (non-inert + correctness);
#         thin2 (A/A of thin) vs thin must be within spread (harness self-test).
set -u
NAT=${NAT:-/dev/shm/tn/release/examples/streaming_thin}
ISAL=${ISAL:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
SEED=${SEED:-20260621}

echo "== route-A removal-oracle race (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "nat sha=$(sha256sum $NAT|cut -c1-12)  isal sha=$(sha256sum $ISAL|cut -c1-12)"

# arms -> which binary
declare -A BIN
BIN[prod]=$NAT; BIN[gzippy]=$NAT; BIN[thin2]=$NAT
BIN[libdeflate]=$NAT; BIN[zlibng]=$NAT; BIN[igzip]=$ISAL
# the actual example mode per arm (thin2 is a 2nd thin run = A/A self-test)
declare -A MODE
MODE[prod]=prod; MODE[gzippy]=gzippy; MODE[thin2]=gzippy
MODE[libdeflate]=libdeflate; MODE[zlibng]=zlibng; MODE[igzip]=igzip
ARMS="prod gzippy thin2 libdeflate zlibng igzip"

for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE"; continue; }
  REF=$(zcat "$F" | wc -c)
  echo "--- $corp  ref_bytes=$REF ---"
  declare -A BEST WORST BYTES
  for a in $ARMS; do BEST[$a]=999999; WORST[$a]=0; BYTES[$a]=0; done
  for r in $(seq 1 "$REPS"); do
    order=$(echo $ARMS | tr ' ' '\n' | shuf --random-source=<(yes "$SEED$r") )
    for a in $order; do
      line=$(taskset -c $PIN "${BIN[$a]}" "${MODE[$a]}" "$F" 2>/dev/null | grep '^RESULT')
      ms=$(echo "$line" | sed -n 's/.*ms=\([0-9.]*\).*/\1/p')
      by=$(echo "$line" | sed -n 's/.*bytes=\([0-9]*\).*/\1/p')
      [ -z "$ms" ] && { echo "  ARM $a FAILED rep $r"; continue; }
      BYTES[$a]=$by
      awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
      awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
    done
  done
  # report
  GATE=PASS
  for a in $ARMS; do
    sp=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
    bok=OK; [ "${BYTES[$a]}" = "$REF" ] || { bok="BAD(${BYTES[$a]})"; [ "$a" = igzip ] && bok="OK*"; GATE=FAIL; }
    printf "  %-11s best=%9s ms  spread=%7s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$sp" "$bok"
  done
  # A/A self-test
  aa=$(awk -v a="${BEST[gzippy]}" -v b="${BEST[thin2]}" -v s1="$(awk -v b="${BEST[gzippy]}" -v w="${WORST[gzippy]}" 'BEGIN{print w-b}')" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.3f", d}')
  printf "  A/A self-test |thin-thin2| = %s ms (should be << inter-arm Δ)\n" "$aa"
  echo "  GATE0(bytes)=$GATE"
  # ratios
  printf "  thin/igzip=%s  thin/libdeflate=%s  prod/igzip=%s  prod/libdeflate=%s\n" \
    "$(awk -v g="${BEST[gzippy]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",g/c}')" \
    "$(awk -v g="${BEST[gzippy]}" -v c="${BEST[libdeflate]}" 'BEGIN{printf "%.3f",g/c}')" \
    "$(awk -v p="${BEST[prod]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",p/c}')" \
    "$(awk -v p="${BEST[prod]}" -v c="${BEST[libdeflate]}" 'BEGIN{printf "%.3f",p/c}')"
  printf "  route-A capture (prod-thin)/prod = %s%%  | residual thin-igzip gap = %s%%\n" \
    "$(awk -v p="${BEST[prod]}" -v t="${BEST[gzippy]}" 'BEGIN{printf "%.1f",(p-t)/p*100}')" \
    "$(awk -v t="${BEST[gzippy]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.1f",(t-c)/c*100}')"
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
