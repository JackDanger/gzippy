#!/usr/bin/env bash
# _igzipbare_oracle_guest.sh — MEASUREMENT 2: de-confounded bare-inner-decode
# removal-oracle. ALL arms run from the SAME isal binary (so gz kernel codegen
# is byte-identical across thin/igzipbare and igzip is the same lib):
#   thin      = gz contig driver + gz run_contig kernel + gz table-build (no CRC)
#   igzipbare = gz contig driver + igzip read_header + igzip _04 kernel  (no CRC)
#   igzip     = full ISA-L monolithic streaming decode (WITH CRC) — the x86 bar
# A/A self-tests: thin2 (thin vs thin), bare2 (igzipbare vs igzipbare).
# Decode-only INTERNAL timing, /dev/null sink, pinned to one core, per-rep
# RANDOMIZED arm order, best-of-N (min ms).
#
# THE BOUND (brief): thin - igzipbare == the inner-decode (kernel+table-build)
# ceiling. Fork:
#   igzipbare << thin  (recovers most of thin-igzip)  => inner decode IS the lever
#   igzipbare ~= thin  (thin-igzipbare ~= 0)          => residual is gz driver/scaffold
# Gate-0: every arm bytes==zcat (igzip computes CRC; thin/bare skip it — noted);
#         A/A spreads << inter-arm Δ.
set -u
BIN=${BIN:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia nasa monorepo"}
GZDIR=${GZDIR:-/root}
SEED=${SEED:-20260621}

echo "== igzipbare bare-inner-decode oracle (pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "bin sha=$(sha256sum "$BIN"|cut -c1-12)"

declare -A MODE
MODE[thin]=gzippy; MODE[thin2]=gzippy
MODE[igzipbare]=igzipbare; MODE[bare2]=igzipbare
MODE[cheap]=igzipbarecheap; MODE[cheap2]=igzipbarecheap
MODE[igzip]=igzip
ARMS="thin thin2 igzipbare bare2 cheap cheap2 igzip"

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
      line=$(taskset -c $PIN "$BIN" "${MODE[$a]}" "$F" 2>/dev/null | grep '^RESULT')
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
    printf "  %-10s best=%9s ms  spread=%7s ms  bytes=%s\n" "$a" "${BEST[$a]}" "$sp" "$bok"
  done
  aa1=$(awk -v a="${BEST[thin]}" -v b="${BEST[thin2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  aa2=$(awk -v a="${BEST[igzipbare]}" -v b="${BEST[bare2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  aa3=$(awk -v a="${BEST[cheap]}" -v b="${BEST[cheap2]}" 'BEGIN{d=a-b;if(d<0)d=-d;printf "%.3f",d}')
  printf "  A/A self-test |thin-thin2|=%s ms  |bare-bare2|=%s ms  |cheap-cheap2|=%s ms\n" "$aa1" "$aa2" "$aa3"
  echo "  GATE0(bytes)=$GATE"
  printf "  thin/igzip=%s  igzipbare/igzip=%s  cheap/igzip=%s  thin/cheap=%s\n" \
    "$(awk -v t="${BEST[thin]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",t/c}')" \
    "$(awk -v g="${BEST[igzipbare]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",g/c}')" \
    "$(awk -v k="${BEST[cheap]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.3f",k/c}')" \
    "$(awk -v t="${BEST[thin]}" -v k="${BEST[cheap]}" 'BEGIN{printf "%.3f",t/k}')"
  # NON-INERT proof + decomposition:
  #   header artifact removed = (igzipbare - cheap)/igzip  (must exceed A/A spread)
  #   CLEAN scaffold          = (cheap - igzip)/igzip      (gz contig driver vs monolith, gz skips CRC)
  #   inner-decode (cheap)    = (thin - cheap)/thin        (gz kernel+tables vs igzip _04+read_header)
  printf "  HEADER-ARTIFACT removed (bare-cheap)/igzip = %s%%  [non-inert if > A/A]\n" \
    "$(awk -v b="${BEST[igzipbare]}" -v k="${BEST[cheap]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.1f",(b-k)/c*100}')"
  printf "  CLEAN SCAFFOLD (cheap-igzip)/igzip = %s%%  | INNER-DECODE(cheap) (thin-cheap)/thin = %s%%\n" \
    "$(awk -v k="${BEST[cheap]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.1f",(k-c)/c*100}')" \
    "$(awk -v t="${BEST[thin]}" -v k="${BEST[cheap]}" 'BEGIN{printf "%.1f",(t-k)/t*100}')"
  printf "  (legacy contaminated: inner(thin-bare)=%s%%  residual(bare-igzip)=%s%%)\n" \
    "$(awk -v t="${BEST[thin]}" -v g="${BEST[igzipbare]}" 'BEGIN{printf "%.1f",(t-g)/t*100}')" \
    "$(awk -v g="${BEST[igzipbare]}" -v c="${BEST[igzip]}" 'BEGIN{printf "%.1f",(g-c)/c*100}')"
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
