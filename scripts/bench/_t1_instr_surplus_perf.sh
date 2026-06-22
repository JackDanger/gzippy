#!/usr/bin/env bash
# _t1_instr_surplus_perf.sh — LOCATE the gzippy-NATIVE T1 instruction surplus vs
# igzip via `perf stat` retired-instruction counts + removal/perturbation oracles.
#
# Ship target = gzippy-NATIVE prod (pure-Rust kernel, no FFI). Bar = igzip (ISA-L
# monolith WITH CRC). Two binaries:
#   BIN_N = gzippy-native streaming_thin (prod = decompress_parallel(&[u8],…,1))
#   BIN_I = gzippy-isal   streaming_thin (igzip = ISA-L monolith bar)
#
# Arms (all /dev/null, decode-only) and the BUCKET each isolates:
#   igzip       bar (isal monolith)                              — instr bar
#   prod        native prod, default 1MiB chunk                  — total native instr
#   prodcrcoff  native prod + GZIPPY_ORACLE_CRC_OFF=1            — CRC 2nd-touch bucket
#                  = (prod_instr - prodcrcoff_instr)
#   prod2048    native prod + GZIPPY_CHUNK_KIB=2048              — per-chunk bucket (slope)
#   prod4096    native prod + GZIPPY_CHUNK_KIB=4096              — per-chunk bucket (slope)
#                  = (prod_instr - prod4096_instr) = per-chunk bookkeeping instr
#
# perf-stat instruction counts are deterministic to ~0.1% run-to-run; we take the
# MIN-instructions run of REPS as the representative (no jitter from scheduling).
# Non-inert proof: each oracle prints a GZIPPY_DEBUG banner/stride first.
set -u
BIN_N=${BIN_N:-/dev/shm/tn/release/examples/streaming_thin}
BIN_I=${BIN_I:-/dev/shm/ti/release/examples/streaming_thin}
PIN=${PIN:-4}
REPS=${REPS:-3}
CORPORA=${CORPORA:-"silesia nasa monorepo squishy"}
GZDIR=${GZDIR:-/root}
EV="instructions,cycles,minor-faults"

declare -A BINOF MODE ENV
BINOF[igzip]=$BIN_I;      MODE[igzip]=igzip;     ENV[igzip]=""
BINOF[prod]=$BIN_N;       MODE[prod]=prod;       ENV[prod]=""
BINOF[prodcrcoff]=$BIN_N; MODE[prodcrcoff]=prod; ENV[prodcrcoff]="GZIPPY_ORACLE_CRC_OFF=1"
BINOF[prod2048]=$BIN_N;   MODE[prod2048]=prod;   ENV[prod2048]="GZIPPY_CHUNK_KIB=2048"
BINOF[prod4096]=$BIN_N;   MODE[prod4096]=prod;   ENV[prod4096]="GZIPPY_CHUNK_KIB=4096"
ARMS="igzip prod prodcrcoff prod2048 prod4096"

echo "== T1 INSTRUCTION-SURPLUS LOCATE (perf stat, pin=cpu$PIN reps=$REPS) =="
echo "load_start: $(cat /proc/loadavg)"
echo "BIN_N sha=$(sha256sum "$BIN_N" | cut -c1-12)  BIN_I sha=$(sha256sum "$BIN_I" | cut -c1-12)"
echo "grep T1ResidentScope in BIN_N: $(strings "$BIN_N" 2>/dev/null | grep -c T1ResidentScope || echo 0)"

echo "-- non-inert proofs (GZIPPY_DEBUG, first corpus) --"
F0=$GZDIR/$(echo $CORPORA | awk '{print $1}').gz
for a in prod prodcrcoff prod2048 prod4096; do
  dbg=$(env ${ENV[$a]} GZIPPY_DEBUG=1 taskset -c $PIN "${BINOF[$a]}" "${MODE[$a]}" "$F0" 2>&1 >/dev/null | grep -i 'thin-T1\|stride\|CRC_OFF\|resident' | head -2 | tr '\n' ' ')
  printf "  %-11s : %s\n" "$a" "${dbg:-<no debug line>}"
done

for corp in $CORPORA; do
  F=$GZDIR/$corp.gz
  [ -f "$F" ] || { echo "  $corp: NO FILE ($F)"; continue; }
  REF=$(zcat "$F" | wc -c)
  echo "--- $corp  ref_bytes=$REF ---"
  declare -A INS CYC MF
  for a in $ARMS; do INS[$a]=0; CYC[$a]=0; MF[$a]=0; done
  for a in $ARMS; do
    best=999999999999999
    for r in $(seq 1 "$REPS"); do
      out=$(env ${ENV[$a]} taskset -c $PIN perf stat -e "$EV" "${BINOF[$a]}" "${MODE[$a]}" "$F" 2>&1)
      ins=$(echo "$out" | grep -iw 'instructions' | awk '{gsub(/,/,"",$1);print $1}')
      cyc=$(echo "$out" | grep -iw 'cycles'       | awk '{gsub(/,/,"",$1);print $1}')
      mf=$(echo "$out"  | grep -i  'minor-faults' | awk '{gsub(/,/,"",$1);print $1}')
      [ -z "$ins" ] && continue
      if awk -v x="$ins" -v b="$best" 'BEGIN{exit !(x<b)}'; then
        best=$ins; INS[$a]=$ins; CYC[$a]=$cyc; MF[$a]=$mf
      fi
    done
    printf "  %-11s instr=%-16s cycles=%-16s minor-faults=%s\n" "$a" "${INS[$a]}" "${CYC[$a]}" "${MF[$a]}"
  done
  # Derived buckets (instruction counts)
  printf "  SURPLUS (prod-igzip)/igzip        = %s%%  (Δinstr=%s)\n" \
    "$(awk -v p="${INS[prod]}" -v i="${INS[igzip]}" 'BEGIN{printf "%.2f",(p-i)/i*100}')" \
    "$(awk -v p="${INS[prod]}" -v i="${INS[igzip]}" 'BEGIN{printf "%.0f",p-i}')"
  printf "  CRC 2nd-touch (prod-prodcrcoff)    = %s instr  (%s%% of surplus)\n" \
    "$(awk -v p="${INS[prod]}" -v v="${INS[prodcrcoff]}" 'BEGIN{printf "%.0f",p-v}')" \
    "$(awk -v p="${INS[prod]}" -v v="${INS[prodcrcoff]}" -v i="${INS[igzip]}" 'BEGIN{s=p-i; if(s>0)printf "%.1f",(p-v)/s*100; else printf "n/a"}')"
  printf "  per-chunk @4096 (prod-prod4096)    = %s instr  (%s%% of surplus)  [prod2048 Δ=%s]\n" \
    "$(awk -v p="${INS[prod]}" -v v="${INS[prod4096]}" 'BEGIN{printf "%.0f",p-v}')" \
    "$(awk -v p="${INS[prod]}" -v v="${INS[prod4096]}" -v i="${INS[igzip]}" 'BEGIN{s=p-i; if(s>0)printf "%.1f",(p-v)/s*100; else printf "n/a"}')" \
    "$(awk -v p="${INS[prod]}" -v v="${INS[prod2048]}" 'BEGIN{printf "%.0f",p-v}')"
  printf "  residual surplus @4096 (prod4096-igzip)/igzip = %s%%\n" \
    "$(awk -v v="${INS[prod4096]}" -v i="${INS[igzip]}" 'BEGIN{printf "%.2f",(v-i)/i*100}')"
done
echo "load_end: $(cat /proc/loadavg)"
echo "== DONE =="
