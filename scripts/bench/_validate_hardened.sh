#!/usr/bin/env bash
# _validate_hardened.sh — validate the hardened adaptive-chunk binary.
#   * byte-exact (sha vs gzip -dc oracle, and rapidgzip cross-tool) per corpus
#   * wall best-of-N to /dev/null (SINK LAW: all arms same sink), HARD vs BASE vs RG
# Usage: CORPUS=/root/silesia.gz TS="1 4 7" N=8 bash _validate_hardened.sh
set -u
HARD=${HARD:-/dev/shm/gz-hard-target/release/gzippy}
BASE=${BASE:-/dev/shm/gz-b22-target/release/gzippy}
RG=${RG:-rapidgzip}
CORPUS=${CORPUS:?set CORPUS}
TS=${TS:-"1 4 7"}
N=${N:-8}

ref=$(gzip -dc "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "# corpus=$CORPUS ref_sha(gzip)=${ref:0:12}"

# --- byte-exact pass (T7) ---
for label in HARD BASE RG; do
  case $label in
    HARD) cmd="env GZIPPY_FORCE_PARALLEL_SM=1 $HARD -d -c -p 7 $CORPUS";;
    BASE) cmd="env GZIPPY_FORCE_PARALLEL_SM=1 $BASE -d -c -p 7 $CORPUS";;
    RG)   cmd="$RG -d -c -P 7 $CORPUS";;
  esac
  s=$($cmd 2>/dev/null | sha256sum | cut -d' ' -f1)
  if [ "$s" = "$ref" ]; then echo "  byte-exact $label: OK (${s:0:12})"; else echo "  byte-exact $label: MISMATCH got ${s:0:12} != ${ref:0:12}"; fi
done

# --- wall pass (best-of-N to /dev/null) ---
bestms() { # $@ = command; echoes best wall ms over N
  local best=9999999 i s e ms
  for i in $(seq 1 "$N"); do
    s=$(date +%s.%N); "$@" >/dev/null 2>/dev/null; e=$(date +%s.%N)
    ms=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.1f",(b-a)*1000}')
    best=$(awk -v m="$ms" -v bb="$best" 'BEGIN{print (m<bb)?m:bb}')
  done
  echo "$best"
}
echo "# T   HARD_ms  BASE_ms   RG_ms  hard/base  hard/rg"
for T in $TS; do
  h=$(bestms env GZIPPY_FORCE_PARALLEL_SM=1 "$HARD" -d -c -p "$T" "$CORPUS")
  b=$(bestms env GZIPPY_FORCE_PARALLEL_SM=1 "$BASE" -d -c -p "$T" "$CORPUS")
  r=$(bestms "$RG" -d -c -P "$T" "$CORPUS")
  hb=$(awk -v a="$h" -v b="$b" 'BEGIN{printf "%.3f",a/b}')
  hr=$(awk -v a="$h" -v b="$r" 'BEGIN{printf "%.3f",a/b}')
  printf "%3s  %8s  %8s  %7s  %9s  %7s\n" "$T" "$h" "$b" "$r" "$hb" "$hr"
done
