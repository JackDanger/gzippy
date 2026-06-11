#!/usr/bin/env bash
# _slabrec_rss_sha_guest.sh — slab-t-conditional verification (perf/slab-t-conditional):
#   1. RSS deltas per cell, bin-slabrec-native vs baseline (x3, unfrozen, canonical masks)
#   2. sha grid {silesia,model,bignasa,storedmix} x T{1,4,8,16} for the new binary
# Output is plain text; every line prefixed for machine grep.
set -u

NEW="${NEW:-/root/bin-slabrec-native}"
BASE="${BASE:-/root/bin-f2-native}"
RUNS="${RUNS:-3}"

pin_mask() {
  case "$1" in
    1) echo "0";; 4) echo "0,2,4,6";;
    8) echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *) echo "";;
  esac
}

SINK=/dev/shm/.slabrec_sink.bin
TF=/dev/shm/.slabrec_time.txt

run_rss() { # <bin> <corpus> <T> -> "secs rss_mb"
  local bin="$1" f="/root/$2.gz" t="$3" mask s e
  mask="$(pin_mask "$t")"
  s=$(date +%s.%N)
  /usr/bin/time -f '%M' -o "$TF" taskset -c "$mask" "$bin" -d -c -p "$t" "$f" > "$SINK" 2>/dev/null
  e=$(date +%s.%N)
  echo "$(echo "$e $s" | awk '{printf "%.4f", $1-$2}') $(( $(cat "$TF") / 1024 ))"
}

echo "== RSS GRID (unfrozen, x${RUNS}, canonical masks) new=$NEW base=$BASE =="
for cell in silesia:1 silesia:4 silesia:8 silesia:16 model:8; do
  c="${cell%%:*}"; t="${cell##*:}"
  for ((i=1;i<=RUNS;i++)); do
    read -r bs brss < <(run_rss "$BASE" "$c" "$t")
    read -r ns nrss < <(run_rss "$NEW" "$c" "$t")
    echo "RSS cell=$c:T$t iter=$i base_s=$bs base_rss_mb=$brss new_s=$ns new_rss_mb=$nrss"
  done
done

echo "== SHA GRID new=$NEW =="
for c in silesia model bignasa storedmix; do
  f="/root/$c.gz"
  [ -f "$f" ] || { echo "SHA corpus=$c MISSING"; continue; }
  ref="$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)"
  for t in 1 4 8 16; do
    mask="$(pin_mask "$t")"
    got="$(taskset -c "$mask" "$NEW" -d -c -p "$t" "$f" 2>/dev/null | sha256sum | cut -d' ' -f1)"
    if [ "$got" = "$ref" ]; then v=OK; else v="MISMATCH got=$got ref=$ref"; fi
    echo "SHA corpus=$c T=$t $v"
  done
done

rm -f "$SINK" "$TF"
echo "SLABREC_GUEST_DONE"
