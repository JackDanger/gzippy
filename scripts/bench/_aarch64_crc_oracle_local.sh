#!/usr/bin/env bash
# _aarch64_crc_oracle_local.sh — Gate-2 CRC REMOVAL ORACLE AB: production T1
# (CRC on, default) vs CRC-removed (GZIPPY_ORACLE_CRC_OFF=1, bytes still correct),
# both vs libdeflate-gunzip, on ONE binary (env toggles CRC — controls for build
# variance). Sizes CRC32's share of the gz T1 wall = (crcon - crcoff)/crcon.
# Gate-0: /dev/null all arms; sha-verified up front (CRC-off bytes MUST still ==
# libdeflate ref — UNLIKE nostore, CRC-off keeps bytes correct); A/A self-test.
# Gate-1: interleaved, randomized per-rep order, best-of-N min + spread.
# macOS aarch64 (no perf) => wall-time only. NOT-YET-LAW (single box, indicative).
set -u
GZ=${GZ:-/home/user/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_crc_oracle_local.sh <file.gz> [reps]}
N=${2:-15}

now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

# Gate-0 sha self-check (ALL arms incl. CRC-off must equal libdeflate).
REF=$(libdeflate-gunzip -c "$F" | shasum | cut -d' ' -f1)
ON=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
OFF=$(GZIPPY_ORACLE_CRC_OFF=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" 2>/dev/null | shasum | cut -d' ' -f1)
if [ "$REF" != "$ON" ] || [ "$REF" != "$OFF" ]; then
  echo "GATE-0 FAIL: sha mismatch ref=$REF crcon=$ON crcoff=$OFF"; exit 1
fi

declare -A BEST WORST
ARMS="crcon crcoff ld crcon2"
for a in $ARMS; do BEST[$a]=999999999; WORST[$a]=0; done

run_arm() {
  case "$1" in
    crcon|crcon2) GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    crcoff)       GZIPPY_ORACLE_CRC_OFF=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    ld)           libdeflate-gunzip -c "$F" >/dev/null 2>&1 ;;
  esac
}

for a in $ARMS; do run_arm "$a"; done   # warm

for r in $(seq 1 "$N"); do
  order=$(echo $ARMS | tr ' ' '\n' | sort -R | tr '\n' ' ')
  for a in $order; do
    t0=$(now); run_arm "$a"; t1=$(now)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e6}')
    awk -v m="$ms" -v b="${BEST[$a]}" 'BEGIN{exit !(m<b)}' && BEST[$a]=$ms
    awk -v m="$ms" -v w="${WORST[$a]}" 'BEGIN{exit !(m>w)}' && WORST[$a]=$ms
  done
done

echo "== CRC REMOVAL ORACLE AB  file=$F reps=$N (best-of-N min ms, /dev/null all arms) =="
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-7s best=%9s ms  spread=%8s ms\n" "$a" "${BEST[$a]}" "$spread"
done
echo "--- analysis ---"
awk -v on="${BEST[crcon]}" -v off="${BEST[crcoff]}" 'BEGIN{printf "  CRC share of gz wall = (crcon-crcoff)/crcon = %.4f  (%.2f ms)\n", (on-off)/on, on-off}'
awk -v on="${BEST[crcon]}" -v l="${BEST[ld]}" 'BEGIN{printf "  crcon  / libdeflate  = %.4f  (PRODUCTION, CRC on)\n", on/l}'
awk -v off="${BEST[crcoff]}" -v l="${BEST[ld]}" 'BEGIN{printf "  crcoff / libdeflate  = %.4f  (residual w/o CRC; libdeflate still computes CRC)\n", off/l}'
awk -v on="${BEST[crcon]}" -v on2="${BEST[crcon2]}" 'BEGIN{printf "  A/A crcon/crcon2     = %.4f  (self-test, want ~1.00)\n", on/on2}'
