#!/usr/bin/env bash
# _aarch64_crc_kernel_ab_local.sh — GATED CRC KERNEL AB on ONE binary (env toggles
# the CRC kernel → controls for build variance), all vs libdeflate-gunzip:
#   crc3   = production 3-way interleaved crc32x (default)
#   crcleg = crc32fast 1.5.0 single-crc32x-chain  (GZIPPY_CRC_LEGACY=1)
#   crcoff = CRC removed entirely (GZIPPY_ORACLE_CRC_OFF=1, ceiling oracle)
#   ld     = libdeflate-gunzip (the bar; it ALSO computes CRC via PMULL)
# Gate-0: /dev/null all arms; sha-verified up front (crc3 & crcleg MUST == ref);
# A/A self-test (crc3 vs crc3b). Gate-1: interleaved, randomized order, best-of-N.
# macOS aarch64 (no perf) => wall-time only. NOT-YET-LAW (single box, indicative).
set -u
GZ=${GZ:-/home/user/www/gz-aarch64-routeb/target/release/gzippy}
F=${1:?usage: _aarch64_crc_kernel_ab_local.sh <file.gz> [reps]}
N=${2:-15}

now() { python3 -c 'import time;print(time.perf_counter_ns())'; }

REF=$(libdeflate-gunzip -c "$F" | shasum | cut -d' ' -f1)
S3=$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
SL=$(GZIPPY_CRC_LEGACY=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" | shasum | cut -d' ' -f1)
if [ "$REF" != "$S3" ] || [ "$REF" != "$SL" ]; then
  echo "GATE-0 FAIL: sha mismatch ref=$REF crc3=$S3 crcleg=$SL"; exit 1
fi

declare -A BEST WORST
ARMS="crc3 crcleg crcoff ld crc3b"
for a in $ARMS; do BEST[$a]=999999999; WORST[$a]=0; done

run_arm() {
  case "$1" in
    crc3|crc3b) GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    crcleg)     GZIPPY_CRC_LEGACY=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    crcoff)     GZIPPY_ORACLE_CRC_OFF=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p1 "$F" >/dev/null 2>&1 ;;
    ld)         libdeflate-gunzip -c "$F" >/dev/null 2>&1 ;;
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

echo "== CRC KERNEL AB  file=$F reps=$N (best-of-N min ms, /dev/null all arms) =="
for a in $ARMS; do
  spread=$(awk -v b="${BEST[$a]}" -v w="${WORST[$a]}" 'BEGIN{printf "%.3f", w-b}')
  printf "  %-7s best=%9s ms  spread=%8s ms\n" "$a" "${BEST[$a]}" "$spread"
done
echo "--- analysis ---"
awk -v l="${BEST[crcleg]}" -v t="${BEST[crc3]}" 'BEGIN{printf "  crcleg -> crc3 wall  = %.2f ms saved  (%.4f x)\n", l-t, t/l}'
awk -v l="${BEST[crcleg]}" -v o="${BEST[crcoff]}" 'BEGIN{printf "  legacy CRC cost      = %.2f ms\n", l-o}'
awk -v t="${BEST[crc3]}" -v o="${BEST[crcoff]}" 'BEGIN{printf "  3-way  CRC cost      = %.2f ms\n", t-o}'
awk -v t="${BEST[crc3]}" -v ld="${BEST[ld]}" 'BEGIN{printf "  crc3   / libdeflate  = %.4f  (PRODUCTION)\n", t/ld}'
awk -v l="${BEST[crcleg]}" -v ld="${BEST[ld]}" 'BEGIN{printf "  crcleg / libdeflate  = %.4f  (pre-lever)\n", l/ld}'
awk -v t="${BEST[crc3]}" -v t2="${BEST[crc3b]}" 'BEGIN{printf "  A/A crc3/crc3b       = %.4f  (self-test, want ~1.00)\n", t/t2}'
