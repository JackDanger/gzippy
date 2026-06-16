#!/usr/bin/env bash
# _cliff_sweep.sh — map the adaptive-chunk boundary cliff.
# Sweeps GZIPPY_CHUNK_KIB (FIXED compressed spacing, bypasses adaptive) on a
# corpus at a given T, best-of-N internal decode wall (the v0.6 total=Xms line),
# prints chunk_size, chunk_count, cnt_mod_T, best_ms, sha_ok.
set -u
BIN=${BIN:-/dev/shm/gz-b22-target/release/gzippy}
CORPUS=${CORPUS:-/root/nasa.gz}
T=${T:-4}
N=${N:-5}
SIZES=${SIZES:-"256 384 448 480 500 510 515 520 530 560 640 768 825 1024 1536 2048 4096"}
COMP=$(stat -c%s "$CORPUS")
SHA_REF=$(GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p "$T" "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "# corpus=$CORPUS comp=$COMP T=$T N=$N ref_sha=${SHA_REF:0:12}"
echo "# kib  chunk_bytes  chunk_count  cnt_mod_T  best_ms  sha_ok"
for kib in $SIZES; do
  cb=$((kib*1024))
  cnt=$(( (COMP + cb - 1) / cb ))
  best=999999.0
  okall=1
  for i in $(seq 1 "$N"); do
    err=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_CHUNK_KIB=$kib "$BIN" -d -c -p "$T" "$CORPUS" 2>/tmp/cs.err >/tmp/cs.out)
    sha=$(sha256sum /tmp/cs.out | cut -d' ' -f1)
    [ "$sha" = "$SHA_REF" ] || okall=0
    ms=$(grep -oE 'total=[0-9.]+ms' /tmp/cs.err | head -1 | grep -oE '[0-9.]+')
    [ -n "$ms" ] && best=$(awk -v a="$ms" -v b="$best" 'BEGIN{print (a<b)?a:b}')
  done
  printf "%5d  %9d  %6d  %4d  %8s  %s\n" "$kib" "$cb" "$cnt" "$((cnt % T))" "$best" "$okall"
done
