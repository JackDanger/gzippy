#!/usr/bin/env bash
# _fence_knob_sweep.sh — drive the ADAPTIVE knob (GZIPPY_TARGET_DECODED_KIB)
# across targets that WOULD (un-fenced) yield bad chunk sizes, and confirm the
# 1 MiB fence pins the resulting chunk_size >= 1 MiB and the wall stays clean.
set -u
HARD=${HARD:-/dev/shm/gz-hard-target/release/gzippy}
CORPUS=${CORPUS:-/root/nasa.gz}
T=${T:-4}
N=${N:-6}
# targets in KiB; for nasa ratio ~9.93 these map raw-> ~414..1700 KiB, the bad
# band [448,672] included — all must be fenced to >=1 MiB.
TARGETS=${TARGETS:-"4000 4500 5000 5300 5600 5900 6000 6700 7000 8000 10000 16000"}
ref=$(GZIPPY_FORCE_PARALLEL_SM=1 "$HARD" -d -c -p "$T" "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "# corpus=$CORPUS T=$T N=$N ref=${ref:0:12}"
echo "# target_kib  chunk_kib  best_ms  sha_ok"
for tk in $TARGETS; do
  cs=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TARGET_DECODED_KIB=$tk "$HARD" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -oE 'chunk_size=[0-9]+' | head -1 | grep -oE '[0-9]+')
  ck=$(( ${cs:-0} / 1024 ))
  best=999999.0; ok=1
  for i in $(seq 1 "$N"); do
    GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TARGET_DECODED_KIB=$tk "$HARD" -d -c -p "$T" "$CORPUS" >/dev/shm/fk.out 2>/dev/null
    s=$(sha256sum /dev/shm/fk.out | cut -d' ' -f1); [ "$s" = "$ref" ] || ok=0
    ms=$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_TARGET_DECODED_KIB=$tk "$HARD" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -oE 'total=[0-9.]+ms' | grep -oE '[0-9.]+')
    [ -n "$ms" ] && best=$(awk -v a="$ms" -v b="$best" 'BEGIN{print (a<b)?a:b}')
  done
  printf "%10d  %8d  %8s  %s\n" "$tk" "$ck" "$best" "$ok"
done
