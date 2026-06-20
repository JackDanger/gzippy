#!/bin/bash
# NIGHT39 correctness gate: byte-exact (new vs zcat vs base) across
# silesia/nasa/monorepo x T1/T4/T8 (T4/T8 EXERCISE the speculative path), plus
# a malformed-input safety probe (over-subscribed dynamic Huffman table) — new
# must return non-zero exit (Err) and NOT segfault/hang.
set -u
NEW=/dev/shm/gz-new
BASE=/dev/shm/gz-base
CORPORA="/root/silesia.gz /root/nasa.gz /root/monorepo.gz"
echo "### BYTE-EXACT new vs zcat vs base (sha256, T1/T4/T8) ###"
fail=0
for f in $CORPORA; do
  ref=$(zcat "$f" 2>/dev/null | sha256sum | cut -c1-16)
  for T in 1 4 8; do
    sn=$(GZIPPY_FORCE_PARALLEL_SM=1 "$NEW"  -d -c -p$T "$f" 2>/dev/null | sha256sum | cut -c1-16)
    sb=$(GZIPPY_FORCE_PARALLEL_SM=1 "$BASE" -d -c -p$T "$f" 2>/dev/null | sha256sum | cut -c1-16)
    ok="OK"; if [ "$sn" != "$ref" ] || [ "$sn" != "$sb" ]; then ok="MISMATCH"; fail=1; fi
    printf "  %-22s T%-2s zcat=%s new=%s base=%s -> %s\n" "$(basename $f)" "$T" "$ref" "$sn" "$sb" "$ok"
  done
done
echo "### PRODUCTION PATH ASSERTION (GZIPPY_DEBUG) ###"
GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$NEW" -d -c -p1 /root/silesia.gz >/dev/null 2>/dev/shm/n39_dbg.txt
grep -o "path=[A-Za-z]*" /dev/shm/n39_dbg.txt | head -1

echo "### MALFORMED-INPUT SAFETY (over-subscribed dynamic table) ###"
# Build a tiny gzip whose deflate dynamic block over-subscribes the litlen tree.
python3 /dev/shm/n39_make_malformed.py /dev/shm/n39_bad.gz
for label in new base; do
  bin=$NEW; [ "$label" = base ] && bin=$BASE
  for T in 1 4; do
    timeout 20 env GZIPPY_FORCE_PARALLEL_SM=1 "$bin" -d -c -p$T /dev/shm/n39_bad.gz >/dev/null 2>/dev/null
    rc=$?
    verdict="ERR-OK(rc=$rc)"; [ $rc -eq 0 ] && verdict="!! ACCEPTED-BAD(rc=0)"; [ $rc -eq 124 ] && verdict="!! HANG(timeout)"; [ $rc -ge 128 ] && verdict="!! CRASH(sig $((rc-128)))"
    printf "  %-5s T%-2s -> %s\n" "$label" "$T" "$verdict"
    [ $rc -eq 0 -o $rc -eq 124 -o $rc -ge 128 ] && fail=1
  done
done
echo "### CORRECTNESS_FAIL=$fail ###"
