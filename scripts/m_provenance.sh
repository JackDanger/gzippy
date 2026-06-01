#!/usr/bin/env bash
# Provenance witness for the measurement run (RUN ON GUEST 199).
# Asserts: pure-rust build (isal_inflate dynsym=0), path=IsalParallelSM,
# sha == rapidgzip oracle, RAW size match. Emits RUN_TRUSTWORTHY.
set -u
B=/root/gzippy/target/release/gzippy
F=/root/gzippy/benchmark_data/silesia-large.gz
RAPID=/usr/local/bin/rapidgzip
RAW_EXPECT=503627776
export GZIPPY_FORCE_PARALLEL_SM=1

echo "## PROVENANCE WITNESS"
echo "gzippy_head=$(cd /root/gzippy && git rev-parse --short HEAD) branch=$(cd /root/gzippy && git branch --show-current)"
echo "dirty=$(cd /root/gzippy && git status --porcelain --ignore-submodules=dirty | grep -vE '(Cargo\.lock|target/?)' | wc -l | tr -d ' ')"

# pure-rust proof: ISA-L inner-decode symbols must be ABSENT (dynsym=0)
ISAL_DYN=$(nm -D "$B" 2>/dev/null | grep -c 'isal_inflate' || echo 0)
ISAL_ANY=$(nm "$B" 2>/dev/null | grep -c 'isal_inflate' || echo 0)
echo "isal_inflate_dynsym=$ISAL_DYN  isal_inflate_anysym=$ISAL_ANY  (pure-rust => both 0)"

# path proof
PATHLINE=$(GZIPPY_DEBUG=1 "$B" -d -c -p 8 "$F" 2>&1 >/dev/null | grep -iE 'path=' | head -1)
echo "path_proof=$PATHLINE"

# sha oracle: gzippy == rapidgzip == gzip(1)
GS=$("$B" -d -c -p 8 "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
RS=$("$RAPID" -d -c -P 8 "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
ZS=$(gzip -dc "$F" 2>/dev/null | sha256sum | cut -d' ' -f1)
RAW=$(gzip -dc "$F" 2>/dev/null | wc -c)
echo "sha_gzippy=$GS"
echo "sha_rapid =$RS"
echo "sha_gzip  =$ZS"
echo "raw_bytes=$RAW expect=$RAW_EXPECT"

OK=1
[ "$ISAL_DYN" = 0 ] || { echo "FAIL: isal dynsym != 0 (not pure-rust)"; OK=0; }
echo "$PATHLINE" | grep -q IsalParallelSM || { echo "FAIL: path != IsalParallelSM"; OK=0; }
[ "$GS" = "$RS" ] && [ "$GS" = "$ZS" ] || { echo "FAIL: sha mismatch"; OK=0; }
[ "$RAW" = "$RAW_EXPECT" ] || { echo "FAIL: raw size"; OK=0; }
echo "RUN_TRUSTWORTHY=$([ $OK = 1 ] && echo true || echo false)"
