#!/bin/bash
# Verify every tool decompresses a corpus to bytes identical to `gzip -dc`.
set -u
C=/tmp/corpora/monorepo.gz
GZIPPY=/mnt/internal/bin/gzippy-native
MINIGZIP=/mnt/internal/bin/minigzip-ng
RG=/root/oracle_c/rapidgzip-native
LD=/usr/bin/libdeflate-gunzip
REF=$(gzip -dc "$C" | sha256sum | cut -d' ' -f1)
echo "REF(gzip)      = $REF"
check() { local name="$1"; shift; local s; s=$("$@" 2>/dev/null | sha256sum | cut -d' ' -f1); if [ "$s" = "$REF" ]; then echo "OK   $name = $s"; else echo "FAIL $name = $s"; fi; }
check "gzippy-native" "$GZIPPY" -d -c -p 4 "$C"
check "igzip"         igzip -d -c "$C"
check "libdeflate"    "$LD" -d -c "$C"
check "minigzip-ng"   "$MINIGZIP" -d -c "$C"
check "pigz"          pigz -d -c -p 4 "$C"
check "rapidgzip"     "$RG" -d -c -f -P 4 "$C"
echo "=== version/provenance ==="
echo -n "gzippy: "; "$GZIPPY" --version 2>&1 | head -1
echo -n "igzip:  "; igzip --version 2>&1 | head -1
echo -n "libdef: "; "$LD" 2>&1 | head -1 || true
echo -n "minigz: "; "$MINIGZIP" --version 2>&1 | head -1 || true
echo -n "pigz:   "; pigz --version 2>&1 | head -1
echo -n "rg:     "; "$RG" --version 2>&1 | head -1
echo "=== sha realdata decodes (gzippy vs gzip) ==="
RR=$(gzip -dc /tmp/corpora/squishy_realdata.gz | sha256sum | cut -d' ' -f1)
GR=$("$GZIPPY" -d -c -p 4 /tmp/corpora/squishy_realdata.gz | sha256sum | cut -d' ' -f1)
echo "realdata gzip=$RR gzippy=$GR $([ "$RR" = "$GR" ] && echo MATCH || echo MISMATCH)"
