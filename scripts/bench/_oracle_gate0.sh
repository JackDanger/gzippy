#!/bin/bash
# _oracle_gate0.sh — GATE-0 self-validation for the removal-oracle arms on the
# PINNED b22e1b14 binary. Untimed (correctness/non-inertness only).
#   arg1 CORPUS (silesia), arg2 T, arg3 MASK, arg4 RECFILE
set -u
C=${1:?corpus}; T=${2:?t}; MASK=${3:?mask}; REC=${4:?recfile}
BIN=/dev/shm/gz-b22-target/release/gzippy
echo "## BIN=$BIN"; ls -la "$BIN"
echo "## /dev/null is char-special BEFORE:"; test -c /dev/null && echo "  OK /dev/null char-special" || { echo "  FATAL /dev/null not char-special"; exit 9; }

echo "=== A. baseline sha (real decode+emit) ==="
BASE_SHA=$(env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "  base_sha=$BASE_SHA"

echo "=== B. NOSTORE banner fires (non-inert) + bytes DIFFER from base ==="
NS_SHA=$(env GZIPPY_ORACLE_NOSTORE=1 GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" 2>/tmp/g0.ns.err | sha256sum | cut -d' ' -f1)
echo "  nostore banner count=$(grep -c 'NOSTORE ACTIVE' /tmp/g0.ns.err)"
echo "  nostore_sha=$NS_SHA"
[ "$NS_SHA" != "$BASE_SHA" ] && echo "  OK nostore bytes DIFFER (garbage, as designed)" || echo "  WARN nostore bytes == base (store elide may be inert!)"

echo "=== C. RECORD pass (generate symbol stream) ==="
env GZIPPY_ORACLE_RECORD="$REC" GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" >/dev/null 2>/tmp/g0.rec.err
grep -E 'RECORD (ACTIVE|wrote)' /tmp/g0.rec.err
ls -la "$REC" 2>&1

echo "=== D. NODECODE replay: hits>0, byte-EXACT vs base ==="
ND_SHA=$(env GZIPPY_ORACLE_NODECODE="$REC" GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p"$T" "/root/$C.gz" 2>/tmp/g0.nd.err | sha256sum | cut -d' ' -f1)
echo "  nodecode banner: $(grep -c 'NODECODE ACTIVE' /tmp/g0.nd.err)"
grep -E 'warm_replay|replay: hits' /tmp/g0.nd.err
echo "  nodecode_sha=$ND_SHA"
[ "$ND_SHA" = "$BASE_SHA" ] && echo "  OK nodecode byte-EXACT vs base" || echo "  WARN nodecode bytes DIFFER from base (replay miss/incorrect!)"

echo "## /dev/null is char-special AFTER:"; test -c /dev/null && echo "  OK /dev/null char-special" || echo "  FATAL /dev/null clobbered"
