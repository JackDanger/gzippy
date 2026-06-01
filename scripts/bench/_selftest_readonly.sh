#!/usr/bin/env bash
# Read-only self-test of lib primitives. Mutates NOTHING (writes only to /tmp).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/lib_state.sh"
. "$HERE/lib_gate.sh"

echo "=== base_mhz_from_msr ==="
base_mhz_from_msr

echo "=== write_state to /tmp/st ==="
write_state /tmp/st && echo "write ok" && cat /tmp/st

echo "=== state_get round-trip ==="
echo "NO_TURBO=$(state_get /tmp/st NO_TURBO)"
echo "UNCORE=$(state_get /tmp/st UNCORE_0x620)"
echo "THP=$(state_get /tmp/st THP)"
echo "GOVERNORS(head)=$(state_get /tmp/st GOVERNORS | cut -c1-60)..."

rm -f /tmp/st
echo "=== DONE (nothing mutated) ==="
