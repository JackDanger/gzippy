#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${BRANCH:-reimplement-isa-l}"
ART_LOCAL="${ART_LOCAL:-/tmp/gzippy-same-sink-$(date +%Y%m%d-%H%M%S)}"
NEUROTIC=(ssh -o ConnectTimeout=15 neurotic)
GUEST=(ssh -o ConnectTimeout=15 -J neurotic root@10.30.0.199)
BHD=/root/gzippy-bench
mkdir -p "$ART_LOCAL"
"${NEUROTIC[@]}" "mkdir -p $BHD && cat > $BHD/guest_same_sink_floor.sh" <"$ROOT/scripts/bench/guest_same_sink_floor.sh"
"${NEUROTIC[@]}" "scp -o StrictHostKeyChecking=accept-new $BHD/guest_same_sink_floor.sh root@10.30.0.199:$BHD/"
"${GUEST[@]}" "chmod +x $BHD/guest_same_sink_floor.sh"
"${NEUROTIC[@]}" "GUEST_SCRIPT=guest_same_sink_floor.sh bash $BHD/host_lock_and_bench.sh BRANCH=${BRANCH} THREADS=8 N=9" 2>&1 | tee "$ART_LOCAL/host-guest.log"
"${GUEST[@]}" "tar czf - -C /root/gzippy-bench artifacts-same-sink 2>/dev/null" | tar xzf - -C "$ART_LOCAL" || true
echo "Done. log: $ART_LOCAL/host-guest.log"
