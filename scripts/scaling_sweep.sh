#!/usr/bin/env bash
# scaling_sweep.sh — bench gzippy + rapidgzip across T=1,2,4,8,16 to
# locate the parallel-scaling cliff.
#
# Context: the user observed gzippy
# beats rapidgzip at T=1 but loses at T=16. This script measures
# the speedup-per-thread to identify where gzippy's efficiency
# diverges from vendor's.
#
# Usage:
#   scripts/scaling_sweep.sh \
#       --compressed benchmark_data/silesia-large.gz \
#       --original  benchmark_data/silesia-large.bin \
#       --gzippy    target/release/gzippy \
#       --rapidgzip vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip \
#       [--trials 10]   # per-T bench trials
#       [--threads "1 2 4 8 16"]  # quoted, space-sep list
#       [--out-dir target/tooling/scaling-TIMESTAMP]
#
# Output for each T in $OUT_DIR/T-$t/:
#   - bench.json + bench.md (from bench_stats.py)
#   - gzippy.trace.jsonl    (per-event log)
#   - timeline.md           (timeline.py output)
# Plus $OUT_DIR/scaling.md — summary table across all T values.

set -euo pipefail

COMPRESSED=""
ORIGINAL=""
GZIPPY=""
RAPIDGZIP=""
TRIALS=10
THREAD_LIST="1 2 4 8 16"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compressed) COMPRESSED="$2"; shift 2 ;;
    --original)   ORIGINAL="$2"; shift 2 ;;
    --gzippy)     GZIPPY="$2"; shift 2 ;;
    --rapidgzip)  RAPIDGZIP="$2"; shift 2 ;;
    --trials)     TRIALS="$2"; shift 2 ;;
    --threads)    THREAD_LIST="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    -h|--help)    sed -n '3,21p' "$0" >&2; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$COMPRESSED" && -n "$ORIGINAL" && -n "$GZIPPY" && -n "$RAPIDGZIP" ]] || {
  echo "ERROR: --compressed --original --gzippy --rapidgzip all required" >&2
  exit 2
}
[[ -n "$OUT_DIR" ]] || OUT_DIR="target/tooling/scaling-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== scaling_sweep ===" >&2
echo "  out_dir: $OUT_DIR" >&2
echo "  T values: $THREAD_LIST" >&2
echo "  trials per T: $TRIALS" >&2

SCALING_MD="$OUT_DIR/scaling.md"
{
  echo "# Scaling sweep — gzippy vs rapidgzip across T"
  echo ""
  echo "Fixture: \`$COMPRESSED\` | trials per T: $TRIALS"
  echo ""
  echo "| T | gzippy med MB/s | rapidgzip med MB/s | gzippy speedup | rapidgzip speedup | g/r ratio | gzippy CI lo/hi | rapidgzip CI lo/hi |"
  echo "|---|---|---|---|---|---|---|---|"
} > "$SCALING_MD"

# First pass: collect per-T medians.
declare -A GZIPPY_MED RAPIDGZIP_MED GZIPPY_LO GZIPPY_HI RAPIDGZIP_LO RAPIDGZIP_HI
for T in $THREAD_LIST; do
  echo "" >&2
  echo "[T=$T] bench …" >&2
  TDIR="$OUT_DIR/T-$T"
  mkdir -p "$TDIR"
  python3 "$SCRIPT_DIR/bench_stats.py" \
    --compressed "$COMPRESSED" \
    --original   "$ORIGINAL" \
    --gzippy     "$GZIPPY" \
    --rapidgzip  "$RAPIDGZIP" \
    --trials     "$TRIALS" \
    --threads    "$T" \
    --output     "$TDIR/bench.json" \
    --md         "$TDIR/bench.md" 2>&1 | tail -3

  # gzippy timeline trace (single-shot, untimed — for per-thread analysis).
  echo "[T=$T] timeline trace …" >&2
  GZIPPY_LOG_FILE="$TDIR/gzippy.trace.jsonl" \
    "$GZIPPY" -d --processes "$T" \
    < "$COMPRESSED" > /dev/null 2>"$TDIR/gzippy.stderr" || {
      echo "WARN: gzippy trace run failed at T=$T (see $TDIR/gzippy.stderr)" >&2
    }
  if [[ -s "$TDIR/gzippy.trace.jsonl" ]]; then
    python3 "$SCRIPT_DIR/timeline.py" \
      --log "$TDIR/gzippy.trace.jsonl" \
      --output "$TDIR/timeline.md" 2>&1 | tail -1
  fi

  # Extract medians from bench.json.
  GZIPPY_MED[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['gzippy']['throughput_mbps']['median']:.1f}\")")
  RAPIDGZIP_MED[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['rapidgzip']['throughput_mbps']['median']:.1f}\")")
  # CIs from bootstrap.
  GZIPPY_LO[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['gzippy']['throughput_mbps']['p10']:.1f}\")")
  GZIPPY_HI[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['gzippy']['throughput_mbps']['p90']:.1f}\")")
  RAPIDGZIP_LO[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['rapidgzip']['throughput_mbps']['p10']:.1f}\")")
  RAPIDGZIP_HI[$T]=$(python3 -c "import json; d=json.load(open('$TDIR/bench.json')); print(f\"{d['summary']['rapidgzip']['throughput_mbps']['p90']:.1f}\")")
done

# Second pass: emit scaling table with speedup vs T=1.
GZ_BASE=${GZIPPY_MED[1]:-${GZIPPY_MED[$(echo $THREAD_LIST | awk '{print $1}')]}}
RG_BASE=${RAPIDGZIP_MED[1]:-${RAPIDGZIP_MED[$(echo $THREAD_LIST | awk '{print $1}')]}}

for T in $THREAD_LIST; do
  GM=${GZIPPY_MED[$T]}
  RM=${RAPIDGZIP_MED[$T]}
  GSP=$(python3 -c "print(f'{$GM/$GZ_BASE:.2f}x')")
  RSP=$(python3 -c "print(f'{$RM/$RG_BASE:.2f}x')")
  RATIO=$(python3 -c "print(f'{$GM/$RM:.2f}x')")
  echo "| $T | $GM | $RM | $GSP | $RSP | $RATIO | ${GZIPPY_LO[$T]}/${GZIPPY_HI[$T]} | ${RAPIDGZIP_LO[$T]}/${RAPIDGZIP_HI[$T]} |" >> "$SCALING_MD"
done

# Identify the cliff: largest negative delta in (gzippy_speedup_ratio - rapidgzip_speedup_ratio).
echo "" >> "$SCALING_MD"
echo "## Cliff detection" >> "$SCALING_MD"
python3 <<EOF >> "$SCALING_MD"
import json, sys
ts = "$THREAD_LIST".split()
rows = []
for t in ts:
    with open("$OUT_DIR/T-" + t + "/bench.json") as f:
        d = json.load(f)
    rows.append((int(t),
                 d['summary']['gzippy']['throughput_mbps']['median'],
                 d['summary']['rapidgzip']['throughput_mbps']['median']))
rows.sort()
gz_base = rows[0][1]
rg_base = rows[0][2]
deltas = []
for t, gz, rg in rows:
    eff_gz = gz / gz_base / t
    eff_rg = rg / rg_base / t
    deltas.append((t, eff_gz, eff_rg, eff_gz - eff_rg))
    print(f"- T={t}: gzippy efficiency={eff_gz:.3f}, rapidgzip efficiency={eff_rg:.3f}, gap={eff_gz - eff_rg:+.3f}")
worst = min(deltas, key=lambda r: r[3])
print(f"\n**Cliff at T={worst[0]}** (efficiency gap {worst[3]:+.3f}). "
      f"See \`target/tooling/scaling-*/T-{worst[0]}/timeline.md\` for the per-thread Gantt.")
EOF

echo "" >&2
echo "=== done ===" >&2
echo "Summary: $SCALING_MD" >&2
echo "Per-T artifacts: $OUT_DIR/T-*/" >&2
