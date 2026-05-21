#!/usr/bin/env bash
# profile_compare.sh — side-by-side flamegraph + perf-stat for gzippy vs
# rapidgzip on the canonical Silesia fixture.
#
# Designed to run on neurotic (x86_64 Linux + perf). On macOS this is a
# no-op with a helpful message; use scripts/profile_compare_macos.sh if you
# need samply-based profiling on Darwin.
#
# Outputs (one fresh dir per run):
#   $OUT_DIR/gzippy.perfstat.txt
#   $OUT_DIR/rapidgzip.perfstat.txt
#   $OUT_DIR/gzippy.flamegraph.svg
#   $OUT_DIR/rapidgzip.flamegraph.svg
#   $OUT_DIR/perfstat_diff.md
#
# Usage:
#   scripts/profile_compare.sh \
#       --compressed benchmark_data/silesia-large.bin.gz \
#       --gzippy    target/release/gzippy \
#       --rapidgzip vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip \
#       --threads 16 \
#       --out-dir target/tooling/profile-$(date +%Y%m%d-%H%M%S)

set -euo pipefail

# Handle --help before the Linux/perf preflight so docs work everywhere.
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    sed -n '3,22p' "$0" >&2
    exit 0
  fi
done

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "ERROR: this script needs Linux + perf. On macOS, run on neurotic" >&2
  echo "       or write scripts/profile_compare_macos.sh using samply." >&2
  exit 2
fi

if ! command -v perf >/dev/null; then
  echo "ERROR: perf not installed (apt install linux-perf-tools or distro equivalent)" >&2
  exit 2
fi

COMPRESSED=""
GZIPPY=""
RAPIDGZIP=""
THREADS=16
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compressed) COMPRESSED="$2"; shift 2 ;;
    --gzippy)     GZIPPY="$2"; shift 2 ;;
    --rapidgzip)  RAPIDGZIP="$2"; shift 2 ;;
    --threads)    THREADS="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '3,22p' "$0" >&2; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$COMPRESSED" ]] || { echo "ERROR: --compressed required" >&2; exit 2; }
[[ -n "$GZIPPY" ]]     || { echo "ERROR: --gzippy required"     >&2; exit 2; }
[[ -n "$RAPIDGZIP" ]]  || { echo "ERROR: --rapidgzip required"  >&2; exit 2; }
[[ -n "$OUT_DIR" ]]    || OUT_DIR="target/tooling/profile-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"

# Pick a flamegraph generator. Prefer inferno (Rust port, single binary,
# no Perl). Fall back to FlameGraph.pl from cloned brendangregg/FlameGraph
# if inferno isn't installed.
FLAMEGRAPH_TOOL=""
if command -v inferno-collapse-perf >/dev/null && command -v inferno-flamegraph >/dev/null; then
  FLAMEGRAPH_TOOL="inferno"
elif command -v flamegraph.pl >/dev/null; then
  FLAMEGRAPH_TOOL="perl"
fi

# perf record events.
PERF_EVENTS="cycles:u,instructions:u,cache-references:u,cache-misses:u,branches:u,branch-misses:u,task-clock,context-switches"

run_perf_stat() {
  local label="$1" cmd_path="$2" arg_list_var="$3"
  local cmd_args=()
  eval "cmd_args=(\"\${$arg_list_var[@]}\")"
  echo "" >&2
  echo "[perf stat] $label" >&2
  perf stat -e "$PERF_EVENTS" -- "$cmd_path" "${cmd_args[@]}" \
    < "$COMPRESSED" > /dev/null 2> "$OUT_DIR/${label}.perfstat.txt"
}

run_perf_record() {
  local label="$1" cmd_path="$2" arg_list_var="$3"
  local cmd_args=()
  eval "cmd_args=(\"\${$arg_list_var[@]}\")"
  local pdata="$OUT_DIR/${label}.perf.data"
  local folded="$OUT_DIR/${label}.folded"
  local svg="$OUT_DIR/${label}.flamegraph.svg"
  echo "" >&2
  echo "[perf record] $label" >&2
  perf record -F 999 --call-graph=dwarf -o "$pdata" -- \
    "$cmd_path" "${cmd_args[@]}" \
    < "$COMPRESSED" > /dev/null 2>>"$OUT_DIR/perf.stderr.log"

  if [[ "$FLAMEGRAPH_TOOL" == "inferno" ]]; then
    perf script -i "$pdata" 2>>"$OUT_DIR/perf.stderr.log" | \
      inferno-collapse-perf > "$folded"
    inferno-flamegraph --title "$label" < "$folded" > "$svg"
  elif [[ "$FLAMEGRAPH_TOOL" == "perl" ]]; then
    perf script -i "$pdata" 2>>"$OUT_DIR/perf.stderr.log" | \
      stackcollapse-perf.pl > "$folded"
    flamegraph.pl --title "$label" "$folded" > "$svg"
  else
    echo "WARN: no flamegraph tool found; skipping SVG generation" >&2
    echo "      install inferno: cargo install inferno" >&2
    echo "      or clone brendangregg/FlameGraph and add to PATH" >&2
    rm -f "$folded"
    return 0
  fi
  echo "  → $svg" >&2
}

GZIPPY_ARGS=("-d" "--processes" "$THREADS")
RAPIDGZIP_ARGS=("-d" "-P" "$THREADS")

run_perf_stat gzippy    "$GZIPPY"    GZIPPY_ARGS
run_perf_stat rapidgzip "$RAPIDGZIP" RAPIDGZIP_ARGS

run_perf_record gzippy    "$GZIPPY"    GZIPPY_ARGS
run_perf_record rapidgzip "$RAPIDGZIP" RAPIDGZIP_ARGS

# ---------- Side-by-side perf stat diff ----------
DIFF_MD="$OUT_DIR/perfstat_diff.md"
{
  echo "# perf stat — gzippy vs rapidgzip"
  echo ""
  echo "Fixture: \`$COMPRESSED\`  threads: $THREADS"
  echo ""
  echo "## gzippy"
  echo ""
  echo '```'
  cat "$OUT_DIR/gzippy.perfstat.txt"
  echo '```'
  echo ""
  echo "## rapidgzip"
  echo ""
  echo '```'
  cat "$OUT_DIR/rapidgzip.perfstat.txt"
  echo '```'
} > "$DIFF_MD"

echo "" >&2
echo "=== done ===" >&2
echo "perf stat diff: $DIFF_MD" >&2
[[ -f "$OUT_DIR/gzippy.flamegraph.svg"    ]] && echo "gzippy flamegraph:    $OUT_DIR/gzippy.flamegraph.svg" >&2
[[ -f "$OUT_DIR/rapidgzip.flamegraph.svg" ]] && echo "rapidgzip flamegraph: $OUT_DIR/rapidgzip.flamegraph.svg" >&2
