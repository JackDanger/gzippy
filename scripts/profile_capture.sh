#!/usr/bin/env bash
# profile_capture.sh — capture gzippy + rapidgzip profiles and produce a
# machine-readable diff.
#
# Linux + perf: uses perf record + stackcollapse-perf.pl → folded stacks.
# macOS:        uses samply record → samply JSON.
# Both feed into scripts/profile_diff.py for a band-classified JSON diff.
#
# Output (one fresh dir per run):
#   $OUT_DIR/gzippy.{folded,profile.json}
#   $OUT_DIR/rapidgzip.{folded,profile.json}
#   $OUT_DIR/diff.json     (machine-readable: bands + L1 distance)
#   $OUT_DIR/diff.md       (human summary table)
#
# Usage:
#   scripts/profile_capture.sh \
#       --compressed benchmark_data/logs.txt.gz \
#       --gzippy    target/release/gzippy \
#       --rapidgzip vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip \
#       --threads 8 \
#       --iterations 20 \
#       --out-dir target/tooling/profile-$(date +%Y%m%d-%H%M%S) \
#       [--baseline target/tooling/profile-LAST/diff.json]

set -euo pipefail

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
REPO_DIR=$( cd "$THIS_DIR/.." && pwd )

COMPRESSED=""
GZIPPY=""
RAPIDGZIP=""
THREADS=8
ITERATIONS=20
OUT_DIR=""
BASELINE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --compressed) COMPRESSED="$2"; shift 2 ;;
        --gzippy)     GZIPPY="$2";     shift 2 ;;
        --rapidgzip)  RAPIDGZIP="$2";  shift 2 ;;
        --threads)    THREADS="$2";    shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2";    shift 2 ;;
        --baseline)   BASELINE="$2";   shift 2 ;;
        *)            echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$COMPRESSED" || -z "$GZIPPY" || -z "$RAPIDGZIP" || -z "$OUT_DIR" ]] && {
    echo "usage: $0 --compressed FILE --gzippy BIN --rapidgzip BIN --out-dir DIR [...]" >&2
    exit 1
}

[[ -f "$COMPRESSED" ]]    || { echo "missing: $COMPRESSED" >&2; exit 1; }
[[ -x "$GZIPPY" ]]        || { echo "not executable: $GZIPPY" >&2; exit 1; }
[[ -x "$RAPIDGZIP" ]]     || { echo "not executable: $RAPIDGZIP" >&2; exit 1; }

mkdir -p "$OUT_DIR"

run_iter_loop() {
    local bin="$1"; local args="$2"; local n="$3"
    for ((i=0; i<n; i++)); do
        eval "$bin $args" >/dev/null
    done
}

UNAME=$(uname)

capture_tool() {
    local tool_label="$1"; local bin="$2"; local extra_args="$3"
    local args="-d -c -p $THREADS $extra_args \"$COMPRESSED\""

    echo "── $tool_label: capturing $ITERATIONS iterations ──" >&2

    if [[ "$UNAME" == "Linux" ]]; then
        local perf_data="$OUT_DIR/$tool_label.perf.data"
        local folded="$OUT_DIR/$tool_label.folded"
        perf record -F 999 -g --call-graph=fp -o "$perf_data" -- \
            bash -c "for i in \$(seq $ITERATIONS); do $bin $args >/dev/null; done"
        # stackcollapse-perf.pl is the canonical Brendan Gregg tool. Try the
        # FlameGraph repo path first, then PATH, then fall back to inferno.
        if command -v stackcollapse-perf.pl >/dev/null 2>&1; then
            perf script -i "$perf_data" | stackcollapse-perf.pl > "$folded"
        elif command -v inferno-collapse-perf >/dev/null 2>&1; then
            perf script -i "$perf_data" | inferno-collapse-perf > "$folded"
        else
            echo "ERROR: need stackcollapse-perf.pl or inferno-collapse-perf in PATH" >&2
            exit 1
        fi
        echo "$folded"
    elif [[ "$UNAME" == "Darwin" ]]; then
        local samply_json="$OUT_DIR/$tool_label.profile.json"
        samply record --save-only -o "$samply_json" -- \
            bash -c "for i in \$(seq $ITERATIONS); do $bin $args >/dev/null; done"
        echo "$samply_json"
    else
        echo "ERROR: unsupported OS $UNAME" >&2
        exit 1
    fi
}

GZIPPY_OUT=$(capture_tool gzippy    "$GZIPPY"    "")
# rapidgzip flags: -d decompress, -c stdout, -P thread count.
RAPIDGZIP_ARGS_OVERRIDE="-d -c -P $THREADS"
echo "── rapidgzip: capturing $ITERATIONS iterations ──" >&2
if [[ "$UNAME" == "Linux" ]]; then
    perf record -F 999 -g --call-graph=fp -o "$OUT_DIR/rapidgzip.perf.data" -- \
        bash -c "for i in \$(seq $ITERATIONS); do $RAPIDGZIP $RAPIDGZIP_ARGS_OVERRIDE \"$COMPRESSED\" >/dev/null; done"
    RAPIDGZIP_OUT="$OUT_DIR/rapidgzip.folded"
    if command -v stackcollapse-perf.pl >/dev/null 2>&1; then
        perf script -i "$OUT_DIR/rapidgzip.perf.data" | stackcollapse-perf.pl > "$RAPIDGZIP_OUT"
    elif command -v inferno-collapse-perf >/dev/null 2>&1; then
        perf script -i "$OUT_DIR/rapidgzip.perf.data" | inferno-collapse-perf > "$RAPIDGZIP_OUT"
    fi
elif [[ "$UNAME" == "Darwin" ]]; then
    RAPIDGZIP_OUT="$OUT_DIR/rapidgzip.profile.json"
    samply record --save-only -o "$RAPIDGZIP_OUT" -- \
        bash -c "for i in \$(seq $ITERATIONS); do $RAPIDGZIP $RAPIDGZIP_ARGS_OVERRIDE \"$COMPRESSED\" >/dev/null; done"
fi

# Determine input flags for profile_diff.py based on capture format.
DIFF_ARGS=""
if [[ "$GZIPPY_OUT" == *.folded ]]; then
    DIFF_ARGS="--gzippy-folded $GZIPPY_OUT --rapidgzip-folded $RAPIDGZIP_OUT"
else
    DIFF_ARGS="--gzippy-samply $GZIPPY_OUT --rapidgzip-samply $RAPIDGZIP_OUT"
fi

BASELINE_ARG=""
if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
    BASELINE_ARG="--baseline $BASELINE"
fi

echo "── computing band-classified diff ──" >&2
python3 "$THIS_DIR/profile_diff.py" \
    $DIFF_ARGS \
    --fixture "$COMPRESSED" \
    --out    "$OUT_DIR/diff.json" \
    --out-md "$OUT_DIR/diff.md" \
    $BASELINE_ARG

echo "" >&2
echo "✓ profile capture + diff complete" >&2
echo "  JSON: $OUT_DIR/diff.json" >&2
echo "  MD:   $OUT_DIR/diff.md" >&2
echo "" >&2

# Final summary to stdout (so the harness/log captures it).
cat "$OUT_DIR/diff.md"
