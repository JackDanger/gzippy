#!/usr/bin/env bash
# trace_parity_check.sh — correctness gate + optional profile diff per format.
#
# Usage:
#   scripts/trace_parity_check.sh
#   scripts/trace_parity_check.sh --profile --format single-member \
#       --source benchmark_data/silesia-large.bin --threads 16
#   scripts/trace_parity_check.sh --profile --all-formats --source /path/to/raw.bin
#
# Steps:
#   1. cargo test --release trace_parity  (all formats, thread sweep)
#   2. Optional: profile_capture.sh for gzippy vs rapidgzip (one format)

set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$THIS_DIR/.." && pwd)
cd "$REPO_DIR"

PROFILE=0
FORMAT="single-member"
ALL_FORMATS=0
SOURCE=""
THREADS=16
OUT_DIR=""
BASELINE=""
GZIPPY="${GZIPPY:-target/release/gzippy}"
RAPIDGZIP="${RAPIDGZIP:-vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)      PROFILE=1; shift ;;
        --format)       FORMAT="$2"; shift 2 ;;
        --all-formats)  ALL_FORMATS=1; shift ;;
        --source)       SOURCE="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --baseline)     BASELINE="$2"; shift 2 ;;
        --gzippy)       GZIPPY="$2"; shift 2 ;;
        --rapidgzip)    RAPIDGZIP="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "══ trace parity: correctness (cargo test trace_parity) ══" >&2
cargo test --release trace_parity

if [[ "$PROFILE" -eq 0 ]]; then
    echo "OK: correctness gate passed (no --profile)." >&2
    exit 0
fi

[[ -x "$GZIPPY" ]] || { echo "build gzippy first: cargo build --release" >&2; exit 1; }
[[ -x "$RAPIDGZIP" ]] || { echo "missing rapidgzip: $RAPIDGZIP" >&2; exit 1; }

if [[ -z "$SOURCE" ]]; then
    echo "--profile requires --source (raw uncompressed file)" >&2
    exit 1
fi
[[ -f "$SOURCE" ]] || { echo "missing source: $SOURCE" >&2; exit 1; }

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="target/tooling/trace-parity-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

profile_one() {
    local slug="$1"
    local fixture="$OUT_DIR/${slug}.fixture"
    case "$slug" in
        single-member)
            if command -v gzip >/dev/null 2>&1; then
                gzip -9 -c "$SOURCE" >"$fixture"
            else
                "$GZIPPY" -c <"$SOURCE" >"$fixture"
                echo "WARN: gzip(1) not found; using gzippy -c fixture" >&2
            fi
            ;;
        multi-member)
            python3 - "$SOURCE" "$fixture" <<'PY'
import subprocess, sys
from pathlib import Path
src = Path(sys.argv[1]).read_bytes()
dst = Path(sys.argv[2])
parts = 8
chunk = (len(src) + parts - 1) // parts
with dst.open("wb") as out:
    for i in range(0, len(src), chunk):
        out.write(subprocess.run(["gzip", "-c"], input=src[i : i + chunk], check=True, capture_output=True).stdout)
PY
            ;;
        bgzf)
            if command -v bgzip >/dev/null 2>&1; then
                bgzip -c "$SOURCE" >"$fixture"
            else
                echo "SKIP bgzf: bgzip not installed" >&2
                return 0
            fi
            ;;
        gz-subfield)
            "$GZIPPY" -c -i <"$SOURCE" >"$fixture"
            ;;
        *)
            echo "unknown format: $slug" >&2
            return 1
            ;;
    esac

    local prof_dir="$OUT_DIR/profile-${slug}"
    mkdir -p "$prof_dir"
    echo "── profile $slug → $prof_dir ──" >&2
    local extra=()
    [[ "$slug" == "gz-subfield" ]] && extra=(--gzippy-only)
    if [[ ${#extra[@]} -eq 0 ]]; then
        "$THIS_DIR/profile_capture.sh" \
            --compressed "$fixture" \
            --gzippy "$GZIPPY" \
            --rapidgzip "$RAPIDGZIP" \
            --threads "$THREADS" \
            --iterations 15 \
            --out-dir "$prof_dir" \
            ${BASELINE:+--baseline "$BASELINE"}
    else
        echo "NOTE: $slug has no rapidgzip leg; profile gzippy only via manual run" >&2
    fi
}

if [[ "$ALL_FORMATS" -eq 1 ]]; then
    for slug in single-member multi-member bgzf gz-subfield; do
        profile_one "$slug" || true
    done
else
    profile_one "$FORMAT"
fi

echo "OK: trace parity check complete → $OUT_DIR" >&2
