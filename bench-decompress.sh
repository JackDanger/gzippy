#!/bin/bash
set -e

# =============================================================================
# Decompression Benchmark
# =============================================================================
# Benchmarks gzippy (Rust) vs libdeflater crate (C libdeflate)
# Tests 3 archive types: silesia (mixed), software (source code), logs (repetitive)
#
# Usage:
#   ./bench-decompress.sh                  Run speed benchmark (10 iterations)
#   ./bench-decompress.sh --runs 50        Run with 50 iterations for stable results
#   ./bench-decompress.sh --analyze        Run detailed analysis (block types, cache stats)
#   ./bench-decompress.sh --prepare        Only download/prepare benchmark data
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare benchmark data using shared script
"$SCRIPT_DIR/scripts/prepare_benchmark_data.sh" all

export RUSTFLAGS="-C target-cpu=native"

# Parse arguments
ANALYZE=false
PROFILE=false
PREPARE_ONLY=false
RUNS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --analyze)
            ANALYZE=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --prepare)
            PREPARE_ONLY=true
            shift
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./bench-decompress.sh [--runs N] [--analyze] [--profile] [--prepare]"
            exit 1
            ;;
    esac
done

# Exit early if only preparing data
if [[ "$PREPARE_ONLY" == "true" ]]; then
    echo "Benchmark data prepared successfully."
    exit 0
fi

# Export BENCH_RUNS if specified
if [[ -n "$RUNS" ]]; then
    export BENCH_RUNS="$RUNS"
fi

if [[ "$PROFILE" == "true" ]]; then
    cargo test --release --features profile bench_profile -- --nocapture
elif [[ "$ANALYZE" == "true" ]]; then
    cargo test --release bench_analyze -- --nocapture
else
    cargo test --release bench_decompress -- --nocapture
fi
