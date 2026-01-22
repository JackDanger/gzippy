#!/bin/bash
set -e

echo "Running gzippy Consume-First benchmarks..."
echo "Datasets: silesia (mixed), software (source code), logs (repetitive)"
echo ""

# Ensure we use native CPU optimizations
export RUSTFLAGS="-C target-cpu=native"

# Run all benchmarks matching 'bench_cf_'
cargo test --release bench_cf_ -- --nocapture
