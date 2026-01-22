#!/bin/bash
RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_silesia -- --nocapture
