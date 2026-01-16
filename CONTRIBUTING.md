# Contributing to rigz

## Quick Start

```bash
make         # Build and run quick benchmarks (<30s)
make validate  # Verify output works with gunzip
cargo test   # Run unit tests
```

## Architecture

```
src/
├── main.rs                     # CLI entry point
├── cli.rs                      # gzip-compatible argument parsing
├── compression.rs              # File compression orchestration
├── decompression.rs            # File decompression
├── parallel_compress.rs        # Rayon-based parallel gzip (the core)
├── optimization.rs             # Content detection, thread tuning
├── simple_optimizations.rs     # SimpleOptimizer wrapper
├── error.rs                    # Error types
├── format.rs                   # Gzip/zlib format detection
└── utils.rs                    # File utilities
```

**Critical paths:**
- **Single-threaded**: `compression.rs` → direct `GzEncoder` (no overhead)
- **Multi-threaded**: `compression.rs` → `mmap` → `ParallelGzEncoder` → rayon

## Hard-Won Lessons

### 1. Use zlib-ng for maximum performance

```toml
# zlib-ng is 2-3x faster than standard zlib (SIMD-optimized)
flate2 = { version = "1.0", default-features = false, features = ["zlib-ng"] }
```

zlib-ng produces valid gzip with equal or better compression ratios. The only reason to use standard zlib is byte-for-byte identical output to `gzip`, which is rarely needed.

### 2. Fixed block size beats dynamic

Use 128KB blocks like pigz. Dynamic sizing based on file size caused 14% regression.

### 3. Single-threaded must be zero-overhead

```rust
if thread_count == 1 {
    // Go directly to flate2, skip all optimizer logic
    let mut encoder = GzEncoder::new(writer, compression);
    io::copy(&mut reader, &mut encoder)?;
    encoder.finish()?;
}
```

### 4. Benchmarking needs many runs

| File Size | Minimum Runs | Why |
|-----------|--------------|-----|
| 1MB | 20 | Coefficient of variation can be 20%+ |
| 10MB | 10 | CV is 2-5% |
| 100MB | 5 | CV is <2% |

Use **median** not min. 3 runs gave us false failures.

### 5. mmap for zero-copy I/O

For compression: mmap files >128KB (parallel mode benefits from zero-copy access).
For decompression: always use mmap (faster than buffered reads on all platforms).

### 6. Global thread pool

Creating a rayon thread pool per compression is expensive:

```rust
static THREAD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
```

## Performance Testing

```bash
make quick      # Fast iteration (<30s)
make perf-full  # Full suite (10+ min)
```

**Targets:**
- Single-threaded: Match gzip (within 5%)
- Multi-threaded: Beat pigz

## Pull Request Checklist

- [ ] `cargo test` passes
- [ ] `make validate` passes (output works with gunzip)
- [ ] `make quick` shows no regressions
- [ ] Single-threaded compression ratio matches gzip (within 0.1%)

## Future Optimization Ideas

- **libdeflate** - Even faster single-block compression (but non-streaming API)
- **Intel ISA-L** - Hardware-accelerated on Intel CPUs
- **Shared dictionaries** - Like pigz, improve compression between adjacent blocks
- **Parallel decompression** - Scan for gzip member boundaries and inflate in parallel