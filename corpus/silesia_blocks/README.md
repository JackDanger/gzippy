# silesia_blocks corpus

Canonical microbench corpus for the inner-loop optimization work (Phase 1.1
of `plans/inner-loop-execution.md`).

## What's here

- `NNN.bin` — one DEFLATE block per file, in the binary format
  documented in `tools/extract_blocks/src/main.rs` preamble (search for
  `GZBLK01` magic).
- `INDEX.json` — human-readable index with per-block metadata
  (`block_idx`, `decoded_len`, `compressed_bit_len`,
  `max_litlen_code_len`, `max_dist_code_len`,
  `literal_code_count`, `distance_code_count`).

Each block is self-contained: the predecessor 32 KiB window, the
compressed bitstream, the expected decoded output, and the litlen +
distance code lengths.

## Source

Extracted from `/root/benchmark_data/silesia-gzip.tar.gz` (silesia tarball
gzipped, standard benchmark fixture) on neurotic. The full silesia gzip
has ~3357 dynamic-Huffman blocks; this corpus picks 30 via stratified
sampling on `max_litlen_code_len` to cover diverse Huffman shapes.

## Selection criteria

Phase 1.1 spec calls for 30-50 blocks stratified by LUT pressure:

- shallow (max_litlen ≤ 9): LUT-friendly
- typical (10-12): the bulk
- deep (≥ 13): LUT-stressing
- back-ref heavy: high decoded/compressed-bit ratio

Silesia happens to have no `max_litlen ≤ 9` blocks (compressed at typical
gzip default levels), so this corpus picked 20 typical + 10 deep + 0
back-ref-heavy distinct entries.

## Reproducibility

The corpus is **frozen**. Do not regenerate. To verify a block file
hasn't bit-rotted, the `gzippy-extract-blocks` tool's `verify_extraction`
path re-decodes each block via the same `set_initial_window` + body
loop and asserts byte-equality with the snapshot.

A `make extract-corpus-verify` target (added in Phase 1.2) re-runs
verification on every block in this directory and fails if any
re-decode diverges. This catches:
- Bit-rot in the binary files.
- Refactor of `Block` that breaks the seed-window contract.
- Toolchain regressions affecting deterministic decode.

## Usage

- Rust microbench: `benches/inflate_block.rs` (Criterion) reads
  these files via the parser in `gzippy::testing::block_corpus`.
- C++ vendor harness: `tools/vendor_inflate_bench/main.cpp` reads the
  same files via a mirror of the binary format.
- Differential tests: `tests/inflate_inner_differential.rs` uses a
  trimmed-down committed subset (`tests/fixtures/inflate_blocks/`)
  for fast `cargo test` runs.

## Regenerating from scratch (if absolutely necessary)

If `Block` changes break extraction:

```bash
cd tools/extract_blocks
cargo build --release
./target/release/gzippy-extract-blocks \
  /path/to/silesia-gzip.tar.gz \
  ../../corpus/silesia_blocks 40
```

Then audit the diff against the committed corpus carefully. Any change
to a `.bin` file means the inner-loop fingerprint of that block has
moved; bench-baseline data committed before the change is no longer
comparable to data after.
