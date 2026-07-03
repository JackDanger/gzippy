# gzippy parallel-SM fuzzing

cargo-fuzz targets for the pure-Rust parallel single-member (`ParallelSM`) decode
path — the sole production single-member decoder.

## Target: `parallel_sm_roundtrip`

Two arms per input:

1. **Round-trip correctness.** Compress the fuzz bytes with flate2/zlib-ng (an
   independent gzip encoder), decode with gzippy at T=8 (parallel caps disabled so
   the parallel pipeline runs instead of the small-output serial floor), and assert
   byte-exact recovery. A mismatch or a decode `Err` on a well-formed stream is a bug.
2. **Malformed-input robustness.** Feed raw fuzz bytes to the decoder — restricted to
   inputs the router classifies as `ParallelSM`/`StoredParallel` so this target fuzzes
   its own path. Must not panic (an `Err` is fine).

The fuzz `[profile.release]` disables `overflow-checks` to match the shipped release
profile's arithmetic (wrapping, not panicking), so the target reports real production
panics (bounds OOB, unwraps) and mismatches rather than benign bit-reader wraps.

## Run

```
cargo +nightly fuzz run parallel_sm_roundtrip -- -max_total_time=180
```

## Bugs found (all fixed in this branch)

- **`decompress/format.rs` `is_likely_multi_member` slice OOB panic** — a malformed
  first header (FLG=0xbc) made `parse_gzip_header_size` report a size past the buffer,
  so `data[pos..scan_end]` panicked. Fixed with a `pos >= scan_end` guard. This is a
  real production panic (bounds checks are profile-independent) on the classify step
  that precedes ParallelSM dispatch. Regression test in `format.rs`.
- **`inflate/consume_first_decode.rs` `Bits::consume` subtract-with-overflow** — the
  bit-reader decremented `bitsleft` with `-=`; on malformed deflate headers it
  underflowed. Fixed to `wrapping_sub`, matching the sibling `consume_entry` and
  libdeflate's "garbage in high bits" model (byte-identical for valid streams).
  Fuzz-build-only panic (production wraps), fixed for consistency + panic-freedom.

## Separate finding (NOT this target's path; documented, not fixed here)

- **`decompress/mod.rs` `decompress_multi_member_sequential` OOM** — the output buffer
  doubles unboundedly (line ~515) on a crafted multi-member stream, so a ~50-byte input
  can drive a multi-GB allocation (DoS). This is the multi-member sequential decoder,
  not the parallel-SM path; needs its own bounded-growth fix + tests. ARM 2 now skips
  multi-member-classified inputs so it does not mask parallel-SM findings.
