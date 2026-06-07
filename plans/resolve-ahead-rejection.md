# resolve-ahead §3.1 key re-key — REJECTION RECORD

Required by CLAUDE.md ("a rejection needs a mechanism, not a code comment"). This
captures the *reproduced* CRC32 break that retires §3.1 of `plans/resolve-ahead-fix.md`.

`cf` = `src/decompress/parallel/chunk_fetcher.rs`.
Reproduced at HEAD `0a3e9a3` (Part A live counters present), scratch edit uncommitted,
then fully reverted (`chunk_fetcher.rs` == committed HEAD, verified clean).

---

## 1. The §3.1 change attempted (exact scratch diff)

§3.1 proposed: in `queue_prefetched_marker_postprocess` (cf:2396), instead of keying the
predecessor-window lookup on `chunk_consumer_handoff_bit(arc) = arc.max_acceptable_start_bit`,
key it on the OFFSET-SORTED PREVIOUS chunk's published end
(`prev.encoded_offset_bits + prev.encoded_size_bits`), per vendor `GzipChunkFetcher.hpp:557`.

Applied minimally over the offset-sorted `contents` loop (cf:2411). Faithful to the §3.1
"After:" block — the window-eligibility probe was ALSO re-keyed (keeping only the
`has_been_post_processed` guard), because §3.1 explicitly replaces the
`chunk_may_resolve_markers_early` window-probe with `window_map.get(handoff_bit)`:

```diff
     let mut submitted = 0usize;
+    let mut prev: Option<ChunkArc> = None; // §3.1 SCRATCH
     for (cache_key, arc) in contents {
+        // §3.1 SCRATCH: key on the offset-sorted predecessor's published end.
+        let scratch_handoff_bit = match prev.as_ref() {
+            Some(p) => p.encoded_offset_bits + p.encoded_size_bits, // vendor windowOffset, GzipChunkFetcher.hpp:557
+            None => chunk_consumer_handoff_bit(arc.as_ref()),       // frontier seed: keep current behavior
+        };
+        prev = Some(Arc::clone(&arc));
         if skip_cache_keys.contains(&cache_key) { continue; }
         let real_offset = arc.encoded_offset_bits;
         if skip_real_offsets.contains(&real_offset) { continue; }
-        if !chunk_may_resolve_markers_early(arc.as_ref(), window_map) { continue; }
+        // keep has_been_post_processed guard, gate window presence on the predecessor key
+        if arc.has_been_post_processed(false) { continue; }
+        if window_map.get(scratch_handoff_bit).is_none() { continue; }
         if in_flight.contains_key(&real_offset) { continue; }
         if arc.data_with_markers.is_empty() { continue; }
         ...
-        let handoff_bit = chunk_consumer_handoff_bit(arc.as_ref());
+        let handoff_bit = scratch_handoff_bit; // §3.1 SCRATCH
         ...
         let Some((pred_key, predecessor_window)) =
             confirmed_predecessor_window(window_map, handoff_bit) else { continue; };
```

> NOTE: a FIRST attempt that re-keyed ONLY the final `handoff_bit` lookup (cf:2436) and left
> the `chunk_may_resolve_markers_early` eligibility probe on `max_acceptable_start_bit`
> PASSED — that gate filtered the offending chunks out before the wrong-key lookup. That is
> not §3.1: §3.1's "After:" block replaces the gate too. Only the faithful version (both the
> gate AND the lookup re-keyed) exercises the broken path. This matters: an unfaithful
> partial apply silently hides the bug.

## 2. Failing test + ACTUAL captured failure (verbatim)

Test: `tests::routing::tests::test_coalesce_fixed_huffman_multithread_byte_exact`
Build: `cargo test --release --no-default-features --features pure-rust-inflate <test> -- --nocapture`

Verbatim panic (from /tmp/stage2-test.log):

```
running 1 test

thread 'tests::routing::tests::test_coalesce_fixed_huffman_multithread_byte_exact' (3437705) panicked at src/tests/routing.rs:365:37:
coalesce decode failed at T=8: Decompression("parallel SM: CRC32 mismatch")
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test tests::routing::tests::test_coalesce_fixed_huffman_multithread_byte_exact ... FAILED

failures:
    tests::routing::tests::test_coalesce_fixed_huffman_multithread_byte_exact

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 876 filtered out; finished in 0.71s
```

The test (routing.rs:354) builds a 32 MiB BTYPE=00/01 fixed-Huffman-heavy fixture, compresses
it (> 10 MiB, so the parallel single-member gate fires), and decodes at T ∈ {2,4,8,16}
asserting byte-exact output. The break surfaces at **T=8** with a hard
`Decompression("parallel SM: CRC32 mismatch")` — i.e. the pipeline produced WRONG BYTES; the
gzip-trailer CRC32 caught it terminally (no silent fallback, per CLAUDE.md rule 5). The same
seed key (HEAD) is green; the predecessor key is red.

## 3. MECHANISM — why offset-sorted `prev` is NOT the true contiguous decode predecessor at resolve-ahead time

Resolve-ahead (`queue_prefetched_marker_postprocess`) runs over the SPECULATIVE prefetch cache
**before block-finder confirmation**. The cache holds range-speculative chunks whose metadata
was rewritten in `try_speculative_decode_candidate` (cf:3096-3189):

```
cf:3181  chunk.encoded_offset_bits   = partition_seed;            // = offset.first  (partition GUESS)
cf:3182  chunk.encoded_size_bits     = encoded_end - partition_seed;
cf:3183  chunk.max_acceptable_start_bit = decode_start;           // = offset.second (block-aligned bit the worker actually seeked to)
```

So for a speculative chunk N, `prev.encoded_offset_bits + prev.encoded_size_bits` is
`prev`'s partition SEED plus its decoded size — anchored to the **partition guess**, never
re-anchored until the consumer's `set_encoded_offset` at drain. The offset-sorted scan orders
the cache by these *guessed* seeds, so the "previous element" is the previous-by-guess chunk,
NOT the proven contiguous decode predecessor of N. Two ways it diverges:

- **Mis-speculation / deep prefetch:** N's true decode-start (`max_acceptable_start_bit`,
  offset.second) need not equal `prev`'s published end. The §3.1 key
  `prev.encoded_offset_bits + prev.encoded_size_bits` then names a DIFFERENT published window
  that the dense window map usually DOES contain → the `window_map.get` guard passes → N is
  resolved against the WRONG predecessor window → corrupt back-reference bytes → CRC32 fail.
- **Partition gaps / overshoot:** a failed/absent prefetch or an overshoot chunk that swallowed
  a partition index makes `prev` non-adjacent in encoded space, so the key is simply wrong.

Contrast with the CONSUMER, which is correct precisely because it does NOT key on a stored
speculative chunk field. It keys on the **block-finder-CONFIRMED** real boundary:

```
cf:1049  let next_block_offset = block_finder.get(next_unprocessed_block_index) ...  // confirmed real offset
cf:1410  let handoff_bit = decode_start;                                            // == confirmed boundary
```

Real boundaries enter the block finder only after a predecessor is confirmed and appended
(`block_finder.insert(chunk_end_bit)` / `insert(sc.encoded_offset_bits + sc.encoded_size_bits)`,
vendor `GzipChunkFetcher.hpp:374`). At resolve-ahead time those confirmations have NOT happened
for the speculative cache, so there is no confirmed predecessor identity to key on. Vendor's
literal `get(chunkData->encodedOffsetInBits)` (GzipChunkFetcher.hpp:544) is safe only because
vendor's BlockFinder yields the *real* offset for a confirmed index; gzippy pins
`encoded_offset_bits` to the partition seed (cf:3181) and never re-anchors a cached prefetched
chunk before consumption, so neither the seed nor the sorted-prev's published end reconstructs
the confirmed boundary pre-confirmation.

## 4. VERDICT

**§3.1-as-specified is INCORRECT.** Resolve-ahead cannot safely re-key the predecessor-window
lookup onto the offset-sorted previous neighbor's published end: pre-block-finder-confirmation,
the speculative prefetch cache's offset order is by partition GUESS, so the sorted-prev element
is not the true contiguous decode predecessor — N resolves against the wrong window and the
output fails CRC32 (`test_coalesce_fixed_huffman_multithread_byte_exact`, T=8, reproduced and
reverted). The existing HEAD key (`max_acceptable_start_bit`, gated by
`chunk_may_resolve_markers_early`) stays. Direction set aside on a reproduced mechanism, not a
comment.
