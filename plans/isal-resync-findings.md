# JOB 2 — ISA-L stored/fixed resync: SOURCE + EMPIRICAL findings (OWNER, 2026-06-08, worktree isal-resync-stored-fixed @ d56cb0f5)

## Premise (charter / advisor caveat in GOAL#2 orchestrator-status entry)
"gzippy-isal degrades to native (byte-exact, ZERO ISA-L coverage) on stored/fixed-
block-heavy inputs because ISA-L's END_OF_BLOCK stop doesn't fire there (nbounds=0).
Port rapidgzip's non-dynamic read_in resync (isal.hpp:392-405) to keep ISA-L coverage."

## SOURCE-VERIFIED (first-hand)
- gzippy decline site: src/decompress/parallel/gzip_chunk.rs:302-333. INEXACT path: pick
  first boundary at-or-past stop_hint; if NONE and `end_bit <= stop_hint` accept (genuine
  BFINAL), ELSE `return Ok(false)` => DECLINE to pure-Rust (counted ISAL_ENGINE_ORACLE_FALLBACKS).
  EXACT path (until_exact): require a boundary whose bit_offset == stop_hint exactly; else DECLINE.
- gzippy ISA-L wrapper: src/backends/isal_decompress.rs:643 (with_boundaries) + :810
  (_into, the PRODUCTION copy-free path). Both record a BlockBoundary ONLY when
  `state.stopped_at == ISAL_STOPPING_POINT_END_OF_BLOCK` (lines 755/893).
- rapidgzip isal.hpp:392-405 is `readBytes()` (byte-aligned FOOTER reader), NOT a block
  resync. The actual rapidgzip mechanism is `IsalInflateWrapper::readStream` (isal.hpp:255-360):
  a LOOP of isal_inflate calls fed by its own BitReader; ISA-L's read_in/read_in_length carry
  sub-byte bit state across ALL block types; it terminates at block_state==ISAL_BLOCK_FINISH
  (BFINAL). It sets points_to_stop_at and records boundaries via tellCompressed() (read_in-aware).
- VENDORED patched ISA-L (vendor/isa-l/igzip/igzip_inflate.c) ALREADY emits
  `stopped_at = END_OF_BLOCK` after BOTH decode_literal_block (stored/TYPE0, lines 2386-2389
  + 2507-2510) AND decode_huffman_code_block_stateless (fixed/dynamic). Stored/fixed DO
  participate in the stopping-point machinery in the C source.

## EMPIRICAL PROBE (guest x86_64, gzippy-isal feature, ISA-L FFI; src/tests/isal_stored_fixed_probe.rs)
Ran decompress_deflate_from_bit_with_boundaries AND the production _into variant:
- 6 hand-built STORED (BTYPE=00) blocks  => **6 boundaries** recorded, bytes exact.
- 64x 64KiB STORED blocks (4 MiB)        => **64 boundaries** (with_boundaries AND _into), exact.
- flate2 FIXED-Huffman (BTYPE=01) streams (23B/1KiB/256KiB) => boundaries recorded
  (1-2, == block count), bytes exact.
- All byte-exact, NO declines at the wrapper level.

## VERDICT (provisional — to advisor)
The premise "ISA-L's END_OF_BLOCK doesn't fire on stored/fixed => nbounds=0" is EMPIRICALLY
REFUTED at the FFI/boundary-recording level: the gzippy-vendored patched ISA-L records EOB
boundaries on stored AND fixed blocks, including via the production copy-free _into path.
=> There is NO wrapper-level coverage gap to "port read_in resync" for. The
`decompress_deflate_from_bit_with_boundaries`/`_into` ALREADY behave like rapidgzip's
readStream for non-dynamic blocks (boundaries present).

The gzip_chunk DECLINE can still fire for a DIFFERENT reason (not nbounds=0):
(a) EXACT path needs a boundary EXACTLY at stop_hint — if the chunk's exact stop is mid-block
    or the partition guess misaligns, no exact-match boundary exists => decline (correct).
(b) INEXACT with end_bit>stop_hint and no boundary past hint — but probes show boundaries ARE
    present, so this requires a narrower trigger (e.g. a chunk that ends exactly at BFINAL with
    the last boundary BEFORE stop_hint, or an under-reserve None in _into).
This means JOB 2 as framed (port read_in resync to fix nbounds=0) targets a PHANTOM. The real
question: is there a REAL stored/fixed input where gzippy-isal's production clean tail DECLINES
with isal_fallbacks>0? That must be reproduced via the FULL finish_decode_chunk_impl on a
btype01/stored chunk BEFORE any port — else a port fixes nothing measurable.

Note: test_coalesce_fixed_huffman_multithread_byte_exact (routing.rs:353) ALREADY decodes a
32 MiB btype01-heavy fixture through the full production pipeline byte-exact at T2/4/8/16 —
correctness is NOT at risk; only ISA-L *coverage* (perf) on such inputs is the open question.

## PRODUCTION-PATH MEASUREMENT (guest x86_64, gzippy-isal RELEASE binary, /tmp scratch — NOT the
## JOB-1 pinned bench root; GZIPPY_VERBOSE coverage counters)
40 MB btype01-heavy payload (routing.rs make_btype01_heavy_data shape), gzip -1 => 16.6 MB gz
(>10 MiB parallel gate), FIXED-Huffman-heavy:
- T4: decoded byte-exact (40,000,000 == orig). **isal_chunks=2  isal_fallbacks=0**. finish_decode=2,
  flip_to_clean=1, window_seeded=1. => ISA-L coverage POSITIVE, ZERO declines on fixed-Huffman.
- T8: byte-exact. **isal_chunks=2  isal_fallbacks=0**. finished_no_flip=14, window_seeded=2.
ALL-STORED input (python gzip level-0, 40 MB, first block BTYPE=0): routes to a DEDICATED
`path=StoredParallel` (commit 37b326d4 "non-speculative parallel decode for stored-dominated
input") — NOT ParallelSM, NOT the ISA-L clean-tail engine. byte-exact. No ISA-L counters because
the clean-tail engine is not on this path at all (specialized stored fast-path).

## CORRECTION — genuine fixed-Huffman fixture (flate2 DEFAULT, the repo's
## test_coalesce_fixed_huffman_multithread_byte_exact source; gzip -1 was wrongly DYNAMIC)
gzip -1 emitted DYNAMIC (btype=2) for make_btype01_heavy_data — NOT fixed. Re-materialized the
GENUINE fixture via flate2 Compression::default() (level 6) on a 40 MB make_btype01_heavy_data
payload => 16.0 MB gz, fixed-Huffman-heavy interior. Ran the release gzippy-isal binary:
- T4:  byte-exact, isal_chunks=1 isal_fallbacks=0
- T8:  byte-exact, isal_chunks=1 **isal_fallbacks=1**, inflate_wrapper=1  <- A REAL DECLINE
- T16: byte-exact, isal_chunks=2 isal_fallbacks=0
- T8 REPEATED x3: isal_fallbacks=0,0,0  <- the T8=1 was a ONE-OFF, NON-DETERMINISTIC.
=> The decline-to-pure-Rust on fixed-Huffman DOES exist (isal_fallbacks can be >0), but it is
RARE + INTERMITTENT + non-deterministic (depends on where speculative chunk partitions land vs
deflate block boundaries), NOT the wholesale "degrades to native, ZERO ISA-L coverage" the
premise stated. Even when one chunk declines, the OTHER clean chunks keep ISA-L coverage
(isal_chunks stays >0). All runs byte-exact.

## ADVERSARIAL FIXTURE (advisor-owed; the lean-to-reject is FLIPPED) — gap is REAL+CONCENTRATED
Built the regime the advisor named: MANY TINY blocks via flate2 L1 + SYNC_FLUSH every 2 KiB on
a 40 MB make_btype01_heavy_data payload => 18.2 MB gz, ~20,480 flushed blocks
(src/tests/isal_stored_fixed_probe.rs::materialize_tiny_block_gz). Production gzippy-isal binary,
byte-exact every run, isal_chunks / isal_fallbacks (3 reps each — STABLE):
  T2:  isal_chunks=4   isal_fallbacks=1     (x3 stable)
  T4:  isal_chunks=9   isal_fallbacks=11    (x3 stable)   <- fallbacks > ISA-L chunks
  T8:  isal_chunks=19  isal_fallbacks=46-48 (stable)
  T16: isal_chunks=38  isal_fallbacks=101-104 (stable)
=> On small-block-dense fixed/stored input the clean tail DECLINES to pure-Rust on the MAJORITY
of tail chunks, STABLY (not the one-off seen on the benign flate2-L6 fixture). This IS the
"common degrade" the premise described. The lean-to-reject is REFUTED by this fixture. All
byte-exact (correctness already holds; the gap is PERF/faithfulness — ISA-L coverage collapses).

## ROOT-CAUSE DISAMBIGUATION (pass-2 advisor's "biggest risk" — RESOLVED by direct measurement)
The pass-2 advisor flagged a contradiction: the production comment (gzip_chunk.rs:316-325 / commit
19add96c) claims "ISA-L records ZERO boundaries on stored/fixed => decline = absent boundaries",
while my wrapper probes said boundaries ARE recorded. Resolved by probing the EXACT adversarial
tiny-block stream through decompress_deflate_from_bit_with_boundaries:
  raw_deflate=19,062,964  decoded=41,943,040  end_bit=152,503,696  **BOUNDARIES=40,960**
  spacing exactly 2048 bytes (2 boundaries per SYNC_FLUSH: data block + flate2's empty stored block).
=> BOUNDARIES ARE RECORDED IN ABUNDANCE (40,960 for ~20,480 flushes). The production comment's
"ZERO boundaries on stored/fixed" is EMPIRICALLY FALSE for this input. The decline is therefore
NOT absent-boundaries; it is the ACCEPT logic — the until_exact path demands a boundary whose
bit_offset == stop_hint_bits EXACTLY, and with dense 2048-byte-cadence blocks the speculative
partition's stop_hint frequently lands BETWEEN recorded boundaries => no exact match => decline.
My gap-location map (accept logic, boundaries present) is CONFIRMED; the production comment is
outdated/wrong for the SYNC_FLUSH case. The 19add96c over-decode guard (inexact end_bit>stop_hint)
is a DIFFERENT branch — with boundaries present the inexact path finds one at-or-past hint and
accepts, so the surviving declines are the exact-match (until_exact) ones.

## FINAL VERDICT (advisor-vetted ×2: GAP IS REAL on small-block-dense input; the CITED port is
## DOUBLY WRONG; real fix = gzip_chunk until_exact accept relax — a NEW gated turn)
1. The gap IS REAL + STABLE + CONCENTRATED on small-block-dense fixed/stored input (the adversarial
   tiny-block fixture: fallbacks 1/11/48/104 at T2/4/8/16, MAJORITY of tail chunks decline). The
   premise's "common degrade" reproduces THERE. So this is NOT a phantom — JOB 2 has a real target.
2. BUT it is BENIGN on ordinary fixed-Huffman (flate2 L6): fallbacks 0 (one-off 1), ISA-L coverage
   intact. And on ALL-STORED input the clean-tail engine isn't even on the path (StoredParallel).
   So the gap matters only on a narrow, adversarial-ish input class (frequent SYNC_FLUSH / tiny
   blocks), not the common silesia-class corpus (where parity already holds, JOB 1: 0 fallbacks).
3. The CITED fix (isal.hpp:392-405) is the WRONG function — that is readBytes() (footer reader).
   The gzippy ISA-L wrapper ALREADY records EOB boundaries on stored+fixed (probes: 6/6, 64/64,
   fixed==block-count). So there is nothing to "resync" at the FFI layer. The DECLINE is in
   gzip_chunk.rs's coalesce/stop-hint ACCEPT logic: it requires a recorded boundary at-or-past
   (inexact) / exactly-at (until_exact) stop_hint; with tiny blocks the chunk's stop_hint rarely
   coincides with a boundary, so it declines. rapidgzip sidesteps this by decoding ACROSS the
   boundary (readStream loops to BFINAL/next-clean-EOB) rather than demanding a boundary at the
   guessed stop — that is the actual faithful mechanism to port, into gzip_chunk, NOT isal.hpp.
=> DECISION: do NOT land the cited port (wrong function, FFI wrapper already correct). The REAL
   JOB-2 lever = make the ISA-L clean-tail accept decode-across-boundary (faithful rapidgzip
   readStream coalesce) so a tiny-block chunk keeps ISA-L coverage instead of declining. That is a
   correctness-sensitive change to gzip_chunk's accept/coalesce logic (NOT a 5-line port), warrants
   its OWN gated turn with the byte-exact dual-sha + the adversarial fixture as the coverage gate.
   This turn DELIVERS: the source refutation of the cited line-range, the empirical map of WHERE
   the gap is/isn't, and the adversarial fixture that PROVES the gap (the missing repro the prior
   advisor caveat asserted but never measured). No production code changed (probes + fixtures only).
