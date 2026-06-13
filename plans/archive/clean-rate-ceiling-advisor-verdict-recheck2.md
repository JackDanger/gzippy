# Advisor recheck #2 — FOLD second-touch split (owner ran the demanded decomposition)

Same disproof advisor. Prior verdict (clean-rate-ceiling-advisor-verdict.md) REFUTED
"36ms == clean engine-rate gap," estimated a 200MB ring→data second-touch at 10-20ms,
and required GZIPPY_FOLD_NODRAIN/FOLD_NOCRC to split it. Owner ran it (N=21, best-of-N
min): drain = -0.6ms, CRC = 1.2ms, total second-touch = ~0.6ms (below noise floor).

I did NOT rule from the numbers. I traced the production path to source. The numbers are
right but the owner's stated *mechanism* is partly wrong, and the real mechanism makes
the conclusion STRONGER. Source grounding:

- Native (default, `not(isal_clean_tail)`; build.rs:110 gates isal_clean_tail to
  x86_64+gzippy-isal only) decodes the bulk clean tail via
  `finish_decode_chunk_contig_native` (gzip_chunk.rs:1108-1246): `marker_inflate::Block`
  clean arm (`decode_clean_into_contig`) writing **u8-DIRECT into chunk.data's
  contiguous tail** — NO ring, NO drain memcpy. CRC is taken directly from
  `chunk.data.decoded_range` (1212). resumable.rs:1187 confirms ~99% of clean bytes go
  here. `MarkerStep::FlipToClean`/`finish_decode_chunk_with_inexact_offset` (the ring+
  drain path) is "never returned on gzippy-native" (gzip_chunk.rs:1320-1323).
- Therefore GZIPPY_FOLD_NODRAIN (which lives only in `ContigFoldSink::push_clean_u8`,
  gzip_chunk.rs:905-914) governs ONLY the ~1% marker-loop dribble — NOT the bulk. Its
  ~0ms is consistent but is not itself the proof. The proof is structural: the bulk has
  no drain to remove.
- ocl_cf (`finish_decode_chunk_isal_oracle`, gzip_chunk.rs:188-: engine-swap-only oracle,
  "pool/consumer/window-publish/ring/CRC all stay") also decodes u8-direct into chunk.data
  (writable_tail_reserve, line 235) with the same CRC. So native-contig and ocl_cf are
  structurally IDENTICAL on covered chunks except the inner decode engine.

## Q1 — does ~0.6ms defeat the 10-20ms estimate? REFUTED (my own estimate falls)
My 10-20ms rested on a 200MB cold ring→data second-touch. That term does NOT exist on
the production native bulk path — it is u8-direct (contig). The slack-masking escape I
worried about is MOOT: there is no bulk drain term to mask. My estimate is refuted by the
source structure (the NODRAIN ~0ms merely corroborates). NOTE TO OWNER: your stated
mechanism ("remaining drain = extend_from_slice from a warm ring slice") is the ~1%
marker-loop path, not the bulk; the bulk has no extend_from_slice at all. Fix the framing.

## Q2 — is 404ms a valid speed-up ceiling for the 4 STEP-2 techniques? UPHELD-WITH-CAVEATS
YES, and more robustly than the NODRAIN argument: native-contig vs ocl_cf differ ONLY in
the inner engine (both u8-direct into chunk.data + identical CRC), so the 36ms IS the
clean symbol-rate delta. The four techniques (table _mm_prefetch, single-level L1 table,
static-Huffman, FASTLOOP yield-elision) attack exactly `decode_clean_into_contig`.
CAVEATS:
  (a) COVERAGE SYMMETRY. ocl_cf fires only when initial_window==MAX_WINDOW_SIZE
      (gzip_chunk.rs:206). It swaps the engine at the SAME call site that hands native
      its window, so symmetry is by-construction — but confirm "14/0" means the 14
      ocl_cf chunks are exactly the chunks native decodes window-seeded-clean, not chunks
      native marker-bootstraps. If a covered chunk is bootstrap-on-native, the gap absorbs
      bootstrap savings the 4 techniques cannot capture and 404 is over-optimistic there.
  (b) STALE DOCS. The comments at gzip_chunk.rs:871-875 and 996-999 ("engine ring write +
      ring→data drain memcpy remain… UPPER BOUND on symbol rate") describe a residual the
      contig bulk path does NOT pay. They will make the next reader re-derive a phantom
      ring confound. Correct them to say: bulk = u8-direct, residual ≈ pure symbol rate.

## Q3 — proceed? UPHELD
Proceed to the four STEP-2 techniques bounded by 404ms. Pre-flight: (1) confirm coverage
symmetry (a) so 404 isn't shaving bootstrap; (2) measure each technique on the FULL path,
interleaved + sha-verified, never the slice (standing rule). C2 STANDS: the 21ms
0.945×→1.0× residual is OUTSIDE the clean engine (marker-region pure-Rust + bootstrap +
placement) — STEP-2 alone will not reach rapidgzip parity; do not expect it to.

BOTTOM LINE: My 10-20ms drain estimate is REFUTED — the production native bulk is u8-direct
(no ring, no drain), so native-vs-ocl_cf differ only in the inner engine and the ~36ms is
genuinely symbol rate; 404ms is a valid STEP-2 ceiling provided ocl_cf's 14 covered chunks
are window-seeded on native too — proceed, but fix the stale "ring drain remains" comments
and keep C2's 21ms as a separate non-engine bar.
