Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
I traced the production path to the bytes rather than ruling from the numbers. The numbers are right, but the owner's *mechanism* is partly wrong — and the real mechanism makes their conclusion **stronger**, not weaker.

## The decisive source fact

On the native build (default, `not(isal_clean_tail)` — build.rs:110 gates the ISAL path off here), the bulk clean tail does **not** go through the ring-drain path the owner's NODRAIN knob lives in. It goes through `finish_decode_chunk_contig_native` (gzip_chunk.rs:1108-1246): `marker_inflate::Block`'s clean arm (`decode_clean_into_contig`) writing **u8-direct into `chunk.data`'s contiguous tail — no ring, no drain memcpy**, CRC taken straight from `decoded_range`. resumable.rs:1187 confirms ~99% of clean bytes flow here; the ring/drain path (`FlipToClean`) is "never returned on gzippy-native."

So `GZIPPY_FOLD_NODRAIN` (which only lives in `ContigFoldSink::push_clean_u8`) governs **only the ~1% marker-loop dribble, not the bulk.** Its ~0ms is consistent, but it is not the proof — the proof is that the bulk has no drain to remove.

## Verdicts

**Q1 — REFUTED (my own 10-20ms estimate).** My estimate assumed a 200MB cold ring→data second-touch. That term does not exist on the production native bulk — it's u8-direct. The slack-masking escape I reserved is moot: there's no bulk drain term to mask. *Owner correction owed:* your stated mechanism ("remaining drain = extend_from_slice from a warm ring slice") describes the ~1% path; the bulk has no `extend_from_slice` at all.

**Q2 — UPHELD-WITH-CAVEATS.** Yes, 404ms is a valid speed-up ceiling — and more robustly than NODRAIN shows: native-contig and ocl_cf are structurally identical on covered chunks (both u8-direct into `chunk.data` + identical CRC), differing **only in the inner decode engine**. So the 36ms is genuinely clean symbol rate, and the four techniques attack exactly `decode_clean_into_contig`. Two caveats: (a) **coverage symmetry** — confirm the "14/0" chunks ocl_cf covers are ones native also decodes window-seeded-clean, not chunks native marker-bootstraps (else the gap absorbs bootstrap savings the four techniques can't capture); (b) the doc comments at gzip_chunk.rs:871-875 and 996-999 ("engine ring write + drain remain… upper bound on symbol rate") are **stale** for the contig bulk path — fix them so the next reader doesn't re-derive a phantom ring confound.

**Q3 — UPHELD.** Proceed, bounded by 404ms. Pre-flight: confirm coverage symmetry; measure each technique on the full path interleaved + sha-verified, never the slice. **C2 stands**: the 21ms (0.945×→1.0×) residual is outside the clean engine — STEP-2 alone won't reach rapidgzip parity.

**Bottom line:** My 10-20ms drain estimate is REFUTED — the production native bulk is u8-direct, so native-vs-ocl_cf differ only in the engine and the ~36ms is real symbol rate; 404ms is a valid STEP-2 ceiling provided ocl_cf's covered chunks are window-seeded on native too — proceed, fix the stale "ring drain remains" comments, and keep C2's 21ms as a separate non-engine bar.
