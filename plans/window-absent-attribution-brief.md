# DISPROOF-ADVISOR BRIEF — window-absent decodeBlock 1.6× attribution

You are an INDEPENDENT, READ-ONLY disproof advisor. Source-verify every claim
first-hand against the repo at /home/user/www/gzippy-reimplement-isal
(branch reimplement-isa-l, HEAD 7aae6c4a + measurement-only overlay). Try to BREAK
the owner's attribution. Do NOT trust the owner's prose; read the cited code.

## The claim to disprove (owner's attribution of the T8 window-absent decodeBlock 1.6× gap)
gzippy decodeBlock SUM 0.805-0.831s vs rapidgzip 0.4995s = 1.6×. The owner attributes
this NOT to the u16 marker inner loop, NOT to u16-width-over-clean-bulk, NOT to
table-build — but to the **CLEAN u8 tail decoder being ~2.3× slower than ISA-L**. The
measured gzippy-ISAL build's post-flip clean tail runs through pure-Rust `resumable.rs`
(`unified::Inflate<Clean>` via `StreamingInflateWrapper`), NOT ISA-L FFI, whereas
rapidgzip's WITH_ISAL build hands its clean tail to real ISA-L.

Full attribution + numbers: plans/window-absent-attribution.md (read it).

## Evidence the owner relies on (verify each)
1. **rg --verbose T8 unseeded:** decodeBlock 0.4995s; "custom inflate" 0.4748s; "ISA-L"
   0.2065s; markers 34.5% (73.1M). gzippy --verbose T8: decodeBlock 0.805-0.831s;
   marker bootstrap body 0.323s (73.0M @ 226-235 MB/s); flip_to_clean=12,
   finished_no_flip=4, window_seeded=2.
2. **Causal SLOW_MODE A/B (ΔdecodeBlock SUM, freq-neutral sleep control):**
   baseline 0.831s; MARKER+100% → 0.965s (+134ms, body 323→483); CLEAN+100% →
   1.025s (+194ms, marker body UNCHANGED 312). Sleep controls: marker +142ms, clean
   +248ms. CLEAN inject lands in resumable.rs:1199; MARKER inject in marker_inflate.rs.
3. **Source: flip threshold byte-identical** marker_inflate.rs:1116-1119 vs vendor
   deflate.hpp:1282-1284. **Two-phase routing** gzip_chunk.rs:1397-1410 (isal_clean_tail
   → FlipToClean → StreamingInflateWrapper). **Clean engine** inflate_wrapper.rs:154-161
   (unified::Inflate<Clean,Generic,Streaming>). resumable.rs:1182-1192 note. build.rs:101
   isal_clean_tail def.

## Specifically try to break (adversarial angles)
A. Is the CLEAN-tail attribution a subtraction artifact? The 2.3× ratio = (decodeBlock −
   marker body) vs rg ISA-L. Is the SLOW_MODE +194ms causal evidence airtight that the
   clean tail (resumable.rs) is the BIGGER decodeBlock term — or could the inject hit
   something else? Verify resumable.rs is the post-flip clean engine on isal_clean_tail.
B. Is "marker loop is FASTER than rg" fair? gzippy marker body 0.323s for 73.0M vs rg
   "custom inflate" 0.4748s — but does rg's "custom inflate" cover the SAME work
   (window-absent marker prefix only) or more (e.g. it includes clean bytes rg decodes
   before switching to ISA-L)? Could the comparison be denominator-mismatched?
C. Does rg's "ISA-L 0.2065s" decode the SAME bytes gzippy's clean tail decodes (≈139M)?
   If rg's ISA-L covers MORE/FEWER bytes, the 2.3× per-byte claim is off.
D. The owner did NOT land any fix and did NOT run a clean-engine removal oracle this
   turn. Is the attribution actionable as-is, or does it need the Phase-0 ISA-L oracle
   re-run (clean tail through ISA-L) to confirm the clean engine reaches the WALL (not
   just the decodeBlock SUM)? Note decodeBlock SUM is slack-masked at Fill 85%.
E. Faithfulness: per governing memory the target is ONE u8-direct clean engine (no
   two-phase). Is "route clean tail through ISA-L FFI" (candidate 1) faithful to rg's
   WITH_ISAL build, or a divergence? Is it the gzippy-faithful goal #2 or goal #1?

Output a verdict: which owner claims are UPHELD / UPHELD-WITH-CAVEATS / REFUTED, the
single most load-bearing correction, and whether the attribution is sound enough to
scope a fix WITHOUT another measurement. Be concise and specific (cite file:line).
