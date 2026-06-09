# Disproof advisor verdict — CONVERGE-BOOTSTRAP checkpoint (branch reimplement-isa-l @ d56cb0f5)

Read-only Opus disproof. Source-verified first-hand at HEAD d56cb0f5. Attack posture.

## TL;DR per claim

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Window-absent residual is dominated by the CLEAN u8 TAIL decoder (pure-Rust inner Huffman), not marker/FFI/placement | **FIX-NEEDED** — TRUE for gzippy-NATIVE; FALSE as stated for gzippy-ISAL (its clean tail is ISA-L FFI, not pure-Rust). The leader's LOCATE conflates the two builds. |
| 2 | "gzippy-isal clean tail is pure-Rust resumable.rs, NOT ISA-L FFI; build.rs:98-101 is stale" | **REFUTED** — at HEAD the gzippy-isal PRODUCTION clean tail DOES route through real ISA-L FFI. The build.rs:98-110 comment was reverted to a WRONG state and now contradicts the live wiring. The banked GOAL #2 (19add96c) is the correct description. |
| 3 | asm necessary-but-NOT-sufficient for T4 0.99; even perfect ISA-L = ocl_cf 0.899x; residual has co-primary non-engine term | **SOUND (with one caveat)** — the 0.899x ceiling is tightly measured and the insufficiency conclusion holds; but "non-engine" is an UPPER-BOUND bucket that still contains engine (marker prefix) + FFI-handoff, not proven-scheduling. |
| 4 | clean tail ~2.3x slower/byte than rg's ISA-L; is that intrinsic-to-asm or closable-in-Rust? | **My read: PARTLY closable in pure-Rust-without-asm, but NOT all the way to ISA-L parity. Matching rg's clean-tail instruction behavior faithfully = matching ISA-L, and rg's clean tail LITERALLY IS ISA-L (deflate.hpp:175, GzipChunk.hpp:520-524).** |
| 5 | OPEN-2 = consumer-imminent eviction, distinct from DIS-6 offset-supply | **FIX-NEEDED** — plausible and formally distinct from DIS-6, but the stated MECHANISM ("eviction") is asserted, not measured; the ledger's own OPEN-2 says the dispatched-then-evicted vs never-dispatched discriminator is UNRUN. It is a re-label of an OPEN question, not yet an established mechanism. |

---

## Claim 1 — LOCATE: residual dominated by the clean u8 tail decoder. FIX-NEEDED.

Source-true for **gzippy-native**:
- FlipToClean is NOT returned on native; native returns `FlipToContig` (gzip_chunk.rs:1453 `#[cfg(not(isal_clean_tail))]`) → `finish_decode_chunk_contig_native` (:1224) → `block.decode_clean_into_contig` (:1303). That IS the pure-Rust ISA-L-LUT-ported multi-symbol fast loop (marker_inflate.rs:2071). So native's clean tail = pure-Rust inner Huffman. ✓
- The u16 marker prefix being 0.68x and the SLOW_MODE causal A/B are prior-banked and not re-attacked here; I take them as given.

WRONG as stated for **gzippy-isal**:
- On the isal build the same window-absent chunk hits the OTHER match arm, `MarkerStep::FlipToClean` (:1169), which requires a full 32 KiB clean window (`last_32kib_window_vec`, :1179) and calls `finish_decode_chunk_with_inexact_offset` (:1185) → `finish_decode_chunk_impl(... allow_isal=true)` (:620-631) → the gate `if allow_isal && isal_engine_oracle_enabled()` (:669) → `finish_decode_chunk_isal_oracle` = **real ISA-L FFI `decompress_deflate_from_bit_into`** (:275). The window-len guard (:223) is satisfied (full 32 KiB), so ISA-L runs, not the pure-Rust fallback.
- So the leader's sentence "ISAL build: FlipToClean → StreamingInflateWrapper = pure-Rust resumable.rs (DESPITE the name isal_clean_tail)" is **source-false** for the window-absent FlipToClean chunk. That chunk is decoded by ISA-L on the isal build (or, on the all-dynamic parity corpus, always — fallbacks=0 per GOAL #2).

Why this matters: the leader's whole LOCATE leans on "BOTH builds decode the clean tail in pure Rust." That is only true for native. The conflation is exactly the build.rs comment error (claim 2). The CORRECT locate is build-specific:
- native: clean tail = pure-Rust `decode_clean_into_contig` (claim 1 right).
- isal: clean tail = ISA-L FFI, and its T4 number IS the 0.899x ocl_cf ceiling. There is no "pure-Rust clean tail" to blame on the isal build's covered chunks.

VERDICT: FIX-NEEDED. Re-state as "the native build's residual is dominated by the pure-Rust clean tail; the isal build already runs ISA-L there and STILL only reaches 0.899x@T4 — which is the actual evidence for claim 3, not claim 1's pure-Rust-engine framing."

## Claim 2 — the build.rs-vs-GOAL#2 contradiction. REFUTED (build.rs comment is the wrong one).

This is the load-bearing reconciliation. First-hand chain at HEAD d56cb0f5:

1. `Cargo.toml:84`: `gzippy-isal = ["pure-rust-inflate", "isal-compression"]`. So the isal build enables BOTH `pure-rust-inflate` AND `isal-compression`.
2. `build.rs:94-95`: `parallel_sm = (x86_64||aarch64) && has_pure_rust_inflate; pure_inflate_decode = parallel_sm`. → on isal build, `pure_inflate_decode` is set.
3. `build.rs:110`: `isal_clean_tail = is_x86_64 && has_gzippy_isal && parallel_sm` → set on the isal build.
4. `gzip_chunk.rs:154-163` `isal_engine_oracle_enabled()` defaults (env unset) to `cfg!(isal_clean_tail)` = **true** on the isal build.
5. `finish_decode_chunk_isal_oracle` real body is `#[cfg(all(parallel_sm, feature="isal-compression", target_arch="x86_64"))]` (:205). On the isal build all three hold → the REAL FFI body (not the `Ok(false)` stub at :390) is compiled.
6. Production reaches it: `finish_decode_chunk_with_inexact_offset` passes `allow_isal=true` (:630); `finish_decode_chunk_impl` calls the FFI at :669-678; `Ok(true) ⇒ return Ok(())` — ISA-L's bytes are committed copy-free into chunk.data.

⇒ On the gzippy-isal build, with env unset (production), the clean tail IS decoded by real ISA-L FFI. This matches the banked GOAL #2 entry (19add96c, which IS in git history) and the measured isal coverage (14/14 fallbacks=0 @T4/T8, T8 1.030x TIE).

The `StreamingInflateWrapper` (pure-Rust `Inflate<Clean>`, inflate_wrapper.rs:154 `#[cfg(pure_inflate_decode)]`) is only reached on the isal build as the **fallback** when ISA-L declines (until_exact with no exact boundary, or reserve-overrun) — counted in `ISAL_ENGINE_ORACLE_FALLBACKS`, asserted ==0 on the parity corpus. The second `StreamingInflateWrapper` (inflate_wrapper.rs:340) is a `#[cfg(not(parallel_sm))]` unavailable stub — NOT an ISA-L variant; there is no ISA-L `StreamingInflateWrapper`. So "the isal clean tail = pure-Rust resumable.rs" is true ONLY on the fallback (rare, asserted-zero), never on the covered production path.

The build.rs:98-110 comment ("BOTH topologies decode the clean tail in PURE RUST … Real ISA-L FFI … reachable ONLY under GZIPPY_ISAL_ENGINE_ORACLE=1 … Prior comment claimed REAL ISA-L FFI; that was stale/aspirational") is **factually wrong at HEAD** and directly contradicts gzip_chunk.rs:141-164 + :669. The env var is an OVERRIDE, not the only enable; the build default IS ISA-L on the isal build. Whoever wrote the new build.rs comment inverted the truth — likely reasoning from the native build (where it's correct) and over-generalizing, the same error as claim 1.

VERDICT: REFUTED. The leader adopted the stale/wrong build.rs comment as ground truth. The GOAL #2 banked claim is the source-correct one. **This should block the escalation as written** — any decision predicated on "the isal clean tail is pure-Rust" is built on a false premise. (Recommend a one-line fix to the build.rs comment so the next agent isn't re-poisoned.)

Caveat the leader is half-right about: the GOAL #2 advisor CAVEAT (stored/fixed-heavy input → ISA-L declines → degrades to pure-Rust) is real (confirmed by the JOB-2 entry, 40,960-boundary probe). So on NON-dynamic corpora the isal clean tail CAN be pure-Rust. But on the silesia/all-dynamic parity corpus the leader is measuring, it is ISA-L. The unqualified "is pure-Rust" is false.

## Claim 3 — asm necessary-but-not-sufficient; 0.899x ceiling. SOUND, one caveat.

- 0.899x ceiling: measured on a frozen quiet guest, T4, interleaved best-of-N, ≤5% spread (3-4%), sha-verified, isal_chunks=14 fallbacks=0, path=ParallelSM. This is the FULLEST-fulcrum-grade number the charter demands and it reproduces the time-accounting ocl_REAL. The ceiling is solid. ✓
- Insufficiency: if REAL ISA-L (the fastest known clean engine, = rg's own engine) only reaches 0.899x@T4 with the rest of gzippy's pipeline, then a hand-asm pure-Rust engine — whose BEST case is ISA-L-equivalent — cannot exceed 0.899x@T4 either. So asm cannot reach 0.99 alone. The "necessary-but-not-sufficient" logic is sound and removal-oracle-backed (not extrapolation). ✓ This correctly applies charter rule 3 (remove-the-region oracle, not slow-slope).
- Could the 56ms be ENTIRELY engine (making asm sufficient)? No: ocl_cf@T4 swaps the engine to ISA-L and still loses 0.101x. That residual EXISTS WITH the engine removed ⇒ it is by construction not engine-instruction-rate. The "asm sufficient" alternative is refuted by the same oracle. ✓

CAVEAT (the leader already states it, I'm upholding with sharpening): the ≤0.101x "non-engine residual" is an UPPER BOUND that the JOB-1 advisor explicitly says still contains (a) the pure-Rust MARKER-PREFIX engine (chunk-0 bootstrap + the <32KiB markered prefix stay pure-Rust even on ocl_cf — gzip_chunk.rs:128-131,196-223) and (b) per-chunk ISA-L FFI/handoff overhead. So "co-primary genuinely-non-engine PLACEMENT term (OPEN-2)" is NOT yet isolated from that bucket — the residual is "not-the-clean-engine," which is not the same as "scheduling/placement." Calling OPEN-2 "co-primary" overstates what the oracle proves. The owed disambiguator (an oracle that ALSO ISA-Ls the marker prefix, or an FFI-null run) is unrun. So claim 3's "co-primary non-engine placement term" should read "≤0.101x non-clean-engine residual, composition unresolved (marker-engine + FFI-handoff + possible placement)."

VERDICT: SOUND on the ceiling and the insufficiency. FIX the over-attribution of the residual to "placement/OPEN-2."

## Claim 4 — is the 2.3x clean-tail gap intrinsic-to-asm or closable-in-Rust? (the crux)

Honest read, three-way as asked:

**(a) Pure-Rust WITHOUT asm — partially closable, NOT to ISA-L parity.** Evidence the leader's own ledger supplies: the native binary already emits 433 BZHI/PEXT (BMI2 idioms already lowered by LLVM — manual PEXT/BZHI has no headroom); the 16 KiB decode table is L1-resident (table-prefetch no-headroom). The dist cache was just shrunk 128→8 KiB (cf0c5f62) improving locality. The packed multi-literal store loop (VAR_V) is already ported into decode_clean_into_contig and TIE'd. BUT the prior decode/store-localization turn (25846265) flagged FOUR authorized techniques as UNTRIED at that time: table `_mm_prefetch` (now argued moot), static-Huffman specialization, FASTLOOP_OUTPUT_MARGIN yield-elision, single-level L1 table geometry. Static-Huffman specialization and yield-elision are NOT exhausted and are pure-Rust. My read: these can shave SOME of the gap (the resumable yield-check tax + dynamic-table dispatch are real LLVM-vs-asm overheads), but they will not reach ISA-L's instruction rate because the structural win in ISA-L (one-iteration-ahead literal-table gather across the back-edge F1, speculative dist gather F2, flag-free SHLX/SHRX refill F3, loop state pinned in callee-saved GPRs F4 — Phase-1 source-map, igzip asm:540/550/528/108) is precisely what LLVM does NOT emit and the VAR_VII transliteration NO-GO proved cannot be bolted on per-symbol.

**(b) Pure-Rust WITH asm — the only path that can capture the full engine share, but bounded.** The full-kernel asm (rewrite the whole hot loop in `core::arch::asm!`, NOT per-symbol re-entry) is the only construct that can emit F1-F4. The VAR_VII NO-GO (0.276x) is NOT binding against it — VAR_VII failed specifically on per-symbol asm↔Rust re-entry spills (4 regs to `bits`, LLVM barrier ×300-460K/chunk); a full-kernel rewrite keeps state in registers across the back-edge and exits to Rust only on rare long-codes. So a full-kernel asm COULD reach ~ISA-L. But it is bounded by ocl_cf 0.899x@T4 (claim 3) — it captures the engine share, does not reach 1.0 alone.

**(c) Only via ISA-L instructions themselves.** This is the faithful-convergence truth the charter's NEW GOAL forces: rg's clean tail IS ISA-L (`using LiteralOrLengthHuffmanCoding = HuffmanCodingISAL`, deflate.hpp:175; `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>`, GzipChunk.hpp:441/502/522). So "faithfully match rg's clean-tail RUNTIME BEHAVIOR" literally means "run ISA-L's instructions." A full-kernel hand-asm port of igzip's decode loop IS that, transliterated. Pure-Rust-codegen improvements (a) make gzippy FASTER but do NOT make it converge to rg's instruction stream.

**My honest verdict on the crux:** the 2.3x is NEITHER purely-intrinsic NOR fully-closable-in-Rust. It is ~majority intrinsic-to-handwritten-asm (F1-F4 are LLVM blind spots, BMI2/prefetch headroom already spent), with a closable minority (static-Huffman specialization + yield-elision, untried). Matching rg's clean-tail behavior faithfully (charter goal) requires the ISA-L instructions = the full-kernel asm rewrite. Pure-Rust-without-asm can narrow but not converge. So: **asm is the faithful-convergence path; pure-Rust-codegen is a partial speed path that diverges from rg's structure.** The user's decision is genuinely "authorize the multi-session full-kernel asm OR accept ~0.86-0.90x pure-Rust at low-T and declare faithful-enough." This framing is sound; the leader's stop-and-escalate is correct in shape — but it MUST be re-grounded on the corrected claim 2 (rg's tail is ISA-L; so is gzippy-isal's already; the GAP is on the gzippy-NATIVE build whose tail is pure-Rust).

## Claim 5 — OPEN-2 consumer-imminent eviction. FIX-NEEDED.

- DISTINCT from DIS-6: yes, formally. DIS-6/LEV-7 is offset-SUPPLY (re-target the overshot prefetch index) — REFUTED because gzippy already re-targets (gzip_block_finder.rs:180-182) and needs_confirmed_offset has zero hits. OPEN-2 is a RETENTION/horizon question: a free worker exists at every stall yet the covering chunk is neither in-flight nor resident (NOT_RESIDENT=4, has_nearest_le_start=0). Different failure mode. ✓ Not a re-label of DIS-6.
- BUT the leader's stated MECHANISM — "consumer-lag evicts the prefetched covering chunk before the lagging consumer arrives; cache-pollution stop protects to-be-prefetched but not the consumer's imminent chunk" — is ASSERTED, not measured. The ledger's own OPEN-2 (disproof-ledger.md:93) says the DECISIVE discriminator (never-dispatched vs dispatched-then-EVICTED; parked-vs-unspawned idle; N>>3) is UNRUN. So "eviction" is one of two live hypotheses (the other: prefetch horizon simply too shallow / worker saturation), NOT an established mechanism. Presenting "eviction" as THE mechanism prejudges the unrun discriminator — exactly the attribution-without-perturbation the charter forbids (rule 1).
- The leader's own checkpoint text is actually MORE careful than the claim-5 framing handed to me ("the DECISIVE discriminator … is UNRUN ⇒ the faithful fix is NOT yet identified … NOT-yet-actionable"). So the leader internally knows it's open. The claim as PHRASED for this review ("consumer-lag evicts …") overstates that.

VERDICT: FIX-NEEDED. OPEN-2 is a legitimately-distinct OPEN question (good — not a DIS-6 re-label), but it is NOT a confirmed mechanism and must not be escalated to the user as "co-primary located lever." It is "an unrun diagnostic." Escalating it as co-primary inflates the case for stopping.

---

## What I'd tell the user (net)

1. The engine-fork decision SHAPE is sound and well-supported: real ISA-L = 0.899x@T4 (tight, removal-oracle), so no pure-Rust engine reaches T4 0.99 alone — asm is necessary-not-sufficient. That part survives disproof.
2. BUT the checkpoint's LOCATE narrative is **poisoned by a false premise** (claims 1+2): it says both builds' clean tails are pure-Rust because it trusted a build.rs comment that is WRONG at HEAD. Source truth: gzippy-isal's production clean tail runs real ISA-L FFI (Cargo.toml:84 → build.rs:110 → gzip_chunk.rs:161,205,669,275); GOAL #2 (19add96c) is correct. The pure-Rust clean tail is the gzippy-NATIVE story (0.740x) and the isal FALLBACK only.
3. Net effect on the decision: the gap to close in pure-Rust is the gzippy-NATIVE 0.740x → ~0.899x (capturing what ISA-L proves capturable) via full-kernel asm, then both builds remain bounded at 0.899x needing the non-clean-engine residual. The asm-or-accept fork is REAL; the OPEN-2 "co-primary placement" leg is NOT yet established (unrun discriminator) and should be presented as an open diagnostic, not a confirmed co-lever.
4. Before escalating: (a) fix the build.rs:98-110 comment (it actively misleads — costs the NEXT agent a turn), (b) re-word the LOCATE to be build-specific, (c) downgrade OPEN-2 from "co-primary lever" to "unrun discriminator."

No code changed. No orphan processes (read-only).
