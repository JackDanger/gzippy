Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
**VERDICT: UPHELD-WITH-CAVEATS** — reject the port *as specified*, but you have not yet earned "premise is dead." One adversarial fixture is owed.

The reject is not resting on a TIE (clears rule 7): you have a mechanism-level refutation — the user's premise ("declines COMMON → degrade to native → ZERO coverage") names a behavior that does not reproduce, AND the cited source is the wrong function. That's grounds (a) under rule 7, not "Δ < spread."

**Disproof attempts:**

1. **Wrong-source check (your strongest card).** isal.hpp:392-405 = `readBytes()`, a byte-aligned footer reader. The actual resync is `readStream()`, and gzippy's vendored patched ISA-L already sets `stopped_at=END_OF_BLOCK` after stored + fixed stateless decode (igzip 2386/2507). → Porting the literally-cited lines accomplishes nothing; the real mechanism is already in-tree. **Survives. Reject holds.**

2. **"ZERO coverage" claim.** T4/T8/T16 all show isal_chunks≥1 and other clean chunks keep ISA-L even when one declines. → "Degrades to native / zero coverage" is **falsified outright.** Survives.

3. **"COMMON" claim.** T8 fallback=1 was a one-off; x3 repeat → 0,0,0; non-deterministic on partition-vs-boundary landing. → "Common" **falsified for the tested regime.** Survives.

4. **Faithfulness-demands-parity (falsifier a/e).** Does the port become a *correctness/faithfulness* win because rapidgzip never declines a clean fixed chunk? → Partially bites, but the cure is in **gzip_chunk's stop-hint accept logic** (until_exact requires a boundary *exactly* at stop_hint), NOT a missing read_in/readStream capability — the wrapper already records the boundaries (your with_boundaries probe). So even the faithfulness fix is not the proposed port. **Reject of the proposed port holds; relocates the real lever.**

5. **isal_chunks LOW ⇒ coverage marginal anyway (falsifier c).** chunks=1–2 means the marker engine does most of the bulk regardless; converting one declined chunk back to ISA-L is a sliver of a sliver. → **Strengthens** reject.

6. **StoredParallel "dodge" (falsifier d).** All-stored routes around the clean-tail engine entirely. Faithful-port goal targets the *decode pipeline*; a stored fast-path is a routing-layer optimization, not a coverage hole in the ISA-L engine. → Not a coverage gap; a separate faithfulness wrinkle. Neutral-to-supporting.

7. **Adversarial-fixture gap (falsifier b — this is where you're exposed).** Your fixtures (flate2 L6 fixed, 64×64KiB stored) exercise the *benign* regime. The actual decline trigger you named yourself — `until_exact=true` chunks where no recorded boundary lands *exactly* at stop_hint — is **maximized by many tiny fixed blocks**, which you did not build. → **Does NOT survive cleanly.** You measured the easy regime and generalized.

**The single strongest reason you're wrong:** #7 + #4. You proved the premise false *in the regimes you tested*, but the decline mechanism is an exact-match failure on `until_exact` chunks, and small-block-dense fixed/stored input is precisely where exact stop_hint coincidence gets rare and could go *concentrated and stable* — the regime rapidgzip genuinely sidesteps by decoding across the boundary. If that corpus drives fallbacks high+stable, the user's "common degrade" reproduces (as a perf, not correctness, gap) and your reject flips.

**Action to convert UPHELD-WITH-CAVEATS → UPHELD:** build the tiny-fixed-block / `until_exact` stress corpus, run it through ParallelSM, and confirm ISAL_ENGINE_ORACLE_FALLBACKS stays low/intermittent (not concentrated). If it does, the premise is dead and the proposed `readBytes` port is misidentified anyway. If it spikes, re-open — but the fix lands in gzip_chunk accept logic, never isal.hpp:392-405.

---
## PASS-2 NOTE (owner): the advisor's owed adversarial fixture was BUILT + MEASURED and it
## FLIPPED the lean. tiny-block (20480 SYNC_FLUSH blocks) fixture => fallbacks 1/11/48/104 at
## T2/4/8/16, STABLE x3, byte-exact. The gap is REAL+CONCENTRATED there (the regime the advisor
## #7 named). Updated conclusion (plans/isal-resync-findings.md FINAL VERDICT): gap real but
## (a) benign on ordinary fixed-Huffman, (b) cited isal.hpp:392-405 is the WRONG function
## (readBytes footer reader; wrapper already records boundaries), (c) real fix = decode-across-
## boundary in gzip_chunk accept/coalesce (faithful rapidgzip readStream), a correctness-sensitive
## NEW gated turn. This turn delivers the source refutation + gap-location map + the adversarial
## fixture that PROVES the gap (the repro the prior caveat never measured). See pass-2 advisor below.

---
## PASS-2 ADVISOR VERDICT (full text)
**VERDICT: UPHELD-WITH-CAVEATS.** The banked DECISION survives (gap is real on small-block-dense fixed/stored input; do NOT land the cited `isal.hpp:392-405` port; defer the real fix to a gated turn). But the **gap-LOCATION map (owner point 3) is contradicted by production code** and must be re-derived before that gated turn.

**Disproof attempts:**

**(d) — counter semantics [tried hardest, FAILS to break]:** `ISAL_ENGINE_ORACLE_CHUNKS` increments *only* on `Ok(true)` (gzip_chunk.rs:386, after a committed ISA-L accept); `FALLBACKS` *only* on `Ok(false)` (:680). They are mutually exclusive per `finish_decode_chunk` entry into the ISA-L branch. So `fallbacks > chunks` genuinely means "majority of tail chunks that attempted ISA-L declined to pure-Rust." The owner's inference holds. (Denominator caveat: window-absent bootstrap chunks return `Finished` at :666-668 without entering this branch, so the ratio is over *tail chunks attempting ISA-L*, not all chunks — which is exactly what the owner claims.) **Does not break.**

**(c)+gap-map — the real crack:** Owner point (3) says "the ISA-L WRAPPER already records boundaries on stored+fixed; the decline is in gzip_chunk's ACCEPT logic." The production comment (gzip_chunk.rs:316-325) and commit 19add96c say the OPPOSITE: on stored/fixed input ISA-L's END_OF_BLOCK does **not** fire, the decode records **ZERO boundaries** and runs multi-MiB past the hint. That's *why* it reaches the `end_bit <= stop_hint` guard and declines. If boundaries were recorded, the inexact branch (:304-309) would find one at-or-past the hint and ACCEPT. So the decline is **not** an over-strict accept rule over present boundaries — it's *absent boundaries*. The owner's proposed "faithful readStream coalesce / decode-across-boundary" lever is therefore **mis-aimed**: there is no recorded boundary to coalesce across, and forcing acceptance of `end_bit > stop_hint` is *precisely the over-decode mis-seed the guard was added (same day) to prevent* → "Stored block len/nlen mismatch." rapidgzip's `readStream` works because its wrapper records a boundary per block and stops at the *first* one past `untilOffset`; the gzippy defect is in **boundary RECORDING on stored/fixed**, not accept logic.

**(a) — is "defer" a dodge? [partial — CAVEAT]:** Justified, not a dodge: this was a disproof pass, the change touches the speculative seed path, and the guard it would invert landed *today*. Attempting it same-turn without a dedicated falsifier would be reckless. CAVEAT: legitimate *only* if "gated turn" isn't an indefinite shelf — and only after the root cause is relocated (see crack above), else the gated turn builds on a wrong map.

**(b) — too synthetic? [partial]:** 20k SYNC_FLUSHes isn't archive-class, but it's not academic either — flush-dense streams are real (HTTP gzip transfer-encoding flush-per-write, `Z_SYNC_FLUSH`-per-record, log shippers, WAL streaming). "Narrow but real (streaming producers)" is correct framing. Priority is low vs. the headline low-T *engine* deficit; acknowledge, don't over-invest.

**(e) — byte-exact = safe? [FAILS to certify]:** Byte-exact on one fixed-seed fixture proves the gap *exists*, not that a cross-boundary fix is *safe*. The failure mode (mis-seed → len/nlen mismatch) is input-dependent and currently *avoided by declining*. A fix that changes the seed path must gate against many stored/fixed inputs + fuzzing, not this single fixture. The owner's coverage-gate plan is necessary but insufficient — add fuzz + multiple stored/fixed corpora.

**Single biggest risk:** The owner banks a **mislocated root cause**. Production code says stored/fixed blocks record ZERO ISA-L boundaries; the owner's map says the wrapper already records them and blames the accept logic. Build the "cross-boundary accept" fix on that map and it either no-ops (nothing to coalesce across) or reintroduces the exact over-decode mis-seed `19add96c` guards against. Re-derive root cause from boundary-recording behavior (does ISA-L emit per-block boundaries on BTYPE=00/01?) **before** the gated turn writes any accept-path change.

---
## OWNER — pass-2 "biggest risk" (mislocated root cause) RESOLVED by direct measurement:
Probed the EXACT adversarial tiny-block stream: decompress_deflate_from_bit_with_boundaries
records 40,960 boundaries (2/SYNC_FLUSH, 2048-byte cadence). Boundaries ARE present in abundance;
the production comment's "zero boundaries on stored/fixed" is FALSE for this input. The decline is
the until_exact EXACT-match accept (stop_hint rarely == a boundary bit with dense blocks), NOT
absent boundaries. The real fix lands in gzip_chunk accept logic (relax until_exact to coalesce to
the nearest clean EOB like rapidgzip readStream), NOT isal.hpp, NOT the inexact over-decode guard.
Confirms the gated-turn decision is on a CORRECT map. No production code changed this turn.
