Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
I've source-verified the brief, prereg, both prior turns, the charter oscillation block, the oracle (`gzip_chunk.rs:160-281`), the dispatch/guard (`:178`, `:539-554`), the production clean path (`:571-614`), and the vendor decode primitive (`deflate.hpp:1277`, `1285`, `1452-1453`, `1585-1666`). Verdict below.

---

# DISPROOF-ADVISOR VERDICT — clean-tail wall oracle

Independent, read-only, HEAD 7aae6c4a. I tried to break both conclusions; here is what survived.

## Bottom line
**CONCLUSION 1 — REFUTED as-framed (verdict: INCONCLUSIVE).** The falsifier was not validly testable: the oracle's own overhead, by the owner's *own* estimate (~0.17×), is **larger than both the prod→ocl gap (0.05×) and the gap to the 0.85× threshold (~0.15×)**. The instrument cannot read the speed-up ceiling it was built to read.
**CONCLUSION 2 — UPHELD-WITH-CAVEAT.** The window-absent *structure* costs ~0.16× at the wall (clean), but crediting it to "the u16 marker bootstrap **rate**" overcredits: seeding co-removes 13 spec-failure re-decodes (a block-finder/scheduling term), and the 0.120s≈0.322s "match" is a SUM÷overlap step — the charter's SUM-vs-wall trap (CAMPAIGN-CHARTER.md:522-528).

## The single most load-bearing correction
The owner uses the copy confound **asymmetrically, and the two uses contradict**:
- Reconciliation: "ocl_seed 0.86× vs pure seedfull 1.029× — **the ~0.17× gap is the oracle copy**" (brief:48-49). → copy is **large**.
- Result 1: "a copy-free ISA-L clean tail would be **at-best slightly above** ocl's 0.70×, still ≈prod's 0.75×" (brief:50-52). → copy is **negligible**.

Both can't hold. Take the owner's own C≈0.17×: model W_ocl = W_prod − S + C with S = ISA-L clean-engine wall-saving. Result 1 measured W_ocl ≈ W_prod ⇒ **S ≈ C ≈ 0.17×**. That doesn't show the clean engine is slack — it implies the clean engine saves ~0.17× of wall, **exactly masked by the copy**, which would put a copy-free ocl at ~0.84–0.87× — **at/above the 0.85× falsifier threshold**. The owner's numbers, taken together, point the *opposite* way from CONCLUSION 1.

The logical error: a *handicapped* contender (oracle pays a 64 MiB alloc+copy per chunk, `gzip_chunk.rs:203,247-256`, that production's direct-to-`chunk.data.writable_tail()` stream `gzip_chunk.rs:592-614` never pays) **failing to win** is *uninformative* about a speed-UP ceiling — not "conservative." "Conservative" would apply only if the handicapped contender had *won*. This is Measurement-PROCESS Rule 3 again: the oracle was supposed to *remove* the region to bound the speed-up, but it removes the pure-Rust engine and **adds a fresh confound of the same magnitude as the lever**, so it bounds nothing.

## Q1 — Does the copy confound invalidate CONCLUSION 1?
**Yes.** Not because the copy lands "specifically on the critical path" in some subtle way, but because of magnitude: the owner's own reconciliation prices the oracle overhead at ~0.17× of the wall ratio (1.029→0.86, brief:48), and ocl fires on 14 chunks unseeded vs 17 seeded — comparable. The overhead is ≥ the prod→ocl gap AND ≥ the gap-to-threshold. The "only makes ocl look worse ⇒ conservative" argument is invalid for a speed-up-ceiling question (above). Note the 2×2 also shows the oracle is net-slower than pure-Rust in *both* conditions (0.86<1.029 seeded, 0.70<0.73 unseeded) — so it yields **no positive evidence** an ISA-L-speed clean engine helps, but its overhead is too big to yield reliable **negative** evidence either. Null/contaminated instrument.

## Q2 — Is CONCLUSION 2's attribution airtight, or the Phase-0 Claim-3 bundle again?
**Bundle reappears.** Seeding removes three co-varying things (brief:60): (i) the u16 marker bootstrap body, (ii) **13→0 spec-failure re-decodes**, (iii) flip machinery (12→0). The unseed→seed delta (0.70→0.86) is credited wholesale to (i)'s *rate*, but (ii) is wasted re-decode from wrong partition guesses — that's the named head-of-line/block-finder term ([[project_confirmed_offset_prefetch_gap]]), a *scheduling* cost removed by a better block-finder **without touching the marker rate**. The "ΔdecodeBlock 0.120s ≈ bootstrap body 0.322s SUM/overlap" reconciliation (brief:41) divides a SUM by an overlap factor to match a SUM-delta — exactly the arithmetic the charter flags as the SUM-vs-wall trap (lines 522-528) and that has been refuted before (the combine_crc "62ms" phantom, CLAUDE.md rule 8). The CAUSAL claim ("the window-absent *structure* costs ~0.16× at the wall, clean engine held constant") is sound; the *sub-attribution to marker-rate specifically* is not isolated.

## Q3 — Does this end the engine↔scheduling oscillation? Third term?
**No.** Two terms remain entangled, not eliminated: (a) the **spec-failure re-decode** cost bundled into Q2 (scheduling/block-finder), and (b) the **marker→clean resolution pass** — rg spends 0.0348s "applying last window" (attribution.md:22); gzippy's gather/narrow resolve is isolated by neither conclusion. So the proposed "third binder" (bootstrap rate) is itself cross-contaminated by *both* engine (Q1 confound) and scheduling (spec-fail). This does not settle the oscillation — it risks installing a third under-resolved label.

## Q4 — Is no-FFI 1.0× reachable via a faster pure-Rust u16 marker bootstrap, unbounded by the VAR_VI 0.6× clean plateau? Or genuinely forked?
**The fork is NOT established; the premise is structurally wrong.** In rapidgzip the marker decode and the clean decode are the **SAME primitive**: `readBlock` calls `readInternal(..., m_window16)` for markers (deflate.hpp:1277) and `readInternal(..., window)` for clean (1291), both dispatching to the **one** `readInternalCompressedMultiCached` (1452-1453) — a single template over window element width (`containsMarkerBytes` at 1600), same `coding.decode()` symbol loop (1585-1666), with `setInitialWindow()` (1285) flipping width mid-chunk. rg ties unseeded **because that one fast loop serves both**; the marker path's only extra cost is u16 writes + the end resolve.

gzippy's premise — "the marker bootstrap (227 MB/s, marker_inflate.rs) is a separate lever, ISA-L-untouchable, **not** bounded by the clean primitive plateau" — describes gzippy's **divergence** (two engines: marker_inflate u16 + resumable u8 clean), not rapidgzip's design. Under a faithful port there is ONE width-templated multi-cached loop whose ceiling governs **both** rates; VAR_VI measured essentially that loop's clean variant. So the marker rate is most likely bounded by the **same** primitive ceiling, and "speed the marker path independently, free of the clean plateau" is not a real fork — it's an artifact of the un-unified engines. The faithful move is to collapse marker+clean into one `readInternalCompressedMultiCached`-shaped primitive (per [[project_faithful_unified_decoder_over_perf]]); then both rates rise or plateau together. Whether that unified ceiling clears 1.0× pure-Rust is the genuine open question — but it is **one** question, not a fork.

## Verdict table
| claim | verdict |
|---|---|
| C1: clean engine rate is slack-masked, NOT the wall binder | **REFUTED as-framed → INCONCLUSIVE** (instrument overhead ≥ effect ≥ threshold gap; owner's own C≈0.17× implies S≈0.17×, i.e. engine ≈ the whole gap) |
| C2: gap is the window-absent structure (engine held constant) | **UPHELD** for *structure*; **CAVEAT** — sub-attribution to marker *rate* bundles 13 spec-fail re-decodes + uses SUM÷overlap |
| oscillation ended | **No** — spec-fail re-decode and marker-resolve still entangled |
| Q4 fork (marker path unbounded by clean plateau) | **Not established** — vendor uses ONE width-templated primitive for both (deflate.hpp:1277/1452/1585-1666); the "fork" is gzippy's two-engine divergence |

**Most actionable next step:** the oracle is unusable for the speed-up ceiling until the 64 MiB alloc + intermediate-buffer copy is removed (decode ISA-L directly into `chunk.data.writable_tail()`), so C→0 and S becomes readable. Until then, neither "engine is slack" nor "engine is the lever" is decided.
=== ADVISOR EXIT 0 ===
