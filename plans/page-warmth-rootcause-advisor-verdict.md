# Page-warmth root-cause — independent DISPROOF verdict (advisor, HEAD f80294ae)

Adversarial review of plans/page-warmth-rootcause-brief.md. The advisor REFUTED the core fix direction.

## The decisive arithmetic (the frame for everything)
Excess faults = 110,617 − 55,790 ≈ 54,800. A first-touch fault that zero-fills (`__memset_avx2` is in the profile) costs ~0.6–1.0µs ⇒ excess faults = **~33–55ms of CPU work**. The MATCHED wall gap = 0.132 − 0.115 = **17ms**.
The fault work (33–55ms) is **2–3× larger than the entire wall gap (17ms)**. At depth-14 across 8 cores, that is only possible if ≥half (likely ≥70%) of the excess faults are OVERLAPPED/SLACK. ⇒ Driving faults to 56K has a CEILING of ~17ms and a realistic expectation of FAR less. **The warmth thesis can be 100% right about fault COUNT and still be a near-TIE on the wall.** That is the LIKELY mode, not a tail risk.

## Verdicts
- **D1 (push_slice 44% is the primary fault site) — UPHELD-WITH-CAVEATS.** Top fault SITE, yes; but `perf record -e page-faults` samples FAULT events, not wall. "44% SELF of faults" is the attribution rule #1 forbids concluding a lever from. push_slice could be 44% of faults AND fully slack-masked; the clean-tail (18%) could be the wall lever. Correction: rank by CAUSAL PERTURBATION, not fault self%.
- **D2 (cause = fresh-per-chunk u16 segments, no warm reuse) — REFUTED.** (1) MANUAL_BUFFER_POOL, which DOES recycle, was WORSE (120K>110K). (2) gzippy's own chunk_buffer_pool.rs:270 says rpmalloc's thread cache ALREADY retains warm pages ⇒ "no warm reuse" is factually wrong. The allocator-warmth lever family tops out ~−10% (slab), vs the ~50% reduction needed. Real divergence = the cross-thread FREE DESTINATION (gzippy frees on the CONSUMER → evicts the worker's warm thread-cache → next worker alloc cold), not absence of reuse.
- **D3 (fix = thread-local worker-side reusedDataBuffers) — REFUTED-as-stated / premature.** rg's reusedDataBuffers reuses the worker's OWN just-finished buffer in its NEXT decode (reuse window = 1 chunk), BEFORE cross-thread handoff. At gzippy's depth-14 the consumer is 14 chunks behind; the worker starts ~14 chunks before any buffer frees back ⇒ a thread-local LIFO RE-SKINS the manual pool and likely reproduces its WORSE result. Unmeasured prerequisite: rg's actual in-flight DEPTH vs gzippy's 14 — if rg runs shallower, DEPTH is the divergence and the fix is bounding depth / earlier recycle, not a buffer cache.
- **D4 (falsifier faults↓ AND matched wall↓) — UPHELD as correct, but MIS-SEQUENCED.** Run it FIRST and cheaply, BEFORE any big build, because the fault-work ≫ wall-gap predicts faults-drop / wall-TIE.

## (c) The under-investigated counter-thesis (most faithful-port-shaped)
finished_no_flip=4 ⇒ 4/16 chunks stay at u16 width (2 bytes/symbol), never narrow. window_seeded=2. If rg's window-seeding decodes those chunks CLEAN (u8) where gzippy carries them double-width, gzippy first-touches strictly MORE pages — a FOOTPRINT divergence downstream of speculation/seeding policy, not an allocator divergence. This would explain a fault gap allocator tricks demonstrably can't close (−10% ceiling). Reducing finished_no_flip shrinks the u16 footprint at the SOURCE and is vendor-convergence. UNMEASURED: rg's flip/seed ratios.

## THE ONE CHEAPEST DECISIVE EXPERIMENT (advisor-prescribed, run BEFORE any implementation)
Pre-fault the u16 marker arena OFF the hot path (MAP_POPULATE / one-time startup pre-touch+memset of the marker segments), byte-transparent. This is the rule-#3 oracle-removal — eliminates the fault COST without building reusedDataBuffers or touching architecture.
- faults→~56K AND matched gz_null wall→0.115 ⇒ warmth CONFIRMED → then build the faithful worker-side recycle.
- faults→~56K AND wall stays ~0.132 ⇒ faults are SLACK ⇒ **warmth thesis REFUTED** (afternoon, not a port).
The arithmetic predicts the second outcome. Run it first.

---

# Advisor PASS 2 — on the (premature) refutation, BEFORE the clean oracle
Verdict: refutation NOT EARNED — the prefault oracle FAILED (faults up, not down), so no clean fault-removal was ever run. SLAB's −10%/wall-worse is confounded by its lock overhead. Owed: ONE clean oracle — never-freed / MAP_POPULATE'd persistent per-worker decode buffer, measured at /dev/null. Worker-side recycle (DecodedData.hpp:344-388) is owed REGARDLESS as faithful convergence (rule 7a) but coupled to a depth bound (a bolt-on at depth-14 finds an empty pool). Next localization: the clean-only engine oracle (2.3× clean-rate gap), independently owed.

# Advisor PASS 3 — on the CLEAN oracle (slab64k targeting the 44% marker fault site)
Verdict: NOT UPHELD AS STATED — uphold only the NARROW claim.
- EARNED: resident-retain of u16 MARKER segments cannot reach rg's floor (−12% ceiling, confound-independent). Rely on the CEILING, not the flat wall.
- NOT EARNED: "page-warmth refuted as a class" — the 211 MiB OUTPUT materialization (~56% of faults, the site rg avoids via windowed/recycled append) was NEVER warmth-oracle'd.
- The flat matched wall is an UNDERPOWERED TIE: max plausible warmth win (single-digit ms) is 5-10× below the 13-18ms harness spread ⇒ unresolvable; proves nothing either way (rule 5/7). Drop it as evidence.
- The "rg faults less = ISA-L" attribution conflates engine (ISA-L) with DATA STRUCTURE (windowed recycled materialization). The latter is pure-Rust-portable, NOT forbidden, = the owed rule-7a recycle. The 3.26× DTLB regression was FRESH segmentation; RECYCLED warm-TLB windows are a different unrun experiment.
- Next move (clean-only engine oracle) CORRECT AS TRIAGE not closure: 2.3× engine gap ≫ warmth's hidden few-ms. Record "engine-first on triage grounds; page-warmth not refuted as a class."
- STILL OWED for page-warmth: one recycled-window oracle on the OUTPUT materialization (faithful DecodedData.hpp:344-388, reused 128 KiB append windows, warm-TLB).
