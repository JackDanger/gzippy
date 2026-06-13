# Page-warmth — clean oracle RAN → REFUTED as a wall lever (owner turn, HEAD f80294ae)

## The clean fault-removal oracle the advisor owed (run this turn)
Advisor pass 1's prefault oracle FAILED (faults went UP — rpmalloc handed the decode different pages). Advisor pass 2 owed a CLEAN fault-removal oracle: drive marker faults to ~rg's floor with a never-munmap resident slab, measured at /dev/null on the same footing as the gap.

Built GZIPPY_SLAB_THRESHOLD_KIB (lower the SlabAlloc resident-retain threshold so the 128 KiB u16 marker segments — the dominant fault site SegmentedU16::push_slice = 44% — ALSO get resident-retained instead of rpmalloc cross-thread-free + re-fault; sub-MiB blocks round to 128 KiB not 1 MiB; GZIPPY_SLAB_CAP=512 to retain enough). Byte-exact OFF==ON==028bd002…cb410f.

### RESULTS (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, interleaved N=15)
- FAULTS (file sink p8): off 110,619 | slab64k 97,377 (−12%) | rg 55,139. Retaining marker segments resident moved faults only −13K — STILL ~1.77× rg, FAR from the floor.
- MATCHED WALL (/dev/null): gz_null 0.1324 | gz_s64 0.1314 (**1.008× = FLAT/TIE within spread**) | rg_null 0.1132 (1.17×).

### Prior oracles (all confounded or weak, this + last turn)
- SLAB (≥3 MiB only, retains big chunk.data): faults 110K→100K (−10%), matched wall 0.971× = WORSE (lock/side-table overhead). Note: ≥3 MiB threshold MISSED the 128 KiB marker segments entirely.
- MANUAL_BUFFER_POOL: faults 120K (WORSE), wall worse.
- PREFAULT_ARENA: faults 193K/333K (FAILED — added cost, rpmalloc didn't reuse the warm pages).

## VERDICT (advisor pass-3 corrected — do NOT overclaim)
- **EARNED (narrow):** resident-retain of the u16 MARKER segments (the push_slice 44% write site) CANNOT reach rg's fault floor — −12% ceiling, confound-independent (a perfect zero-overhead version still only retains marker segments). It is NOT the wall lever. This is the solid claim; rely on the CEILING, not the flat wall.
- **NOT EARNED (broad):** "page-warmth REFUTED as a class" OVERCLAIMS. The dominant-by-residency fault site — the **monolithic 211 MiB OUTPUT materialization** (~56% of faults, the site rg avoids via windowed/recycled append DecodedData.hpp:344-388) — was NEVER warmth-oracle'd; slab64k only touched the marker segments.
- **The flat matched wall proves little (underpowered TIE):** advisor arithmetic — max plausible warmth win is single-digit ms, the harness spread is 9-14% of ~132ms = 13-18ms ⇒ the effect is 5-10× BELOW the noise floor; the harness physically cannot resolve it. Adding N won't help. Per rule 5/7 a sub-spread TIE is NOT a refutation of the direction. Drop "flat wall" as evidence in either direction.
- **WHY rg faults half (corrected):** TWO fused causes — (1) ISA-L the engine, (2) DecodedData's windowed/recycled materialization (decode into a small reused working buffer, append 128 KiB increments ⇒ faulted working set = the recycle WINDOW, not the full 211 MiB). **(2) is a pure-Rust-portable DATA-STRUCTURE choice, NOT ISA-L, NOT forbidden by goal #1** — and is exactly the rule-7a worker-side recycle already owed. The prior 3.26× DTLB regression was FRESH clean-segmentation (cold TLB per segment); RECYCLED windows reuse virtual addresses (warm TLB) — a DIFFERENT, unrun experiment.

## What is KEPT (rule 7a, byte-exact, OFF-by-default)
- GZIPPY_SLAB_THRESHOLD_KIB + sub-MiB granularity (measurement knob; OFF==identity). Not promoted.
- GZIPPY_PREFAULT_ARENA (failed oracle, kept as a knob; OFF==identity).
NO production fix landed (no warmth lever moved the matched wall).

## The faithful convergence still OWED (NOT a wall lever — structural fidelity, rule 7a)
Advisor pass 2 (c): gzippy frees chunk buffers on the CONSUMER (chunk_data.rs:1665); rg recycles on the WORKER before handoff (reusedDataBuffers, DecodedData.hpp:344-388) COUPLED with its depth bound. This is a structural divergence worth converging EVEN ON A TIE — but it must be ported TOGETHER with rg's depth/backpressure (a bolt-on recycle finds an empty pool at depth-14). Deferred: it does not move the matched wall, so it is fidelity work, not parity work.

## NEXT MOVE = engine-first AS TRIAGE (not as page-warmth closure)
Pivot to the CLEAN-ONLY ENGINE ORACLE — decode the clean bulk via the fastest path with no marker resolution, measure the matched wall; bounds the 2.3× clean-rate engine gap (project_pregate_placement_is_dominant_lever). Rationale (advisor pass-3): the 2.3× engine gap DWARFS warmth's sub-spread few-ms, so engine is the right priority — but record honestly: "engine-first on triage grounds, page-warmth NOT refuted as a class." If engine bounds short, slow-inject sched (handoff/marker-resolution/window-publish) by N% with a frequency-neutral control.

## STILL OWED for page-warmth (rule 7a convergence, the only untested 56%)
ONE recycled-window oracle on the OUTPUT materialization: faithful port of DecodedData.hpp:344-388 — reused 128 KiB APPEND windows (warm-TLB, virtual-address reuse), NOT the fresh per-segment clean-segmentation that caused the 3.26× DTLB regression. This is the only test that targets the ~56% of faults slab64k left untouched AND is the worker-side-recycle convergence item already on the hook. Deferred behind engine triage; do NOT claim page-warmth refuted as a class until it runs.

## Falsifier outcome (corrected)
Pre-registered: faults↓ toward 56K AND matched wall↓ toward 1.0×. RESULT: the MARKER-segment warmth sub-lever moved faults only −12% (NOT toward 56K) ⇒ that sub-lever is dead. The matched wall is FLAT but UNDERPOWERED (effect sub-spread) ⇒ does NOT adjudicate. The OUTPUT-materialization warmth sub-lever is UNTESTED. ⇒ page-warmth's marker sub-lever refuted; the class is NOT refuted; triage to engine while the output recycled-window oracle stays owed.
