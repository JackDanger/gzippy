I've verified every load-bearing citation against source. Here is my independent disproof verdict.

---

## Verification summary (what I read first-hand)

| Cited fact | File:line I checked | Holds? |
|---|---|---|
| known-window → ISA-L | GzipChunk.hpp:440-444 | ✅ exact |
| stream-end → ISA-L | GzipChunk.hpp:501-505 | ✅ (leader didn't cite this 3rd dispatch — strengthens claim) |
| clean≥32KiB → ISA-L | GzipChunk.hpp:520-526 | ✅ exact |
| WITH_ISAL default ON | CMakeLists.txt:14 | ✅ |
| harness builds no -D override | guest_fulcrum_capture.sh build block | ✅ plain `cmake .. -DCMAKE_BUILD_TYPE=Release` |
| u16 ring identical | deflate.hpp:805 | ✅ `array<uint16_t,2*MAX_WINDOW>`; u8 reinterpret is :806 (leader said :893 — wrong line, right fact) |
| in-place flip | deflate.hpp:1282-1289 | ✅ |
| dist = ReversedBitsCached | deflate.hpp:336 | ✅ |
| ISA-L writes u8 direct | isal.hpp:257 `next_out=output` | ✅ |
| dist cache 128 KiB inline | huffman_reversed_bits_cached.rs:15,20 + Symbol=u16 | ✅ `[(u8,u16);1<<15]` = 32768×4 = **exactly 128 KiB**, not boxed |
| gzippy clean loop writes u16 | marker_inflate.rs:1526 | ✅ **and this is the crux — see CLAIM 2** |

Two filing notes: the leader cites `librapidarchive/...` paths; there are **two** source trees (`vendor/rapidgzip/src/` and `…/librapidarchive/src/`) — I confirmed against `librapidarchive/`. And CMakeLists:25 sets the *wrong* variable name (`WITH_ISAL` not `LIBRAPIDARCHIVE_WITH_ISAL`) in the Darwin guard — irrelevant to the Linux x86_64 guest, but it means even macOS keeps ISA-L on.

---

## CLAIM 1 — rapidgzip ~99% ISA-L — **AGREE (refine the "99%")**

The dispatch is exactly as cited, and there are **three** handoffs to `IsalInflateWrapper` (440-444 known-window, 501-505 new-stream, 520-526 post-prefix), all under `#ifdef LIBRAPIDARCHIVE_WITH_ISAL`, which defaults ON and is not overridden by the trace build. The pure `deflate::Block` runs only the ≤32 KiB markered prefix of window-absent chunks. Corroborated empirically by the non-zero `isal_stream_inflate` busy in parity-final. The qualitative conclusion — **igzip decodes ~all bytes in both `d_c` and `d_w`; gzippy's scalar marker engine decodes 100% of both** — is robust.

Refine: "99%" is chunk-size-dependent (32 KiB prefix ÷ chunk). At rapidgzip's multi-MiB chunks it's ~99%; even at a pessimistic 512 KiB chunk it's ~94%. The number is illustrative, the dominance is not in doubt. Don't let "99%" become a load-bearing constant — state it as "≥94%, prefix/chunk-bounded."

## CLAIM 2 — u16-width refuted — **REFUTE the framing (the most important finding)**

The narrow point is true: **ring storage width is identical** (both 128 KiB u16, deflate.hpp:805 ↔ marker_inflate.rs:290). But "u16-width REFUTED" is **too strong and mislabeled**, and the attack in the prompt is correct — I confirmed it on both sides:

- rapidgzip's **production** path is ISA-L, which writes **u8 directly** (`next_out=output`, isal.hpp:257). Its bulk **never touches the u16 ring**.
- gzippy's clean-mode loop (`CONTAINS_MARKERS=false`) shares `run_canonical_loop!` and writes **u16 per literal** into the u16 ring at **marker_inflate.rs:1526** (`ring_ptr:*mut u16 .write(sym)`). Plus §0.4's resolve pass streams the **full ~20 MiB u16 chunk buffer** even though markers exist only in the ≤32 KiB prefix.

So there are **two** distinct "width" facts, and the doc collapses them: (a) *ring-storage width* — genuinely refuted, both 128 KiB; (b) *write-traffic + resolve-traffic width on the clean bulk* — **NOT refuted, it is a live ~2×-write + full-chunk-reread asymmetry on ~100% of bytes that rapidgzip pays on ~0%.** This is a **bandwidth** difference, and CLAIM 3 says T8 is partly bandwidth-bound — so this is plausibly a primary T8 lever, not a refuted non-lever.

The design *does* propose u8-direct clean writes (§1.2A) but buries it as a sub-bullet of "make the clean loop SIMD," conflating a **low-risk traffic fix** with a **high-risk AVX2 project**. It also leaves the 20 MiB u16 resolve-pass-over-clean-bytes in place. Note one subtlety that sharpens the priority: rapidgzip's *pure* decoder (the ShortBitsCached bench rows) **also** writes the u16 ring — so gap-A (gzippy-pure vs rapidgzip-pure) is pure compute/decoder-class, while gap-B (pure-u16 vs ISA-L-u8) **bundles asm compute AND the u8-traffic/no-resolve advantage**. You therefore **cannot** close gap-B with SIMD compute alone; part of it is architectural traffic.

## CLAIM 3 — vendor ceiling — **AGREE on "scalar can't reach igzip"; REFINE the bandwidth attribution**

Numbers verified verbatim (deflate.hpp:72-93 multi-thread, 137-145 single-thread): s-t silesia ISA-L 720.5, ShortBitsMultiCached-11 337.3, ShortBitsCached-11 327.6, DoubleLiteralCached 252.5; 20× ISA-L 5024 vs ShortBitsCached-11 3927 = 1.28×. Point 1 (a pure **scalar** loop can't reach 720 s-t; asm matters) is sound.

Two refinements:
1. **CPU transplant.** These are a "Frankensystem" with ±10–20% variance explicitly noted (:96-107). Using absolute MB/s (337/720) as guest *targets* is illegitimate. The leader mostly escapes this via the guest-measured ISA-L oracle + ratio control in PROOF-1 — but the PASS criteria still name absolute 337/720 (§2 PROOF-1). Purge absolutes; gate on **guest-measured ratios** only.
2. **The 2.1×→1.28× shrinkage is not purely DRAM bandwidth.** Vendor's own text (:170-172) attributes DoubleLiteralCached's multi-thread degradation to **its LUT being too large for cache when two HW threads share a core** — i.e. **cache contention**, a *different* mechanism with a *different* remedy (shrink the working set → the dist-cache shrink B4) than DRAM bandwidth (move fewer bytes → the u8-write fix). The high-T convergence is therefore consistent with **cache/traffic** binding, which is exactly what the design **ranks lowest**. If vendor's own diagnosis is right, the design has the lever priority **inverted** at the T-regime that the bar is measured in.

## CLAIM 4 — design feasibility — **AGREE the tension is real & the resolution is the only charter-legal one; REFINE the lever ranking**

(a) Hand-rolled Rust+inline-asm reaching igzip-asm (years-tuned `igzip_decode_block_stateless_04.asm`) is *plausible in principle* (same ISA) but **high-risk, multi-week-plus**, with a real chance of landing at the ~337 pure ceiling not 720. The design honestly gates this on PROOF-1 and says "if (ii) plateaus at 337… NOT proven" — good, the risk is acknowledged and fenced.
(b) The one-engine/no-FFI invariant **is** in genuine tension with the TIE (to match igzip bulk you need igzip or an equal); the 3rd option (FFI in gzippy-isal, native accepted-slower) is charter-rejected. The resolution is the only legal path — but it's the highest-risk path, so the proof gate placement is appropriate.
(c) **Lever vs phantom — partial mis-aim.** SIMD compute is *necessary* (CLAIM 3.1), but the design pre-ranks vector-copy #1 / refill #3 / dist-cache-shrink #4, and explicitly "do NOT shrink the u16 ring," **before** PROOF-2 measures which term binds at T8. Given CLAIM 2 (live u16 traffic on 100% of bytes) and CLAIM 3 (cache-contention, not just bandwidth), the **traffic/residency levers (u8 clean writes, eliminate clean-byte resolve, dist-cache shrink) are plausibly co-equal or dominant at T8** and are far lower-risk. Ranking them below an AVX2 project is a soft tier-discipline violation (pre-judging the lever before the perturbation).

## CLAIM 5 — footprint — **AGREE (with one amplification)**

Confirmed exactly: `code_cache: [(u8, Symbol); 1<<MAX_CODE_LENGTH]`, `MAX_CODE_LENGTH=15`, `Symbol=u16` → 32768×4 B = **128 KiB inline (not boxed)**, dist_hc held inline in `Block` (marker_inflate.rs:353-356). With `output_ring` 128 KiB + ~16 KiB litlen LUT → ~279 KiB steady state, overflowing a 256 KiB L2. No shared table: the canonical paths build **local, per-block** `HuffmanCodingReversedBitsCached` instances (marker_inflate.rs:1482, 1587) — "shared tables done" did check a dead path.

Amplification: the **FixedHuffman** path (marker_inflate.rs:1587) instantiates *another* `HuffmanCodingReversedBitsCached::<LITLEN_CAP>` locally = **a second 128 KiB cache**, so fixed-block transient footprint spikes to ~400 KiB, not 279. Minor for silesia (mostly dynamic) but the "279 KiB" is a dynamic-steady-state figure, not a worst case.

---

## TIER-2 proof plan assessment

**PROOF-1 (isolation bench + ISA-L oracle + ~2.1× positive control): sound.** The instrument self-test (reproduce vendor's known ratio before trusting variant ii) is exactly the discipline CLAUDE.md demands. Keep it, but state the PASS gate in guest-measured ratio terms, not absolute 337/720.

**PROOF-2 (max() bandwidth/compute model validated by reproducing both current walls): has a real hole — the prompt's worry is correct.** A 3-term `max()` validated by **reproducing the two walls it was tuned on** is near-circular: enough free constants always refit their anchors, so reproduction ≠ predictive power, and the faster-engine projection is exactly the regime no anchor constrains. Fixes:
- **Hold out.** You already capture per-T (1,2,4,8,16). Fit on T1/T2/T4, **predict T8/T16 as held-out**; trust the faster-engine projection only if held-out lands within spread.
- **Measure the binding term directly, don't infer it from a fit.** The listed `perf stat` MPKI/mem-stall + DRAM-ceiling probe (with the `GZIPPY_MEM_BALLAST` control) can *directly* assert whether T8 is compute- or bandwidth/cache-bound. Use them as the verdict; the model is only a projector.

**A cheaper, more decisive gate is missing — run it FIRST.** The design's whole premise is "clean-loop compute is the T8 lever." CLAUDE.md's own method says test that with a **causal perturbation before a work-stretch**: slow *only* the clean-mode inner loop by a known factor (the `GZIPPY_SLOW_BOOTSTRAP`/ballast template) and measure the interleaved **T8 wall response** + a frequency-neutral control. Monotonic ⇒ compute is on the critical path, build PROOF-1. **Flat ⇒ bandwidth/publish already binds, the AVX2 project is moot**, and you've falsified it for ~an hour instead of after a multi-week SIMD build + a model. Pair it with a tiny **u8-clean-write A/B** (write clean bytes u8 + skip resolve for clean bytes) to isolate the traffic component of gap-B. These two cheap experiments gate the expensive ones and directly resolve the lever mis-ranking above.

---

## Bottom line

- **Is the diagnosis sound enough to design on?** **Yes, with one correction.** Every source citation checks out (modulo two cosmetic line-number slips: u8-reinterpret is :806 not :893; paths are `librapidarchive/`). CLAIM 1 (ISA-L dominance), CLAIM 5 (128 KiB inline dist cache, no shared table), and the BMI2 rejection all hold. **But CLAIM 2 is mis-framed:** ring-storage width is refuted; **clean-bulk write+resolve traffic width is a live, ~100%-of-bytes asymmetry that is NOT refuted** — and it's a bandwidth lever in the exact T8 regime CLAIM 3 says is bandwidth/cache-sensitive.
- **Is the design aimed at the real lever?** **Partially.** SIMD compute is correctly identified as *necessary* (you can't reach igzip s-t with a scalar loop). But the design **under-weights and pre-ranks-low the traffic/residency levers** (u8 clean writes, eliminating clean-byte resolve over the 20 MiB u16 buffer, dist-cache shrink) that vendor's own cache-contention note (:170-172) and the live u16 write at marker_inflate.rs:1526 point to as plausibly dominant at T8 — and it pre-commits to "don't shrink the u16 ring" *before* PROOF-2. That's a lever-aim risk and a soft tier-discipline violation.
- **Is the proof plan a valid gate?** **PROOF-1 yes; PROOF-2 no, as written** (validate-by-reproduction is overfit-circular — needs held-out cross-validation + direct perf-stat term attribution). And the plan **skips the cheapest decisive gate**: a clean-loop slow-injection perturbation + a u8-write A/B, which per CLAUDE.md's own method should run *before* committing to the AVX2 build.

**What must change before TIER-3:** (1) relabel CLAIM 2 — separate ring-width (refuted) from clean-bulk traffic-width (live lever); promote the u8-clean-write + drop-clean-resolve to a first-class, separately-measured item. (2) Add the clean-loop causal perturbation + u8-write A/B as a **pre-gate** ahead of PROOF-1/2; let it, not the priority list, rank SIMD-vs-traffic. (3) Fix PROOF-2 to hold-out-validate and to read the binding term from perf-stat directly. (4) Strip absolute 337/720 targets in favor of guest-measured ratios. With those, the diagnosis is solid enough and the gates become real falsifiers rather than confirmations.
EXIT_ADV=0
