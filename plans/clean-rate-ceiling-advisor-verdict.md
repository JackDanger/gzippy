Confirmed the decisive point in the owner's own code. Lines 872–875 state plainly: production native pays "the engine ring write + the ring->data drain memcpy ... the ISA-L `ocl_cf` ceiling pays neither, so the residual to that 0.925× ceiling is an UPPER BOUND on intrinsic symbol rate, not pure rate." That is the disproof. Verdict follows.

---

# DISPROOF VERDICT — clean engine-rate ceiling (ocl_cf 404ms)

## The break (governs C1 and C3)

The oracle does **not** isolate the clean engine. It removes **two** things at once:

1. pure-Rust Huffman symbol rate → ISA-L symbol rate (the *intended* removal), **and**
2. the **MarkerRing write + ring→`chunk.data` drain memcpy** — a second-touch of *every* clean byte (~200 MB) that production pays and ISA-L-into-`chunk.data` does not.

This is not my inference — it is asserted verbatim in the owner's own source (`gzip_chunk.rs:872-875`) and a dedicated knob (`GZIPPY_FOLD_NODRAIN`/`FOLD_NOCRC`, lines 149-175, 905-911) exists *specifically* to subtract term #2. **The brief did not run that split.** So 36ms (440→404) = engine-rate gain **+** ring-write/drain elimination, jointly. At memory bandwidth a 200 MB second-touch is plausibly 10–20ms — i.e. up to *half* the "engine" headroom may be the ring copy, not symbol rate.

The four STEP-2 techniques the brief authorizes (table prefetch, single-level table, static-Huffman, FASTLOOP yield-elision) attack **symbol rate only**. None of them removes the ring write or drain. So they **cannot reach 404ms** — the realistic engine-only floor sits at `404 + (ring_write + drain)` ms.

## Claim-by-claim

- **C1 — REFUTED as stated.** "The clean engine-rate gap is 36ms" is false: 36ms is engine-rate **+ ring architecture**, an *upper bound* on the engine lever (owner's own words). The two are distinct levers (one is the inner loop; the other is the faithful u8-direct/no-drain change in memory `[[project_faithful_unified_decoder_over_perf]]`). Run `FOLD_NODRAIN`+`FOLD_NOCRC` and report the split before calling engine-rate "the largest single lever."

- **C2 — UPHELD-WITH-CAVEATS.** The 21ms (404 vs 383) residual is real and genuinely *outside the clean engine*. Caveat: its composition is unattributed and it includes **marker-region pure-Rust decode** (the 12 flipped + 2 seeded chunks' pre-flip portion), which is engine-*class* compute, not "bootstrap structure + placement" only. Directionally sound; don't over-claim it as pure scheduling.

- **C3 — REFUTED as stated.** "404ms is the ceiling; no pure-Rust engine technique can go below 404 / anything below 404 is noise" is wrong in both directions. (i) The authorized engine techniques cannot *reach* 404 (they don't remove the drain), so 404 is a loose, unreachable target for them. (ii) A pure-Rust change that *also* eliminates the ring drain (the authorized u8-direct path) can legitimately approach 404 by a **non-engine** mechanism — so a sub-404 measurement is NOT automatically noise. 404 is only a valid *hard lower bound* for "ISA-L-class clean path **with** direct write," not for "engine rate."

- **C4 — UPHELD-WITH-CAVEATS.** The 1.4 GHz pin improves determinism (7-8% spread) and the *structure* (engine gap + non-engine residual) is frequency-robust. But the **absolute** 36/21ms are frequency-specific; the brief itself concedes the engine fraction shrinks at low freq, which means the engine lever is likely **worth more** at turbo. Do not port absolute ms to a turbo target — re-measure at production frequency.

## The three questions

- **(a) Valid speed-up ceiling? UPHELD-WITH-CAVEATS.** Copy-free verified in code (`writable_tail_reserve`+`commit`, no intermediate Vec). Coverage 14/0 and matched `/dev/shm` sink verified. It satisfies Rule 3 (removal, not slope). **But it over-removes** — it brackets engine-rate **+ ring write/drain** together. Valid ceiling for "ISA-L-class direct-write clean path"; **invalid** as a ceiling for the engine-rate lever alone.

- **(b) Is the 21ms residual real, or oracle-specific structure rg doesn't pay? REAL but partly mis-labeled.** It is genuinely non-clean-engine and survives matched-pairing. The oracle still pays gzippy-specific marker bootstrap + ring (for marker chunks) + per-byte CRC + consumer that rg-with-ISAL structures differently — so 21ms is a legitimate *non-engine* gap, but it is a mix of (bootstrap/placement) **and** (marker-region pure-Rust decode + residual ring cost on marker chunks), not a clean "scheduling" number.

- **(c) Does the 1.4 GHz pin invalidate? NO** — it does not invalidate the structure or the ranking. It **does** make the absolute ms non-transferable to turbo (engine lever larger there).

## What's owed before STEP-2 uses this number
Run the decomposition the owner already built: `GZIPPY_FOLD_NODRAIN=1` and `+FOLD_NOCRC=1` on native, interleaved against native-fold and ocl_cf, to split the 36ms into {symbol-rate, ring-drain, CRC}. Only the symbol-rate slice bounds the four engine techniques. Until then, ocl_cf 404 over-credits them.

**Bottom line:** ocl_cf 404ms is a real removal-based ceiling for an *ISA-L-class direct-write* clean path, but it is **NOT** a clean engine-rate ceiling — by the owner's own code it also strips the ring write+drain, so it is an *upper bound* on the symbol-rate lever; the 21ms non-engine residual is real but compositionally mixed, and the absolute ms are frequency-specific — split the 36ms with FOLD_NODRAIN before bounding STEP-2.
