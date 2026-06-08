## Single most important disproof — the configs are mispaired

The headline conclusion ("removing gzippy's output makes it TIE rg ⇒ OUTPUT is the binder, engine+sched at parity") is built on an **apples-to-oranges pairing**: it compares gzippy-*with-output-removed* against rg-*with-output-present*. Your own Claim-3 matched A/B refutes it. With both sinks neutralized to /dev/null:

- gzippy /dev/null **0.1306** vs rg /dev/null **0.1138** → gzippy **1.148×**, a ~17 ms gap that **survives output removal on both sides.**

And `gzippy_skip` (0.1310) ≈ gzippy /dev/null (0.1306), confirming "skip" just removes output. So the real chain is: gzippy-no-output ties rg-*with*-output, but **loses 15% to rg-no-output.** The TIE in Claim 2 is a coincidence of subtracting gzippy's *more expensive* output (~34 ms) down to where rg's *total* (engine + cheap 14 ms output) happens to sit. Matched-config, gzippy is **not** at engine+sched parity at T8.

This means there are **two** T8 binders, which the owner collapsed into one: (1) ~20 ms excess OUTPUT cost (gzippy's serial copy is dearer than rg's), and (2) ~17 ms engine+sched residual that output-removal cannot touch.

## Per-attack verdicts

**A — oracle validity: UPHELD-WITH-CAVEATS.** The removal oracle is a legitimate *bound* (output costs ≤34 ms on the wall, Rule 3). Your contention worry is actually **rebutted by your own numbers**: T1 skip saves **48 ms**, T8 skip saves only **34 ms** on the same 211 MiB. Bandwidth contention would make the T8 copy *more* expensive (≥48 ms), not less. The 34<48 says the copy is partly *hidden* by WAIT-slack at T8 (Amdahl-serial tail), not amplified. Caveat: the oracle still does not separate "serial placement on the consumer" from "the copy being intrinsically expensive because of an engine/arch choice" — see E.

**B — "skip==rg ⇒ engine at parity": REFUTED.** As above: skip==rg pairs unequal configs. The matched /dev/null comparator (your Claim 3) leaves gzippy 15% behind. The wrong-bytes skip doesn't break overlap to flatter gzippy; it simply removes gzippy's *heavier* output, masking the engine deficit.

**C — granularity refuted: UPHELD.** The cap sweep {2048,256,95 KiB} all TIE-or-worse vs off (0.992/1.000/0.979×) → syscall shape is not the lever. Sound. The cost is the page-cache copy, not the call boundary.

**D — why is rg's serial output ~20 ms cheaper: the wrong question.** Both are serial-synchronous (ParallelGzipReader.hpp:521), and granularity is out. The remaining explanation is **data flow**: rg decodes into the chunk's final u8 buffer the consumer hands straight to write(); gzippy's u16 marker ring forces a **second 211 MiB pass** (apply-window / u16→u8 narrowing) leaving the consumer's source buffer colder and double-touched. So the "20 ms output excess" is engine-architecture, not an output-syscall lever.

**E — phantom risk: UPHELD, and it's the unifying read.** Both binders (the 17 ms /dev/null residual *and* the 20 ms output excess) plausibly reduce to the same cause: a slower per-thread decode and the **extra output-side memory pass from the u16 ring**. "Output" is the engine/arch residual wearing an output mask — exactly the standing thesis in `project_faithful_unified_decoder_over_perf`.

## The one faithful, wall-moving technique

**Decode-and-narrow directly into the final consumer u8 buffer — eliminate the u16→u8 second pass.** Make the chunk's decode write the bytes the consumer will `write()`, the way rg does (decode → final u8 buffer → consumer writes it). This is **faithful to rg's data flow** (not a divergent background writer, which D rules out), and it attacks **both** binders at once: it removes the 211 MiB narrowing pass (the ~20 ms output excess, now a single warm pass) *and* the u16-ring engine tax (the ~17 ms /dev/null residual). Validate with the matched comparator: **gzippy /dev/null vs rg /dev/null**, interleaved, sha-verified — that, not skip-vs-rg-file, is the parity gate.
