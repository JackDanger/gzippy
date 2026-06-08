# Output-reconciliation — independent DISPROOF verdict (advisor, HEAD 20084c91)

Adversarial review of `plans/output-reconciliation-brief.md`. Numbers cited from Battery 3
(N=15, the contention-paired matched battery): gz_file 0.1677, gz_null 0.1324,
gz_skip(null) 0.1318, gz_ovl(file) 0.1600, rg_file 0.1305, rg_null 0.1148.

## Per-claim verdict

- **C1 (matched /dev/null is the correct parity gate; ~1.16× ⇒ two binders): UPHELD-WITH-CAVEATS.**
  The methodology is right and overturns the prior mispairing. The earlier "skip==rg TIE"
  compared gz-output-REMOVED (0.131) against rg-output-PRESENT (0.130) — apples-to-oranges,
  a Rule-6 same-config violation. The matched comparator (both /dev/null) is the legitimate
  gate and it shows gz_null/rg_null = 0.1324/0.1148 = 1.153× (brief rounds to 1.16×). The
  additive partition is clean and self-checks: 19.6ms output-excess + 17.6ms null-residual =
  37.2ms = gz_file−rg_file. **Caveat that matters:** "TWO binders" overstates *independence*.
  Both terms plausibly reduce to ONE root cause (page warmth / arena — see disproof below),
  so they are two *masks*, not two *levers*. Calling them two binders is fine for accounting,
  dangerous for planning (it invites two separate fixes when one fix moves both).

- **C2 (output-overlap lever ceiling ~0.88× rg): REFUTED AS DERIVED; the ceiling is real but
  the 0.88× number is a construction, not a measurement.** The derivation
  "exposure can at best drop to rg's 15.7ms shared floor ⇒ gz_null+15.7 ≈ 0.148 ≈ 0.88×"
  is unsound in BOTH directions:
  - *It can't necessarily reach 0.88× by overlap alone.* The 19.6ms "output excess" is, by the
    /dev/null asymmetry (below), substantially a COLD-SOURCE copy_from_user penalty, not a
    write-timing penalty. Moving the writev to a background thread does not warm the source
    pages — the cold read just happens on the writer thread, still consuming the same bandwidth.
    Overlap HIDES timing; it does not reduce a copy that competes for the same memory bus the
    8 decode workers are saturating.
  - *Conversely, the floor reasoning is wrong — perfect overlap is not bounded below by rg's
    15.7ms.* rg's 15.7ms is rg's *exposed* output; it is not a law that 15.7ms must stay exposed
    in gzippy. A perfect overlap that hid output fully under the 64–67% consumer WAIT would
    bound at gz_null ≈ 0.1324 ≈ 1.015× rg, not 0.88×. The true ceiling is set by memory-bandwidth
    headroom while 8 cores decode, which is unmeasured here.
  - *What IS measured:* the shipped OverlapWriter captures only 0.1677→0.1600 = 7.7ms / 4.6%
    (0.81× rg). So the only demonstrated overlap ceiling is ~0.81×; everything beyond is
    unproven arithmetic. Report the lever as "measured 0.81×, ceiling unknown and probably
    bandwidth-bound," not "ceiling 0.88×."

- **C3 (rg's 15.7ms is the genuine SHARED memory-bandwidth floor): UPHELD-WITH-CAVEATS as a
  LOWER BOUND, REFUTED as a "floor gzippy also pays."** 211 MiB copied once, warm, at ~14 GB/s
  ≈ 15ms — so ~15.7ms is physically consistent with a single warm materialization, and it is a
  sound LOWER bound on any tool's output cost. BUT the brief uses it to assert "~15.7 of gzippy's
  35.3 is NOT a gzippy deficit." That inference is wrong: gzippy does not get rg's 15.7ms for
  free — it gets 35.3ms *because its source pages are cold*. 15.7ms is rg's *warm-source* cost;
  gzippy can only reach it by fixing page warmth, which is precisely the engine/arena deficit.
  So 15.7ms is a floor gzippy must EARN, not one it already pays.

- **C4 (next lever is the engine+sched residual / cold-page-fault tax, not output-overlap):
  UPHELD, and it is the strongest claim in the brief — but its stated *mechanism* needs
  correcting (see disproof).** Direction is right: stop chasing overlap (sub-parity AND
  non-faithful), attack page warmth. The specific mechanism must shift from advisor-D's
  "eliminate gzippy's u16→u8 second pass" to "keep the narrowed buffer warm (per-Vec arena)."

## Strongest single disproof — the "u16→u8 second pass is gzippy's deficit" premise is FALSE; rg does it too

The brief's reconciliation (§35–36) and advisor-D attribute the 19.6ms output excess to gzippy's
"u16-ring → u8 second pass leaves the consumer source buffer cold/double-touched," implying rg
avoids it by decoding straight into a final u8 buffer. **Source refutes this.** rapidgzip carries
the identical two-stage marker machinery and runs the SAME u16→u8 narrowing pass:

- `vendor/.../rapidgzip/DecodedData.hpp:164,306` — `applyWindow()` "Replaces all 16-bit wide
  marker symbols by looking up the referenced 8-bit symbols in window."
- `ChunkData.hpp:84-85,247` — "Elements can be 16-bit wide markers … applyWindow is called during
  the second decompression stage." The in-source stats show 34.9–95.5% *replaced-marker* buffers.
- This matches CLAUDE.md's own traced-rg figure (31.25% replaced markers, 0.113s apply-window).

So rg pays a u16→u8 second pass and still lands rg_null 0.1148 and 15.7ms output. The EXISTENCE
of the second pass therefore cannot be the 19.6ms gzippy-specific excess. The real differentiator
is **page warmth**: rg's `FasterVector<RpmallocAllocator>` (FasterVector.hpp:120-128) keeps the
narrowed buffer's pages hot in a per-thread arena, so applyWindow's output and the subsequent
write() hit warm pages; gzippy's `std::alloc` munmaps large-Vec frees and re-faults on next alloc
(`chunk_buffer_pool.rs:72-81`: 40% page-fault CPU vs rg's 17%).

This is corroborated by the **/dev/null asymmetry** the brief never accounts for: Linux
`write_null` returns immediately WITHOUT `copy_from_user`, so a writev to /dev/null does not read
the source iovecs. Hence `file − null` bundles the source READ. For gzippy with cold post-narrow
pages that read is dear; for rg's warm pages it's cheap. So the "19.6ms output excess" is in large
part the SAME cold-page tax that sits in the 17.6ms /dev/null residual — measured through two
different windows. **Both binders are the arena/page-warmth deficit wearing two masks.** The
additive partition (19.6+17.6) is arithmetically valid but causally NOT two independent levers.
(Note: this is *not* a double-count — the two terms are measured at different levels, file−null
delta vs null-level, and sum correctly — but it does mean fixing warmth moves both.)

Net effect on the brief: C4 is *more* right than the brief argues (it's the unifying cause of
both binders), but C4's recommended fix must change. "Decode-and-narrow into the final u8 buffer
to eliminate the second pass" would make gzippy UNLIKE rapidgzip (which keeps the pass) — a
forbidden divergence under the bias guardrails. The faithful fix is to mirror rg's arena, not to
delete a pass rg also runs.

## Is the overlap lever even faithful? No — secondary reason to drop it

`ParallelGzipReader.hpp:~512-545`: rg's `writeFunctor` calls `writeAll(...)` **inline,
synchronously, inside the consumer `read()` loop** — there is no background writer thread. gzippy's
`output_writer.rs` OverlapWriter (separate `gzippy-out-writer` thread + MPSC) is a DIVERGENCE from
rg's structure. Under the governing guardrails ([[feedback_bias_guardrails]]:
"a change that makes gzippy UNLIKE rapidgzip is forbidden even if it helps the wall"), the overlap
writer should not be the lever regardless of its ceiling. C2's lever is rejectable on faithfulness
alone, independent of the 0.81×/0.88× dispute.

## The ONE most wall-moving faithful next lever

**Wire rg's per-Vec arena: `Vec<T, RpmallocAlloc>` (via allocator-api2 polyfill) for `ChunkData`'s
u16 + u8 buffers only** — the faithful transliteration of `FasterVector<RpmallocAllocator>`
(`FasterVector.hpp:120-128`). This attacks the UNIFIED root: warm pages cut the 40%→17%
page-fault tax (the 17.6ms null-residual) AND warm the applyWindow output buffer so the inline
write reads hot pages (the 19.6ms output-excess) — both masks, one faithful change, no structural
divergence. Gate it on the matched comparator (gz_null vs rg_null, interleaved, sha-verified) per
C1 — that, not the file-sink wall, is the parity gate. Pre-register the falsifier: if gz_null does
not move toward rg_null, page warmth is not the binder and the residual must be re-localized before
any further arena work.

Caveats on the lever: per-Vec arena is unproven (the prewarm and global mimalloc/jemalloc tries
regressed; `chunk_buffer_pool.rs:82-95`), and a fresh CLI process gets a cold arena each run, so
some of rg's advantage may be daemon-lifetime that a one-shot binary can't recover — measure
before believing. But it is the only lever that is simultaneously (a) aimed at the confirmed root,
(b) faithful to vendor, and (c) capable of moving BOTH binders.
