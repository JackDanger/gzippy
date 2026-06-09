# DIS-24 high-T gate — independent disproof advisor (read-only, source-verified)

Scope: gate the conclusion-reversing high-T result (gzippy-isal LOSES T9..T32) and
scrutinize the named binder ("gz over-partitions / chunk count scales with T while
rg holds ~17 constant"). Source-first, no edits.

## Bottom line

- **High-T LOSS is BANKABLE.** gz-isal loses every cell T9..T32 (best T9-E 0.938,
  trough T12-14 0.77) under the binding ≥0.99x bar. Measured frozen/quiet,
  interleaved N=9, sha-OK, path=ParallelSM asserted, rg -P matched. The S-floor
  "gz keeps winning as T→∞" story INVERTS past T8. Bank it.
- **The named BINDER (chunk-count growth as the cause) is REFUTED.** Two independent
  disproofs: (1) the owner's OWN per-cell table has cells with IDENTICAL chunk
  counts but materially different ratios that track TOPOLOGY, not count; (2) the
  "rg holds ~17 constant" premise is UNMEASURED above T8 and SOURCE-CONTRADICTED —
  rg uses the byte-identical parallelization-scaled shrink formula, so rg's chunk
  count grows with -P exactly as gz's does.
- **Constant-chunk-count is NOT the faithful lever and would not be expected to
  close the loss.** It makes gz UNLIKE rg (vendor scales chunks with -P), and
  equal-chunk cells already differ by topology, so capping count cannot be the
  primary lever.

## Per-claim

### Q1 — BINDER ATTRIBUTION: gz side SOUND, rg side REFUTED, causal claim REFUTED

**gz over-partitions as T grows: SOURCE-TRUE.**
- `src/decompress/parallel/single_member.rs:205` — production path calls
  `adjusted_chunk_size_bytes(gzip_data.len(), num_threads, default_chunk)`. (The
  `#[allow(dead_code)]` comments on the fn at :72 are stale/misleading — it IS the
  production chunk sizer, invoked unconditionally under `cfg(parallel_sm)`.)
- `single_member.rs:72-88` — the shrink: when `default*2*threads <= file_size`
  return default, else `chunk_size = max(512Ki, ceilDiv(ceilDiv(file_size,
  3*threads), 512Ki)*512Ki)`. So above the threshold `file_size/(2*default)`,
  chunk_size falls ~1/T and chunk count rises ~3T.
- `chunk_fetcher.rs:516` — `split_chunk_size` is the block-finder spacing, so chunk
  count ≈ file_size/split_chunk_size. gz chunk count growing 14→34 with T is
  source-true and faithfully expected.

**rg holds chunk count constant: REFUTED (source) / UNMEASURED (this sweep).**
- Vendor `ParallelGzipReader.hpp:294-306` is the EXACT same formula gzippy ports —
  `m_chunkSizeInBytes * 2U * parallelization > fileSize` ⇒
  `max(512_Ki, ceilDiv(ceilDiv(fileSize, 3U*parallelization), 512_Ki)*512_Ki)`.
  rapidgzip's chunk count is parallelization-scaled IDENTICALLY. Same `fileSize`
  (compressed input), same `-P`.
- Harness `_hicurve_guest.sh:167`: rg is run `-P "$P"` (same T as gz) but ONLY
  `rsec`/`rsha` are captured — **rg's chunk count is never measured in DIS-24.**
- The "rg ~17 constant" is imported from `amdahl-verdict-gate.md:26-27`, which
  measured rg `--verbose` Total Fetched = 17 at **T2=T4=T8 only** — every point
  ≤ T8. At ~68 MB compressed silesia, the shrink threshold is P > fileSize/(2*4Mi)
  ≈ 8.5, so 17-constant below T9 is exactly "shrink never fires," NOT "rg caps
  chunks." Above T8 the formula fires for rg too. The "constant" was illegitimately
  extrapolated into the one regime where source says it grows.

**Causal claim (chunk-count growth CAUSES the high-T loss): REFUTED by own data.**
The DIS-24 table has cells with EQUAL chunk counts and EQUAL machinery counters but
different ratios:
- T9-E vs T9-SMT: both 19 chunks / fb=1 / flip=18 → ratios 0.938 vs 0.890.
- T16-Ephys vs T16-SMT: both 28 chunks / fb=1 / flip=25 → ratios 0.861 vs 0.912.
If chunk count were the binder, equal count ⇒ equal ratio. It does not. The
differentiator at fixed chunk count is TOPOLOGY (E-core vs SMT placement). Chunk
count is a CORRELATE of T, not the demonstrated cause.

### Q2 — TOPOLOGY CONTROL: E-core handling, not chunk count, separates the cells

T16-SMT (0.912, 8P×2 SMT) BEATS T16-Ephys (0.861, 8P+8E) at IDENTICAL chunk
count (28). So spreading onto E-cores hurts gz MORE than SMT oversubscription —
at equal machinery. But rg on the SAME E-cores, same -P (so same-or-more chunks),
stays FLAT 336-378ms. Therefore E-cores are not intrinsically the killer; gz's
HANDLING of heterogeneous cores is. Disentanglement: the high-T loss is
**(b)+(c)** — gz pays more per chunk AND schedules heterogeneous cores worse than
rg — NOT (a) "gz partitions more than rg." rg likely stays flat via its prefetch
cache + work distribution (ChunkFetcher) absorbing slow-core stragglers and its
cheaper per-chunk window-publish/marker path; gz's serial publish-chain + per-chunk
flip_to_clean + fallback (DIS-18/DIS-14 mechanisms) is what rises.

### Q3 — THE FIX: constant-chunk-count is NOT faithful and not the lever — FIX-NEEDED

- **Not faithful.** Vendor scales chunk count with parallelization
  (`ParallelGzipReader.hpp:295,305`). Capping gz to a constant ~17 makes gz
  STRUCTURALLY UNLIKE rg — a divergence forbidden by the faithful-port mandate
  and by the bias guardrail. rg does not floor chunk count; it floors chunk SIZE
  at 512Ki and lets count grow.
- **Would not close the loss.** Equal-chunk cells already lose by topology, so
  reducing count cannot be the primary lever. Capping below core count would idle
  cores (the very thing rg tolerates by NOT capping). At best it trims per-chunk
  overhead while leaving the E-core/scheduling wall — i.e. it would expose the
  next binder (rule 3).
- The real lever set: cut per-chunk machinery cost (publish-chain serialization,
  flip_to_clean speculative marker decode, fallback re-decode) and improve
  heterogeneous-core scheduling toward rg's prefetch/work-distribution.

### Q4 — RECONCILE: yes, two true regimes, but the high-T binder must be renamed

- engine-W (DIS-18) stands ONLY in its banked box (silesia × T3-T8 × the
  shrink-OFF, ≈fixed-chunk regime). Above T8 W is T-dependent (fb 0→2, flip 12→31)
  and the fit cannot be extrapolated — the owner correctly flagged this for gz.
- The campaign verdict should read: **engine-W at low-mid T + a TOPOLOGY /
  per-chunk-machinery binder at high T.** NOT "chunk-count over-partition." The
  high-T LOSS is real and bankable; the high-T CAUSE labeled "chunk count" is not.

### Q5 — SCOPE: silesia-only, and silesia is rg's home field — FIX-NEEDED

Vendor comment (`ParallelGzipReader.hpp:300`): the formula is "mostly optimized for
silesia.tar.bz2." silesia is rg's tuning corpus. flip_to_clean rate, block
boundaries, and shrink threshold all move with compressibility, so the T9-T32
picture is silesia-specific. Re-run the curve on squishy variety before generalizing
"gz loses 16+ threads."

## Highest-value next move (free, and it directly kills or confirms the named binder)

1. **Measure rg's per-T chunk count** (rapidgzip `--verbose` "Total Fetched") across
   T9..T32 under the same -P, in the SAME sweep. Prediction from source: rg grows
   ≈3*P too. If rg's count grows while its wall stays flat, the chunk-count binder
   is dead and the verdict is locked to topology/per-chunk-machinery.
2. Then attack the demonstrated differentiator: at FIXED chunk count, why does
   E-core placement cost gz 0.05-0.07 ratio that rg eats for free? Compare gz's
   chunk dispatch/publish-chain to rg's ChunkFetcher prefetch + work distribution
   on the heterogeneous mask.

## Disproof attempts run against THIS verdict
- Tried: "maybe gz's growth IS the cause because rg really is constant." Broken by
  vendor `ParallelGzipReader.hpp:295` (identical scaled formula) + harness never
  measuring rg count above T8 + own equal-count/different-ratio cells.
- Tried: "maybe E-cores are intrinsically the killer." Broken by rg staying flat on
  the same E-cores at the same -P.
- Tried: "maybe constant-chunk would still help." Not refuted as a minor trim, but
  refuted as the primary lever (equal-count cells already lose) and as faithful.
