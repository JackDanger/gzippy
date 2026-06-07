# Lever decision — resolve-ahead vs decode-rate @ HEAD 85ad00a

Read-only analysis. `cf` = `src/decompress/parallel/chunk_fetcher.rs`,
`vendor` = `vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`.
Every claim cites a `file:line` I actually opened.

---

## VERDICT — (b)

**Resolve-ahead reuse is already faithfully keyed and firing; it is at its
correctness-bounded ceiling pre-confirmation. No faithful corrected §3.1 exists —
the only re-key that would raise firing applies the WRONG predecessor window and
breaks CRC. The binding residual is LEVER B (decode rate, `d_w` 1.77×), which must
be confirmed by a `GZIPPY_SLOW_BOOTSTRAP` causal perturbation before any inner-loop
edit.**

---

## 1. The §3.1 correction does not exist faithfully — it is already TRIED and REJECTED in the tree

The proposed §3.1 fix (resolve-ahead-fix.md:136-149) threads the offset-sorted
*previous* element through the scan and keys the predecessor lookup on
`prev.encoded_offset_bits + prev.encoded_size_bits`. That exact experiment is
recorded as REJECTED in the production code, with the break mechanism:

> cf:2428-2435 — "Re-keying this lookup to the offset-sorted predecessor's published
> end (`prev.encoded_offset_bits + prev.encoded_size_bits`, the §3.1 experiment) was
> REJECTED: for range-speculative chunks that key names a DIFFERENT published window,
> and the dense window map usually contains it, so the chunk gets resolved against the
> wrong predecessor window → CRC32 mismatch (test_coalesce_fixed_huffman / silesia
> parallel-SM, green on the seed key, red on the predecessor key)."

This is exactly the §5 "`prev` = true predecessor assumption" caveat
(resolve-ahead-fix.md:194-199) realized as a *correctness* failure, not a missed
opportunity: the caveat assumed a wrong key would simply fail `window_map.get` and be
skipped. It does not — the window map is dense, so the wrong key *hits a real-but-wrong
window*, `apply_window` consumes the chunk's markers against it, and the bytes are
wrong. The consumer's reuse guard `consumer_pred_key == Some(eager_pred_key)` (cf:1612)
would reject the mismatch, but only AFTER the in-place pool `apply_window` already
mutated the cached `Arc` — the damage is done. So §3.1 cannot be made safe by the
pred-key guard.

## 2. HEAD's resolve-ahead is ALREADY the faithful key, and IS firing

HEAD keys the resolve-ahead predecessor lookup on the chunk's OWN decode-start:

- `chunk_consumer_handoff_bit(chunk) = chunk.max_acceptable_start_bit` (cf:2335-2337),
  used as the lookup key at cf:2436 and the eligibility gate at cf:2348
  (`chunk_may_resolve_markers_early`).
- `max_acceptable_start_bit = decode_start` — the bit the worker actually decoded
  byte 0 from (cf:3177, set in `try_speculative_decode_candidate`). Because a
  window-absent decode can only start on a real deflate block header (seed-first at
  cf:3209-3216, else the boundary-search candidate at cf:3256), **this is always a
  REAL block boundary**, never a raw guess.

This mirrors vendor `queuePrefetchedChunkPostProcessing` line-for-line: vendor keys on
`m_windowMap->get(chunkData->encodedOffsetInBits)` (vendor:544), where vendor's
`encodedOffsetInBits` is the chunk's own real decoded start. HEAD looking up the
chunk's own real decode-start is the SAME faithful pattern — and is **more faithful
than §3.1** (which invents a threaded-prev key vendor never uses).

The chain even propagates within a single sorted pass exactly as vendor does:
`queue_prefetched_marker_postprocess` scans the whole cache sorted (cf:2408,
`prefetch_cache_contents_sorted`) and publishes each chunk's tail window
*synchronously before* submitting the pool `apply_window` (cf:2453 →
`publish_end_window_before_post_process`, then cf:2454 submit) — vendor:557-575 publish,
vendor:579-582 pool submit. So when predecessor P (lower cache key) is processed it
publishes its real-end window, and successor N (higher key) can pick it up later in the
same pass — *iff* `N.max_acceptable_start_bit == P.encoded_offset_bits +
P.encoded_size_bits`.

Tiling makes that equality the COMMON case, not the rare one: P stops at the first real
boundary ≥ partition[k+1] (`stop_hint_bit_for`, cf:1831-1843) and N starts at the first
real boundary ≥ its seed = partition[k+1] — the **same** boundary. So resolve-ahead is
correctly scoped and DOES fire: captured `EAGER_PROBE_SUBMITTED ≈ 30 / 77`,
`EAGER_PROBE_REUSED ≈ 12` (structural-gap-analysis.md artifact + cf:2469/1614). It is
not a dead lever; it is a *partially-effective, correctly-bounded* one.

## 3. Why reuse can't go higher faithfully — the ceiling is pre-confirmation publish timing, set by decode rate

Reuse needs P's tail window **published before the in-order consumer reaches N**. The
captured causal number is the ceiling: only **1/37 ≈ 0** chunks had their predecessor
window published before the chunk's own decode start (structural-gap-analysis.md §0/§3;
task brief). The keys CAN align (§2); the *windows aren't ready in time*.

The publish-ahead chain anchors on a consumer-confirmed window and walks outward; each
link `get_last_window_vec(predecessor_window)` (cf:2385) needs the predecessor's
resolved data, whose heavy `apply_window` runs on the pool (cf:2454, off the consumer
wall — `apply_window` wall-critical = 0 ms, structural-gap-analysis.md §2 row 7). The
chain reaches the consumer ahead-of-time only if the pool finishes those resolutions
before the in-order cursor arrives. gzippy's pool decode is **1.77× slower**
(`d_w` 125.5 ms vs rapidgzip 70.95 ms, structural-gap-analysis.md §3.1 Lever B), at the
**same** prefetch depth (2·T = 16) and the **same** full-cache scan breadth (cf:2408 ↔
vendor:524-529). A 1.77×-slower producer at matched depth falls behind the in-order
consumer → windows publish late → reuse caps low and `consumer.dispatch_recv` stays
~278 ms wall-critical (structural-gap-analysis.md §1.4).

**Why vendor escapes:** NOT a different key (vendor:544 is the same own-offset key HEAD
uses) and NOT earlier structural confirmation of *which* key — it is that vendor's
faster pool decode + faster `apply_window` (per-link `L_resolve` median 0.065 ms vs
gzippy 4.89 ms, structural-gap-analysis.md §1.4) finishes the resolutions and PUBLISHES
the windows before the consumer arrives, so the SAME-shaped resolve-ahead chain runs
ahead. The 1/37-published-before-start is the *symptom* of a pool falling behind, not a
mis-keyed lookup. Therefore the residual is decode rate (Lever B).

## 4. Pre-registered falsifier (for verdict (b))

Capture with `scripts/run_locked_fulcrum.sh` at T8, silesia-large, production path
asserted (`GZIPPY_DEBUG=1 → path=IsalParallelSM`, `--no-default-features --features
pure-rust-inflate`, `GZIPPY_FORCE_PARALLEL_SM=1`), interleaved best-of-N≥7, sha-verified.

**(b) is FALSIFIED** (i.e. a structural resolve-ahead lever still exists) if ANY holds:
- `EAGER_PROBE_REUSED` rises materially (> ~20 of 77) from a change that touches ONLY
  post-process scheduling (no inner-loop edit) AND `consumer.dispatch_recv` wall-critical
  drops from ~278 ms toward rapidgzip's ~3 ms, with **window-absent ≥ ~90%** preserved
  (causal `window_present:false`; if it drifts toward 31% the change turned workers into
  clean-decoders — forbidden divergence, reject).
- wall drops toward rapidgzip's 0.535 s from a resolve-ahead/scheduling change alone.

**(b) is CONFIRMED** if, with resolve-ahead verified firing (`EAGER_PROBE_SUBMITTED > 0`,
`RESOLVE_AHEAD_OK > 0`, cf:2449), `consumer.dispatch_recv` and `L_resolve`-median do NOT
move from scheduling changes, AND a `GZIPPY_SLOW_BOOTSTRAP=N` perturbation of
`worker.block_body` moves the wall ~proportionally (survives a sleep-based
frequency-neutral control) — i.e. the consumer stall is decode-rate-bound.

**Mandatory gate before any inner-loop (Lever B) edit:** run the `GZIPPY_SLOW_BOOTSTRAP`
causal perturbation + frequency-neutral sleep control FIRST (CLAUDE.md Measurement
PROCESS 1-3). Do not edit `decode_huffman_body_resumable` / `block_body` on the strength
of the 1.77× attribution alone — attribution is analyst-biasable; only the perturbation
is the verdict.

## 5. Disproof attempt against my own conclusion

The strongest counter: 12/77 reuse and 1/37-published-before-start are from a capture
that may **predate or not reflect** HEAD `85ad00a`'s resolve-ahead key fix
(structural-gap-analysis.md §6 explicitly flags `handoff_key=0` in the report and "no
re-capture proving 85ad00a fired"). If HEAD now fires far more than the captured 30/12,
the ceiling I cite is stale and a scheduling lever might still pay — so (b) rests on a
number I could not re-measure read-only (resolve-ahead-fix.md §5 same caveat). Mitigant:
even granting perfect key alignment and a full-cache scan (both already present, §2), the
chain still cannot publish a predecessor window ahead of the consumer faster than the
pool resolves it, and the pool is 1.77× slower at matched depth — so the timing ceiling
is decode-rate-bound by construction regardless of the exact reuse count. The falsifier
in §4 (re-measure `EAGER_PROBE_REUSED` + `dispatch_recv` from a scheduling-only change)
is precisely the test that would overturn this if I am wrong.

INVESTIGATION COMPLETE
