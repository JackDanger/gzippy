Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
Confirmed: 1 file, 453 insertions, **0 deletions** — purely additive. I have verified every claim first-hand against the source. Verdict follows.

---

# DISPROOF ADVISOR VERDICT — commit c224aaad (Copy-free-to-final, Stage 1)

## Claim A — BYTE-EXACT TREE (additive, zero production callers): **UPHELD**

First-hand verification:
- `git show --shortstat`: **1 file changed, 453 insertions(+), 0 deletions**. No existing line was modified or removed — every hunk inserts a new `fn`/test between existing code.
- `grep` for both new symbols across `src/`: `decode_clean_into_contig` is called **only** at `marker_inflate.rs:3938` (inside the test `decode_contig_clean`); `emit_backref_contig` is called **only** at `marker_inflate.rs:2142` (inside `decode_clean_into_contig`).
- Both new fns carry the same `#[cfg(...)]` gate as the surrounding clean-path code; neither is referenced by `read`, `read_internal_compressed*`, `single_member`, or any router.

The production decode graph is untouched. Bytes are identical on both features and both archs by construction. **No disproof found.**

## Claim B — CONTIGUOUS BACK-REF CORRECTNESS: **UPHELD**

I reduced `emit_backref_ring_u8` (3317–3381) to the non-wrapping case and compared arm-by-arm with `emit_backref_contig` (3239–3287):

- **Arm reduction is exact.** For a non-wrapping buffer, `src_round_fits`/`dst_round_fits` and every `% U8_RING_SIZE` are no-ops, so the ring's `distance>=8 && fits` → word-copy, `else non-wrap` → exact byte copy, `else` → wrap-straddle collapses to exactly contig's `distance>=8 ? word : byte`. RLE and general-overlap arms map 1:1.
- **distance==1 RLE source.** `src = base+(*pos-distance) = base+*pos-1`; `let v = *src` reads `base[*pos-1]` — the byte just before the write head, identical to the ring's `ring8[(*pos-1)%RING]`. Correct.
- **Underflow of `*pos-distance`.** Caller enforces `distance==0 → Err` (2125) and `distance > *pos → Err` (2138) **before** the call, so `1 <= distance <= *pos` ⇒ `*pos-distance ∈ [0, *pos)`. No underflow.
- **Word-overshoot bound — proven sufficient, and this is the crux.** I traced the max advance per outer iteration. `lut_litlen_decode` packs `sym_count ∈ {1,2,3}`, and the ISA-L pair/triple packing **stops at any symbol ≥256** (comment 1381–1386, "a packed pair/triple cannot end in EOB", igzip_inflate.c:473-476) — so a multi-symbol slot is **all literals**; a back-ref appears **only** at `sym_count==1`. Therefore one iteration writes *either* ≤3 literals *or* one back-ref ≤258, never both. With `out_room = cap-266` and the top-of-loop guard `emitted < local_cap ≤ out_room-*pos_init`, the write head at any back-ref satisfies `*pos ≤ out_room-1 = cap-267`; the word path touches `[*pos, *pos+((258+7)&!7)=*pos+264)`, max end `= cap-3 < cap`. The literal-triple path ends at `≤ cap-264`. **The 266-byte reservation is provably sufficient with 3 bytes to spare.**

Caveat (forward-looking, not a defect): this proof *depends* on `sym_count ≤ 3` and "multi-symbol ⇒ all literals." If a future LUT change packed a length code into a multi-symbol slot, the headroom proof breaks. Worth a comment, but the current code is correct.

## Claim C — RANGE-CHECK EQUIVALENCE (`distance > *pos` ⟺ `distance > decoded_bytes+emitted`): **UPHELD**

`set_initial_window_impl` sets `decoded_bytes = decoded_bytes_at_block_start = initial_window.len()` (719–720); the test starts `*pos = window.len()`. Both `*pos` and `decoded_bytes+emitted` then advance by exactly the output byte count (literals +1, back-refs +length). So invariantly **`*pos = decoded_bytes + emitted`** (both `= window_len + cumulative_output`). The contig check `distance > *pos` (2138) is byte-exactly the ring careful-loop check `distance > self.decoded_bytes + emitted` (1958). The check is cumulative (not per-block) on both sides — `decoded_bytes_at_block_start` is not used by either, so a multi-block body preserves equivalence (a back-ref may legally reach across block boundaries into earlier contiguous output).

Edge cases:
- **Empty window**: `set_initial_window` returns early leaving `contains_marker_bytes = true` (688–694), which trips `decode_clean_into_contig`'s `debug_assert!(!contains_marker_bytes)` — i.e. the contig fn is correctly *inapplicable* to the marker phase, not silently wrong. Equivalence is not claimed there and not needed.
- **Stored block**: see Claim D — handled differently (the one substantive divergence), but does not break the *range-check* equivalence for Huffman bodies.

## Claim D — DIFFERENTIAL ADEQUACY: **UPHELD-WITH-CAVEATS**

What the 3 tests genuinely exercise (verified):
- **distance reaching exactly the window start**: test 1 places `window[..4096]` as the first payload bytes under `set_dictionary`, which drives a `distance == *pos` (==32768 at the first clean byte) back-ref resolving to `base[0]` — the off-by-one boundary of `distance <= *pos`. Good.
- **cross-call back-ref sources**: test 2 (`per_call=4096`, ~10 calls, dict back-refs of distance up to 20000) validates a back-ref in call N sourcing output written in call N-1 — the real resumable risk.
- **RLE + short-distance + word-copy arms**: test 3 (distance==1 run of 5000, distance==2 overlap) plus the word arm in tests 1–2.
- `set_dictionary` **is** a faithful forcing function: it primes the encoder's window so emitted distances reach into the dict, and the decode side seeds the identical 32 KiB as both `set_initial_window` (ring) and the `[0,window_len)` prefix (contig). Faithful.

Owed-for-Stage-2 gaps (name them explicitly so they're not silently assumed retired):
1. **Stored/uncompressed clean block is NOT handled** — `decode_clean_into_contig` returns `Err(InvalidCompression)` for `Uncompressed` (2046), whereas the ring `read()` *does* decode stored blocks in clean mode (`read_internal_uncompressed`, clean branch 1172/1195). Stage-2 wiring must route stored blocks or extend the fn. (Not unsafe — a loud early Err — but a real functional gap.)
2. **Multi-deflate-block clean phase is UNTESTED.** Both test loops call `read_header` exactly once; `read()` decodes a single block to EOB and does not auto-advance headers. So the tested payloads are single-block, and the contig path's behavior across a block boundary (caller must detect EOB, call `read_header`, continue with persisted `*pos`/`decoded_bytes`) is unexercised.
3. **The out_room cap-saturation + actual buffer regrow is UNTESTED.** Both tests size `cap = window+payload+512`, so the binding cap is always `n_max`/`per_call`, never `out_room - *pos`; the test comment itself says "without a regrow." The "grow-between-calls" contract the commit names as the hardest landed risk is present in the *math* but not driven by any test (no call actually saturates out_room and regrows `base`/`cap`).
4. "Per-call split landing mid-back-ref" is a **non-case** by construction (the cap is checked at outer-iteration granularity; a ≤258 back-ref always completes within the 266-byte headroom) — nothing to test there.

## Claim E — SCOPE HONESTY: **UPHELD** (correct checkpoint; nothing to revert)

The change banks no wall (unwired) and that is honestly stated. Per the charter it is legitimate de-risking, not busywork: it lands a real, tested, byte-exact primitive (addressing retarget + contiguous back-ref + cap arithmetic) and validates it via the advisor-required ring-vs-**production**-decode differential (the test drives `read()`, i.e. the real production loop, so the differential is stronger than "vs the careful-loop reference"). The genuinely dangerous wiring (data_prefix_len activation, CRC-prefix exclusion, decode_bypass serialization) is correctly deferred. **I found nothing WRONG that must be reverted** — the two divergences from `read()` (stored-block Err; no internal header advance) are honest, loud incompleteness scoped to clean Huffman bodies, not silent miscompiles. The only caller is the test; the release-compiled-out `debug_assert!(*pos <= out_room)` cap contract has no production caller to violate yet.

---

### Summary
| Claim | Verdict |
|---|---|
| A — additive, no prod caller | **UPHELD** |
| B — contig back-ref byte-equiv + bound | **UPHELD** (proof depends on `sym_count≤3`/multi-sym-all-literals) |
| C — range-check equivalence | **UPHELD** |
| D — differential adequacy | **UPHELD-WITH-CAVEATS** (owed: stored block, multi-block header advance, out_room/regrow) |
| E — scope honesty | **UPHELD**, nothing to revert |

**Recommendation:** Land/keep as Stage 1. Before Stage 2 flips the default and deletes the drain, the three owed cases (stored clean block routing, cross-block `read_header` advance, and an actual out_room-saturating regrow in the differential) must be added to the test or explicitly routed — and the byte-exactness landmine the commit already names (CRC-prefix exclusion + decode_bypass round-trip) re-gated. Add a one-line assertion/comment tying the 266-byte headroom proof to `MAX sym_count = 3` so a future LUT change can't silently invalidate it.

=== ADVISOR EXIT 0 ===
