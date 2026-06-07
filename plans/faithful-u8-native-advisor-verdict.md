# Independent DISPROOF verdict — faithful u8-direct flip-in-place (fc1c965b)

Advisor: independent Opus, adversarial. Method: read `git show fc1c965b` + current
source first-hand (no build, per instructions). I attacked each claim; where I could
not break it I say so, where I found a hole I name it.

TL;DR — **A: UPHELD. B: UPHELD-WITH-CAVEATS. C: UPHELD (TIE sound; mechanism inference
valid; one cross-harness comparison is not load-bearing).** The faithful-u8 *mandate*
(structural, "u8 wherever rapidgzip uses u8") IS satisfied even though the wall is a TIE.
No correctness hole found that produces wrong bytes on a legal input. The two caveats are
about *what the tests prove*, not about a live bug.

---

## CLAIM A — BYTE-EXACT → **UPHELD**

### What I verified in the source

**The marker (`<true>`) path is behaviorally byte-identical.** The diff only wraps the
existing stores in `if CONTAINS_MARKERS { …old… } else { …u8… }` (marker_inflate.rs
:1493-1500 literal store, :1585-1598 back-ref). For `CONTAINS_MARKERS=true`:
`ring_modulus = RING_SIZE` (:1394-1398, same as the old hard-coded `RING_SIZE`),
literal store `ring_ptr.add(pos % RING_SIZE).write(code & 0xFF)` (unchanged),
back-ref `emit_backref_ring::<true>` (unchanged), drain `push_slice` of u16 (:758-766,
unchanged). `init_marker_zone` / marker store helpers are NOT in the diff. Const-folding
removes the `if` per instantiation. ⇒ pre-flip output is bit-for-bit the baseline.

**The flip guard guarantees a full clean 32 KiB window.** Both arms of `just_flipped`
(:1116-1119) require `distance_to_last_marker_byte >= MAX_WINDOW_SIZE`: arm 1 is
`>= RING_SIZE` (65536 ⇒ trivially ≥ 32768), arm 2 is `>= MAX_WINDOW_SIZE && ==
decoded_bytes` (no marker EVER ⇒ all bytes clean). So the window `[ring_pos-32768,
ring_pos)` repacked by `flip_repack_to_u8` is always wholly clean, and
`decoded_bytes >= MAX_WINDOW_SIZE` always holds (the `debug_assert` at :flip is real).
This is the SAME invariant the *baseline* relied on for its post-flip `drain_to_output`
clean branch (which carried the identical `(v as usize) < 256` assert) — so this is **no
new release-mode-corruption risk**; it is the pre-existing one, and the production
silesia sha (window-absent chunks DO flip — 16/18 chunks per the leader's own
GZIPPY_VERBOSE count) empirically validates it.

**`drain_transition_narrow_u16` only ever sees clean bytes.** `[ring_drained, ring_pos)`
at the flip read = exactly this read's output (the previous drain set
`ring_drained=ring_pos`). I checked the cross-product: a marker emitted *in this read*
forces `distance_to_last_marker < decoded_bytes` and `< RING_SIZE` (this read is
≤ 65278 bytes), so neither flip arm fires. The flip can only fire when the last marker is
≥ 32768 (arm 2: never; arm 1: ≥ 65536 ⇒ older than this whole read) ⇒ this read's bytes
are post-last-marker ⇒ clean. The narrow is safe; the `< 256` assert is provable, not
hopeful.

**The repack index transform is correct (attack target).** `flip_repack_to_u8`
(:flip) reads u16 logical `[p-32768+k]` via `(p + RING_SIZE - MAX_WINDOW_SIZE + k) %
RING_SIZE` into a **scratch** `tmp` (mandatory — u8 dest `[98304,131072)` physically
aliases u16 source slots `[49152,65536)`; reading all-then-writing avoids mid-loop
corruption, mirrors vendor `conflatedBuffer`), value-downcasts `(v & 0xFF) as u8`
(lossless because guard ⇒ v<256; **not** a LE reinterpret), writes to u8 slot
`U8_RING_SIZE-MAX_WINDOW_SIZE+k = 98304+k`, then rebases `ring_pos=ring_drained=
U8_RING_SIZE`. I checked the consistency algebraically: post-flip a distance-`d` back-ref
(`d∈1..=32768`) reads u8 slot `(U8_RING_SIZE - d)`, which = `98304 + (32768-d)` = the
repacked byte logically `d` before the seam. **Matches the baseline u16 semantics
exactly** (baseline read `(p-d)%RING_SIZE`, the byte `d` before the seam). d=32768 ⇒ slot
98304 (oldest); d=1 ⇒ slot 131071 (newest). Correct.

**`emit_backref_ring_u8` (attack target — overshoot / aliasing / distance≥8).** I
attempted to break the 8-byte word path and could not:
- **No OOB.** Word path is gated on `src_round_fits && dst_round_fits`
  (`phys + rounded ≤ U8_RING_SIZE`); otherwise it falls to a non-wrap exact copy or the
  per-byte wrap-straddle modular fallback. The ≤7-byte overshoot is bounded inside
  `U8_RING_SIZE`.
- **No live-source corruption from the overshoot.** The overshoot lands at
  `[dst+length, dst+rounded)` — strictly *ahead* of the advanced `*pos`, i.e. future
  write positions, which are never a back-ref source (sources are always `< pos`). They
  get overwritten before they can be read. Confirmed.
- **`distance >= 8` (BYTES) is the correct invariant.** I initially suspected the forward
  word loop corrupts overlapping `length>distance` copies, then re-derived: src and dst
  *both* advance by 8 and `src = dst - distance`, so each word's read region
  `[dst+8j-distance, dst+8j-distance+8)` ENDS at `(dst+8j) - (distance-8) ≤` the current
  write start when `distance ≥ 8`. Reads only touch already-finalized bytes. This is the
  standard libdeflate fastloop invariant and equals the u16 path's `distance >= 4` (u16)
  = 8 bytes. No corruption.
- **RLE (`distance==1`) and general-overlap arms** are byte-for-byte the u16 baseline's
  arms (compared at marker_inflate.rs :2583-2618) at u8 width. Output is correct LZ77.

**Stored block after flip (`read_internal_uncompressed`, clean mode).** Writes
`ring8.add(pos % U8_RING_SIZE)` per byte (:1184-1190), bounded ≤65535 < U8_RING_SIZE, no
back-refs (no source-overwrite hazard), drained modularly. Wrap-safe.

**`set_initial_window_impl` seeds the u8 view, self-consistently** (:708-729): u8 seed at
slots `[0,len)`, `ring_pos=len` (u8-logical), `contains_marker_bytes=false`. A first
back-ref `d ≤ len` reads `(len + U8_RING_SIZE - d) % U8_RING_SIZE = len-d ∈ [0,len)` —
correct, no underflow (the `+U8_RING_SIZE` guards it). **Reachability: this Block path is
test-only.** Production window-SEEDED chunks decode via `ResumableInflate2`
(gzip_chunk.rs:598-629); production window-ABSENT chunks seed with `&[]` (empty ⇒ early
return :692-694, stays marker mode, then flips). So the u8-seed change is correctness-
neutral in production and self-consistent where it is exercised (the `set_initial_window_*`
unit tests, which pass).

### Why the dual-feature sha is strong (and what it actually covers)
- **gzippy-native (arm64)**: `isal_clean_tail` OFF (build.rs:101 — needs x86_64 + gzippy-isal).
  The FOLD runs (gzip_chunk.rs:1205-1219): Engine M keeps decoding the clean tail in-place
  ⇒ `emit_backref_ring_u8` + the u8 drain are FULLY exercised over the whole post-flip tail.
- **gzippy-isal (x86_64 Rosetta)**: `isal_clean_tail` ON ⇒ the loop hands off to Engine C
  after ≥32 KiB clean. But the hand-off threshold is `clean_appended_len() >= MAX_WINDOW_SIZE`,
  and `clean_appended_len` only advances *after* Block's internal flip (only the clean
  drain / transition drain push u8). So Block's u8 path still runs for ~32 KiB before the
  hand-off — the u8 store/back-ref/repack ARE exercised on this feature too.

Two independent builds, two partly-different exercises of the new code, **same sha
028bd00…cb410f**. Combined with the structural mirror analysis above, I cannot construct a
legal input that diverges. **UPHELD.** (Honest residual: byte-exactness is proven
*empirically + by structural mirror*, not by exhaustive proof; but the silesia corpus
flips on 16/18 chunks and the seam test adds the adversarial case, so coverage of the new
code is real, not incidental.)

---

## CLAIM B — THE SEAM TEST → **UPHELD-WITH-CAVEATS**

**Routing is production, not a shortcut.** The test calls `decode_chunk_window_absent`
(gzip_chunk.rs:1694) → `decode_chunk_with_rapidgzip_impl(initial_window=&[])` →
(window absent) `decode_chunk_unified_marker` → `marker_decode_step_vendor_block` → Engine
M `Block::read`. Under the arm64 native test build (`isal_clean_tail` OFF) this is exactly
the FOLD path that runs `flip_repack_to_u8` + `emit_backref_ring_u8`. ✓.

**The repack + max-distance u8 back-ref IS validated.** The whole-stream
`assert_eq!(out, oracle)` against an independent flate2 inflate covers the post-flip region:
back-refs in `[flip_point, flip_point+32768)` read the value-downcasted repacked window, and
the 3rd A + the distance-32768 refs are well past the flip. A wrong dest offset, missing
rotation, or LE-reinterpret-instead-of-downcast would corrupt those bytes and the array
compare would fail. So rotation/downcast/dest-offset are genuinely under test. ✓.

**CAVEAT 1 (the named sentinel is not a guaranteed pinpoint).** The test's headline
assertion `out[32*1024] == 0xA5` is advertised as "the distance-32768 back-ref read the
correct u8 slot." But the flip does NOT necessarily fire at byte 32768: `read()` caps a
single marker-mode call at `RING_SIZE - MAX_RUN_LENGTH = 65278` bytes, and the flip is
checked only *after* the call returns. With flate2 level-6 block structure, the flip most
likely fires at ~65278 (or a block boundary), so `out[32768]` is plausibly decoded in
MARKER mode (`emit_backref_ring::<true>`), i.e. the sentinel line may be testing the
baseline path, not the u8 repack. This does not weaken correctness — the *array* compare
still covers the u8 seam — but the test comment over-claims which byte proves the repack.
Recommend (non-blocking): assert a byte known to be ≥ `flip_point` (e.g. read the actual
flip offset from a counter, or place a sentinel at byte 70000) to make the u8-path
coverage deterministic rather than block-layout-dependent.

**CAVEAT 2 (off-by-one at d=32768 vs d=32769).** The distance==32768 boundary is exercised
(payload is A‖A‖A with |A|=32768). d=32769 (out-of-window) is *rejected upstream* by
`distance > MAX_WINDOW_SIZE` (:1575) before reaching `emit_backref_ring_u8`, so the u8 code
never sees it — correct, but the test does not explicitly construct a 32769 case (it
can't; the encoder won't emit it). The in-window boundary (32768) is covered; the
out-of-window rejection is covered by other tests/the distance guard. Acceptable.

Net: the seam's real risk surface (repack rotation, value-downcast, max-distance u8
back-ref) IS under an independent-oracle test on the production path. The sentinel line is
softer than advertised. **UPHELD-WITH-CAVEATS.**

---

## CLAIM C — THE WALL (TIE) → **UPHELD** (interpretation sound; one comparison not load-bearing)

I cannot reproduce the guest numbers (no build / no guest, per instructions), so I assess
methodology and internal consistency.

**The TIE band reasoning is sound.** u8 vs base: T8 1.004x, T1 0.976x. Reported spreads:
base 7–31%, u8 4–14%. 0.976–1.004x (±2.4%) sits *well inside* even the tighter 4% arm of
the u8 spread. Per the charter's own rule "Δ < inter-run spread ⇒ TIE, full stop," this is
a TIE, not a hidden regression. I see no statistical sleight: best-of-N relative with the
band = max(spread) is the project's standard and is being applied honestly (the leader
reports the *less* favorable T1=0.976 rather than hiding it).

**The mechanism inference is a VALID causal perturbation, not an attribution.** This is the
strongest part of Claim C and it satisfies the project's own measurement law. The u8
rewrite ~halves the clean-path memory traffic (u16→u8 ring + u8 back-ref copies + no
narrow-at-drain). That is a ~2× perturbation of one resource (memory traffic) with a
**flat wall response** ⇒ traffic is SLACK, not the binding resource ⇒ the binding term is
per-symbol LUT-decode compute. This is exactly "perturb, don't attribute" done right, and
it *falsifies* the round-2 hypothesis ("the u8 clean ring is the main lever",
orchestrator-status.md round-2) — an honest negative result, consistent with the round-2
PLATEAU verdict that already pegged the engine at ~2.4× ISA-L on *compute*. The new cost
the u8 path adds is minimal and const-folded (the `<false>` instantiation has NO
`CONTAINS_MARKERS` branch in the hot loop; the transition drain is one-shot at the seam),
so the perturbation is reasonably clean — it is not the case that a traffic win is being
masked by a new branch tax in steady state.

**The constant ~1.70x at BOTH T1 and T8 supports "decode binds per-thread, not
scheduling."** A scheduling/placement gap would widen with T; a flat ratio across T1→T8 is
the signature of a per-thread throughput gap. Consistent with
[[project_pregate_placement_is_dominant_lever]] naming engine + placement co-primary: this
result isolates the *engine* (per-thread decode) component as a real, placement-independent
~1.7× deficit.

**CAVEAT (not load-bearing): the 393 MB/s / 0.13–0.52s absolutes vs the charter's 0.604s
"same-sink wall" are NOT apples-to-apples** and should not be read as "u8 beat 0.604s."
The 0.604s is rapidgzip's same-sink figure from a different harness; the leader's numbers
are measure.sh (stdout→mktemp), a different sink/config. The TIE verdict is **relative**
(u8 vs base on the *same* measure.sh harness) and is valid on its own terms; the
cross-harness absolute comparison the charter asked for ("close the gap vs 0.604s") is
left genuinely open and should not be inferred from these relative numbers. The leader is
right to keep the verdict relative; the charter's 0.604s question is answered only as "u8
did not change the relative gap to rapidgzip."

**Steelman of the opposite (where would u8 show a delta?).** A workload dominated by
window-SEEDED chunks decodes via `ResumableInflate2` (already u8) — u8-rewriting Engine M
touches nothing there, so no delta expected (not a counterexample to the TIE). A very
low-redundancy / rare-block-boundary corpus maximizes window-ABSENT Engine-M share — but
the traffic-slack finding predicts still no wall delta there. Silesia is already 89%
window-absent (Engine M's worst case for traffic), so it is the *favorable* corpus for a
u8-traffic win, and it TIE'd. I therefore cannot construct a realistic config where the u8
traffic reduction would flip the wall — which strengthens, not weakens, the
compute-bound conclusion. **UPHELD.**

---

## Is the faithful-u8 MANDATE satisfied? — **YES (structurally), despite the perf TIE**

The user directive ([[project_faithful_unified_decoder_over_perf.md]],
[[project_engine_plateau_pure_rust]]) was explicitly: "u8 wherever rapidgzip uses u8, FULL
STOP," and the u16-keep + narrow-at-drain was named as *the shortcut/deviation* that had
to be removed. This commit:
- decodes the window-absent clean bulk **u8-direct** into the u8 view of the same backing
  store (vendor `getWindow()` reinterpret, deflate.hpp:890-894);
- flips the SAME buffer u16→u8 **width** in place at the seam via a faithful
  `setInitialWindow` rotate + **value-downcast** (deflate.hpp:1762-1782);
- emits back-refs u8 with no marker scan (vendor const-folds it for the u8 window);
- removes the narrow-at-drain.

That is the deviation the user ordered removed, removed. **The mandate is met.** The perf
TIE does NOT violate it — the mandate was about faithfulness, and the honest finding is
that faithfulness here buys structural correctness, not wall, because the engine is
compute-bound (LUT per-symbol), not traffic-bound.

## The REAL lever (per this result)
Memory traffic is slack; **per-symbol LUT-decode compute** is the binding term and the
remaining ~1.70× per-thread gap to rapidgzip. That points the next move at the inner
Huffman decode kernel (the igzip-class packed-table / speculative-store / preload pipeline
the asm-kernel-feasibility work scoped), NOT at any further ring/traffic/placement change.
The faithful-u8 step was a *necessary* structural correction and a *necessary* disproof of
the traffic hypothesis; it was correctly NOT expected to move the wall on its own, and it
didn't. Layer it and keep it (byte-exact, faithful) per
[[feedback_layer_dont_revert_whole_system]].

---

## Verdict summary
| Claim | Verdict | Load-bearing reason |
|---|---|---|
| A — byte-exact | **UPHELD** | marker path const-folded-identical; flip guard ⇒ clean 32 KiB window provable; repack rotation/downcast/dest algebraically matches baseline; emit_backref_ring_u8 OOB/aliasing/distance≥8 all sound; dual-feature sha 028bd00…cb410f on two partly-different exercises of the new code. No legal input diverges. |
| B — seam test | **UPHELD-WITH-CAVEATS** | routes through production fold path; whole-stream flate2 oracle validates repack + max-distance u8 back-ref. Caveat: the `out[32768]==0xA5` sentinel may land on the MARKER path (flip fires at ~65278, block-layout-dependent), so that named line is softer than advertised — the array compare, not the sentinel, is what proves the u8 seam. |
| C — wall TIE | **UPHELD** | Δ(2.4%) ≪ spread ⇒ TIE; the u8 rewrite is a clean ~2× traffic perturbation with a flat wall response ⇒ traffic slack, compute (LUT) binds — a valid causal disproof of the traffic hypothesis. Caveat: the 393 MB/s vs 0.604s cross-harness absolute is NOT apples-to-apples and is not load-bearing; the verdict is relative and stands. |

No live correctness defect found. Two test-coverage caveats (B sentinel pinpoint, B 32769
boundary) are non-blocking hardening suggestions. Mandate satisfied; next lever is the
inner Huffman compute kernel, not traffic/placement.
