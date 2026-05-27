# Unified pure-Rust deflate decoder

**Goal.** Replace the two-decoder split (Phase 1 bootstrap +
Phase 2 ResumableInflate2) with one parameterized decoder that handles
both modes — markers-aware (speculative, no window) and clean (window
known) — sharing one hot loop, one set of Huffman tables, one
resumable contract.

**Anchor commit.** `c09e8d4` (post tight-Huffman Wave 1 + T3 fix +
neurotic A/B at 0.91× of ISA-L). All work here builds on
`ResumableInflate2` because that's where the perf-tuning landed.

---

## 1. The two-decoder reality today

### Phase 1 — `bootstrap_with_deflate_block` (`src/decompress/parallel/gzip_chunk.rs:962`)

Used by parallel-SM workers when the predecessor chunk's window is
NOT yet known. The decoder:

- Operates on a hand-rolled state machine `deflate_block::Block`
  (`src/decompress/parallel/deflate_block.rs:139`, ~2,372 lines).
- Maintains `output_ring: Box<[u16; RING_SIZE]>` — every output
  position is a u16, where the low 8 bits are a literal and the high
  range encodes a back-reference marker. The ring is initialized so
  that back-refs landing in the marker zone produce
  self-identifying u16s; on resolution, a window XOR turns markers
  into real bytes.
- Uses Huffman tables `HuffmanCodingShortBitsMultiCached` (litlen) +
  `HuffmanCodingReversedBitsCached` (distance) +
  `rfc_tables::get_distance_dynamic`. These are *different
  structurally* from the libdeflate-style tables used by Phase 2.
- Tracks `distance_to_last_marker_byte` and
  `contains_marker_bytes`. Flips clean-mode when the counter
  reaches `MAX_WINDOW_SIZE = 32768` AND equals total decoded — i.e.,
  the entire trailing 32 KiB is markerless.
- Loops until: (a) 32 KiB clean run + block boundary
  (`clean_handoff_armed`), (b) BFINAL, (c) `stop_hint_bits` reached.
- Returns `DeflateBootstrap { markers: Vec<u16>, clean_window: Option<Vec<u8>>, end_bit_offset }`.

**Cost on real silesia (neurotic perf record, commit `c09e8d4`):**
*~83% of total cycles*, dominated by
`bootstrap_with_deflate_block_inner` self-time (8.5%) +
`huffman_short_bits_multi_cached::decode` (8.0%) +
`huffman_reversed_bits_cached::decode` (3.2%) +
`rfc_tables::get_distance_dynamic` (2.3%).

### Phase 2 — `ResumableInflate2` (`src/decompress/inflate/resumable.rs`)

Used when (a) predecessor window is known (skip Phase 1 entirely),
or (b) Phase 1 has accumulated 32 KiB of clean output and handed off.
The decoder:

- Operates on a clean-only state machine with `SlidingWindow: Box<[u8; 32768]>`
  (`resumable.rs:80`).
- Uses libdeflate-shape `LitLenTable` (TABLE_BITS=12) + `DistTable`
  (TABLE_BITS=9) (`libdeflate_entry.rs:329, :540`). Different from
  Phase 1's tables.
- Writes directly to caller's `&mut [u8]` slice (no intermediate
  ring).
- Resumable yield-on-output-fill via `BlockState` + `PendingMatch`.
- All B1–B6 + T0/T3/T4/T5 perf work landed here.

**Cost on real silesia:** `copy_match_windowed` 5.6% + the inner
`decode_huffman_body_resumable` is folded into surrounding noise
(<1% each symbol-shaped). Total Phase 2 share: ~10%.

### The cost asymmetry

> Bootstrap (Phase 1): ~83% of cycles, 2,372 lines of mostly
> un-tuned decoder code, three separate Huffman variants.
>
> ResumableInflate2 (Phase 2): ~10% of cycles, ~1,600 lines of
> aggressively-tuned hot loop, two libdeflate-shape tables.

This is the wrong distribution. We tuned the cold path.

### Why two decoders exist

Historical accident, not design intent. The bootstrap was the FIRST
pure-Rust decoder, ported from `vendor/rapidgzip/.../gzip/deflate.hpp`
literally. ResumableInflate2 was built later as a libdeflate-shape
replacement for the ISA-L FFI's Phase 2 role. They never got merged
because the bootstrap needs marker emission (a vendor-specific
trick) and ResumableInflate2 doesn't.

---

## 2. The C/C++ libraries we still rely on (and why)

The unification design must respect these, since none are going away
in this scope:

| Library | Used for | Stays as-is? |
|---|---|---|
| **libdeflate (libdeflater crate)** | Sequential single-member decode, BGZF parallel, multi-member parallel. The fast one-shot decoder. | YES. We don't have a one-shot decoder to replace it. |
| **flate2 / zlib-ng** | Streaming decode for >1 GiB single-member when ISA-L unavailable; compression L6-L9. | YES. Different role from inflate-in-parallel-SM. |
| **isal-rs (FFI to patched ISA-L)** | Current production parallel-SM Phase 2. Replaced by `ResumableInflate2` when `--features pure-rust-inflate` is on. | DELETE eventually (the plan's Tier 1 done-when). |
| **vendor/rapidgzip** | Reference for parallel-SM architecture port. Code shape we mirror. | Reference only; not linked. |

**What this design does NOT do:**
- Replace libdeflate one-shot. It's the right tool for non-parallel-SM
  paths.
- Replace ISA-L for compression. Out of scope; only inflate.
- Compete with rapidgzip's pipeline architecture (consumer drain,
  prefetcher, etc.). That's higher up the stack.

**What this design DOES do:**
- Replace Phase 1 (`Block::read`) AND Phase 2 (`ResumableInflate2`)
  with one parameterized decoder.
- Reuse all B1–B6 + T0/T3/T4/T5 perf work for both modes.
- Delete `Block` (~2,372 lines), `huffman_short_bits_multi_cached`,
  `huffman_reversed_bits_cached`, `rfc_tables` once parity confirmed.

---

## 3. The unified design

### Shape: one decoder, two output modes via const generic

```rust
/// Output type discriminator. `Markers` = u16 output ring (Phase 1
/// speculative); `Clean` = u8 caller slice (Phase 2 windowed).
pub enum OutputMode { Markers, Clean }

pub struct UnifiedInflate<'a> {
    bits: Bits<'a>,
    block_state: BlockState,
    pending_match: Option<PendingMatch>,
    // Reused from ResumableInflate2 — same Huffman tables for both
    // modes. The B1–B6 work applies to both.
    dynamic_tables: Option<(LitLenTable, DistTable)>,
    // Marker-mode state (zero-cost in Clean mode via const generic).
    window: SlidingWindow,  // empty when marker mode active
    contains_marker_bytes: bool,  // rapidgzip pattern
    distance_to_last_marker_byte: usize,
    // Resumable + stopping-point state (shared)
    last_bfinal: bool,
    last_btype: u8,
    points_to_stop_at: StoppingPoint,
    stopped_at: StoppingPoint,
    encoded_until_bits: usize,
    pending_stream_header_stop: bool,
}

impl<'a> UnifiedInflate<'a> {
    /// Clean mode: write u8 bytes to caller's slice. Same surface as
    /// ResumableInflate2 today.
    pub fn read_stream_clean(&mut self, output: &mut [u8]) -> Result<InflateStreamResult> {
        self.read_stream_inner::<{ OutputMode::Clean }>(/* output as u8 slice */)
    }

    /// Marker mode: write u16s (markers + literals) to caller's slice.
    /// Used by parallel-SM speculative bootstrap.
    pub fn read_stream_markers(&mut self, output: &mut [u16]) -> Result<InflateStreamResult> {
        self.read_stream_inner::<{ OutputMode::Markers }>(/* output as u16 slice */)
    }

    /// Per-call mode transition: when Markers mode accumulates 32 KiB
    /// clean output at a block boundary, the caller flips to
    /// `read_stream_clean` with a window seeded from the trailing
    /// 32 KiB. Internal state (Bits, BlockState, etc.) is preserved
    /// across the flip.
    pub fn set_window(&mut self, window: &[u8]) -> Result<()> { ... }
}

fn read_stream_inner<const MODE: OutputMode>(
    state: &mut UnifiedInflate,
    output: OutputSlice<MODE>,  // const-generic-discriminated u8/u16 slice
) -> Result<InflateStreamResult> {
    // Same FASTLOOP + SAFE LOOP shape as decode_huffman_body_resumable.
    // const-generic flag selects:
    //   - literal write: `*ptr.add(out_pos) = lit` (u8) vs
    //                    `*ptr.add(out_pos) = lit as u16` (u16, no markers)
    //   - match copy: `copy_match_fast`/`copy_match_windowed` (u8) vs
    //                 `emit_backref_ring` (u16, produces markers for
    //                  refs past output start)
    //   - clean-count tracking: noop (u8) vs distance_to_last_marker_byte
    //                           update (u16)
    ...
}
```

LLVM monomorphises into two function specializations. The const
generic ensures zero runtime overhead per branch:

- `read_stream_inner::<{ OutputMode::Clean }>` ≡ today's
  `decode_huffman_body_resumable` (modulo `set_window` plumbing).
- `read_stream_inner::<{ OutputMode::Markers }>` is the marker-aware
  variant — the bootstrap's hot loop but written in
  `ResumableInflate2` style (raw-pointer writes, FASTLOOP, T3
  multi-literal, B1–B6 locals).

### Mid-stream mode flip (at block boundary only)

The rapidgzip pattern flips `contains_marker_bytes` at block
boundaries, not mid-block. We honor that:

1. `read_stream_markers` runs.
2. At each block-finish, the caller checks `state.contains_marker_bytes`.
3. If FALSE (32 KiB clean accumulated AND no markers in trailing
   window), caller (a) extracts trailing 32 KiB from the u16 ring as
   u8 bytes, (b) seeds `state.window` via `set_window`, (c) switches
   to `read_stream_clean` for the rest of the chunk.
4. The decoder's `Bits`, `BlockState`, `dynamic_tables` carry across
   unchanged. Only the OUTPUT mode and the window-seed differ.

This matches the existing two-decoder handoff but eliminates the
intermediate `Vec<u16>` materialization + decoder reconstruction.

### Sharing the Wave 1 + T-series wins

Every optimization that landed in `decode_huffman_body_resumable`
applies to both monomorphisations:

- **B1 (register-local bitbuf/bitsleft/in_pos)**: same lifted locals.
- **B2 (PRELOAD + huffdec! end-of-branch)**: same.
- **B3 (collapsed exceptional branch)**: same.
- **B4 (BMI2 BZHI for extras)**: same.
- **B5 (DistTable::lookup_subtable_direct)**: same.
- **B6 (LitLenTable::lookup_subtable_direct)**: same.
- **T0 (raw-pointer literal writes)**: trivially extends to u16
  via `*out_ptr.add(out_pos) = lit as u16`.
- **T3 (multi-literal lookahead with refill-before-carry)**: same
  pattern; the u16 store is a `write_unaligned::<u64>` for 4 u16s
  packed into 64 bits.
- **T4 (FASTLOOP yield-elide)**: same; the FASTLOOP_MARGIN bound is
  in bytes (u8 mode) or u16 elements (u16 mode = double the byte
  margin).
- **T5 (arm64 prefetch)**: same.

### Match-copy specialization

The hardest piece. Clean mode uses `copy_match_fast` (libdeflate-shape
SIMD u8 copies). Markers mode uses `emit_backref_ring`'s u16 ring
arithmetic: a single u16 memcpy over the ring that produces correct
markers AND correct in-chunk bytes simultaneously (per the
deflate_block.rs:119-131 doc comment).

The const generic dispatches to the right variant. Both fit the same
function signature (`(state, output_or_ring, out_pos, distance,
length) -> usize`). Marker-mode's variant comes verbatim from
`Block::emit_backref_ring`; clean-mode's is today's
`copy_match_windowed` (with the T0/T3 fast path).

---

## 4. Migration sequence

1. **Build `UnifiedInflate` alongside existing decoders.** New file
   `src/decompress/inflate/unified.rs`. Const-generic monomorphisation
   into `<Clean>` and `<Markers>`. Initially:
   - `<Clean>` is a literal copy of `ResumableInflate2`. Should
     produce identical output and identical perf (verify with
     existing 638 tests + tight_huffman_baseline bench).
   - `<Markers>` is a port of `Block::read`'s hot loop, sharing every
     primitive with `<Clean>` modulo the u16 emit path. Initial perf
     target: equal to current `Block::read` (don't regress bootstrap).

2. **Add unified differential test.** Real silesia × both modes ×
   chunked sizes. Cross-check `<Markers>` output ring against the
   current `Block::read`'s ring byte-for-byte (post-`apply_window`
   resolution).

3. **Switch parallel-SM workers to `UnifiedInflate`.** `gzip_chunk.rs`'s
   `decode_chunk_marker_bootstrap_then_isal` becomes:
   ```rust
   let mut decoder = UnifiedInflate::new(...);
   // Phase 1: markers
   decoder.read_stream_markers(&mut chunk.data_with_markers)?;
   // Mid-flip check
   if decoder.contains_marker_bytes() == false {
       let window = chunk.last_clean_window(32_KiB);
       decoder.set_window(window)?;
       decoder.read_stream_clean(&mut chunk.data_spare)?;
   }
   ```
   No `Block` construction. No `ResumableInflate2` construction. One
   decoder lifecycle per chunk.

4. **Bench on neurotic.** Tier 2 gate ≥ 0.9× of `isal-compression`.
   Expected result: better than the current 0.91× because we're
   replacing the bootstrap (the 83%-of-cycles hot path) with the
   tight `<Markers>` variant.

5. **Delete the old code.** Once neurotic confirms parity:
   - `src/decompress/parallel/deflate_block.rs` (Block + ring)
   - `src/decompress/parallel/huffman_short_bits_multi_cached.rs`
   - `src/decompress/parallel/huffman_reversed_bits_cached.rs`
   - `src/decompress/parallel/huffman_short_bits_cached_deflate.rs`
   - `src/decompress/parallel/huffman_symbols_per_length.rs`
   - `src/decompress/parallel/rfc_tables.rs`
   - `src/decompress/parallel/huffman_base.rs`
   - `src/decompress/inflate/resumable.rs` (folded into unified)
   - The `IsalInflateWrapper` `pure-rust-inflate` cfg branch (replaced
     by direct `UnifiedInflate` use).

   Net: ~5,000 lines of decoder code deleted, replaced by ~1,800
   in `unified.rs`. Single hot loop. Single tuning surface.

---

## 5. Risk register

1. **Bootstrap perf regression.** `<Markers>` mode's hot loop has
   marker-bookkeeping the libdeflate-shape inner loop doesn't.
   Mitigation: monomorphise so the bookkeeping is const-folded out
   in `<Clean>` mode; benchmark both vs current `Block::read` AND
   current `ResumableInflate2` before deletion.

2. **Const-generic enum is unstable in stable Rust.** Workarounds:
   (a) two separate trait-bound types `Clean` / `Markers` instead of
   `enum`; (b) two `fn`s instead of one const-generic. Both compile
   on stable.

3. **Mode-flip mid-stream correctness.** The state preserved across
   flip is `Bits`, `BlockState`, `dynamic_tables`. The window must
   match exactly what the prior u16 output decoded to. Mitigation:
   real-silesia differential test (per the
   `feedback_real_corpus_test_with_lever` rule) that flips at every
   plausible boundary.

4. **The "rapidgzip-faithful" framing weakens.** Today
   `Block::read` cites vendor `deflate.hpp` file:line for every method.
   The unified decoder is structurally different from vendor (one
   monomorphised body instead of vendor's templated class). Vendor
   citations move from "literal port" to "ported the technique."
   Per CLAUDE.md update 2026-05-27 (final), inner-loop innovation is
   authorized — this is in scope.

5. **Apply-window step needs to handle both u8 (today) and u16
   (Phase 1 ring tail) outputs.** Already handled in
   `replace_markers_lut_narrow` and `apply_window` in
   `chunk_fetcher.rs`; no change needed.

6. **`isal-compression` feature continues to use the OLD `Block` +
   ISA-L FFI path until the FFI is also deleted.** Both code
   paths coexist behind the cfg until Tier 2's neurotic A/B confirms
   we can delete. No production routing change required for this
   unification — it lands behind the `pure-rust-inflate` feature
   only.

---

## 6. Done-when

**Conditional gates per the advisor's pre-flight critique** (Amdahl
on the 17% identifiable-Huffman vs 61% unattributed-children split
in the bootstrap profile is the load-bearing perf claim; design lives
or dies by what's in the 61%).

1. `UnifiedInflate` lands; 638-test suite remains green; real-silesia
   differential test (both modes + flip) passes byte-perfect.
2. `tight_huffman_baseline` bench shows `<Clean>` mode at parity with
   today's `ResumableInflate2` (no regression from the unification).
3. **Neurotic A/B conditional bands** (`make bench-sm-pure-rust`):
   - **≥ 0.95× of isal-compression**: full success. Ship.
   - **≥ 0.93× and < 0.95×**: partial win; ship + write follow-up
     plan for the next bottleneck (likely ring drain or
     `BootstrapBuffer` allocator).
   - **< 0.93× (i.e. matches current 0.91× or worse)**: REVERT.
     The unification didn't materialize the predicted gain, and we
     now own two divergent decoder shapes for negative payoff.
     Roll back to the two-decoder split and write a different plan
     attacking the actual 61% unattributed cost.
4. **7-day soak** between step 3 result and step 5 (deletion).
   Catches load-dependent perf cliffs and any latent correctness
   issues that the synthetic + silesia + dual-oracle tests miss.
5. Opus advisor sign-off on the neurotic measurement and the
   structural delete.

## 6.5 Pre-flight requirement (gate before writing any unified.rs code)

Per advisor: **run a 30-minute `perf record -g --call-graph dwarf` on
neurotic against commit `c09e8d4` and produce a breakdown of the 61%
unattributed bootstrap children.** Two possible findings:

- **(A) Unattributed 61% is dominated by Huffman table calls that
  got inlined out** (likely). The unification gain materializes as
  predicted (~6.6 pp wall improvement); proceed.
- **(B) Unattributed 61% is dominated by ring drain +
  `BootstrapBuffer` + allocator + header parse** (also plausible).
  The unification produces 1–2% gain at best — not worth the churn.
  Pivot the plan to "fix the ring drain" (or whichever symbol
  dominates).

Without this pre-flight we'd commit to ~3 weeks of work whose
payoff is unverified. The cost is 30 minutes.

## 6.6 The mid-stream flip correctness assertion (load-bearing)

Per advisor: today's two-decoder handoff invariant lives at
`gzip_chunk.rs:1290-1295`:

> the trailing `MAX_WINDOW_SIZE` values of `output` are
> marker-free (assert tracked by `trailing_clean`).

This assertion is what makes the bootstrap→ResumableInflate2 handoff
sound. The unified decoder's mid-stream flip needs the same
assertion at the same point in the flow:

- After the `<Markers>` `read_stream_markers` call returns, BEFORE
  flipping to `<Clean>`, the decoder MUST verify that the 32 KiB
  tail of the just-written ring is marker-free.
- If verification fails (decoder reached a block boundary with
  fewer than 32 KiB of clean trailing bytes), the caller MUST NOT
  flip — keep running `<Markers>` until the next block boundary
  with clean tail, OR until BFINAL.

This invariant becomes a method on `UnifiedInflate`:
`fn can_flip_to_clean(&self) -> bool` returning
`self.distance_to_last_marker_byte >= MAX_WINDOW_SIZE
&& self.block_state == BlockState::AwaitingHeader`. Without this
method explicitly named in the design, the flip is unsafe.

## 6.7 Two-codepath maintenance during transition

For the duration of work (until `isal-rs` FFI is also deleted —
Tier 1 done-when from `plans/pure-rust-isa-l.md`), BOTH the old
`Block`-based bootstrap AND `UnifiedInflate<Markers>` will exist:

- `--features isal-compression`: `Block` + ISA-L FFI Phase 2
- `--features pure-rust-inflate`: `UnifiedInflate<Markers>` +
  `UnifiedInflate<Clean>`

The `gzip_chunk.rs::bootstrap_with_deflate_block` function becomes
feature-conditional. Every future B-series lever has to be ported
to BOTH (until ISA-L FFI deletion lands). Estimated transition
window: 1-3 months. Not great, but the alternative (deleting ISA-L
FFI in the same PR) is too big.

## 6.8 Test migration (before any deletion)

Tests directly using `ResumableInflate2` (must migrate to
`UnifiedInflate::<Clean>` before §4 step 5 deletion):
- `src/tests/resumable_correctness.rs`
- `src/tests/step25_production_instrumentation.rs`
- `benches/tight_huffman_baseline.rs`
- `benches/inflate_isal_vs_pure_rust.rs`

---

## 7. Non-goals (explicit)

- Replacing `libdeflate` in BGZF / multi-member / sequential SM. Out
  of scope.
- Replacing ISA-L for compression. Out of scope.
- Touching the chunk_fetcher / consumer / prefetcher architecture.
  Out of scope — this is a decoder unification, not a pipeline
  rewrite.
- Beating rapidgzip. The pipeline gap (0.51× rapidgzip on our latest
  bench) is architectural, not decoder.
- Changing the `IsalInflateWrapper` public surface. Callers stay the
  same (same `read_stream`, same `set_window`, same stopping points).
