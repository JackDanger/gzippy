# DISPROOF VERDICT — FOLD cadence-vs-intrinsic split (GZIPPY_FOLD_CONTIG)

Independent, read-only. Source-verified first-hand against HEAD 7aae6c4a + overlay,
branch reimplement-isa-l. I tried to break C1/C2/C3. Summary up front:

- **C2 (FOLD does not use the two-phase refetch loop): CONFIRMED.** Source + runtime
  counters both hold. This is solid.
- **C1 (cadence/grow/double-copy tax is NEGLIGIBLE): REFUTED IN SCOPE.** The control
  removed only ONE of the (at least) two post-ring copies — the `pending_clean`
  middle-man + its regrow. A DISTINCT, larger, per-block drain copy
  (`drain_to_output` → fresh `u8buf` alloc + byte-by-byte ring fill) is paid by BOTH
  arms and is invisible to this A/B. So C1 is proven for the `pending_clean`
  component only, not for "the cadence tax" as a whole.
- **C3 (residual gap is dominated by intrinsic symbol rate): NOT LICENSED YET.** The
  untested drain copy sits INSIDE the 0.678→0.925 distance and is a harness cost, not
  symbol rate. The conclusion is premature, not wrong.
- **TIE verdict on the tested component: VALID.** Keep the change: OK, with a caveat.

---

## What the control actually removed (source-counted, not brief-counted)

The brief's mental model of the clean-tail byte journey omits the FIRST copy. Tracing
the real path (`marker_inflate::Block::read` → `drain_to_output` →
`MarkerSink::push_clean_u8`):

`drain_to_output` (marker_inflate.rs:745–773), post-flip clean branch (755–761):
```rust
let mut u8buf = Vec::with_capacity(new_bytes);   // (1) FRESH ALLOC per drain
for i in 0..new_bytes {
    let b = unsafe { *ring8.add((self.ring_drained + i) % U8_RING_SIZE) };
    u8buf.push(b);                                //     byte-by-byte fill
}
output.push_clean_u8(&u8buf);                     //     hand to the sink
```

**OFF** clean-tail copies (per block):
1. `ring → u8buf`  — fresh `Vec::with_capacity` + byte-by-byte loop (drain_to_output)
2. `u8buf → pending_clean` — `UnifiedMarkerSink::push_clean_u8` extend; fresh `Vec::new()` per loop iter (gzip_chunk.rs:1050, 864–868)
3. `pending_clean → chunk.data` — `append_clean` → `data.extend_from_slice` (gzip_chunk.rs:1061; chunk_data.rs:716–727)

**ON** clean-tail copies (per block):
1. `ring → u8buf`  — **UNCHANGED** (still drain_to_output's alloc + byte loop)
2. `u8buf → chunk.data` — `ContigFoldSink::push_clean_u8` → `data.extend_from_slice` (gzip_chunk.rs:938–955)

So the control eliminated copy **#2 (the `pending_clean` middle-man) plus its per-loop
`Vec::new()` and the `append_clean` regrow**. It did NOT eliminate copy **#1**, which
is the LESS-optimized of the two ring drains: the marker branch drains via at most two
contiguous `push_slice` slices (no byte loop, no per-call alloc — lines 762–770), but
the post-flip CLEAN branch materializes a fresh `Vec` and copies byte-by-byte. That
per-block alloc + linear byte pass over every clean byte (~all of the chunk's output)
survives on both arms.

The brief's framing "eliminates BOTH copies + regrow" / "OFF: drain copy 1, append
copy 2" is off by one: there are **three** buffers OFF (`u8buf`, `pending_clean`,
`data`), **two** ON. The control removed **one** copy, not two. The remaining
`ring → u8buf` copy is the one most likely to cost (it is the byte-loop + per-drain
heap allocation), and the A/B is blind to it because both arms pay it identically.

## C1 — REFUTED IN SCOPE

The N=15 4-pass interleaved OFF-vs-ON result (1.005 / 0.974 / 1.004 / 1.019, ±2.6%,
freq-neutral ratio under 4.7–6.2 load) is a clean TIE and I accept it **for what it
tested**: removing the `pending_clean` intermediate + `append_clean` regrow does not
move the wall. That is a real, honest finding. A single block-sized `extend_from_slice`
copy per block plausibly is <1% of wall, below this resolution — chasing it is not
worth it vs the ~0.25× remainder. Agreed there.

But C1 as written ("the cadence/grow/double-copy tax … is NEGLIGIBLE … it is NOT a
meaningful free component of the 0.14× gap") generalizes a one-component result to the
whole cadence tax. The `ring → u8buf` per-block drain (fresh alloc + byte-by-byte fill)
is a distinct cadence/copy component, it is larger (it touches every clean byte with a
materializing copy, and the marker path proves it could be a two-slice memcpy instead),
and it is **untested**. Under the campaign's own Rule 3 ("to bound a speed-up lever you
must REMOVE the region and measure — never extrapolate"), C1 cannot claim the cadence
tax is closed while the dominant cadence copy was never removed.

## C2 — CONFIRMED (both halves)

1. *Source.* The two-phase handoff that feeds `finish_decode_chunk_with_inexact_offset`
   (and thence the `writable_tail` refetch loop) is reached ONLY via
   `MarkerStep::FlipToClean` (gzip_chunk.rs:1076 → 1092). That step is returned ONLY
   inside `#[cfg(isal_clean_tail)]` (gzip_chunk.rs:1524–1539); the gzippy-native build
   has that cfg OFF and explicitly "falls through — no handoff, Engine M continues."
2. *Runtime.* `flip_to_clean=0 finished_no_flip=16` is the live proof the FlipToClean
   arm was never taken across all 16 chunks; `finish_decode_chunk_with_inexact_offset`
   is therefore never called on this path. So the advisor's (b)/(c) cadence mechanism,
   which was framed around `finish_decode_chunk_impl:591–628`'s `writable_tail` refetch,
   genuinely does NOT exist on the native FOLD path. C2's premise is correct.

The catch: C2 then says "therefore the gap is dominated by INTRINSIC SYMBOL RATE." That
does not follow. The advisor's *specific* cadence (the refetch loop) is absent — but a
*different* per-128-KiB-equivalent cadence (the per-block `drain_to_output` materializing
copy) is present and unremoved. C2 disproved the advisor's named mechanism; it did not
establish that no cadence component remains.

## C3 — DIRECTIONALLY RIGHT, MIS-ATTRIBUTED

C3 bounds the remaining inner-loop work by the full native_fold→ocl_cf distance
(0.678→0.925 ≈ 0.25×) rather than VAR_VI's 0.6× slope — correct method (oracle-removed
ceiling, not slow-down slope; Rule 3). But it labels that whole 0.25× "intrinsic symbol
rate." It is not all symbol rate: the ISA-L copy-free oracle (`ocl_cf`) decodes ring-free
**directly into `chunk.data`** via `writable_tail_reserve` (gzip_chunk.rs:220–233) — it
pays NEITHER the `ring → u8buf` drain NOR any of the post-ring copies. So part of the
0.678→0.925 advantage is drain/copy elimination, which is a harness cost. The contig
control cannot separate the two because contig still pays copy #1. C3 over-attributes.

## Disproof angles, answered

1. **Is the control valid? Page-fault confound from the big reserve?** No confound that
   masks a win. `reserve_clean` → `Vec::reserve` (chunk_data.rs:736–738) is a lazy
   capacity reservation — it does NOT memset; pages fault in on first write either way,
   and both arms ultimately write the same bytes. If anything the bias runs the OTHER
   way: the OFF path grows `chunk.data` by amortized doubling, each realloc copying live
   data; the up-front reserve removes those realloc-copies. So contig is mildly favored
   to show a win and still TIE'd — that STRENGTHENS the "this copy is negligible"
   finding, it does not hide one. **But** the control did NOT remove the right cadence:
   the per-block `u8buf` alloc + byte-by-byte ring fill in `drain_to_output` is the
   bigger, untouched cost, and it is the granularity that actually corresponds to "no
   per-block refetch/materialize." The control did the GROW half (one up-front reserve)
   and removed one copy; it did not make the drain itself copy-free.
2. **Is TIE right given the noise?** Yes, for the tested component. A one-copy-per-block
   delta is plausibly sub-1% — below the ±2.6% spread at 4.7–6.2 load. Resolving 0.02×
   would need a quiet box (load <1) and N≥15 interleaved, and it is NOT worth it vs the
   0.25× remainder. The error is not the TIE; it is generalizing the TIE to "all cadence
   is closed."
3. **Does C2's source claim hold?** Yes — verified above (cfg gate + live counters).
4. **Keep the byte-exact contig change?** Defensible (rule 7a: correct, byte-identical,
   TIE, removes a real alloc+copy, lazy reserve so no RSS regression). One caveat: it now
   carries TWO env-gated sink paths for the same bytes. If kept, the clean resolution is
   to make contig the DEFAULT and DELETE the `pending_clean`/`UnifiedMarkerSink` clean
   path (it is strictly a superfluous extra copy), rather than maintaining both behind an
   env var indefinitely. Carrying both as a permanent measurement fork is the kind of
   dead-code-by-flag the charter warns against.

## Required next step before C2/C3 can stand

Run the control the advisor's recommendation actually pointed at: make the **post-flip
clean drain copy-free**. Have `drain_to_output`'s clean branch push the (≤2) contiguous
ring slices straight into `chunk.data` (eliminate `u8buf` entirely and the per-block
alloc + byte loop), exactly as the marker branch already does with `push_slice`. Then
re-measure interleaved.
- If that ALSO TIEs → C1/C2/C3 are vindicated; the cadence really is closed and the
  0.25× is symbol rate. KEEP both contig changes (or fold them into the default).
- If it moves the wall → a free cadence component was being mis-booked as "intrinsic
  symbol rate," and C3's inner-loop target shrinks accordingly.

Until that second control runs, C2's "dominated by symbol rate" and C3's "0.25× is all
intrinsic" are UNPROVEN (not disproven) — the A/B that was run cannot see the copy that
matters most.

## Bottom line
- C2 mechanism claim (no refetch loop): **CONFIRMED.**
- C1 (cadence negligible): **REFUTED as stated** — proven only for the `pending_clean`
  copy; the dominant `ring → u8buf` drain copy is untested and shared by both arms.
- C2 inference / C3 attribution (remainder = intrinsic symbol rate): **UNPROVEN** — one
  more, correctly-targeted, copy-free-drain control is owed before that split is real.
- Keep the change: yes, but prefer default-and-delete-the-OFF-path over a permanent env
  fork.

=== ADVISOR EXIT 0 ===
