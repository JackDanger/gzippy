# DISPROOF-ADVISOR VERDICT — copy-free clean-tail WALL oracle

Independent, read-only, HEAD 7aae6c4a + copy-free overlay. I source-verified
first-hand: the copy-free oracle (`gzip_chunk.rs:160-282`), the dispatch/guard
(`:178`, `:540-555`), the production clean path (`finish_decode_chunk_impl`
`:524-628`, esp. the 128 KiB `writable_tail()` loop), `writable_tail_reserve` +
`decoded_range` + `commit` (`segmented_buffer.rs:181-245`), and the FFI-into entry
(`isal_decompress.rs:794-899`). I tried to break all three claims.

## Bottom line
**C1 — UPHELD-WITH-CAVEATS.** The clean-tail decode region IS a genuine wall lever
(0.14× delta, sign- and magnitude-stable across 3 passes, ≫ within-variant spread;
copy confound genuinely gone; the prior advisor's S≈C≈0.17× prediction confirmed).
This validly REVERSES the prior INCONCLUSIVE. CAVEAT: the 0.14× is **not** pure
"engine RATE" — it bundles a second structural difference (per-call cadence + buffer
growth) the brief doesn't account for. Calling it "the ISA-L clean-engine RATE"
over-attributes.

**C2 — UPHELD.** Residual (0.89×→1.0×) is the window-absent structure. Routing-
identical / window-absent-preserved control strengthens this vs the predecessor.

**C3 — UPHELD as DIRECTION, but its pessimistic COROLLARY is REFUTED/NOT-ESTABLISHED.**
Collapsing to one width-templated primitive is the right faithful move. But "pure-Rust
1.0× requires matching ISA-L's clean RATE, and VAR_VI 0.6× says it can't" is
unsupported, *because of the same confound that taints C1*.

## The single most load-bearing correction
**The copy-free fix removed one confound and, in the same stroke, left a second,
opposite-in-implication structural difference unaccounted for.** The oracle decodes
the ENTIRE clean tail in ONE `isal_inflate` call into a 64 MiB contiguous reserve
(`gzip_chunk.rs:204-217`, `isal_decompress.rs:863-891` is a single loop over one
`out` buffer). Production decodes the tail in a **128 KiB per-call resumable cadence**:
`writable_tail()` hands back at most `ALLOCATION_CHUNK_SIZE` (`segmented_buffer.rs:193`),
and `finish_decode_chunk_impl`'s outer `while` (`:591-628`) re-enters `read_stream`,
re-fetches the tail, drains `take_block_boundaries`, and re-checks stop conditions
**every 128 KiB**, with the Vec grown incrementally by amortized `reserve`.

So the measured 0.14× = (a) ISA-L's intrinsic symbol-decode rate **+** (b) elision of
the per-128-KiB resumable yield/refetch/boundary-drain tax **+** (c) one-shot reserve
vs incremental grow. (b) is exactly the "resumable yield-check tax" CLAUDE.md lists as
a *separately attackable* inner-loop target — it is part of the pure-Rust harness, not
of ISA-L's symbol rate. The instrument therefore measures the clean-decode **subsystem**
(engine + its resumable harness), and cannot apportion the 0.14× between intrinsic rate
and cadence. "Engine RATE is the lever" overreaches to a sub-attribution the instrument
doesn't support; "the clean-tail decode subsystem is a real wall lever, not slack" is
what is actually proven — and that is enough to reverse INCONCLUSIVE.

## Q1 — Is the copy-free oracle a VALID instrument now (C→0)? Residual confounds?
**The copy IS gone — verified.** `decompress_deflate_from_bit_into` writes through
`out.as_mut_ptr().add(out_pos)` directly (`:871`), no internal `Vec`, no `to_vec`,
no CRC; the oracle `commit(keep_len)` bumps length with zero copies and CRCs a
`decoded_range` view (`gzip_chunk.rs:251-262`). So the prior ~0.17× alloc+copy term
→ 0. Good.

Residual-confound audit:
- **64 MiB reserve / page faults:** `ensure_buf(len+64MiB)` (`segmented_buffer.rs:207`)
  almost certainly exceeds the warm pool buffer → fresh allocation → cold pages faulted
  on first ISA-L write. ISA-L only writes the few-MiB it actually decodes, so faults ≈
  the written region ≈ what production faults. Any *excess* (a one-shot big malloc/
  realloc-move) makes the oracle look **worse** → conservative, not artificially good.
- **Boundary rebase to `decode_start = chunk.data.len()` vs production `decode_base =
  chunk.decoded_size()`:** these coincide here (the clean tail appends to an already-
  decoded prefix; `data.len()` == decoded byte count). The byte-exact sha self-test
  (OFF==identity, all 14 chunks, 0 fallbacks) is the proof — wrong boundaries/keep_len
  would diverge the output. Correctness instrument PASSES (Rule 4).
- **The cadence/grow difference (above):** this is the ONE residual confound that is
  NOT conservative and NOT priced. It does not invalidate the region-is-a-lever
  conclusion, but it does taint the *rate* sub-attribution and C3's corollary.

So: valid for "is the clean-decode subsystem a lever" (yes); NOT clean enough to
isolate intrinsic ISA-L symbol rate from cadence tax.

## Q2 — Does C1 survive Δ>spread, or is 0.89× inside TIE?
**Survives, decisively.** Per-pass prod→ocl deltas: 0.140 / 0.146 / 0.137 — the gap is
larger than the within-variant spread (prod ~0.022, ocl ~0.025) AND essentially
constant across passes (sign- and magnitude-stable, drifts together with load). This is
not a TIE; the decode region gates the wall. The Rule-3 speed-up ceiling genuinely
FIRED: making clean decode ISA-L-fast moves the wall to ~0.89× (at/above the 0.85×
threshold), and the *remaining* 0.11× does not move with the engine swap → that residual
is structure (C2), not engine. The one caveat is the label, not the magnitude: 0.14× is
the ceiling for "swap to an ISA-L-class clean subsystem (rate + cadence)," not for
"raise the per-symbol rate alone."

## Q3 — Does this change the no-FFI 1.0× picture (C3)?
**Yes — it makes C3's pessimism unsupported.** C3 argues pure-Rust 1.0× needs the
unified primitive to match ISA-L's clean rate, and that VAR_VI's 0.6× standalone
plateau says it can't. But the gap a pure-Rust unified primitive must close is **< 0.14×**,
because a meaningful (unquantified-by-this-instrument) slice of the 0.14× is cadence/
grow tax (b)+(c), which a pure-Rust path recovers WITHOUT touching its symbol rate —
by decoding into a large contiguous window and eliding the per-128-KiB resumable
yield-check (explicitly authorized in CLAUDE.md's inner-loop license). VAR_VI measured
a *standalone* clean variant; it does not bound a primitive that also sheds the harness
tax. So "VAR_VI 0.6× ⇒ pure-Rust 1.0× unreachable" is NOT established. The faithful
direction (one `readInternalCompressedMultiCached`-shaped width-templated primitive,
per [[project_faithful_unified_decoder_over_perf]]) stands; the claim that it
*provably can't* reach 1.0× does not. To actually bound the intrinsic-rate fraction
you'd need a control that gives the **pure-Rust** path the same one-shot large-window
cadence (decode into a >128 KiB contiguous window, no per-call refetch) and re-measures
— that isolates (a) from (b)+(c).

## Q4 — Anything making ocl artificially GOOD (symmetric of the prior copy confound)?
Two candidates examined; both net conservative or neutral except the cadence one:
- **Bounded slice / less work?** No — the oracle slices `input[..stop_byte+256 KiB]`
  and ISA-L decodes PAST the stop, then trims via `keep_len`
  (`gzip_chunk.rs:188,219-246`). It decodes **more** than production then discards →
  conservative.
- **Skips pipeline work?** No — it stays in the real pipeline: replays
  `append_block_boundary_at` per boundary and calls `finalize_with_deflate`
  (`:269-279`); window-publish/ring/consumer unchanged; sha-exact.
- **CRC:** oracle does one `update` over the kept region; production updates
  incrementally. Same total bytes, negligible per-call delta on hw CRC. Neutral.
- **The cadence/grow elision (Q1/Q3):** this is the ONE thing that makes ocl look
  better for a reason that is NOT "ISA-L's symbol rate." It is the symmetric residual
  the brief's Angle 4 was asking about. It does not flip C1's region-is-a-lever
  conclusion but it is the reason the *rate* attribution and the *VAR_VI-ceiling*
  corollary are unsafe.

## Verdict table
| claim | verdict |
|---|---|
| C1: ISA-L clean-engine RATE is a wall lever worth 0.14-0.16×; ceiling fired; reverses INCONCLUSIVE | **UPHELD-WITH-CAVEATS** — region (clean-decode subsystem) is a real lever, Δ≫spread, copy gone, ceiling fired at ~0.89×; but 0.14× bundles per-128-KiB cadence + grow tax, so "RATE" over-attributes |
| C2: residual above ocl is window-absent STRUCTURE | **UPHELD** — engine held to ISA-L, residual doesn't move with the swap; routing-identical control supports it |
| C3: faithful fix = unify marker+clean into one width-templated primitive; pure-Rust 1.0× requires matching ISA-L clean rate (VAR_VI 0.6× = open risk) | **DIRECTION UPHELD; COROLLARY REFUTED/NOT-ESTABLISHED** — gap to close is <0.14× (cadence/grow recoverable in pure-Rust), VAR_VI does not bound a harness-tax-shedding primitive |
| instrument valid now (C→0) | **YES for region-lever; NO for isolating intrinsic rate** — cadence/grow confound remains |

**Most actionable next step:** before crediting any fraction of the 0.14× to the
*inner symbol rate*, run the symmetric control — give the **pure-Rust** clean path one
large contiguous window (decode without the per-128-KiB `writable_tail` refetch/yield
cadence) and re-measure. The portion the wall recovers is cadence/grow (free to
pure-Rust); only the remainder is the intrinsic-rate gap the unified primitive must
close against ISA-L. That, not VAR_VI's standalone number, sets the real no-FFI 1.0×
bar.

=== ADVISOR EXIT 0 ===
