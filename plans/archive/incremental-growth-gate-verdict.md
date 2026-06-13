# DIS-23 incremental-growth gate — advisor verdict (read-only)

Change: `owner/isal-incremental-growth` @ 153da9d1 (base d56cb0f5). Adds
`decompress_deflate_from_bit_into_growable` + `IncrementalOutSink` (impl on
`SegmentedU8`), wired into `finish_decode_chunk_isal_oracle`
(`gzip_chunk.rs:297-360`) behind `GZIPPY_ISAL_INCREMENTAL_GROWTH=1`
(default OFF == the 8x upfront reserve, identity).

Bottom line: **BYTE-EXACT and structurally sound. KEEP (gated) is unconditionally
justified. DEFAULT-ON is gated on ONE more test: a byte-exact differential that
FORCES multiple regrows** — the grow-boundary is the novel code and silesia at
factor-4 likely never exercises it (the same blind spot that hid the fallback bug).

---

## Claim 1 — byte-exactness of grow-mid-decode — **SOUND**

The realloc happens ONLY between `isal_inflate` calls (loop-top `remaining == 0`
branch, `isal_decompress.rs:1019-1027`), never mid-call. A fresh, possibly-moved
`next_out` across calls is the standard zlib/ISA-L streaming contract, and ISA-L
resolves cross-call back-references from its internal 32 KiB `tmp_out_buffer`
history, not from `next_out`.

Three independent proofs:
- **Vendor existence proof.** rapidgzip allocates a *fresh* `DecodedVector(128 KiB)`
  every outer iteration (`GzipChunk.hpp:309`) and feeds fully NON-contiguous
  segment buffers to ISA-L across iterations (`isal.hpp:257-258` sets
  `next_out = output` to each fresh buffer). It relies solely on ISA-L's internal
  history (`isal.hpp:291` comment names `tmp_out_buffer`) and is correct. gzippy's
  growable path is *strictly safer*: one contiguous Vec, so back-refs resolve both
  via ISA-L history AND via `next_out[-D]` (the moved bytes stay contiguous behind
  `base` after realloc).
- **Dict proof.** The initial window/dict is loaded via `isal_inflate_set_dict`
  into ISA-L's history, NOT into `next_out` (which starts empty). The first
  back-ref into the dict therefore resolves from `tmp_out_buffer` — direct evidence
  ISA-L uses internal history for distances beyond the current call's output.
- **Straddling back-ref.** Max DEFLATE distance is 32768 = ISA-L's history size; a
  back-ref across a grow point resolves from `tmp_out_buffer`, which is unaffected
  by the buffer move. Covered.

The growable loop (`isal_decompress.rs:967-1095`) is a faithful copy of the fixed
loop (`:863-913`) — identical stopping-point/boundary/block-state handling — with
the only delta being commit+grow+continue instead of `return None` on exhaustion.
`output_offset` stays measured from `decode_start` because all committed bytes are
contiguous (`commit` = `set_len`, `truncate` keeps the bytes physically present).

## Claim 2 — the fallback-bug fix (truncate(decode_start)) — **SOUND / complete**

The fix is correct AND complete for this function. `truncate(decode_start)` runs
UNCONDITIONALLY on both the `Some` branch (before binding `v`,
`gzip_chunk.rs:333`) and the `None` branch (`:344`), restoring `len ==
decode_start` with the decoded bytes still physically in spare (`Vec::truncate`
keeps capacity + contents; `u8` has no Drop). This makes the post-decode state
byte-identical to the fixed-buffer path at EVERY downstream exit:
- `until_exact` decline — `Ok(false)` at `:376`
- inexact-offset coalesce decline — `Ok(false)` at `:405`
- multi-subchunk boundary replay — `:444-461` credits `[prev_off, output_offset)`
  segments from `decode_start`, unchanged by growth
- success — `commit(keep_len)` at `:418` re-extends to exactly the kept region;
  CRC `decoded_range(decode_start, keep_len)` reads the right bytes.

No code reads/mutates `chunk.data` between truncate and commit. There is exactly
ONE call site of the growable function (`gzip_chunk.rs:311`), and the three paths
the gate names (until_exact / coalesce-decline / multi-subchunk) all live inside
this one function — all covered by the single truncate. The other production ISA-L
path (`decompress_gzip_stream`, the T1 streaming single-member route) is NOT
touched by this change. Set-len arithmetic is safe: each `commit(cur_pos)` adds
`cur_pos <= spare`, so `set_len <= capacity` always (debug_assert guards it).

Reservation rolled into Claim 5: the fixtures cover stored/fixed/btype01/
coalesce-decline, but they cover the COMMIT/TRUNCATE accounting, not necessarily
the multi-regrow boundary itself.

## Claim 3 — default-flip safety / pathological regrow — **SOUND (no perf cliff); coverage gap noted**

No perf cliff. `commit_and_reserve` → `writable_tail_reserve(GROW_MIB)` →
`Vec::reserve`, which uses amortized growth (`max(cap*2, required)`). So GROW_MIB=4
is only a FLOOR on requested spare; actual capacity at least DOUBLES per realloc,
and the loop then fills the WHOLE doubled spare (`cap = full spare`, not capped at
GROW_MIB). Regrow count is therefore LOGARITHMIC in output size (~7 reallocs for a
64 MiB max-expansion chunk), not linear — total copy cost O(n) amortized. A
highly-compressible chunk (zeros/JSON/logs, >4x ratio) is correct and bounded.

Correctness on under-reserve: growth never fails, so the factor-5/6/7 fallbacks
DIS-14 hit are genuinely dissolved — an under-sized initial no longer forces a
fallback; it regrows.

Gap: "factor-4 is genuinely 0-regrow on the corpus" is UNVERIFIED, and `fallbacks=0`
does NOT prove `regrows=0` (regrows don't increment the fallback counter). Real
inputs DO contain >4x-compressible chunks, so the regrow boundary WILL fire in
production at factor-4. That is fine *given Claim 1*, but see Claim 5.

## Claim 4 — scope (TLB half met; DIS-14/17 correction) — **SOUND**

- "TLB half met" is sound: dTLB-miss MPKI is a direct HW counter (not wall, not
  spin-susceptible), measured -41/-42% to BELOW rg at T4/T8.
- The DIS-14/DIS-17 correction is vendor-confirmed: rapidgzip DOES grow
  incrementally (`GzipChunk.hpp:309` fresh 128 KiB per iter, never reserves the
  chunk output upfront), refuting "rg feeds the whole buffer." And peak-RSS-unchanged
  after removing the over-reserve is consistent with "RSS = touched working set
  under lazy faulting, not the lazy over-reserve" — refuting DIS-17's
  "RSS is mostly the 8x over-reserve." The split verdict (dTLB closed, RSS not
  reachable via this lever) is sound.

## Claim 5 — keep / revert / default — **KEEP gated: YES. DEFAULT-ON: gate on one regrow test.**

- KEEP gated is unconditional under rule 7a (byte-exact, wall-neutral, correct).
- DEFAULT-ON is a real production change: on the `isal_clean_tail` build,
  `isal_engine_oracle_enabled()` defaults true (`gzip_chunk.rs:161`), so this is
  the x86_64 parallel-SM production clean-tail allocator.
- The byte-exact ARGUMENT (Claim 1) is strong, but the EMPIRICAL coverage of the
  genuinely-new code — commit→reserve→refetch→resume ACROSS a realloc — is the weak
  point. Silesia at factor-4 with only 14 isal_chunks may never trigger a single
  regrow, leaving the grow boundary near-uncovered by the dual-sha. This is exactly
  the blind spot that already hid one bug (silesia missed the fallback bug; in-tree
  fixtures caught it).

Recommended gate before flipping default: run the dual-sha (and the in-tree
routing/stored/fixed fixtures) with a config that FORCES several regrows —
`GZIPPY_ISAL_INITIAL_FACTOR=1` plus a small `GZIPPY_ISAL_GROW_MIB` (e.g. 1), and/or
a highly-compressible fixture (zeros / repeated JSON, >4x). If byte-exact there,
default-ON factor-4 is safe (factor-4 is a superset-safe, fewer-regrow case of a
proven-exact factor-1). Until that one run exists, keep gated-OFF.
