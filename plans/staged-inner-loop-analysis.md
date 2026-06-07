# Stage D — Inner Huffman loop analysis (window-absent / marker decode)

Read-only analysis. Goal: speed the window-absent inner Huffman decode so
`d_w` 125.5ms → ~71ms (rapidgzip parity). Branch `reimplement-isa-l`,
build `--no-default-features --features pure-rust-inflate` (`pure_inflate_decode == parallel_sm`).

## 1. The EXACT production window-absent decode function + call chain

The parallel-SM worker, for a window-absent (empty-window, u16-marker-emitting)
chunk, calls — top to bottom:

```
chunk_fetcher.rs:3156   decode_chunk_window_absent(input, decode_start, stop_hint_bit, cfg)
gzip_chunk.rs:567       decode_chunk_window_absent           (#[cfg(parallel_sm)], window = &[])
gzip_chunk.rs:585       decode_chunk_with_rapidgzip_impl     (initial_window.len() != MAX_WINDOW_SIZE)
gzip_chunk.rs:634       decode_chunk_unified_marker          (loop over blocks)
gzip_chunk.rs:743       marker_decode_step                   (per deflate block)
gzip_chunk.rs:1017      marker_decode_step_vendor_block      (DEFAULT; MarkerRing only if GZIPPY_MARKER_RING=1)
gzip_chunk.rs:1106-1116 marker_decode_step_loop  -> block.read(bits, output, usize::MAX)
marker_inflate.rs:947   Block::read
marker_inflate.rs:1093  Block::read_internal_compressed
marker_inflate.rs:1098-1102  read_internal_compressed_specialized::<CONTAINS_MARKERS = true|false>
marker_inflate.rs:1226       read_internal_compressed_specialized   <-- THE HOT LOOP
```

**THE inner Huffman hot loop is `marker_inflate.rs:1326-1457`** (the
`while emitted < n_max_to_decode` body inside
`read_internal_compressed_specialized<CONTAINS_MARKERS>`).

The lit/len and dist table primitives are **NOT** `LitLenTable`/`DistTable` and
**NOT** `decode_huffman_body_resumable` (resumable.rs). Those, plus the entire
`src/decompress/inflate/` family (`double_literal.rs`, `vector_huffman.rs`,
`libdeflate_decode.rs`, `jit_decode.rs`, `specialized_decode.rs`,
`consume_first_table.rs`, `bmi2.rs`), are **DEAD relative to the live
window-absent path** — `grep` for them in `src/decompress/parallel/*.rs`
returns nothing. Any technique they implement is still "to do" on the live path.

The LIVE primitives are in `src/decompress/parallel/`:
- **lit/len table:** `isal_huffman_pure::IsalLitLenCodePure` — `decode()` at
  `isal_huffman_pure.rs:1022`, built by `make_inflate_huff_code_lit_len`
  (`:458`) / `set_and_expand_lit_len_huffcode` (`:325`).
- **dist table:** `isal_huffman_pure::IsalDistCodePure` — `decode()` at
  `isal_huffman_pure.rs:1158`.
- **Bits:** `inflate::consume_first_decode::Bits` — `refill` at `:245`,
  `peek` `:292`, `consume` `:297` (libdeflate-style 56-bit LSB buffer).

The window-PRESENT (post-flip clean tail) path is different
(`finish_decode_chunk_*` → IsalInflateWrapper) and is NOT `d_w`. `d_w` is the
marker phase above.

## 2. Characterization of the current inner loop (present / absent)

The decode is a faithful port of ISA-L's `readInternalCompressedMultiCached`
(two-level 12-bit short LUT + long-code escape, multi-symbol packing).

PRESENT already:
- **Multi-LITERAL DECODE (ISA-L triple packing).** One 12-bit
  `short_code_lookup` load can return `sym_count ∈ {1,2,3}` literals packed at
  bit offsets 0/8/16; unpacked at `marker_inflate.rs:1362-1387` (`sym >>= 8`).
- **Length-extra baked into the LUT** (`symbol - 254`), no RFC extra-bit read
  for lengths — `isal_huffman_pure.rs:325 set_and_expand_lit_len_huffcode`,
  consumed via `bits.consume(bit_count)` at `marker_inflate.rs:1357`.
- **`const CONTAINS_MARKERS` specialization** (`marker_inflate.rs:1226`,
  dispatched `:1098-1102`) — const-folds all marker-counter paperwork out once
  the chunk goes clean.
- **Single refill at top of outer iter** (`marker_inflate.rs:1343`) sized so
  inner `available()<32` checks in `decode()` are predictably-false.
- **Inlined distance-extra** (`marker_inflate.rs:1417-1433`) — no function call,
  no `n_lowest_bits_set` helper.
- **Fixed-Huffman handled** (`build_huffman_luts_for_block`,
  `marker_inflate.rs:838-842`) — but it REBUILDS the ISA-L LUT from
  `FIXED_LIT_LEN_LENGTHS` per fixed block (no cached static table). Low impact:
  silesia gzip dynamic blocks dominate; fixed blocks are rare.

ABSENT on the live path:
- **Multi-literal packed WRITES.** The decode packs 2–3 literals per lookup,
  but they are STORED one u16 at a time in the scalar unpack loop
  (`marker_inflate.rs:1362-1387`: `ring_ptr.add(pos % RING_SIZE).write(code & 0xFF)`
  per symbol). No wide (u32/u64) store.
- **BMI2 PEXT/BZHI.** `bmi2.rs` has `extract_bits_bmi2`/BZHI helpers but they are
  unused on this path. Variable-width masks use `next_bits & ((1u64<<n)-1)`
  (`isal_huffman_pure.rs:1047`, and dist-extra `marker_inflate.rs:1427-1428`).
  No runtime dispatch. (Most masks are *constant*-width 12/10-bit AND → BZHI
  wouldn't help those; only the long-code mask and dist-extra are variable.)
- **Table prefetch.** None. NOTE: the lit/len index is data-dependent on the
  bits just read, so the primary `short_code_lookup[next_12_bits]` load is
  inherently non-prefetchable. (16KB short table fits L1 anyway.)
- **FASTLOOP yield-check elision.** There is NO per-symbol resumable yield tax
  to elide here — the loop is a plain `while emitted < n_max_to_decode` with
  `n_max` capped to `RING_SIZE - 258` (`marker_inflate.rs:1293`); the resumable
  contract lives in the OUTER `marker_decode_step_loop`, not per-symbol. The only
  per-iter overhead is the unconditional `bits.refill()` (1343), which is NOT
  wasteful — a backref iter can consume up to ~48 bits (≤20 lit/len + ≤28 dist),
  so a 56-bit buffer holds only ONE worst-case iter; refill-batching across iters
  is unsafe for backref-heavy data. (This is why per-iter refill is structurally
  necessary; do NOT pursue refill-batching as a first move.)

## 3. THE PICK (first move): multi-literal packed WRITE in the unpack loop

**Technique:** when the ISA-L lookup returns `sym_count ≥ 2` (all guaranteed
literals — see landmine #2), replace the 2–3 scalar `u16` ring writes with a
single unaligned wide store (u32 for 2, u64 low-48 for 3), guarded against ring
wrap. Decode/bit-accounting is UNCHANGED — only the store changes.

**Where:** `src/decompress/parallel/marker_inflate.rs:1362-1387` (the
`loop { let code = sym & 0xFFFF; if code <= 255 || sym_count > 1 { ... } }`
unpack body). Add a fast pre-check before the loop: if `sym_count >= 2` and
`(pos & (RING_SIZE-1)) + sym_count <= RING_SIZE`, build the packed u16 lane
value from `sym` bytes and do one `(ring_ptr.add(pos & (RING_SIZE-1)) as *mut u64).write_unaligned(packed)`, then `pos += sym_count; emitted += sym_count;
distance_marker += sym_count (if CONTAINS_MARKERS)`. Else fall through to the
existing scalar loop.

**Rationale (impact × low risk):**
- *Risk = lowest of all candidates.* The decoded symbols, bit positions, and
  `distance_to_last_marker_byte` accounting are byte-identical; only the store
  width changes. A wrong store shows up instantly as a sha mismatch on silesia.
- *Impact = broad.* silesia literals have ~5–6 bit entropy, so `sym_count = 2`
  is common and `3` occurs — every such lookup currently pays 2–3 indexed u16
  stores + 2–3 `pos`/`emitted` increments; the packed path collapses them to one
  store + one increment. This is exactly libdeflate's multi-literal write that
  the live path lacks (it's only "decoded" multi, never "written" multi).
- Other candidates rejected as first move: fixed-Huffman caching (rare blocks,
  low impact); BMI2 BZHI (only variable masks benefit — long-code + dist-extra —
  both dwarfed by the adjacent table load / ring memcpy; marginal); prefetch
  (lit/len index non-prefetchable); FASTLOOP refill-batching (unsafe for
  backref-heavy data — 56-bit budget covers only one worst-case iter).

**Per-technique falsifier:**
- *Correctness gate (must pass or REVERT):* `cargo test --release` (635+ lib
  tests) + silesia differential — output sha256 byte-identical to the pre-change
  binary AND to flate2/libdeflate oracle. Any single-byte diff = revert.
- *Perf gate (interleaved, sha-verified, best-of-N≥7, `GZIPPY_FORCE_PARALLEL_SM=1`,
  assert `path=IsalParallelSM`):* measure `d_w` (window-absent decode) and wall.
  - SUCCESS: `d_w` drops measurably below 125.5ms (target trajectory toward
    ~71ms) AND wall does not regress, Δ > inter-run spread.
  - TIE (Δ < spread): KEEP if byte-identical (correct change layered per
    measurement rule #7) but it is NOT the lever — proceed to BMI2-BZHI on
    dist-extra (`marker_inflate.rs:1417-1433`) or fixed-table caching next.
  - REVERT only on correctness failure or a measured wall regression with a
    named mechanism.

## 4. Correctness landmines on the marker-emitting (u16) path

1. **Ring wrap.** `RING_SIZE = 65536` (power of two → index is `pos & 65535`;
   `marker_inflate.rs:232`). A wide store must NOT cross the physical ring
   boundary — guard `(pos & (RING_SIZE-1)) + sym_count <= RING_SIZE`, else use
   the scalar path. `n_max` is already capped to `RING_SIZE-258`
   (`:1293`) so the *logical* room exists; the wrap guard is purely physical.
2. **Packed group is ALL literals.** The branch `code <= 255 || sym_count > 1`
   (`:1364`) means a `sym_count ≥ 2` group can only be literals — EOB
   (`END_OF_BLOCK_SYMBOL`) and length codes (`code > 255`) appear ONLY at
   `sym_count == 1` (`:1388-1456`). So a packed write never needs to test for
   EOB/length mid-group. Each lane value must be `code & 0xFF` (the low byte),
   NOT the raw 13-bit symbol.
3. **Marker sentinel separation.** `MARKER_BASE = 32768`
   (`replace_markers.rs:23`); literals are `< 256`, so packed-literal stores can
   never alias a marker value. Markers are produced ONLY by
   `emit_backref_ring` (`:1444-1452`) into the pre-initialized marker zone
   (ring upper half) — untouched by literal writes. Do not let a packed store
   spill into a slot the next backref will read as a marker.
4. **`distance_to_last_marker_byte` must advance by exactly `sym_count`** on the
   packed path when `CONTAINS_MARKERS` (mirrors per-literal `+= 1` at `:1379`).
   Under `CONTAINS_MARKERS = false` it const-folds away.
5. **`drain_to_output` watermark / `decoded_bytes`.** `emitted` and
   `self.decoded_bytes` (committed via `commit!`/tail at `:1458-1460`) must equal
   the literal count — the wide store must bump both by `sym_count`, exactly as
   the scalar loop does cumulatively.
