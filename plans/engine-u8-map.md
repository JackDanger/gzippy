# Engine u8-architecture map — vendor rapidgzip vs gzippy (P1 deliverable)

Campaign: plans/engine-campaign.md (funded 2026-06-10). Every vendor claim below was
verified FIRST-HAND against `vendor/rapidgzip/librapidarchive/src/` at submodule
`d2350e9c` (the segmented-clean-storage scar — segmented_buffer.rs:17-21 — is why no
vendor citation here is taken from prior session notes).

Vendor paths are relative to `vendor/rapidgzip/librapidarchive/src/rapidgzip/`;
gzippy paths relative to `src/decompress/parallel/`.

---

## (a) vendor `deflate::Block` — where decode emits u8 vs u16

| Behavior | Vendor citation | Fact |
|---|---|---|
| Window storage is ONE allocation, two widths | `gzip/deflate.hpp:805-806` | `PreDecodedBuffer = std::array<uint16_t, 2*MAX_WINDOW_SIZE>` (65536 u16 = 128 KiB); `DecodedBuffer = WeakArray<uint8_t, 131072>` over the SAME bytes. |
| u8 view is a reinterpret_cast, not a copy | `gzip/deflate.hpp:890-894` | `getWindow()` returns `DecodedBuffer{ reinterpret_cast<uint8_t*>(m_window16.data()) }`. The "flip" is a WIDTH reinterpretation of the same memory. |
| Width selector | `gzip/deflate.hpp:936-939` | `m_containsMarkerBytes` — true ⇒ decode into `m_window16` (u16); false ⇒ decode into `getWindow()` (u8). |
| ONE templated engine for both widths | `gzip/deflate.hpp:1428-1464, 1514-1582, 1589-1666` | `readInternal{Compressed,Uncompressed,CompressedMultiCached}` are templated on `Window`; the u16 and u8 instantiations are the SAME code; marker bookkeeping is `if constexpr (containsMarkerBytes)` (`:1311, :1367, :1523, :1600`). |
| BULK literal/backref write width | `gzip/deflate.hpp:1319, 1341, 1376` | `window[m_windowPosition] = decodedSymbol` / `memcpy(... length * sizeof(window.front()))` — element width of the ACTIVE window: u16 pre-flip, u8 post-flip. The bulk clean path is u8-direct. |
| Marker pre-init zone | `gzip/deflate.hpp:875-888` | upper half of `m_window16` pre-filled with `i + MAX_WINDOW_SIZE` so an out-of-history backref memcpy yields the correct marker value with no marker branch. |
| Marker-distance counter | `gzip/deflate.hpp:1311-1317 (literals), 1379-1389 (backward scan after backref memcpy)` | `m_distanceToLastMarkerByte`: ++ per clean symbol, reset on marker, recomputed by backward scan over the just-copied run. |
| **Clean-window arming (THE flip)** | `gzip/deflate.hpp:1282-1284` | after each u16 `readInternal` call: flip if `distanceToLastMarkerByte >= m_window16.size()` (65536) OR (`>= MAX_WINDOW_SIZE` AND `== m_decodedBytes`). |
| Flip mechanics | `gzip/deflate.hpp:1740-1785` (`setInitialWindow`) | zero the not-yet-decoded ring remainder (`:1762-1764`), `replaceMarkerBytes` over the whole u16 ring against the (possibly empty) initial window (`:1765`), value-downcast the rotated window through a stack `conflatedBuffer` (`:1772-1776`), memcpy it to the TAIL of the u8 view (`:1778-1780`), `m_windowPosition = 0`, `m_containsMarkerBytes = false` (`:1782-1784`). |
| Seam output | `gzip/deflate.hpp:1285-1286` | the flip-triggering call's own bytes are returned as **u8** views: `result.data = lastBuffers(window, m_windowPosition, nBytesRead)` over the just-conflated u8 window — no temp buffer. |
| Pre-decode window seed | `gzip/deflate.hpp:1750-1759` | if nothing decoded yet: memcpy seed into u8 view at [0, len), `m_windowPosition = m_decodedBytes = len`, `m_containsMarkerBytes = false` — the WHOLE chunk decodes u8-direct. |
| Stored-block early-flip special cases | `gzip/deflate.hpp:1212-1256` | three special cases in `read()` for UNCOMPRESSED blocks: (1) `m_uncompressedSize >= MAX_WINDOW_SIZE` ⇒ read straight into u8 window start, flip (`:1214-1219`); (2) markers present AND `distanceToLastMarkerByte + uncompressedSize >= MAX_WINDOW_SIZE` ⇒ downcast surviving prefix + read, flip (`:1220-1242`); (3) already clean ⇒ bulk `bitReader.read` memcpy into u8 ring ("~400 MB/s → ~6 GB/s", `:1243-1255`). |
| Window-range check only when clean | `gzip/deflate.hpp:1569-1573, 1652-1655` | `distance > m_decodedBytes + nBytesRead ⇒ error` is `if constexpr (!containsMarkerBytes)` — marker mode lets backrefs reach the unknown predecessor (= the marker zone). |

## (b) vendor ChunkData/DecodedData — marker vs clean storage, width transition, apply

| Behavior | Vendor citation | Fact |
|---|---|---|
| Two storages, fixed order | `DecodedData.hpp:222-234` | `dataWithMarkers: vector<FasterVector<uint16_t>>` then `data: vector<VectorView<uint8_t>>` (backed by `dataBuffers`). Append of marker data after clean data THROWS (`:267-271`) — markers are strictly a chunk PREFIX. |
| Marker storage granularity | `DecodedData.hpp:241-275` | `appendToEquallySizedChunks`: 128 KiB-capped equal segments (`ALLOCATION_CHUNK_SIZE = 128_Ki`). |
| Clean storage granularity | `DecodedData.hpp:278-289` | clean views are copied into ONE contiguous `dataBuffers.emplace_back(); reserve(dataSize())` per append — explicitly NOT 128 KiB-segmented (comment `:278-281`; this is the segmented_buffer.rs scar's ground truth). |
| **applyWindow: one in-place u16→u8 LUT pass** | `DecodedData.hpp:305-391` | per u16 segment: `target = reinterpret_cast<uint8_t*>(chunk.data()); for i: target[i] = LUT[chunk[i]]` — forward in-place narrow (std::transform rejected for ordering, `:327-335, :345-347`). markerCount ≥ 128 Ki ⇒ 64 KiB full LUT (identity [0,256) + window at [32768,65536), `:314-338`); else branchy `MapMarkers` (`MarkerReplacement.hpp:24-46`). Buffers are RETAINED and re-exposed as u8 views (`std::swap(reusedDataBuffers, dataWithMarkers)` + `reinterpret_cast` views, `:365-388`) — zero new allocation, half the segment space wasted (vendor's own @todo `:374-380`). |
| Marker encoding | `MarkerReplacement.hpp:24-42` | value ≤255 literal; ≥32768 ⇒ `window[value - 32768]` (index from OLDEST window byte); (255, 32768) invalid. |
| CRC of marker bytes | `ChunkData.hpp:231, :247-307` | marker data is CRC'd inside `ChunkData::applyWindow` (post-resolve), on the thread-pool post-processing task (`GzipChunkFetcher.hpp:474, :581`). |
| Window extraction | `DecodedData.hpp:394-488` | `getLastWindow/getWindowAt`: walk dataWithMarkers (through MapMarkers) then data; short previous windows padded with leading zeros (`:421-432`). |
| Chunk decode driver (window absent) | `chunkdecoding/GzipChunk.hpp:413-658` | ONE `deflate::Block` loop (`:456-458` seed if window known); per-block-iteration `result.append(bufferViews)` stores u16 or u8 per the Block's current width. |
| **ISAL-build clean-tail handoff** | `chunkdecoding/GzipChunk.hpp:465-467, 520-526` | `#ifdef LIBRAPIDARCHIVE_WITH_ISAL`: once `cleanDataCount >= MAX_WINDOW_SIZE` (32 KiB of u8 emitted by the Block, i.e. post-engine-flip), delegate the chunk REMAINDER to `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>` with `result.getLastWindow({})`. Window-known chunks go to ISA-L immediately (`:440-444`). **Vendor itself is a flip-hybrid in the ISAL build.** |
| **Non-ISAL build: no fork at all** | same `#ifdef`s compiled out | window-known chunks run the SAME Block loop seeded (`:456-458`); window-absent chunks flip INSIDE Block and continue u8-direct in the same loop. ONE engine end-to-end. The exact-offset (index) path uses `decodeChunkWithInflateWrapper<ZlibInflateWrapper>` (`:190-268, :702-712`). |

## (c) gzippy — Block + sinks + flip sites + apply

| Behavior | gzippy citation | Status vs vendor |
|---|---|---|
| Dual-width ring, one allocation | `marker_inflate.rs:232-243, 301` (`RING_SIZE = 65536` u16, `U8_RING_SIZE = 131072` u8 over the same `Box<[u16; RING_SIZE]>`) | CONVERGED (vendor deflate.hpp:805-806, 890-894). |
| Marker pre-init zone | `marker_inflate.rs:245-269` (`init_marker_zone`) | CONVERGED (deflate.hpp:875-888). |
| Width selector + counter | `marker_inflate.rs:306-339` (`ring_pos`, `contains_marker_bytes`, `distance_to_last_marker_byte`) | CONVERGED. |
| Flip arming | `marker_inflate.rs:1152-1171` (`just_flipped`: `>= RING_SIZE` OR `>= MAX_WINDOW_SIZE && == decoded_bytes`) | CONVERGED (deflate.hpp:1282-1284). |
| Flip mechanics | `marker_inflate.rs:827-895` (`drain_transition_narrow_u16` + `flip_repack_to_u8`: scratch downcast → u8-view tail, cursor re-based to `U8_RING_SIZE` ≡ vendor `m_windowPosition = 0`) | CONVERGED in effect; seam emission differs (DIV-4). |
| Pre-decode window seed | `marker_inflate.rs:710-773` (`set_initial_window_impl`: seed u8 view [0,len), clean mode from byte 0) | CONVERGED (deflate.hpp:1750-1759). |
| Bulk clean emit width | `marker_inflate.rs:1376-2070` (`read_internal_compressed_specialized::<false>` writes `ring8 % U8_RING_SIZE`); post-flip drain is a plain u8 copy (`drain_to_output` clean branch, `:792-813`) | CONVERGED — the historic "ring stays u16 for the bulk" divergence (memory `project_engine_plateau_pure_rust`) is CLOSED in code. |
| Marker fast loop | `marker_inflate.rs:1700-1830` (u16 mirror of vendor `readInternalCompressedMultiCached`, deflate.hpp:1585-1666) | CONVERGED (landed for the 1.69× window-absent decode gap). |
| Marker storage | `segmented_markers.rs` (`SegmentedU16`, 128 KiB segments, in-place fused resolve+narrow) | CONVERGED (DecodedData.hpp:241-275, 305-391) with a documented safe-Rust deviation (resolve into the low half of the same segment instead of UB `Vec` transmute). |
| Clean storage | `segmented_buffer.rs` (`SegmentedU8`, ONE contiguous buffer; module header records the segmentation scar) | CONVERGED (DecodedData.hpp:278-289). |
| applyWindow | `apply_window.rs:22-45`; production fused path `chunk_fetcher.rs:2745-2758` (`resolve_and_narrow_markers_in_place`, 64 KiB u8 LUT, one pass, `update_narrowed_crc` on the pool task, zero-copy narrowed emit) | CONVERGED (DecodedData.hpp:305-391; CRC placement matches ChunkData.hpp:231). |
| Chunk driver fork at 32 KiB clean | `gzip_chunk.rs:1962-2007` (`clean_appended_len() >= MAX_WINDOW_SIZE`: isal build ⇒ `FlipToClean` → ISA-L tail; native ⇒ `FlipToContig` → resume SAME thread-local Block) | CONVERGED with vendor-ISAL (GzipChunk.hpp:520-526); native continuation is same-engine (vendor-native has no exit, control-flow-only difference). |
| Native clean bulk destination | `gzip_chunk.rs:1399-1560` (`finish_decode_chunk_contig_native`) → `marker_inflate.rs:2071+` (`decode_clean_into_contig`: u8-direct into contiguous `chunk.data`, backrefs from `base[pos-d]`, no ring, no drain) | DELIBERATE KEPT DEVIATION (DIV-2): vendor writes the clean bulk into the u8 ring then COPIES views into contiguous buffers (DecodedData.hpp:282-289) — two touches; gzippy does one. |
| Window-seeded chunks (native) | `gzip_chunk.rs:1107-1141` → `finish_decode_chunk_impl:822-897` → `StreamingInflateWrapper` = `inflate::unified::Inflate<Clean,Generic,Streaming>` (`inflate_wrapper.rs:153-161`) | **DIVERGED (DIV-1)** — a second engine where vendor-native uses the same Block. |
| Legacy engines | `lut_bulk_inflate.rs:731` (`MarkerRing`, env-gated `GZIPPY_MARKER_RING=1`), `marker_inflate.rs:2676` (canonical fallback), `decode_bypass.rs` | **DIVERGED (DIV-3)** — vendor compiles exactly ONE readInternal family (deflate.hpp:1451-1463). |

---

## (d) DIVERGENCE LIST (rewrite targets, in priority order)

**DIV-1 — Second clean engine on window-seeded / clean-tail paths (native build).**
gzippy-native routes window-seeded chunks (and the until-exact paths) through
`StreamingInflateWrapper` backed by `inflate::unified::Inflate` (inflate_wrapper.rs:153-161,
gzip_chunk.rs:822-897) — a wholly separate decoder from `marker_inflate::Block`. Vendor's
non-ISAL build seeds the SAME `deflate::Block` (`block->setInitialWindow`, GzipChunk.hpp:456-458;
the ISA-L fork at :440-444 is compiled out) and decodes u8-direct in the one engine. Only the
index/exact path uses a wrapper engine in vendor (ZlibInflateWrapper, GzipChunk.hpp:190-268) —
and that is C-FFI, which gzippy bans, so even there the convergent target is Block-with-
exact-stop. **Rewrite target: seeded + until-exact native chunks run `Block` (seed →
clean-from-byte-0 → `decode_clean_into_contig`), delete `unified::Inflate` from the native
chunk graph.** This is the largest unification: it puts ALL native decode instructions on the
one engine that P3's arsenal will optimize.

**DIV-2 — Clean bulk destination: ring+copy (vendor) vs contig-direct (gzippy). KEPT.**
Vendor: u8 ring write + memcpy into contiguous DecodedData buffers (deflate.hpp:1292 +
DecodedData.hpp:282-289). gzippy-native: `decode_clean_into_contig` writes once, backrefs
resolve from the output buffer itself. One store pass vs two; banked sha-exact win
(gzip_chunk.rs:1150-1169). Meets-or-beats clause applies — keep, but the design (engine-u8-design.md)
makes it the SINGLE clean-destination contract so DIV-1's unification lands on it too.

**DIV-3 — Engine multiplicity / dead alternates.**
Legacy `MarkerRing` (lut_bulk_inflate.rs:731, `GZIPPY_MARKER_RING=1` A/B only),
`read_internal_compressed_canonical` (marker_inflate.rs:2676), `read_clean_e234`
(marker_inflate.rs:2488), `decode_bypass.rs`. Vendor ships ONE compiled readInternal family.
Rewrite target: delete `MarkerRing` + the env fork (`marker_decode_step_marker_ring`,
gzip_chunk.rs:1786-1856) and the unused clean variants once DIV-1 lands; canonical loop stays
only if a non-LUT fallback is provably required (vendor has none).

**DIV-4 — Seam emission temp copy.**
At the flip, gzippy narrows the seam call's bytes through a temp `Vec` (`drain_transition_narrow_u16`,
marker_inflate.rs:834-850). Vendor emits them as u8 views into the just-conflated window — zero
temp (deflate.hpp:1285-1286: `result.data = lastBuffers(window, m_windowPosition, nBytesRead)`
after `setInitialWindow()`). Once-per-chunk, ≤64 Ki elements; low priority but free to fix when
the flip moves into the WidthRing core (conflate already holds the bytes).

**DIV-5 — Vendor's stored-block early-flip special cases are unported.**
Vendor `read()` flips on UNCOMPRESSED blocks via three special cases (deflate.hpp:1212-1256):
≥-window stored block ⇒ straight u8 read + flip; markers + `distanceToLastMarkerByte +
uncompressedSize >= MAX_WINDOW_SIZE` ⇒ downcast + flip; already-clean ⇒ bulk byte read
(~6 GB/s vs per-byte). gzippy `read_internal_uncompressed` (marker_inflate.rs:1196-1260) writes
per-byte and relies only on the GENERIC arming (which needs 65536 clean, not vendor's 32768 for
the stored case) — correct output, later flip + slower stored path. Rewrite target for the
WidthRing migration (M2/M3); matters for stored-heavy corpora (bgzf-like, igzip -0 shapes).

**DIV-6 — (verify-only) `drain_clean_u8` bench helper reads the u16 ring in clean mode**
(marker_inflate.rs:910-926 indexes `% RING_SIZE` while clean mode is u8-logical `% U8_RING_SIZE`).
Bench-only (`NOT used by production routing` per its own doc), but it is WRONG post-flip and
should be deleted with DIV-3's cleanup rather than fixed.

### Explicitly NOT divergences (verified converged; do not re-litigate)
- Flip arming condition, marker encoding/MapMarkers, marker pre-init zone, 128 KiB u16
  segmentation, contiguous clean storage, fused one-pass apply + CRC-on-pool, window-range
  check only-when-clean, ISA-L clean-tail handoff threshold and window source (vendor is the
  same flip-hybrid in its ISAL build, GzipChunk.hpp:520-526).
- The chunk-level "decode-all-then-apply" shape: on marker-heavy chunks where the flip never
  arms, BOTH vendor and gzippy marker-decode the whole chunk u16 and resolve in ONE apply pass —
  that shape already exists and is not a rework target (see design doc §3).
