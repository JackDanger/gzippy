# Page-warmth root cause — perf-localized fault site (owner turn, HEAD f80294ae)

## Pre-registered falsifier (set BEFORE any fix)
A fix succeeds ONLY if BOTH hold on the locked guest, interleaved, sha-exact 028bd002…cb410f:
1. **page-faults DROP** toward rg's ~56K (perf stat, file sink, p8) — i.e. gzippy < ~85K (closing ≥half the 110K→56K gap).
2. **matched gz_null/rg_null wall MOVES toward 1.0×** (Battery: both tools → /dev/null, interleaved, N≥13). Current 1.16×.
A change that drops faults but does NOT move the matched wall is a TIE on the wall (kept 7a if byte-exact), NOT progress.
A change that moves neither REFUTES the page-warmth thesis → re-localize before more arena work.

## What was source-verified + measured THIS turn
### Baseline (perf stat, locked guest REDACTED_IP, file sink, p8, native sha 028bd002…cb410f)
- gzippy DEFAULT: **110,617 page-faults / 1182 ms task-clock**
- gzippy GZIPPY_SLAB_ALLOC=1: 99,881 faults (−10%, modest)
- gzippy GZIPPY_MANUAL_BUFFER_POOL=1: 120,400 faults (WORSE — lock+late-recycle)
- gzippy MANUAL+SLAB: 106,622
- **rapidgzip: 55,790 faults / 619 ms** ⇒ gzippy = **1.98× rg faults**, root cause reproduced.
- All shas 028bd002…cb410f (byte-exact).

### DECISIVE: perf record -e page-faults --call-graph fp (symboled release build, sha-exact)
Top fault sites (SELF%):
- **`SegmentedU16::push_slice` — 44.52% of faults, 43.95% SELF** ← the u16 MARKER buffer write, the dominant fault site.
- `decode_huffman_body_resumable` 18.80% / 18.52% self ← the clean-tail decode (SECONDARY).
- `__memset_avx2_unaligned_erms` 14.27% ← zeroing freshly-faulted cold pages.
- `__memmove_avx` 7.26% ← grow/drain copies.
- `ChunkData::finalize_with_deflate` 6.49% self.
- `_int_malloc/sysmalloc/__libc_malloc` ~3.26% ← non-rpmalloc (glibc) allocs on the path (scratch Vecs / global allocator is glibc).

### Routing (GZIPPY_VERBOSE)
flip_to_clean=12 finished_no_flip=4 window_seeded=2; in-flight depth=14; Buffer pool hits=0 misses=0 (manual pool OFF = vendor default; segments freed via rpmalloc cross-thread free).

## Root cause (perf-localized + source-verified)
The dominant page-fault load is **cold first-touch of FRESH u16 marker segments**, NOT the clean tail:
- gzippy decodes the window-absent prefix (and the 4 `finished_no_flip` chunks ENTIRELY) into `SegmentedU16` (`data_with_markers`) as u16 = **2 bytes/symbol = double footprint**.
- `take_marker_segment` with the manual pool OFF (vendor-faithful default) allocates a FRESH rpmalloc 128 KiB segment every call — NO warm reuse, because:
  - The ChunkData is dropped on the CONSUMER thread (cross-thread free → rpmalloc deferred list on the worker's heap).
  - With depth=14 in flight, the worker re-allocates MANY chunks before its own freed segments return ⇒ always fresh mmap ⇒ cold fault. This is exactly vendor's documented `ChunkData.hpp:24` "memory slab reuse issues in rpmalloc".

### What is already faithful (do NOT touch)
- `resolve_and_narrow_in_place` / `narrow_markers_to_u8_in_place` — gzippy ALREADY narrows u16→u8 IN PLACE reusing the same warm segment pages (port of `DecodedData.hpp:344-388` reinterpret_cast). The advisor's "gzippy decodes clean tail into a separate cold chunk.data" is only the SECONDARY 18% term; the PRIMARY is the cold u16 marker WRITE itself.
- `SegmentedU8` (clean data) is contiguous-per-chunk by design — segmenting it to 128 KiB was already proven to REGRESS (3.26× DTLB-walks, segmented_buffer.rs:17-21). Do NOT re-segment clean data. Vendor (`DecodedData.hpp:278-281`) explicitly keeps clean data contiguous.

## How rapidgzip avoids it (the existence proof)
rg's `MarkerVector = FasterVector<uint16_t>` segments + `reusedDataBuffers` keep marker buffers warm: after `applyWindow` narrows in place, the buffer is SWAPPED into `reusedDataBuffers` and reused by the SAME thread's next decode (DecodedData.hpp:344-388) — the decoder reuses its OWN thread's warm buffers BEFORE any cross-thread handoff, sidestepping the rpmalloc cross-thread issue. Also rg delegates the clean tail to ISA-L once 32 KiB exists, so its per-chunk u16 footprint is SMALL.

## Proposed faithful fix (to be advisor-vetted BEFORE implementation)
Mirror rg's `reusedDataBuffers`: a per-WORKER (thread-local, lock-free LIFO) warm marker-segment cache that the WORKER reuses directly on its next decode, recycled at decode-completion on the WORKER thread (not the consumer), so freed warm segments are reused before being faulted cold — without the mutex-pool's contention/late-recycle that made MANUAL worse.
SECONDARY (if primary insufficient): reduce the u16 footprint for `finished_no_flip` chunks (they decode ~13 MiB as u16 for chunks that are mostly clean) — investigate why 4/16 never flip.

## Claims for the disproof advisor
- D1: The PRIMARY fault site is `SegmentedU16::push_slice` (44%), not the clean-tail chunk.data (18%) — the advisor verdict's "separate cold chunk.data" mechanism mislocates the primary term.
- D2: The cold-fault cause is fresh-per-chunk u16 segments with no warm reuse (cross-thread free + depth-14 in flight), reproduced: manual pool WORSE, slab only −10%.
- D3: The faithful fix is rg's thread-local reusedDataBuffers (worker-side warm reuse), NOT the cross-worker mutex pool (tested worse) and NOT re-segmenting clean data (proven regression).
- D4: Falsifier is correctly two-pronged (faults↓ AND matched gz_null wall↓); a faults-only win is a wall TIE.
