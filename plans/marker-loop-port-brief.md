# Disproof-advisor brief — marker-loop port (commit f4c58fba)

You are an independent disproof advisor. Read-only. Your job is to BREAK the
findings below, not ratify them. Source-verify every premise first-hand against
the cited file:line. NO build, NO measure.

## The change (commit f4c58fba)
Added a speculative software-pipelined FAST LOOP to gzippy's u16 MARKER decode
path (`read_internal_compressed_specialized::<true>` in
src/decompress/parallel/marker_inflate.rs, the new `if CONTAINS_MARKERS && ...
'mfast: loop`), mirroring the clean path's existing fast loop. Claimed as a
faithful port of rg's `readInternalCompressedMultiCached`
(vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp:1585-1666).

Three u16 deltas vs the clean fast loop:
1. Literal store widened to u16 via an 8-byte speculative store
   `(p&0xFF)|((p&0xFF00)<<8)|((p&0xFF0000)<<16)` into the u16 ring.
2. `distance_marker += lit_prefix` per packet; back-refs via
   `emit_backref_ring::<true>` (marker scan maintained inside).
3. No `distance > decoded+emitted` window-range check (vendor const-folds it for
   marker windows, deflate.hpp:1652-1655).

## CLAIMS to break

### C1 — BYTE-EXACT + faithful (not a divergence)
- silesia sha 028bd002…cb410f on gzippy-native (arm64) AND gzippy-isal (guest
  x86_64) at T1/T8/T16, path=ParallelSM. 856 lib tests (1 fail = pre-existing
  flaky diff_ratio timing micro-test). Seam + native_fold_parity green.
- The speculative widened-u16 store is value-identical to the careful loop's
  per-literal `ring_ptr.add(pos%RING_SIZE).write(code & 0xFF)`.
- BREAK IT: is the widening correct (b0→slot0, b1→slot1, b2→slot2, slot3=0)? Is
  the 8-byte store ever able to straddle the ring wrap given `dst_phys +
  FAST_OUT_SLOP <= RING_SIZE` with FAST_OUT_SLOP=282 u16 slots? Does the
  fall-through to the careful loop on `break 'mfast` desync the bit cursor (pre
  is decoded-but-unconsumed)? Does dropping the window-range check in the fast
  loop diverge from the careful loop (which ALSO const-folds it out for markers)?
  Does `emit_backref_ring::<true>` maintain `distance_marker` identically whether
  reached from the fast or careful loop?

### C2 — WALL is a TIE (the change did NOT move the T8 wall)
- Locked guest, 16c gov=perf turbo-on, taskset 0,2,4,6,8,10,12,14, T8,
  measure.sh interleaved N=11 RAW=68229982 sha-OK. markerfast vs mergefix(HEAD):
  +1.2% / +3.0% / +0.0% across 3 runs = TIE (spread 10-38%).
- BREAK IT: is a TIE the honest verdict, or could a real win be noise-masked at
  this spread? Is interleaved measure.sh freq-neutral?

### C3 — MECHANISM: the marker u16 path is only ~2% of decode body bytes
- Trace: `post_flip_u16_bytes=1425448 (2.0% of body)`, flip_to_clean=12
  finished_no_flip=4 (T8, silesia). decodeBlock 0.9568→0.9485s (~0.9%).
- THE BIG CLAIM: the charter's "decodeBlock 1.69× gap = the marker loop" was a
  MIS-ATTRIBUTION — the decodeBlock SUM is dominated by the CLEAN u8 decode
  (already fast-looped), NOT the ~2% u16 marker prefix. So the marker loop port
  has a ~2%-of-body ceiling and the TIE is expected, not a failed port.
- BREAK IT: does `post_flip_u16_bytes` actually count u16-marker-decoded bytes,
  or is it (like the prior BOOTSTRAP_POST_FLIP_U16_BYTES naming bug) inverted /
  mislabeled? Source-verify the counter's increment site. Is 2% consistent with
  the charter's "34.5% replaced markers" (which is an apply_window REPLACED-marker
  count in the FINAL output, a different quantity than decode-time u16 bytes)? If
  the counter is sound, does the ~2% ceiling correctly explain the TIE?

## What to deliver
Write your verdict to plans/marker-loop-port-advisor-verdict.md. For each claim:
UPHELD / UPHELD-WITH-CAVEATS / REFUTED, with the file:line evidence. Flag the
single most load-bearing correction. Be adversarial.
